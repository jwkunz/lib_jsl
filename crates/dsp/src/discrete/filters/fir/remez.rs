use std::f64::consts::PI;

use num_complex::Complex;

use lib_jsl_core::ErrorsJSL;

/// The kind of response `remez` designs, as in SciPy's `type` argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemezFilterType {
    /// A symmetric (linear-phase) filter with piecewise-constant desired gains.
    Bandpass,
    /// An antisymmetric filter whose desired response in each band is `desired · f`; bands with
    /// a non-zero slope are weighted by `1/f`, so the error is relative.
    Differentiator,
    /// An antisymmetric filter with piecewise-constant desired gains: a Hilbert transformer.
    Hilbert,
}

/// The dense frequency grid, with the desired response and weight at each point, already divided
/// and multiplied by the factor `Q(f)` that turns each filter type into a cosine polynomial.
struct Grid {
    f: Vec<f64>,
    desired: Vec<f64>,
    weight: Vec<f64>,
    /// The index of the band each grid point lies in.
    band: Vec<usize>,
}

/// `Q(f)` with `H(f) = Q(f) P(f)` and `P` a cosine polynomial: 1 for odd symmetric filters,
/// `cos πf` for even symmetric, `sin 2πf` for odd antisymmetric and `sin πf` for even antisymmetric.
fn q_factor(f: f64, odd: bool, antisymmetric: bool) -> f64 {
    match (odd, antisymmetric) {
        (true, false) => 1.0,
        (false, false) => (PI * f).cos(),
        (true, true) => (2.0 * PI * f).sin(),
        (false, true) => (PI * f).sin(),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_grid(edges: &[f64], desired: &[f64], weight: &[f64], kind: RemezFilterType, odd: bool, functions: usize, density: usize) -> Grid {
    let antisymmetric = kind != RemezFilterType::Bandpass;
    let delf = 0.5 / (density * functions) as f64;
    let mut grid = Grid { f: Vec::new(), desired: Vec::new(), weight: Vec::new(), band: Vec::new() };
    for b in 0..desired.len() {
        let (mut lo, hi) = (edges[2 * b], edges[2 * b + 1]);
        // Antisymmetric responses vanish at f = 0, so the grid starts one step in.
        if b == 0 && antisymmetric && lo < delf {
            lo = delf;
        }
        let mut f = lo;
        loop {
            let at_edge = f >= hi;
            let x = if at_edge { hi } else { f };
            grid.f.push(x);
            grid.band.push(b);
            if at_edge {
                break;
            }
            f += delf;
        }
    }
    // Types II and III vanish at f = 1/2: keep the last point one step below it.
    let last = grid.f.len() - 1;
    if antisymmetric == odd && grid.f[last] > 0.5 - delf {
        grid.f[last] = 0.5 - delf;
    }
    for i in 0..grid.f.len() {
        let (f, b) = (grid.f[i], grid.band[i]);
        let (mut d, mut w) = (desired[b], weight[b]);
        if kind == RemezFilterType::Differentiator {
            d *= f;
            if desired[b] >= 1e-4 {
                w /= f;
            }
        }
        let q = q_factor(f, odd, antisymmetric);
        grid.desired.push(d / q);
        grid.weight.push(w * q);
    }
    grid
}

/// Barycentric weights `1 / Π_{j≠k} (x_k − x_j)`, scaled by a common factor to stay finite.
fn barycentric_weights(x: &[f64]) -> Vec<f64> {
    let logs: Vec<(f64, f64)> = (0..x.len())
        .map(|k| {
            let (mut log, mut sign) = (0.0, 1.0);
            for (j, &xj) in x.iter().enumerate() {
                if j != k {
                    let d = x[k] - xj;
                    log -= d.abs().ln();
                    sign *= d.signum();
                }
            }
            (log, sign)
        })
        .collect();
    let top = logs.iter().map(|l| l.0).fold(f64::NEG_INFINITY, f64::max);
    logs.into_iter().map(|(l, s)| s * (l - top).exp()).collect()
}

/// The polynomial through `(x_k, y_k)` evaluated at `x` by the barycentric formula.
fn barycentric(x: f64, nodes: &[f64], weights: &[f64], values: &[f64]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for ((&xk, &wk), &yk) in nodes.iter().zip(weights).zip(values) {
        let d = x - xk;
        if d == 0.0 {
            return yk;
        }
        num += wk * yk / d;
        den += wk / d;
    }
    num / den
}

/// The new reference set: alternating local extrema of the weighted error with `|E| ≥ floor`,
/// trimmed from the ends (dropping the smaller) to exactly `needed` points.
fn alternating_extrema(err: &[f64], band: &[usize], floor: f64, needed: usize) -> Option<Vec<usize>> {
    let n = err.len();
    let mut cand: Vec<usize> = Vec::new();
    // A local maximum of E where E > 0, or a local minimum where E < 0 (band edges count).
    let beats = |i: usize, j: usize| if err[i] > 0.0 { err[i] >= err[j] } else { err[i] <= err[j] };
    for i in 0..n {
        let left = i == 0 || band[i - 1] != band[i] || beats(i, i - 1);
        let right = i == n - 1 || band[i + 1] != band[i] || beats(i, i + 1);
        if left && right && err[i].abs() >= floor {
            // Within a run of one sign keep only the largest error.
            match cand.last() {
                Some(&j) if err[j].signum() == err[i].signum() => {
                    if err[i].abs() > err[j].abs() {
                        *cand.last_mut().expect("non-empty") = i;
                    }
                }
                _ => cand.push(i),
            }
        }
    }
    while cand.len() > needed {
        if err[cand[0]].abs() < err[cand[cand.len() - 1]].abs() {
            cand.remove(0);
        } else {
            cand.pop();
        }
    }
    (cand.len() == needed).then_some(cand)
}

/// Extrema with `|E| ≥ |δ|`, up to rounding; if rounding hides one (when `δ` is tiny), any
/// alternating extrema.
fn new_extremals(err: &[f64], band: &[usize], delta: f64, needed: usize) -> Option<Vec<usize>> {
    alternating_extrema(err, band, delta.abs() * (1.0 - 1e-6), needed).or_else(|| alternating_extrema(err, band, 0.0, needed))
}

/// Equiripple linear-phase FIR design by the Parks-McClellan algorithm (the Remez exchange),
/// following SciPy's `remez`.
///
/// `bands` holds pairs of band edges and `desired` one gain per band (a slope for
/// [`RemezFilterType::Differentiator`]); `weight` scales the error in each band. The filter
/// minimises the largest weighted error over all bands. Frequencies are in the units of `fs`.
/// Odd and even lengths and all four linear-phase types are supported: symmetric for `Bandpass`
/// and antisymmetric for `Differentiator` and `Hilbert`. Each response is written as
/// `Q(f) P(f)` with `P` a cosine polynomial, the exchange runs on `P` over a dense grid of
/// `grid_density` points per extremal frequency, and the taps are recovered from the optimal
/// response by frequency sampling. Iteration stops when the reference set no longer changes, or
/// after `maxiter` exchanges.
#[allow(clippy::too_many_arguments)]
pub fn remez(
    numtaps: usize,
    bands: &[f64],
    desired: &[f64],
    weight: Option<&[f64]>,
    filter_type: RemezFilterType,
    maxiter: usize,
    grid_density: usize,
    fs: f64,
) -> Result<Vec<f64>, ErrorsJSL> {
    if numtaps < 3 {
        return Err(ErrorsJSL::InvalidInputRange("numtaps must be >= 3"));
    }
    if !fs.is_finite() || fs <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("fs must be finite and > 0"));
    }
    if bands.len() < 2 || !bands.len().is_multiple_of(2) {
        return Err(ErrorsJSL::InvalidInputRange("bands must contain an even number of edge values"));
    }
    if desired.len() * 2 != bands.len() {
        return Err(ErrorsJSL::InvalidInputRange("desired length must equal bands.len()/2"));
    }
    if bands.windows(2).any(|w| w[1] < w[0]) {
        return Err(ErrorsJSL::InvalidInputRange("bands must be monotonic non-decreasing"));
    }
    if bands.iter().any(|&f| f < 0.0 || f > 0.5 * fs) {
        return Err(ErrorsJSL::InvalidInputRange("band edges must be in [0, fs/2]"));
    }
    let weight = match weight {
        Some(w) if w.len() != desired.len() => return Err(ErrorsJSL::InvalidInputRange("weight length must equal desired length")),
        Some(w) => w.to_vec(),
        None => vec![1.0; desired.len()],
    };
    if weight.iter().any(|&x| !x.is_finite() || x <= 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("weights must be finite and > 0"));
    }
    let edges: Vec<f64> = bands.iter().map(|b| b / fs).collect();
    let antisymmetric = filter_type != RemezFilterType::Bandpass;
    let odd = numtaps % 2 == 1;
    // Number of cosine functions in P(f).
    let functions = numtaps / 2 + usize::from(odd && !antisymmetric);
    let grid = build_grid(&edges, desired, &weight, filter_type, odd, functions, grid_density.max(1));
    let needed = functions + 1;
    if grid.f.len() < needed {
        return Err(ErrorsJSL::InvalidInputRange("dense grid too small for requested numtaps"));
    }
    let x: Vec<f64> = grid.f.iter().map(|f| (2.0 * PI * f).cos()).collect();

    // ANCHOR: remez_exchange
    // Start from reference points spread evenly over the grid.
    let mut ext: Vec<usize> = (0..needed).map(|i| i * (grid.f.len() - 1) / (needed - 1)).collect();
    let mut interpolant: (Vec<f64>, Vec<f64>, Vec<f64>);
    let mut iterations = 0;
    loop {
        iterations += 1;
        // Levelled error δ on the reference: P(x_k) = D_k − (−1)^k δ / W_k for all k.
        let xs: Vec<f64> = ext.iter().map(|&i| x[i]).collect();
        let b = barycentric_weights(&xs);
        let (mut num, mut den) = (0.0, 0.0);
        for (k, &i) in ext.iter().enumerate() {
            let s = if k % 2 == 0 { 1.0 } else { -1.0 };
            num += b[k] * grid.desired[i];
            den += b[k] * s / grid.weight[i];
        }
        let delta = num / den;
        // Interpolate P through all but the last reference point.
        let nodes = xs[..functions].to_vec();
        let values: Vec<f64> = ext[..functions]
            .iter()
            .enumerate()
            .map(|(k, &i)| grid.desired[i] - if k % 2 == 0 { 1.0 } else { -1.0 } * delta / grid.weight[i])
            .collect();
        let weights = barycentric_weights(&nodes);
        let err: Vec<f64> = (0..grid.f.len()).map(|i| grid.weight[i] * (grid.desired[i] - barycentric(x[i], &nodes, &weights, &values))).collect();
        interpolant = (nodes, weights, values);
        match new_extremals(&err, &grid.band, delta, needed) {
            Some(next) if next != ext && iterations < maxiter.max(1) => ext = next,
            _ => break,
        }
    }
    // ANCHOR_END: remez_exchange

    // Frequency sampling: the response at f_j = j/N determines the N taps exactly.
    let n = numtaps;
    let centre = (n as f64 - 1.0) / 2.0;
    let (nodes, weights, values) = interpolant;
    let response: Vec<Complex<f64>> = (0..n)
        .map(|j| {
            let f = j as f64 / n as f64;
            let amplitude = q_factor(f, odd, antisymmetric) * barycentric((2.0 * PI * f).cos(), &nodes, &weights, &values);
            let phase = Complex::from_polar(1.0, -2.0 * PI * f * centre);
            if antisymmetric { Complex::new(0.0, 1.0) * phase * amplitude } else { phase * amplitude }
        })
        .collect();
    Ok((0..n)
        .map(|t| (0..n).map(|j| response[j] * Complex::from_polar(1.0, 2.0 * PI * (j * t) as f64 / n as f64)).sum::<Complex<f64>>().re / n as f64)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_csv(csv: &str) -> Vec<f64> {
        csv.lines().filter(|l| !l.trim().is_empty()).map(|l| l.trim().parse::<f64>().unwrap()).collect()
    }

    /// Compares with taps from SciPy 1.17.1's `remez` (in `test_data`).
    fn check(taps: Vec<f64>, csv: &str, tol: f64) {
        let golden = parse_csv(csv);
        assert_eq!(taps.len(), golden.len());
        let scale = golden.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (a, g) in taps.iter().zip(golden.iter()) {
            assert!((a - g).abs() < tol * scale, "actual={a}, scipy={g}");
        }
    }

    #[test]
    fn matches_scipy_for_all_four_filter_types() {
        let lowpass = remez(17, &[0.0, 0.2, 0.3, 0.5], &[1.0, 0.0], None, RemezFilterType::Bandpass, 25, 16, 1.0).unwrap();
        check(lowpass, include_str!("../../test_data/remez_lowpass_numtaps17.csv"), 1e-6);
        let highpass = remez(21, &[0.0, 0.14, 0.2, 0.5], &[0.0, 1.0], Some(&[2.0, 1.0]), RemezFilterType::Bandpass, 25, 16, 1.0).unwrap();
        check(highpass, include_str!("../../test_data/remez_highpass_numtaps21_weighted.csv"), 1e-6);
        let even = remez(20, &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], &[0.0, 1.0, 0.0], None, RemezFilterType::Bandpass, 25, 16, 1.0).unwrap();
        check(even, include_str!("../../test_data/remez_bandpass_numtaps20_even.csv"), 1e-6);
        let hilbert = remez(31, &[0.05, 0.45], &[1.0], None, RemezFilterType::Hilbert, 25, 16, 1.0).unwrap();
        check(hilbert, include_str!("../../test_data/remez_hilbert_numtaps31.csv"), 1e-6);
        let hilbert_even = remez(30, &[0.05, 0.5], &[1.0], None, RemezFilterType::Hilbert, 25, 16, 1.0).unwrap();
        check(hilbert_even, include_str!("../../test_data/remez_hilbert_numtaps30.csv"), 1e-6);
        let differentiator = remez(30, &[0.0, 0.4], &[1.0], None, RemezFilterType::Differentiator, 25, 16, 1.0).unwrap();
        check(differentiator, include_str!("../../test_data/remez_differentiator_numtaps30.csv"), 1e-6);
    }

    #[test]
    fn the_error_is_equiripple() {
        // Lowpass: the passband and stopband errors peak at the same level, and the response
        // stays within it.
        let h = remez(41, &[0.0, 0.2, 0.25, 0.5], &[1.0, 0.0], None, RemezFilterType::Bandpass, 25, 16, 1.0).unwrap();
        let response = |f: f64| h.iter().enumerate().map(|(n, &v)| Complex::from_polar(v, -2.0 * PI * f * n as f64)).sum::<Complex<f64>>().norm();
        let pass = (0..=400).map(|i| (response(0.2 * i as f64 / 400.0) - 1.0).abs()).fold(0.0, f64::max);
        let stop = (0..=400).map(|i| response(0.25 + 0.25 * i as f64 / 400.0)).fold(0.0, f64::max);
        // Equal up to the resolution of the design grid.
        assert!((pass - stop).abs() < 1e-2 * pass, "{pass} {stop}");
    }

    #[test]
    fn validates_its_inputs() {
        assert!(remez(2, &[0.0, 0.5], &[1.0], None, RemezFilterType::Bandpass, 25, 16, 1.0).is_err());
        assert!(remez(11, &[0.0, 0.2, 0.3], &[1.0], None, RemezFilterType::Bandpass, 25, 16, 1.0).is_err());
        assert!(remez(11, &[0.0, 0.3, 0.2, 0.5], &[1.0, 0.0], None, RemezFilterType::Bandpass, 25, 16, 1.0).is_err());
        assert!(remez(11, &[0.0, 0.2, 0.3, 0.6], &[1.0, 0.0], None, RemezFilterType::Bandpass, 25, 16, 1.0).is_err());
        assert!(remez(11, &[0.0, 0.2, 0.3, 0.5], &[1.0, 0.0], Some(&[1.0, -1.0]), RemezFilterType::Bandpass, 25, 16, 1.0).is_err());
    }
}
