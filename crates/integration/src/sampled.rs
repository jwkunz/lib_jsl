//! Integration of sampled data, following `scipy.integrate`: the composite trapezoid and Simpson
//! rules for equally or unequally spaced samples, Romberg integration of `2ᵏ + 1` equally spaced
//! samples, and the cumulative trapezoid and Simpson integrals.
//!
//! See the book entry [Integrating sampled data](https://jwkunz.github.io/lib_jsl/book/integration/sampled-data.html).

use lib_jsl_core::ErrorsJSL;

/// The sample positions: explicit abscissae, or a constant spacing.
#[derive(Clone, Copy, Debug)]
pub enum Spacing<'a> {
    /// Equally spaced samples `dx` apart.
    Uniform(f64),
    /// The abscissa of each sample (increasing or decreasing, not necessarily evenly spaced).
    Points(&'a [f64]),
}

fn steps(y: &[f64], spacing: Spacing) -> Result<Vec<f64>, ErrorsJSL> {
    match spacing {
        Spacing::Uniform(dx) => Ok(vec![dx; y.len().saturating_sub(1)]),
        Spacing::Points(x) => {
            if x.len() != y.len() {
                return Err(ErrorsJSL::IncompatibleArraySizes((x.len(), y.len())));
            }
            Ok(x.windows(2).map(|w| w[1] - w[0]).collect())
        }
    }
}

/// The composite trapezoid rule `Σ (yᵢ + yᵢ₊₁) hᵢ / 2` (SciPy's `trapezoid`).
pub fn trapezoid(y: &[f64], spacing: Spacing) -> Result<f64, ErrorsJSL> {
    let h = steps(y, spacing)?;
    Ok(y.windows(2).zip(&h).map(|(w, hi)| 0.5 * hi * (w[0] + w[1])).sum())
}

/// The running trapezoid integral: element `i` integrates over the first `i + 1` samples
/// (SciPy's `cumulative_trapezoid`). With `initial`, that value is prepended and added, so the
/// result has the length of `y`; otherwise it is one shorter.
pub fn cumulative_trapezoid(y: &[f64], spacing: Spacing, initial: Option<f64>) -> Result<Vec<f64>, ErrorsJSL> {
    let h = steps(y, spacing)?;
    let mut out = Vec::with_capacity(y.len());
    let mut acc = initial.unwrap_or(0.0);
    if initial.is_some() {
        out.push(acc);
    }
    for (w, hi) in y.windows(2).zip(&h) {
        acc += 0.5 * hi * (w[0] + w[1]);
        out.push(acc);
    }
    Ok(out)
}

// ANCHOR: simpson
/// Simpson's rule on pairs of intervals, `(h₀ + h₁)/6 [(2 − h₁/h₀) y₀ + (h₀ + h₁)²/(h₀h₁) y₁ +
/// (2 − h₀/h₁) y₂]`, the integral of the parabola through three samples (SciPy's `simpson`). With
/// an even number of samples, the last interval is integrated by the parabola through the last
/// three samples (Cartwright's correction), which keeps it exact for quadratics (but not cubics).
pub fn simpson(y: &[f64], spacing: Spacing) -> Result<f64, ErrorsJSL> {
    let n = y.len();
    let h = steps(y, spacing)?;
    if n < 2 {
        return Ok(0.0);
    }
    if n == 2 {
        return Ok(0.5 * h[0] * (y[0] + y[1]));
    }
    let pair = |i: usize| -> f64 {
        let (h0, h1) = (h[i], h[i + 1]);
        let (hs, hp, r) = (h0 + h1, h0 * h1, h0 / h1);
        hs / 6.0 * (y[i] * (2.0 - 1.0 / r) + y[i + 1] * hs * hs / hp + y[i + 2] * (2.0 - r))
    };
    if n % 2 == 1 {
        return Ok((0..n - 2).step_by(2).map(pair).sum());
    }
    let mut result: f64 = (0..n - 3).step_by(2).map(pair).sum();
    let (h0, h1) = (h[n - 3], h[n - 2]);
    let alpha = (2.0 * h1 * h1 + 3.0 * h0 * h1) / (6.0 * (h1 + h0));
    let beta = (h1 * h1 + 3.0 * h0 * h1) / (6.0 * h0);
    let eta = h1.powi(3) / (6.0 * h0 * (h0 + h1));
    result += alpha * y[n - 1] + beta * y[n - 2] - eta * y[n - 3];
    Ok(result)
}
// ANCHOR_END: simpson

/// The running Simpson integral (SciPy's `cumulative_simpson`): each interval is integrated by
/// the parabola through it and a neighbour (the next interval's for the first of each pair, the
/// previous one's for the second), so every partial integral is exact for quadratics.
pub fn cumulative_simpson(y: &[f64], spacing: Spacing, initial: Option<f64>) -> Result<Vec<f64>, ErrorsJSL> {
    let n = y.len();
    if n < 3 {
        return cumulative_trapezoid(y, spacing, initial);
    }
    let h = steps(y, spacing)?;
    // Integral over [x_i, x_{i+1}] of the parabola through samples i, i+1, i+2.
    let forward = |f1: f64, f2: f64, f3: f64, x21: f64, x32: f64| {
        let x31 = x21 + x32;
        let (a, b) = (x21 / x31, x21 / x32);
        x21 / 6.0 * ((3.0 - a) * f1 + (3.0 + a * b + a) * f2 - a * b * f3)
    };
    let h1: Vec<f64> = (0..n - 2).map(|i| forward(y[i], y[i + 1], y[i + 2], h[i], h[i + 1])).collect();
    // The same on the reversed data: the integral over [x_{i+1}, x_{i+2}] from samples i..i+2.
    let h2: Vec<f64> = (0..n - 2).map(|i| forward(y[i + 2], y[i + 1], y[i], h[i + 1], h[i])).collect();
    let mut sub = vec![0.0; n - 1];
    for i in 0..n - 1 {
        sub[i] = if i == n - 2 {
            h2[n - 3]
        } else if i % 2 == 0 {
            h1[i]
        } else {
            h2[i - 1]
        };
    }
    let mut out = Vec::with_capacity(n);
    let mut acc = initial.unwrap_or(0.0);
    if initial.is_some() {
        out.push(acc);
    }
    for s in sub {
        acc += s;
        out.push(acc);
    }
    Ok(out)
}

// ANCHOR: romb
/// Romberg integration of `2ᵏ + 1` equally spaced samples (SciPy's `romb`): trapezoid sums on
/// the nested grids of spacing `h, 2h, 4h, …`, combined by Richardson extrapolation
/// `R(i, j) = R(i, j−1) + (R(i, j−1) − R(i−1, j−1)) / (4ʲ − 1)`.
pub fn romb(y: &[f64], dx: f64) -> Result<f64, ErrorsJSL> {
    let intervals = y.len().saturating_sub(1);
    if intervals == 0 || !intervals.is_power_of_two() {
        return Err(ErrorsJSL::InvalidInputRange("romb needs 2^k + 1 samples"));
    }
    let k = intervals.trailing_zeros() as usize;
    let mut h = intervals as f64 * dx;
    let mut prev = vec![(y[0] + y[intervals]) / 2.0 * h];
    let (mut start, mut step) = (intervals, intervals);
    for i in 1..=k {
        start >>= 1;
        let s: f64 = (start..intervals).step_by(step).map(|j| y[j]).sum();
        step >>= 1;
        let mut row = vec![0.5 * (prev[0] + h * s)];
        for j in 1..=i {
            let p = row[j - 1];
            row.push(p + (p - prev[j - 1]) / ((1u64 << (2 * j)) - 1) as f64);
        }
        prev = row;
        h /= 2.0;
    }
    Ok(prev[k])
}
// ANCHOR_END: romb

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simpson_is_exact_for_cubics_and_romb_for_high_degree() {
        let x: Vec<f64> = (0..=8).map(|i| i as f64 * 0.25).collect();
        let y: Vec<f64> = x.iter().map(|v| v * v * v - 2.0 * v).collect();
        assert!((simpson(&y, Spacing::Uniform(0.25)).unwrap() - (4.0 - 4.0)).abs() < 1e-14);
        // An even number of samples: exact for quadratics; SciPy gives 2.345703125 for x³.
        let x: Vec<f64> = (0..=7).map(|i| i as f64 * 0.25).collect();
        let y: Vec<f64> = x.iter().map(|v| v * v).collect();
        assert!((simpson(&y, Spacing::Points(&x)).unwrap() - 1.75f64.powi(3) / 3.0).abs() < 1e-13);
        let y: Vec<f64> = x.iter().map(|v| v * v * v).collect();
        assert!((simpson(&y, Spacing::Points(&x)).unwrap() - 2.345703125).abs() < 1e-13);
        let x: Vec<f64> = (0..=16).map(|i| i as f64 / 16.0).collect();
        let y: Vec<f64> = x.iter().map(|v| v.powi(8)).collect();
        assert!((romb(&y, 1.0 / 16.0).unwrap() - 1.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn cumulative_rules_end_at_the_full_integral() {
        let x: Vec<f64> = (0..11).map(|i| (i as f64 * 0.3).powf(1.2)).collect();
        let y: Vec<f64> = x.iter().map(|v| v * v).collect();
        let c = cumulative_simpson(&y, Spacing::Points(&x), Some(0.0)).unwrap();
        for (ci, xi) in c.iter().zip(&x) {
            assert!((ci - xi.powi(3) / 3.0).abs() < 1e-12, "{ci} vs {}", xi.powi(3) / 3.0);
        }
        let t = cumulative_trapezoid(&y, Spacing::Points(&x), None).unwrap();
        assert!((t.last().unwrap() - trapezoid(&y, Spacing::Points(&x)).unwrap()).abs() < 1e-14);
    }
}
