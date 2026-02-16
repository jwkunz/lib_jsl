use std::f64::consts::PI;

use ndarray::Array1;
use ndarray_linalg::Solve;
use num::Complex;

use crate::prelude::{C2D, ErrorsJSL, R2D};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemezFilterType {
    Bandpass,
    Differentiator,
    Hilbert,
}

fn solve_linear(a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, ErrorsJSL> {
    let n = b.len();
    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return Err(ErrorsJSL::InvalidInputRange(
            "solve_linear expects a square matrix with matching rhs length",
        ));
    }

    let mut ar = R2D::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            ar[[i, j]] = a[i][j];
        }
    }
    let br = Array1::from_vec(b.clone());
    if let Ok(xr) = ar.solve_into(br) {
        return Ok(xr.to_vec());
    }

    // Fallback to complex solve to improve robustness near singular systems.
    let mut ac = C2D::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            ac[[i, j]] = Complex::new(a[i][j], 0.0);
        }
    }
    let bc = Array1::from_vec(b.drain(..).map(|v| Complex::new(v, 0.0)).collect());
    let xc = ac.solve_into(bc).map_err(|_| {
        ErrorsJSL::RuntimeError("Singular/ill-conditioned system while solving remez equations")
    })?;
    Ok(xc.iter().map(|z| z.re).collect())
}

fn find_extrema_indices(err: &[f64], needed: usize) -> Vec<usize> {
    if err.is_empty() {
        return vec![];
    }
    let mut cand = Vec::new();
    cand.push(0usize);
    for i in 1..(err.len() - 1) {
        let a = err[i - 1].abs();
        let b = err[i].abs();
        let c = err[i + 1].abs();
        if b >= a && b >= c {
            cand.push(i);
        }
    }
    cand.push(err.len() - 1);
    cand.sort_unstable();
    cand.dedup();

    let mut groups: Vec<usize> = Vec::new();
    for &idx in &cand {
        if groups.is_empty() {
            groups.push(idx);
            continue;
        }
        let last = *groups.last().expect("not empty");
        let s_last = err[last].is_sign_positive();
        let s_idx = err[idx].is_sign_positive();
        if s_last == s_idx {
            if err[idx].abs() > err[last].abs() {
                let n = groups.len();
                groups[n - 1] = idx;
            }
        } else {
            groups.push(idx);
        }
    }

    if groups.len() > needed {
        while groups.len() > needed {
            let mut min_i = 0usize;
            let mut min_v = err[groups[0]].abs();
            for (i, &g) in groups.iter().enumerate().skip(1) {
                let v = err[g].abs();
                if v < min_v {
                    min_v = v;
                    min_i = i;
                }
            }
            groups.remove(min_i);
        }
    } else if groups.len() < needed {
        let mut extra = cand.clone();
        extra.sort_by(|&a, &b| {
            err[b]
                .abs()
                .partial_cmp(&err[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for idx in extra {
            if groups.len() >= needed {
                break;
            }
            if !groups.contains(&idx) {
                groups.push(idx);
            }
        }
        if groups.len() < needed {
            let n = err.len().saturating_sub(1).max(1);
            for i in 0..needed {
                if groups.len() >= needed {
                    break;
                }
                let idx = i * n / (needed - 1).max(1);
                if !groups.contains(&idx) {
                    groups.push(idx);
                }
            }
        }
    }

    groups.sort_unstable();
    groups.dedup();
    while groups.len() < needed {
        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for (i, &e) in err.iter().enumerate() {
            if groups.contains(&i) {
                continue;
            }
            let v = e.abs();
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        groups.push(best_idx);
        groups.sort_unstable();
        groups.dedup();
    }
    if groups.len() > needed {
        groups.truncate(needed);
    }
    groups
}

/// FIR equiripple design using a simplified Remez exchange algorithm.
///
/// This currently implements the real, odd-length, linear-phase (Type I) path,
/// which corresponds to the common SciPy `type="bandpass"` use.
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
    if numtaps % 2 == 0 {
        return Err(ErrorsJSL::InvalidInputRange(
            "current remez implementation supports odd numtaps only",
        ));
    }
    if !fs.is_finite() || fs <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("fs must be finite and > 0"));
    }
    if bands.len() < 2 || bands.len() % 2 != 0 {
        return Err(ErrorsJSL::InvalidInputRange(
            "bands must contain an even number of edge values",
        ));
    }
    if desired.len() * 2 != bands.len() {
        return Err(ErrorsJSL::InvalidInputRange(
            "desired length must equal bands.len()/2",
        ));
    }
    if bands.windows(2).any(|w| w[1] < w[0]) {
        return Err(ErrorsJSL::InvalidInputRange(
            "bands must be monotonic non-decreasing",
        ));
    }
    let nyq = fs * 0.5;
    if bands.iter().any(|&f| f < 0.0 || f > nyq) {
        return Err(ErrorsJSL::InvalidInputRange(
            "band edges must be in [0, fs/2]",
        ));
    }
    if filter_type != RemezFilterType::Bandpass {
        return Err(ErrorsJSL::NotImplementedYet);
    }
    let w = if let Some(w) = weight {
        if w.len() != desired.len() {
            return Err(ErrorsJSL::InvalidInputRange(
                "weight length must equal desired length",
            ));
        }
        w.to_vec()
    } else {
        vec![1.0; desired.len()]
    };
    if w.iter().any(|&x| !x.is_finite() || x <= 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("weights must be finite and > 0"));
    }
    let gd = grid_density.max(4);
    let maxiter = maxiter.max(1);

    let m = (numtaps - 1) / 2; // number of cosine terms excluding the symmetric mirror
    let p = m + 2; // number of extremal points and equations

    let mut grid_f = Vec::<f64>::new();
    let mut grid_d = Vec::<f64>::new();
    let mut grid_w = Vec::<f64>::new();
    let half_n = nyq;
    for b in 0..desired.len() {
        let f0 = bands[2 * b];
        let f1 = bands[2 * b + 1];
        if f1 <= f0 {
            continue;
        }
        let span = (f1 - f0) / half_n;
        let mut npts = (gd as f64 * (m + 1) as f64 * span).ceil() as usize + 1;
        if npts < 2 {
            npts = 2;
        }
        for i in 0..npts {
            let t = i as f64 / (npts - 1) as f64;
            let f = f0 + (f1 - f0) * t;
            if let Some(&last) = grid_f.last()
                && (f - last).abs() < 1e-15
            {
                continue;
            }
            grid_f.push(f);
            grid_d.push(desired[b]);
            grid_w.push(w[b]);
        }
    }
    if grid_f.len() < p {
        return Err(ErrorsJSL::InvalidInputRange(
            "dense grid too small for requested numtaps",
        ));
    }

    let mut ext = (0..p)
        .map(|i| i * (grid_f.len() - 1) / (p - 1))
        .collect::<Vec<_>>();

    let mut a = vec![0.0; m + 1];
    for _ in 0..maxiter {
        let mut mat = vec![vec![0.0; p]; p];
        let mut rhs = vec![0.0; p];
        for i in 0..p {
            let gi = ext[i];
            let f = grid_f[gi] / fs;
            for (k, mk) in mat[i].iter_mut().enumerate().take(m + 1) {
                *mk = (2.0 * PI * f * k as f64).cos();
            }
            let s = if i % 2 == 0 { 1.0 } else { -1.0 };
            mat[i][m + 1] = s / grid_w[gi];
            rhs[i] = grid_d[gi];
        }
        let sol = solve_linear(mat, rhs)?;
        a[..(m + 1)].copy_from_slice(&sol[..(m + 1)]);

        let mut err = vec![0.0; grid_f.len()];
        for i in 0..grid_f.len() {
            let f = grid_f[i] / fs;
            let mut h = 0.0;
            for (k, &ak) in a.iter().enumerate().take(m + 1) {
                h += ak * (2.0 * PI * f * k as f64).cos();
            }
            err[i] = (h - grid_d[i]) * grid_w[i];
        }

        let new_ext = find_extrema_indices(&err, p);
        if new_ext == ext {
            break;
        }
        ext = new_ext;
    }

    let mut h = vec![0.0; numtaps];
    h[m] = a[0];
    for k in 1..=m {
        let v = 0.5 * a[k];
        h[m - k] = v;
        h[m + k] = v;
    }
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_csv(csv: &str) -> Vec<f64> {
        csv.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<f64>().unwrap())
            .collect()
    }

    #[test]
    fn test_remez_lowpass_golden() {
        let taps = remez(
            17,
            &[0.0, 0.2, 0.3, 0.5],
            &[1.0, 0.0],
            None,
            RemezFilterType::Bandpass,
            25,
            16,
            1.0,
        )
        .unwrap();
        let golden = parse_csv(include_str!("../test_data/remez_lowpass_numtaps17.csv"));
        assert_eq!(taps.len(), golden.len());
        for (a, g) in taps.iter().zip(golden.iter()) {
            assert!((a - g).abs() < 1e-8, "actual={a}, golden={g}");
        }
    }

    #[test]
    fn test_remez_highpass_weighted_golden() {
        let taps = remez(
            21,
            &[0.0, 0.14, 0.2, 0.5],
            &[0.0, 1.0],
            Some(&[2.0, 1.0]),
            RemezFilterType::Bandpass,
            25,
            16,
            1.0,
        )
        .unwrap();
        let golden = parse_csv(include_str!("../test_data/remez_highpass_numtaps21_weighted.csv"));
        assert_eq!(taps.len(), golden.len());
        for (a, g) in taps.iter().zip(golden.iter()) {
            assert!((a - g).abs() < 1e-8, "actual={a}, golden={g}");
        }
    }
}
