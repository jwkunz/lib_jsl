use std::f64::consts::PI;

use super::{extend, len_guard, truncate};

/// The number of eigenvalues of the symmetric tridiagonal matrix (diagonal `d`, off-diagonal `e`)
/// that are smaller than `x`, by Sturm's sign count of the `LDLᵀ` pivots of `T − xI`.
fn count_below(d: &[f64], e: &[f64], x: f64) -> usize {
    let mut count = 0;
    let mut q = d[0] - x;
    for i in 0..d.len() {
        if i > 0 {
            let qp = if q == 0.0 { f64::EPSILON * (e[i - 1].abs() + 1.0) } else { q };
            q = d[i] - x - e[i - 1] * e[i - 1] / qp;
        }
        if q < 0.0 {
            count += 1;
        }
    }
    count
}

/// The discrete prolate spheroidal sequence (Slepian) window with time-half-bandwidth product `nw`:
/// of all sequences of length `m`, the one with the largest fraction of its energy in the band
/// `|f| < nw / m`. It is the eigenvector for the largest eigenvalue of a symmetric tridiagonal
/// matrix that commutes with the concentration problem; that eigenvalue is located by Sturm
/// bisection and the eigenvector found by inverse iteration, as in SciPy's `dpss`.
pub fn dpss(m: usize, nw: f64, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let wband = nw / m2 as f64;
    let c = (2.0 * PI * wband).cos();
    let d: Vec<f64> = (0..m2).map(|n| ((m2 as f64 - 1.0 - 2.0 * n as f64) / 2.0).powi(2) * c).collect();
    let e: Vec<f64> = (1..m2).map(|n| (n as f64) * (m2 as f64 - n as f64) / 2.0).collect();

    // Gershgorin bounds, then bisection for the largest eigenvalue.
    let radius = |i: usize| (if i > 0 { e[i - 1].abs() } else { 0.0 }) + (if i + 1 < m2 { e[i].abs() } else { 0.0 });
    let (mut lo, mut hi) = (0..m2).fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), i| (l.min(d[i] - radius(i)), h.max(d[i] + radius(i))));
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if count_below(&d, &e, mid) < m2 { lo = mid } else { hi = mid }
    }
    let lambda = hi + 4.0 * f64::EPSILON * hi.abs().max(1.0);

    // Inverse iteration: solve (T − λI) v_new = v by the tridiagonal (Thomas) elimination.
    let mut v = vec![1.0_f64; m2];
    for _ in 0..4 {
        let mut cp = vec![0.0; m2];
        let mut dp = vec![0.0; m2];
        let mut b0 = d[0] - lambda;
        if b0 == 0.0 {
            b0 = f64::EPSILON;
        }
        cp[0] = if m2 > 1 { e[0] / b0 } else { 0.0 };
        dp[0] = v[0] / b0;
        for i in 1..m2 {
            let mut den = d[i] - lambda - e[i - 1] * cp[i - 1];
            if den == 0.0 {
                den = f64::EPSILON;
            }
            cp[i] = if i + 1 < m2 { e[i] / den } else { 0.0 };
            dp[i] = (v[i] - e[i - 1] * dp[i - 1]) / den;
        }
        let mut x = vec![0.0; m2];
        x[m2 - 1] = dp[m2 - 1];
        for i in (0..m2 - 1).rev() {
            x[i] = dp[i] - cp[i] * x[i + 1];
        }
        let norm = x.iter().map(|t| t * t).sum::<f64>().sqrt();
        v = x.into_iter().map(|t| t / norm).collect();
    }

    // Match scipy Kmax=None style normalization.
    let maxv = v.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    if maxv > 0.0 {
        for x in &mut v {
            *x /= maxv;
        }
    }
    if m2 % 2 == 0 {
        let correction = (m2 * m2) as f64 / ((m2 * m2) as f64 + nw);
        for x in &mut v {
            *x *= correction;
        }
    }

    if v.iter().sum::<f64>() < 0.0 {
        for x in &mut v {
            *x = -*x;
        }
    }

    truncate(v, needs_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn wide_bandwidths_select_the_most_concentrated_sequence() {
        // With nw / m > 1/4 the matrix has large negative eigenvalues, which power iteration
        // (the previous method) converged to instead. scipy.signal.windows.dpss(7, 2.5):
        let want = [0.06957312, 0.34640869, 0.77714755, 1.0, 0.77714755, 0.34640869, 0.06957312];
        assert!(dpss(7, 2.5, true).iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-8));
    }

    #[test]
    fn test_dpss() {
        assert_window(include_str!("../test_data/dpss.csv"), dpss(7, 2.5, true));
    }
}
