use std::f64::consts::PI;

use super::{extend, len_guard, truncate};

/// The Chebyshev window is defined by a specific mathematical formula that involves the hyperbolic cosine function and the inverse hyperbolic cosine function. The coefficients are computed based on the position within the window, the total length, and the attenuation parameter `at`, which controls the shape of the window and its spectral properties.
pub fn dpss(m: usize, nw: f64, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let wband = nw / m2 as f64;
    let c = (2.0 * PI * wband).cos();
    let mut d = vec![0.0; m2];
    let mut e = vec![0.0; m2 - 1];
    for (n, dn) in d.iter_mut().enumerate() {
        *dn = (((m2 as f64 - 1.0 - 2.0 * n as f64) / 2.0).powi(2)) * c;
    }
    for n in 1..m2 {
        e[n - 1] = (n as f64) * (m2 as f64 - n as f64) / 2.0;
    }

    // Power iteration for dominant eigenvector of symmetric tridiagonal.
    let mut v = vec![1.0_f64; m2];
    let mut norm = (v.iter().map(|x| x * x).sum::<f64>()).sqrt();
    for x in &mut v {
        *x /= norm;
    }
    for _ in 0..200 {
        let mut y = vec![0.0; m2];
        for i in 0..m2 {
            y[i] += d[i] * v[i];
            if i > 0 {
                y[i] += e[i - 1] * v[i - 1];
            }
            if i + 1 < m2 {
                y[i] += e[i] * v[i + 1];
            }
        }
        norm = (y.iter().map(|x| x * x).sum::<f64>()).sqrt();
        if norm == 0.0 {
            break;
        }
        for i in 0..m2 {
            v[i] = y[i] / norm;
        }
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
    fn test_dpss() {
        assert_window(include_str!("../test_data/dpss.csv"), dpss(7, 2.5, true));
    }
}
