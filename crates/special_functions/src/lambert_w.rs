//! The Lambert W function on every complex branch.
//!
//! `W_k(z)` solves `w e^w = z` on branch `k ∈ ℤ` (Corless, Gonnet, Hare, Jeffrey and Knuth,
//! 1996). `W_0` and `W_{−1}` are real on `[−1/e, ∞)` and `[−1/e, 0)`; the branch cut of `W_0` is
//! `(−∞, −1/e]`, and the real interval of `W_{−1}` is reached from the upper half plane (so
//! `W_{−1}(x + 0i)` is real there, matching [`crate::lambert_w_m1`]). For real arguments,
//! [`crate::lambert_w0`] and [`crate::lambert_w_m1`] are faster.

use num_complex::Complex;
use std::f64::consts::{E, PI};

type C64 = Complex<f64>;

/// The Lambert W function `W_k(z)` on branch `k`. Returns `−∞` for `W_k(0)` with `k ≠ 0` and
/// NaN for non-finite input.
pub fn lambert_w(k: i32, z: C64) -> C64 {
    if !z.re.is_finite() || !z.im.is_finite() {
        return C64::new(f64::NAN, f64::NAN);
    }
    if z == C64::new(0.0, 0.0) {
        return if k == 0 { z } else { C64::new(f64::NEG_INFINITY, 0.0) };
    }
    let branch_point = C64::new(-1.0 / E, 0.0);
    let distance = (z - branch_point).norm();
    let upper = z.im >= 0.0;
    let mut w = if (k == 0 && distance < 1.0) || (distance < 0.3 && ((k == -1 && upper) || (k == 1 && !upper))) {
        // Series about the branch point in p = ±√(2(ez + 1)).
        let mut p = (2.0 * (E * z + 1.0)).sqrt();
        if k != 0 {
            p = -p;
        }
        -1.0 + p - p * p / 3.0 + 11.0 / 72.0 * p * p * p
    } else if k == 0 && z.norm() < 0.5 {
        // Near the origin W ≈ z / (1 + z).
        z / (1.0 + z)
    } else if k == 0 && z.norm() < 3.0 {
        (1.0 + z).ln()
    } else {
        // Asymptotic expansion; ln z + 2πik is bounded away from zero here.
        let l1 = z.ln() + C64::new(0.0, 2.0 * PI * k as f64);
        l1 - l1.ln() + l1.ln() / l1
    };
    // Halley iteration on f(w) = w e^w − z.
    for _ in 0..100 {
        let ew = w.exp();
        let f = w * ew - z;
        let wp1 = w + 1.0;
        let denom = ew * wp1 - (w + 2.0) * f / (2.0 * wp1);
        if denom.norm() == 0.0 || !denom.re.is_finite() || !denom.im.is_finite() {
            break;
        }
        let dw = f / denom;
        w -= dw;
        if dw.norm() <= 4.0 * f64::EPSILON * (1.0 + w.norm()) {
            break;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverse_x_log_x::{lambert_w0, lambert_w_m1};

    fn residual(w: C64, z: C64) -> f64 {
        (w * w.exp() - z).norm() / z.norm().max(1e-300)
    }

    #[test]
    fn agrees_with_the_real_branches_and_known_values() {
        for &x in &[-0.3678, -0.36, -0.2, -0.01, 0.0, 0.3, 1.0, 10.0, 1e6] {
            let w = lambert_w(0, C64::new(x, 0.0));
            assert!((w.re - lambert_w0(x)).abs() < 1e-12 * (1.0 + w.re.abs()) && w.im.abs() < 1e-12, "W0({x}) = {w}");
        }
        for &x in &[-0.3678, -0.3, -0.1, -1e-3, -1e-10] {
            let w = lambert_w(-1, C64::new(x, 0.0));
            assert!((w.re - lambert_w_m1(x)).abs() < 1e-11 * w.re.abs() && w.im.abs() < 1e-12, "W-1({x}) = {w}");
        }
        // W_0(i) and the omega constant W_0(1).
        let wi = lambert_w(0, C64::new(0.0, 1.0));
        assert!((wi - C64::new(0.374_699_020_737_117_5, 0.576_412_723_031_435_3)).norm() < 1e-14);
        assert!((lambert_w(0, C64::new(1.0, 0.0)).re - 0.567_143_290_409_783_8).abs() < 1e-15);
        // W_0 below the branch point is complex: W_0(−1) = −0.3181… + 1.3372…i.
        let wm = lambert_w(0, C64::new(-1.0, 0.0));
        assert!((wm - C64::new(-0.318_131_505_204_764_1, 1.337_235_701_430_689_4)).norm() < 1e-13, "{wm}");
        assert_eq!(lambert_w(2, C64::new(0.0, 0.0)).re, f64::NEG_INFINITY);
    }

    #[test]
    fn solves_w_exp_w_on_many_branches_with_the_asymptotic_imaginary_part() {
        for k in -4..=4 {
            for &(re, im) in &[(1.0, 0.0), (-2.0, 0.5), (0.1, -3.0), (-0.36, 1e-3), (50.0, 20.0), (-1e-4, -1e-4)] {
                let z = C64::new(re, im);
                let w = lambert_w(k, z);
                assert!(residual(w, z) < 1e-13, "k={k} z={z}: {w}");
                // Each branch occupies a strip of imaginary parts (Corless et al., figure 4).
                let (lo, hi) = match k {
                    0 => (-PI, PI),
                    k if k > 0 => ((2 * k - 2) as f64 * PI, (2 * k + 1) as f64 * PI),
                    k => ((2 * k - 1) as f64 * PI, (2 * k + 2) as f64 * PI),
                };
                assert!(w.im >= lo - 1e-12 && w.im <= hi + 1e-12, "k={k} z={z}: {w} outside its strip");
            }
            // For large |z|, W_k(z) ≈ ln z + 2πik − ln(ln z + 2πik), which identifies the branch.
            let z = C64::new(3e5, -7e5);
            let l1 = z.ln() + C64::new(0.0, 2.0 * PI * k as f64);
            assert!((lambert_w(k, z) - (l1 - l1.ln())).norm() < 0.5, "branch k={k}");
        }
        // Conjugate symmetry W_k(conj z) = conj W_{−k}(z) off the cut.
        let z = C64::new(0.7, 1.9);
        assert!((lambert_w(2, z.conj()) - lambert_w(-2, z).conj()).norm() < 1e-13);
    }
}
