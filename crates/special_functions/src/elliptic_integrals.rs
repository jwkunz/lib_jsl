//! Elliptic integrals, computed through Carlson's symmetric forms.
//!
//! All Legendre-form functions take the modulus `k` (so the parameter is `m = k²`) and the
//! amplitude `phi` in radians.

use std::f64::consts::PI;

/// Carlson's degenerate integral `R_C(x, y)`, for `x >= 0`, `y != 0`.
pub fn carlson_rc(x: f64, y: f64) -> f64 {
    const ERRTOL: f64 = 0.0006;
    if x < 0.0 || y == 0.0 || x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    let (mut xt, mut yt, w) = if y > 0.0 {
        (x, y, 1.0)
    } else {
        // Cauchy principal value.
        (x - y, -y, x.sqrt() / (x - y).sqrt())
    };
    let mut ave;
    let mut s;
    loop {
        let alamb = 2.0 * xt.sqrt() * yt.sqrt() + yt;
        xt = 0.25 * (xt + alamb);
        yt = 0.25 * (yt + alamb);
        ave = (xt + yt + yt) / 3.0;
        s = (yt - ave) / ave;
        if s.abs() <= ERRTOL {
            break;
        }
    }
    w * (1.0 + s * s * (0.3 + s * (1.0 / 7.0 + s * (0.375 + s * 9.0 / 22.0)))) / ave.sqrt()
}

/// Carlson's elliptic integral of the first kind `R_F(x, y, z)`, for non-negative arguments with
/// at most one equal to zero.
pub fn carlson_rf(x: f64, y: f64, z: f64) -> f64 {
    const ERRTOL: f64 = 0.001;
    if x.is_nan() || y.is_nan() || z.is_nan() || x < 0.0 || y < 0.0 || z < 0.0 {
        return f64::NAN;
    }
    if (x == 0.0) as u8 + (y == 0.0) as u8 + (z == 0.0) as u8 > 1 {
        return f64::NAN;
    }
    let (mut xt, mut yt, mut zt) = (x, y, z);
    let (mut delx, mut dely, mut delz, mut ave);
    loop {
        let (sx, sy, sz) = (xt.sqrt(), yt.sqrt(), zt.sqrt());
        let alamb = sx * (sy + sz) + sy * sz;
        xt = 0.25 * (xt + alamb);
        yt = 0.25 * (yt + alamb);
        zt = 0.25 * (zt + alamb);
        ave = (xt + yt + zt) / 3.0;
        delx = (ave - xt) / ave;
        dely = (ave - yt) / ave;
        delz = (ave - zt) / ave;
        if delx.abs().max(dely.abs()).max(delz.abs()) <= ERRTOL {
            break;
        }
    }
    let e2 = delx * dely - delz * delz;
    let e3 = delx * dely * delz;
    (1.0 + (e2 / 24.0 - 0.1 - 3.0 * e3 / 44.0) * e2 + e3 / 14.0) / ave.sqrt()
}

/// Carlson's elliptic integral of the second kind `R_D(x, y, z)`, for `x, y >= 0` (not both zero)
/// and `z > 0`.
pub fn carlson_rd(x: f64, y: f64, z: f64) -> f64 {
    const ERRTOL: f64 = 0.001;
    if x.is_nan() || y.is_nan() || z.is_nan() || x < 0.0 || y < 0.0 || z <= 0.0 || (x == 0.0 && y == 0.0) {
        return f64::NAN;
    }
    let (mut xt, mut yt, mut zt) = (x, y, z);
    let mut sum = 0.0;
    let mut fac = 1.0;
    let (mut delx, mut dely, mut delz, mut ave);
    loop {
        let (sx, sy, sz) = (xt.sqrt(), yt.sqrt(), zt.sqrt());
        let alamb = sx * (sy + sz) + sy * sz;
        sum += fac / (sz * (zt + alamb));
        fac *= 0.25;
        xt = 0.25 * (xt + alamb);
        yt = 0.25 * (yt + alamb);
        zt = 0.25 * (zt + alamb);
        ave = 0.2 * (xt + yt + 3.0 * zt);
        delx = (ave - xt) / ave;
        dely = (ave - yt) / ave;
        delz = (ave - zt) / ave;
        if delx.abs().max(dely.abs()).max(delz.abs()) <= ERRTOL {
            break;
        }
    }
    let (c1, c2, c3, c4) = (3.0 / 14.0, 1.0 / 6.0, 9.0 / 22.0, 3.0 / 26.0);
    let (c5, c6) = (0.25 * c3, 1.5 * c4);
    let ea = delx * dely;
    let eb = delz * delz;
    let ec = ea - eb;
    let ed = ea - 6.0 * eb;
    let ee = ed + ec + ec;
    3.0 * sum
        + fac * (1.0 + ed * (-c1 + c5 * ed - c6 * delz * ee) + delz * (c2 * ee + delz * (-c3 * ec + delz * c4 * ea)))
            / (ave * ave.sqrt())
}

/// Carlson's elliptic integral of the third kind `R_J(x, y, z, p)`, for `x, y, z >= 0` (at most one
/// zero) and `p > 0`.
pub fn carlson_rj(x: f64, y: f64, z: f64, p: f64) -> f64 {
    const ERRTOL: f64 = 0.001;
    if x.is_nan() || y.is_nan() || z.is_nan() || p.is_nan() || x < 0.0 || y < 0.0 || z < 0.0 || p <= 0.0 {
        return f64::NAN;
    }
    if (x == 0.0) as u8 + (y == 0.0) as u8 + (z == 0.0) as u8 > 1 {
        return f64::NAN;
    }
    let (mut xt, mut yt, mut zt, mut pt) = (x, y, z, p);
    let mut sum = 0.0;
    let mut fac = 1.0;
    let (mut delx, mut dely, mut delz, mut delp, mut ave);
    loop {
        let (sx, sy, sz) = (xt.sqrt(), yt.sqrt(), zt.sqrt());
        let alamb = sx * (sy + sz) + sy * sz;
        let alpha = (pt * (sx + sy + sz) + sx * sy * sz).powi(2);
        let beta = pt * (pt + alamb).powi(2);
        sum += fac * carlson_rc(alpha, beta);
        fac *= 0.25;
        xt = 0.25 * (xt + alamb);
        yt = 0.25 * (yt + alamb);
        zt = 0.25 * (zt + alamb);
        pt = 0.25 * (pt + alamb);
        ave = 0.2 * (xt + yt + zt + pt + pt);
        delx = (ave - xt) / ave;
        dely = (ave - yt) / ave;
        delz = (ave - zt) / ave;
        delp = (ave - pt) / ave;
        if delx.abs().max(dely.abs()).max(delz.abs()).max(delp.abs()) <= ERRTOL {
            break;
        }
    }
    let (c1, c2, c3, c4) = (3.0 / 14.0, 1.0 / 3.0, 3.0 / 22.0, 3.0 / 26.0);
    let (c5, c6, c7, c8) = (0.75 * c3, 1.5 * c4, 0.5 * c2, c3 + c3);
    let ea = delx * (dely + delz) + dely * delz;
    let eb = delx * dely * delz;
    let ec = delp * delp;
    let ed = ea - 3.0 * ec;
    let ee = eb + 2.0 * delp * (ea - ec);
    3.0 * sum
        + fac
            * (1.0
                + ed * (-c1 + c5 * ed - c6 * ee)
                + eb * (c7 + delp * (-c8 + delp * c4))
                + delp * ea * (c2 - delp * c3)
                - c2 * delp * ec)
            / (ave * ave.sqrt())
}

/// Splits `phi` into `(n, r)` with `phi = n π + r` and `|r| <= π/2`.
fn reduce_amplitude(phi: f64) -> (f64, f64) {
    let n = (phi / PI).round();
    (n, phi - n * PI)
}

/// The complete elliptic integral of the first kind `K(k) = ∫₀^{π/2} dθ / √(1 − k² sin²θ)`,
/// for `|k| <= 1` (`+inf` at `|k| = 1`, `NaN` beyond).
pub fn elliptic_integral(k: f64) -> f64 {
    if k.is_nan() || k.abs() > 1.0 {
        return f64::NAN;
    }
    if k.abs() == 1.0 {
        return f64::INFINITY;
    }
    carlson_rf(0.0, 1.0 - k * k, 1.0)
}

/// The complete elliptic integral of the second kind `E(k) = ∫₀^{π/2} √(1 − k² sin²θ) dθ`,
/// for `|k| <= 1`.
pub fn elliptic_integral_second(k: f64) -> f64 {
    if k.is_nan() || k.abs() > 1.0 {
        return f64::NAN;
    }
    if k.abs() == 1.0 {
        return 1.0;
    }
    let q = 1.0 - k * k;
    carlson_rf(0.0, q, 1.0) - k * k / 3.0 * carlson_rd(0.0, q, 1.0)
}

/// The complete elliptic integral of the third kind `Π(n, k) = ∫₀^{π/2} dθ / ((1 − n sin²θ) √(1 − k² sin²θ))`,
/// for `n < 1` and `|k| < 1`.
pub fn elliptic_integral_third(n: f64, k: f64) -> f64 {
    if n.is_nan() || k.is_nan() || n >= 1.0 || k.abs() >= 1.0 {
        return f64::NAN;
    }
    let q = 1.0 - k * k;
    carlson_rf(0.0, q, 1.0) + n / 3.0 * carlson_rj(0.0, q, 1.0, 1.0 - n)
}

/// The incomplete elliptic integral of the first kind `F(φ, k) = ∫₀^φ dθ / √(1 − k² sin²θ)`.
pub fn elliptic_f(phi: f64, k: f64) -> f64 {
    if phi.is_nan() || k.is_nan() || k.abs() > 1.0 {
        return f64::NAN;
    }
    let (n, r) = reduce_amplitude(phi);
    let (s, c) = r.sin_cos();
    let mut v = s * carlson_rf(c * c, 1.0 - s * s * k * k, 1.0);
    if n != 0.0 {
        v += 2.0 * n * elliptic_integral(k);
    }
    v
}

/// The incomplete elliptic integral of the second kind `E(φ, k) = ∫₀^φ √(1 − k² sin²θ) dθ`.
pub fn elliptic_e(phi: f64, k: f64) -> f64 {
    if phi.is_nan() || k.is_nan() || k.abs() > 1.0 {
        return f64::NAN;
    }
    let (n, r) = reduce_amplitude(phi);
    let (s, c) = r.sin_cos();
    let cc = c * c;
    let q = 1.0 - s * s * k * k;
    let mut v = s * (carlson_rf(cc, q, 1.0) - (s * k) * (s * k) / 3.0 * carlson_rd(cc, q, 1.0));
    if n != 0.0 {
        v += 2.0 * n * elliptic_integral_second(k);
    }
    v
}

/// The incomplete elliptic integral of the third kind
/// `Π(φ, n, k) = ∫₀^φ dθ / ((1 − n sin²θ) √(1 − k² sin²θ))`.
///
/// Requires `1 − n sin²θ > 0` over the range of integration (principal value cases are not
/// supported and return `NaN`).
pub fn elliptic_pi(phi: f64, n: f64, k: f64) -> f64 {
    if phi.is_nan() || n.is_nan() || k.is_nan() || k.abs() > 1.0 {
        return f64::NAN;
    }
    let (m, r) = reduce_amplitude(phi);
    let (s, c) = r.sin_cos();
    let cc = c * c;
    let q = 1.0 - s * s * k * k;
    let mut v = s * carlson_rf(cc, q, 1.0) + n * s * s * s / 3.0 * carlson_rj(cc, q, 1.0, 1.0 - n * s * s);
    if m != 0.0 {
        v += 2.0 * m * elliptic_integral_third(n, k);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs()
    }

    #[test]
    fn complete_first_and_second_kind() {
        let cases = [
            (0.0, 1.5707963267948966, 1.5707963267948966),
            (0.3, 1.6080486199305128, 1.5348334649232490),
            (0.5, 1.6857503548125961, 1.4674622093394272),
            (0.9, 2.2805491384227703, 1.1716970527816142),
            (0.999, 4.4955963958421509, 1.0039944099655078),
        ];
        for &(k, kk, ee) in &cases {
            assert!(close(elliptic_integral(k), kk, 1e-14), "K({k})");
            assert!(close(elliptic_integral_second(k), ee, 1e-14), "E({k})");
            assert!(close(elliptic_integral(-k), kk, 1e-14));
        }
        assert!(elliptic_integral(1.0).is_infinite());
        assert_eq!(elliptic_integral_second(1.0), 1.0);
        assert!(elliptic_integral(1.5).is_nan());
    }

    #[test]
    fn incomplete_forms() {
        assert!(close(elliptic_f(1.0, 0.5), 1.0373561200021773, 1e-14));
        assert!(close(elliptic_e(1.0, 0.5), 0.96487645426862749, 1e-14));
        assert!(close(elliptic_pi(1.0, 0.3, 0.5), 1.1386198831894311, 1e-14));
        assert!(close(elliptic_integral_third(0.3, 0.5), 2.0277924458111315, 1e-14));
        assert!(close(elliptic_pi(1.2, -2.0, 0.5f64.sqrt()), 0.84630033528324021, 1e-14));
    }

    #[test]
    fn amplitude_beyond_quarter_period() {
        assert!(close(elliptic_f(5.0, 0.5), 5.3878724275355712, 1e-14));
        assert!(close(elliptic_e(5.0, 0.5), 4.6525864357205965, 1e-14));
        assert!(close(elliptic_pi(5.0, 0.3, 0.5), 6.5503339088946389, 1e-13));
        assert!((elliptic_f(-1.0, 0.5) + elliptic_f(1.0, 0.5)).abs() < 1e-15);
        assert!(close(elliptic_f(std::f64::consts::FRAC_PI_2, 0.7), elliptic_integral(0.7), 1e-14));
    }

    #[test]
    fn legendre_relation() {
        // E K' + E' K - K K' = π/2
        let k: f64 = 0.6;
        let kp = (1.0 - k * k).sqrt();
        let lhs = elliptic_integral_second(k) * elliptic_integral(kp)
            + elliptic_integral_second(kp) * elliptic_integral(k)
            - elliptic_integral(k) * elliptic_integral(kp);
        assert!((lhs - PI / 2.0).abs() < 1e-14);
    }

    #[test]
    fn carlson_identities() {
        // R_C(x, x) = 1/sqrt(x); R_F(x, x, x) = 1/sqrt(x); R_D(x,x,x) = x^-1.5
        assert!(close(carlson_rc(4.0, 4.0), 0.5, 1e-15));
        assert!(close(carlson_rf(2.0, 2.0, 2.0), 2f64.powf(-0.5), 1e-15));
        assert!(close(carlson_rd(3.0, 3.0, 3.0), 3f64.powf(-1.5), 1e-15));
        assert!(close(carlson_rj(2.0, 2.0, 2.0, 2.0), 2f64.powf(-1.5), 1e-15));
        assert!(carlson_rf(0.0, 0.0, 1.0).is_nan());
    }
}
