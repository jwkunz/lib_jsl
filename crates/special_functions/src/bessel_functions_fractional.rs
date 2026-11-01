//! Bessel functions of fractional (real) order.
//!
//! Uses Steed's method with Temme's series for small arguments (Numerical Recipes `bessjy` /
//! `bessik`). Accuracy is typically 1e-14 relative away from zeros of J and Y.
//!
//! See the book entry [Bessel and Airy functions](https://jwkunz.github.io/lib_jsl/book/special_functions/bessel.html#bessel-functions-of-real-order).

use std::f64::consts::PI;

const EPS: f64 = 1e-16;
const FPMIN: f64 = f64::MIN_POSITIVE / EPS;
const MAXIT: usize = 100_000;
const XMIN: f64 = 2.0;

/// Coefficients c_k of 1/Γ(z) = Σ c_k z^k (Abramowitz & Stegun 6.1.34).
const RECIP_GAMMA: [f64; 26] = [
    1.0,
    0.577_215_664_901_532_9,
    -0.655_878_071_520_253_8,
    -0.042_002_635_034_095_2,
    0.166_538_611_382_291_5,
    -0.042_197_734_555_544_3,
    -0.009_621_971_527_877_0,
    0.007_218_943_246_663_0,
    -0.001_165_167_591_859_1,
    -0.000_215_241_674_114_9,
    0.000_128_050_282_388_2,
    -0.000_020_134_854_780_7,
    -0.000_001_250_493_482_1,
    0.000_001_133_027_232_0,
    -0.000_000_205_633_841_7,
    0.000_000_006_116_095_0,
    0.000_000_005_002_007_5,
    -0.000_000_001_181_274_6,
    0.000_000_000_104_342_7,
    0.000_000_000_007_782_3,
    -0.000_000_000_003_696_8,
    0.000_000_000_000_510_0,
    -0.000_000_000_000_020_6,
    -0.000_000_000_000_005_4,
    0.000_000_000_000_001_4,
    0.000_000_000_000_000_1,
];

/// Returns (gam1, gam2, 1/Γ(1+μ), 1/Γ(1-μ)) for |μ| <= 1/2 where
/// gam1 = (1/Γ(1-μ) - 1/Γ(1+μ)) / (2μ) and gam2 = (1/Γ(1-μ) + 1/Γ(1+μ)) / 2.
fn temme_gammas(mu: f64) -> (f64, f64, f64, f64) {
    // 1/Γ(1+μ) = Σ_{k>=1} c_k μ^(k-1); even and odd parts give gam2 and gam1 directly.
    let mut gam2 = 0.0;
    let mut gam1 = 0.0;
    let mut pw = 1.0; // μ^(k-1)
    let mut pw_even = 1.0; // μ^(k-2) for even k
    let mut gampl = 0.0;
    let mut gammi = 0.0;
    for (i, &c) in RECIP_GAMMA.iter().enumerate() {
        let k = i + 1;
        gampl += c * pw;
        gammi += if (k - 1) % 2 == 0 { c * pw } else { -c * pw };
        if k % 2 == 1 {
            gam2 += c * pw;
        } else {
            gam1 -= c * pw_even;
            pw_even *= mu * mu;
        }
        pw *= mu;
    }
    (gam1, gam2, gampl, gammi)
}

/// Computes `(J_ν(x), Y_ν(x))` for `x > 0` and `ν >= 0`.
fn bessel_jy_positive(x: f64, xnu: f64) -> (f64, f64) {
    let nl = if x < XMIN { (xnu + 0.5) as i64 } else { ((xnu - x + 1.5) as i64).max(0) };
    let xmu = xnu - nl as f64;
    let xmu2 = xmu * xmu;
    let xi = 1.0 / x;
    let xi2 = 2.0 * xi;
    let w = xi2 / PI;
    let mut isign = 1.0;
    let mut h = (xnu * xi).max(FPMIN);
    let mut b = xi2 * xnu;
    let mut d = 0.0;
    let mut c = h;
    for _ in 0..MAXIT {
        b += xi2;
        d = b - d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b - 1.0 / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = c * d;
        h *= del;
        if d < 0.0 {
            isign = -isign;
        }
        if (del - 1.0).abs() <= EPS {
            break;
        }
    }
    let mut rjl = isign * FPMIN;
    let mut rjpl = h * rjl;
    let rjl1 = rjl;
    let rjp1 = rjpl;
    let mut fact = xnu * xi;
    for _ in 0..nl {
        let rjtemp = fact * rjl + rjpl;
        fact -= xi;
        rjpl = fact * rjtemp - rjl;
        rjl = rjtemp;
    }
    if rjl == 0.0 {
        rjl = EPS;
    }
    let f = rjpl / rjl;
    let (rjmu, mut rymu, mut ry1);
    if x < XMIN {
        let x2 = 0.5 * x;
        let pimu = PI * xmu;
        let fact = if pimu.abs() < EPS { 1.0 } else { pimu / pimu.sin() };
        let d = -x2.ln();
        let e = xmu * d;
        let fact2 = if e.abs() < EPS { 1.0 } else { e.sinh() / e };
        let (gam1, gam2, gampl, gammi) = temme_gammas(xmu);
        let mut ff = 2.0 / PI * fact * (gam1 * e.cosh() + gam2 * fact2 * d);
        let e = e.exp();
        let mut p = e / (gampl * PI);
        let mut q = 1.0 / (e * PI * gammi);
        let pimu2 = 0.5 * pimu;
        let fact3 = if pimu2.abs() < EPS { 1.0 } else { pimu2.sin() / pimu2 };
        let r = PI * pimu2 * fact3 * fact3;
        let mut c = 1.0;
        let d = -x2 * x2;
        let mut sum = ff + r * q;
        let mut sum1 = p;
        for i in 1..MAXIT {
            let fi = i as f64;
            ff = (fi * ff + p + q) / (fi * fi - xmu2);
            c *= d / fi;
            p /= fi - xmu;
            q /= fi + xmu;
            let del = c * (ff + r * q);
            sum += del;
            let del1 = c * p - fi * del;
            sum1 += del1;
            if del.abs() < (1.0 + sum.abs()) * EPS {
                break;
            }
        }
        rymu = -sum;
        ry1 = -sum1 * xi2;
        let rymup = xmu * xi * rymu - ry1;
        rjmu = w / (rymup - f * rymu);
    } else {
        let mut a = 0.25 - xmu2;
        let mut p = -0.5 * xi;
        let mut q = 1.0;
        let br = 2.0 * x;
        let mut bi = 2.0;
        let mut fact = a * xi / (p * p + q * q);
        let mut cr = br + q * fact;
        let mut ci = bi + p * fact;
        let mut den = br * br + bi * bi;
        let mut dr = br / den;
        let mut di = -bi / den;
        let mut dlr = cr * dr - ci * di;
        let mut dli = cr * di + ci * dr;
        let mut temp = p * dlr - q * dli;
        q = p * dli + q * dlr;
        p = temp;
        for i in 1..MAXIT {
            a += 2.0 * i as f64;
            bi += 2.0;
            dr = a * dr + br;
            di = a * di + bi;
            if dr.abs() + di.abs() < FPMIN {
                dr = FPMIN;
            }
            fact = a / (cr * cr + ci * ci);
            cr = br + cr * fact;
            ci = bi - ci * fact;
            if cr.abs() + ci.abs() < FPMIN {
                cr = FPMIN;
            }
            den = dr * dr + di * di;
            dr /= den;
            di /= -den;
            dlr = cr * dr - ci * di;
            dli = cr * di + ci * dr;
            temp = p * dlr - q * dli;
            q = p * dli + q * dlr;
            p = temp;
            if (dlr - 1.0).abs() + dli.abs() <= EPS {
                break;
            }
        }
        let gam = (p - f) / q;
        let mut rj = (w / ((p - f) * gam + q)).sqrt();
        if rjl < 0.0 {
            rj = -rj;
        }
        rjmu = rj;
        rymu = rjmu * gam;
        let rymup = rymu * (p + q / gam);
        ry1 = xmu * xi * rymu - rymup;
    }
    let fact = rjmu / rjl;
    let rj = rjl1 * fact;
    let _ = rjp1;
    for i in 1..=nl {
        let rytemp = (xmu + i as f64) * xi2 * ry1 - rymu;
        rymu = ry1;
        ry1 = rytemp;
    }
    (rj, rymu)
}

/// Computes `(I_ν(x), K_ν(x))` for `x > 0` and `ν >= 0`.
fn bessel_ik_positive(x: f64, xnu: f64) -> (f64, f64) {
    let nl = (xnu + 0.5) as i64;
    let xmu = xnu - nl as f64;
    let xmu2 = xmu * xmu;
    let xi = 1.0 / x;
    let xi2 = 2.0 * xi;
    let mut h = (xnu * xi).max(FPMIN);
    let mut b = xi2 * xnu;
    let mut d = 0.0;
    let mut c = h;
    for _ in 0..MAXIT {
        b += xi2;
        d = 1.0 / (b + d);
        c = b + 1.0 / c;
        let del = c * d;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    let mut ril = FPMIN;
    let mut ripl = h * ril;
    let ril1 = ril;
    let mut fact = xnu * xi;
    for _ in 0..nl {
        let rtemp = fact * ril + ripl;
        fact -= xi;
        ripl = fact * rtemp + ril;
        ril = rtemp;
    }
    let f = ripl / ril;
    let (mut rkmu, mut rk1);
    if x < XMIN {
        let x2 = 0.5 * x;
        let pimu = PI * xmu;
        let fact = if pimu.abs() < EPS { 1.0 } else { pimu / pimu.sin() };
        let d = -x2.ln();
        let e = xmu * d;
        let fact2 = if e.abs() < EPS { 1.0 } else { e.sinh() / e };
        let (gam1, gam2, gampl, gammi) = temme_gammas(xmu);
        let mut ff = fact * (gam1 * e.cosh() + gam2 * fact2 * d);
        let mut sum = ff;
        let e = e.exp();
        let mut p = 0.5 * e / gampl;
        let mut q = 0.5 / (e * gammi);
        let mut c = 1.0;
        let d = x2 * x2;
        let mut sum1 = p;
        for i in 1..MAXIT {
            let fi = i as f64;
            ff = (fi * ff + p + q) / (fi * fi - xmu2);
            c *= d / fi;
            p /= fi - xmu;
            q /= fi + xmu;
            let del = c * ff;
            sum += del;
            let del1 = c * (p - fi * ff);
            sum1 += del1;
            if del.abs() < sum.abs() * EPS {
                break;
            }
        }
        rkmu = sum;
        rk1 = sum1 * xi2;
    } else {
        let mut b = 2.0 * (1.0 + x);
        let mut d = 1.0 / b;
        let mut h = d;
        let mut delh = d;
        let mut q1 = 0.0;
        let mut q2 = 1.0;
        let a1 = 0.25 - xmu2;
        let mut q = a1;
        let mut c = a1;
        let mut a = -a1;
        let mut s = 1.0 + q * delh;
        for i in 1..MAXIT {
            let fi = i as f64;
            a -= 2.0 * fi;
            c = -a * c / (fi + 1.0);
            let qnew = (q1 - b * q2) / a;
            q1 = q2;
            q2 = qnew;
            q += c * qnew;
            b += 2.0;
            d = 1.0 / (b + a * d);
            delh *= b * d - 1.0;
            h += delh;
            let dels = q * delh;
            s += dels;
            if (dels / s).abs() < EPS {
                break;
            }
        }
        h *= a1;
        rkmu = (PI / (2.0 * x)).sqrt() * (-x).exp() / s;
        rk1 = rkmu * (xmu + x + 0.5 - h) * xi;
    }
    let rkmup = xmu * xi * rkmu - rk1;
    let rimu = xi / (f * rkmu - rkmup);
    let ri = rimu * ril1 / ril;
    for i in 1..=nl {
        let rktemp = (xmu + i as f64) * xi2 * rk1 + rkmu;
        rkmu = rk1;
        rk1 = rktemp;
    }
    (ri, rkmu)
}

/// Returns `(J_ν(x), Y_ν(x))` for real order `ν` and `x > 0`.
///
/// Negative orders use the reflection formulas. Returns `(NaN, NaN)` for `x <= 0`.
pub fn bessel_jy(v: f64, x: f64) -> (f64, f64) {
    if v.is_nan() || x.is_nan() || x <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    if v >= 0.0 {
        return bessel_jy_positive(x, v);
    }
    let nu = -v;
    let (j, y) = bessel_jy_positive(x, nu);
    let (s, c) = (PI * nu).sin_cos();
    // Exact for integer orders: sin(nπ) is replaced by 0 so J_{-n} = (-1)^n J_n.
    let (s, c) = if nu == nu.round() { (0.0, if nu.round() % 2.0 == 0.0 { 1.0 } else { -1.0 }) } else { (s, c) };
    (c * j - s * y, s * j + c * y)
}

/// Returns `(I_ν(x), K_ν(x))` for real order `ν` and `x > 0`.
///
/// Returns `(NaN, NaN)` for `x <= 0`.
pub fn bessel_ik(v: f64, x: f64) -> (f64, f64) {
    if v.is_nan() || x.is_nan() || x <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    if v >= 0.0 {
        return bessel_ik_positive(x, v);
    }
    let nu = -v;
    let (i, k) = bessel_ik_positive(x, nu);
    // I_{-ν} = I_ν + (2/π) sin(νπ) K_ν ; K_{-ν} = K_ν
    let s = if nu == nu.round() { 0.0 } else { (PI * nu).sin() };
    (i + 2.0 / PI * s * k, k)
}

/// The Bessel function of the first kind J_ν(x) for real order `ν`.
///
/// For `x < 0` the order must be an integer (`J_n(-x) = (-1)^n J_n(x)`), otherwise the result is
/// `NaN`. `J_ν(0)` is 1 for `ν = 0`, 0 for `ν > 0`, and infinite for non-integer `ν < 0`.
pub fn bessel_fractional(v: f64, x: f64) -> f64 {
    if v.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return if v == 0.0 {
            1.0
        } else if v > 0.0 || v == v.round() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if x < 0.0 {
        if v != v.round() {
            return f64::NAN;
        }
        let r = bessel_jy(v, -x).0;
        return if v.round() % 2.0 == 0.0 { r } else { -r };
    }
    bessel_jy(v, x).0
}

/// The Bessel function of the second kind Y_ν(x) for real order `ν` and `x > 0`
/// (`-inf` at 0, `NaN` for `x < 0`).
pub fn bessel_y_fractional(v: f64, x: f64) -> f64 {
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    bessel_jy(v, x).1
}

/// The modified Bessel function of the first kind I_ν(x) for real order `ν` and `x >= 0`.
pub fn bessel_i_fractional(v: f64, x: f64) -> f64 {
    if x == 0.0 {
        return if v == 0.0 { 1.0 } else if v > 0.0 || v == v.round() { 0.0 } else { f64::INFINITY };
    }
    bessel_ik(v, x).0
}

/// The modified Bessel function of the second kind K_ν(x) for real order `ν` and `x > 0`
/// (`+inf` at 0).
pub fn bessel_k_fractional(v: f64, x: f64) -> f64 {
    if x == 0.0 {
        return f64::INFINITY;
    }
    bessel_ik(v, x).1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1e-300)
    }

    #[test]
    fn temme_gamma_series_matches_gamma_function() {
        for &mu in &[-0.5, -0.3, 0.0, 0.1, 0.5] {
            let (gam1, gam2, gampl, gammi) = temme_gammas(mu);
            assert!(close(gampl, 1.0 / crate::gamma::gamma(1.0 + mu), 1e-13), "mu={mu}");
            assert!(close(gammi, 1.0 / crate::gamma::gamma(1.0 - mu), 1e-13));
            assert!(close(gam2, 0.5 * (gampl + gammi), 1e-13));
            if mu != 0.0 {
                assert!(close(gam1, (gammi - gampl) / (2.0 * mu), 1e-9));
            }
        }
    }

    #[test]
    fn integer_orders() {
        assert!(close(bessel_fractional(0.0, 1.0), 0.7651976865579666, 1e-14));
        assert!(close(bessel_fractional(1.0, 2.0), 0.5767248077568734, 1e-14));
        assert!(close(bessel_fractional(2.0, 5.0), 0.04656511627775222, 1e-13));
        assert!(close(bessel_fractional(0.0, 10.0), -0.2459357644513483, 1e-13));
        assert!(close(bessel_y_fractional(0.0, 1.0), 0.08825696421567697, 1e-13));
        assert!(close(bessel_y_fractional(1.0, 2.0), -0.10703243154093754, 1e-13));
        assert!(close(bessel_i_fractional(0.0, 1.0), 1.2660658777520082, 1e-14));
        assert!(close(bessel_i_fractional(1.0, 3.0), 3.953370217402609, 1e-13));
        assert!(close(bessel_k_fractional(0.0, 1.0), 0.42102443824070834, 1e-13));
        assert!(close(bessel_k_fractional(1.0, 3.0), 0.04015643112819418, 1e-13));
    }

    #[test]
    fn half_integer_closed_forms() {
        for &x in &[0.1, 0.7, 1.9, 2.5, 6.0, 20.0] {
            let pref = (2.0 / (PI * x)).sqrt();
            assert!((bessel_fractional(0.5, x) - pref * x.sin()).abs() < 1e-14, "J x={x}");
            assert!((bessel_fractional(-0.5, x) - pref * x.cos()).abs() < 1e-14);
            assert!((bessel_y_fractional(0.5, x) + pref * x.cos()).abs() < 1e-14);
            let ik = (PI / (2.0 * x)).sqrt() * (-x).exp();
            assert!(close(bessel_k_fractional(0.5, x), ik, 1e-13));
            let ipre = (2.0 / (PI * x)).sqrt();
            assert!(close(bessel_i_fractional(0.5, x), ipre * x.sinh(), 1e-13));
            assert!(close(bessel_i_fractional(-0.5, x), ipre * x.cosh(), 1e-13));
        }
        // J_{3/2}(x) = sqrt(2/(pi x)) (sin x / x - cos x)
        let x: f64 = 3.3;
        let want = (2.0 / (PI * x)).sqrt() * (x.sin() / x - x.cos());
        assert!((bessel_fractional(1.5, x) - want).abs() < 1e-14);
    }

    #[test]
    fn wronskian() {
        // J_{v+1} Y_v - J_v Y_{v+1} = 2/(pi x)
        for &(v, x) in &[(0.3, 0.5), (1.7, 3.0), (4.2, 1.0), (0.0, 12.0), (10.5, 8.0)] {
            let (j0, y0) = bessel_jy(v, x);
            let (j1, y1) = bessel_jy(v + 1.0, x);
            assert!(((j1 * y0 - j0 * y1) * PI * x / 2.0 - 1.0).abs() < 1e-12, "v={v} x={x}");
        }
    }

    #[test]
    fn modified_wronskian() {
        // I_v K_{v+1} + I_{v+1} K_v = 1/x
        for &(v, x) in &[(0.3, 0.5), (1.7, 3.0), (4.2, 1.0), (0.0, 12.0)] {
            let (i0, k0) = bessel_ik(v, x);
            let (i1, k1) = bessel_ik(v + 1.0, x);
            assert!(((i0 * k1 + i1 * k0) * x - 1.0).abs() < 1e-12, "v={v} x={x}");
        }
    }

    #[test]
    fn negative_arguments_and_orders() {
        assert!(close(bessel_fractional(1.0, -2.0), -0.5767248077568734, 1e-14));
        assert!(close(bessel_fractional(2.0, -5.0), 0.04656511627775222, 1e-13));
        assert!(bessel_fractional(0.5, -1.0).is_nan());
        assert!(close(bessel_fractional(-2.0, 5.0), 0.04656511627775222, 1e-13));
        assert!(close(bessel_fractional(-1.0, 2.0), -0.5767248077568734, 1e-14));
        assert_eq!(bessel_fractional(0.0, 0.0), 1.0);
        assert_eq!(bessel_fractional(2.5, 0.0), 0.0);
    }

    #[test]
    fn small_and_large_arguments() {
        // Small-argument limit J_v(x) ~ (x/2)^v / Gamma(v+1)
        let x: f64 = 1e-4;
        let want = (x / 2.0).powf(0.7) / crate::gamma::gamma(1.7);
        assert!(close(bessel_fractional(0.7, x), want, 1e-7));
        // Asymptotic form for large x
        let x: f64 = 200.0;
        let v: f64 = 0.3;
        let want = (2.0 / (PI * x)).sqrt() * (x - v * PI / 2.0 - PI / 4.0).cos();
        assert!((bessel_fractional(v, x) - want).abs() < 1e-3 * want.abs().max(0.01));
    }
}
