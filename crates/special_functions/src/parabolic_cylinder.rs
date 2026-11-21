//! Parabolic cylinder functions (SciPy's `pbdv`, `pbvv` and `pbwa`): solutions of Weber's
//! equation `y'' − (x²/4 + a) y = 0` (`U(a, x) = D_{−a−½}(x)` and `V(a, x)`) and of
//! `y'' + (x²/4 − a) y = 0` (`W(a, ±x)`).
//!
//! See the book entry [Mathieu, parabolic cylinder and spheroidal wave functions](https://jwkunz.github.io/lib_jsl/book/special_functions/wave-functions.html#parabolic-cylinder-functions).

use crate::gamma::gamma;
use crate::hypergeometric_functions::hypergeometric_1f1;
use std::f64::consts::{FRAC_2_PI, PI};

/// `1/Γ(x)`, zero at the poles.
fn rgamma(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() { 0.0 } else { 1.0 / gamma(x) }
}

/// `∫₀^∞ t^{s−1} e^{−t²/2 − x t} dt` (`s > 0`, `x ≥ 0`) by the exp-sinh rule
/// `t = exp(π/2 · sinh τ)`, which absorbs the endpoint singularity at zero.
fn laplace_gauss(s: f64, x: f64) -> f64 {
    let h = 1.0 / 64.0;
    let mut sum = 0.0;
    let mut k: i64 = -400;
    while k <= 400 {
        let tau = k as f64 * h;
        let t = (0.5 * PI * tau.sinh()).exp();
        let w = t * 0.5 * PI * tau.cosh();
        let g = ((s - 1.0) * t.ln() - 0.5 * t * t - x * t).exp();
        if g.is_finite() {
            sum += g * w;
        }
        k += 1;
    }
    sum * h
}

// ANCHOR: pbdv
/// `D_v(x)` for `x ≥ 0`: for `v < 0` from `D_v(x) = e^{−x²/4}/Γ(−v) ∫₀^∞ t^{−v−1} e^{−t²/2 − xt} dt`
/// (DLMF 12.5.1), and for `v ≥ 0` by the recurrence `D_{w+1} = x D_w − w D_{w−1}` from two
/// such values. Unlike the power series, neither step cancels for large `x`.
fn dv_positive(v: f64, x: f64) -> f64 {
    let e = (-0.25 * x * x).exp();
    if v == 0.0 {
        return e;
    }
    let by_integral = |w: f64| e * rgamma(-w) * laplace_gauss(-w, x);
    if v < 0.0 {
        return by_integral(v);
    }
    let n = v.floor() as i64 + 1;
    let w0 = v - n as f64; // in [−1, 0)
    let (mut dm, mut d) = (by_integral(w0 - 1.0), by_integral(w0));
    let mut w = w0;
    for _ in 0..n {
        let next = x * d - w * dm;
        dm = d;
        d = next;
        w += 1.0;
    }
    d
}
// ANCHOR_END: pbdv

/// The even and odd solutions `u₁ = e^{−x²/4} M(a/2 + ¼, ½, x²/2)`,
/// `u₂ = x e^{−x²/4} M(a/2 + ¾, 3/2, x²/2)` (DLMF 12.4).
fn basis(a: f64, x: f64) -> (f64, f64) {
    let e = (-0.25 * x * x).exp();
    let z = 0.5 * x * x;
    (e * hypergeometric_1f1(0.5 * a + 0.25, 0.5, z), x * e * hypergeometric_1f1(0.5 * a + 0.75, 1.5, z))
}

/// `V(a, x)` for `x ≥ 0` (`a = −v − ½`): the power series `V(a,0) u₁ + V'(a,0) u₂` (DLMF 12.2.20–21)
/// for `x < 8`, the asymptotic series (DLMF 12.9.2) beyond.
fn vv_positive(v: f64, x: f64) -> f64 {
    let a = -v - 0.5;
    if x < 8.0 {
        let v0 = 2f64.powf(0.5 * a + 0.25) * (PI * (0.75 - 0.5 * a)).sin() * rgamma(0.75 - 0.5 * a);
        let d0 = 2f64.powf(0.5 * a + 0.75) * (PI * (0.25 - 0.5 * a)).sin() * rgamma(0.25 - 0.5 * a);
        let (u1, u2) = basis(a, x);
        return v0 * u1 + d0 * u2;
    }
    // √(2/π) e^{x²/4} x^{a−½} Σ (½ − a)_{2k} / (k! (2x²)^k).
    let (mut term, mut sum) = (1.0, 1.0);
    for k in 1..40 {
        let k2 = 2.0 * k as f64;
        let prev = term;
        term *= (0.5 - a + k2 - 2.0) * (0.5 - a + k2 - 1.0) / (k as f64 * 2.0 * x * x);
        if term.abs() > prev.abs() {
            break;
        }
        sum += term;
        if term.abs() < 1e-17 * sum.abs() {
            break;
        }
    }
    FRAC_2_PI.sqrt() * (0.25 * x * x).exp() * x.powf(a - 0.5) * sum
}

/// `D_v(x)` for any real `x`: for `x < 0` by `D_v(−x) = cos(πv) D_v(x) + π/Γ(−v) V_v(x)`
/// (DLMF 12.2.15).
fn dv(v: f64, x: f64) -> f64 {
    if x >= 0.0 {
        return dv_positive(v, x);
    }
    let y = -x;
    let r = PI * rgamma(-v);
    let vv = if r == 0.0 { 0.0 } else { vv_positive(v, y) };
    (PI * v).cos() * dv_positive(v, y) + r * vv
}

/// `V_v(x)` for any real `x`: for `x < 0` by `V_v(−x) = −sin(πv)/Γ(v+1) D_v(x) − cos(πv) V_v(x)`.
fn vv(v: f64, x: f64) -> f64 {
    if x >= 0.0 {
        return vv_positive(v, x);
    }
    let y = -x;
    let s = (PI * v).sin() * rgamma(v + 1.0);
    let dd = if s == 0.0 { 0.0 } else { dv_positive(v, y) };
    -s * dd - (PI * v).cos() * vv_positive(v, y)
}

/// The parabolic cylinder function `D_v(x)` and its derivative (SciPy's `pbdv`), from
/// `D_v' = x/2 · D_v − D_{v+1}`.
pub fn pbdv(v: f64, x: f64) -> (f64, f64) {
    let d = dv(v, x);
    (d, 0.5 * x * d - dv(v + 1.0, x))
}

/// The parabolic cylinder function `V_v(x) = V(−v − ½, x)` and its derivative (SciPy's
/// `pbvv`), from `V_v' = x/2 · V_v − (v + 1) V_{v+1}`.
pub fn pbvv(v: f64, x: f64) -> (f64, f64) {
    let y = vv(v, x);
    (y, 0.5 * x * y - (v + 1.0) * vv(v + 1.0, x))
}

/// `ln |Γ(x + iy)|` by the Lanczos approximation (with reflection for `x < ½`).
fn ln_abs_gamma_complex(x: f64, y: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // |Γ(z)| = π / (|sin πz| |Γ(1 − z)|).
        let s = ((PI * x).sin() * (PI * y).cosh()).hypot((PI * x).cos() * (PI * y).sinh());
        return PI.ln() - s.ln() - ln_abs_gamma_complex(1.0 - x, -y);
    }
    let (zr, zi) = (x - 1.0, y);
    let (mut ar, mut ai) = (C[0], 0.0);
    for (k, c) in C.iter().enumerate().skip(1) {
        let (dr, di) = (zr + k as f64, zi);
        let d2 = dr * dr + di * di;
        ar += c * dr / d2;
        ai -= c * di / d2;
    }
    let (tr, ti) = (zr + G + 0.5, zi);
    // ln|t^{z+½} e^{−t} √(2π) A|.
    let ln_t = (tr.hypot(ti)).ln();
    let arg_t = ti.atan2(tr);
    0.5 * (2.0 * PI).ln() + (zr + 0.5) * ln_t - zi * arg_t - tr + 0.5 * (ar * ar + ai * ai).ln()
}

// ANCHOR: pbwa
/// `W(a, x)` and its derivative (SciPy's `pbwa`, for `|a| ≤ 5`, `|x| ≤ 5` as there): the
/// solution of `y'' + (x²/4 − a) y = 0` with `W(a, 0) = 2^{−3/4} |Γ(¼ + ia/2)/Γ(¾ + ia/2)|^{½}` and
/// `W'(a, 0) = −2^{−1/4} |Γ(¾ + ia/2)/Γ(¼ + ia/2)|^{½}` (DLMF 12.14.4–5), summed as a power series
/// from the recurrence `(n+2)(n+1) c_{n+2} = a c_n − c_{n−2}/4`.
pub fn pbwa(a: f64, x: f64) -> (f64, f64) {
    let lg1 = ln_abs_gamma_complex(0.25, 0.5 * a);
    let lg3 = ln_abs_gamma_complex(0.75, 0.5 * a);
    let w0 = 2f64.powf(-0.75) * (0.5 * (lg1 - lg3)).exp();
    let d0 = -(2f64.powf(-0.25)) * (0.5 * (lg3 - lg1)).exp();
    let series = |c0: f64, c1: f64| -> (f64, f64) {
        let mut c = vec![c0, c1];
        let (mut y, mut dy) = (c0 + c1 * x, c1);
        let mut n = 0;
        loop {
            let prev2 = if n >= 2 { c[n - 2] } else { 0.0 };
            let next = (a * c[n] - 0.25 * prev2) / ((n + 2) as f64 * (n + 1) as f64);
            c.push(next);
            let p = (n + 2) as i32;
            let term = next * x.powi(p);
            y += term;
            dy += p as f64 * next * x.powi(p - 1);
            n += 1;
            // Coefficients can vanish in runs (every other one, or three in four when a = 0):
            // stop on four consecutive negligible terms.
            if n > 20 && (0..4).all(|j| (c[c.len() - 1 - j] * x.powi(p - j as i32)).abs() < 1e-18 * y.abs().max(1e-300)) {
                break;
            }
            if n > 400 {
                break;
            }
        }
        (y, dy)
    };
    let (e, de) = series(1.0, 0.0);
    let (o, dodx) = series(0.0, 1.0);
    (w0 * e + d0 * o, w0 * de + d0 * dodx)
}
// ANCHOR_END: pbwa


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hermite_cases_and_wronskians() {
        // D_n(x) = e^{−x²/4} He_n(x): D_2 = (x² − 1) e^{−x²/4}.
        for x in [-3.0f64, -0.5, 0.0, 1.0, 4.0, 9.0] {
            let want = (x * x - 1.0) * (-0.25 * x * x).exp();
            assert!((pbdv(2.0, x).0 - want).abs() < 1e-14 * want.abs().max(1e-3), "{x}");
        }
        // W{U, V} = √(2/π) for every v and x.
        for (v, x) in [(0.3, 1.2), (-1.7, 0.4), (2.5, -2.0), (4.0, 3.0)] {
            let (d, dd) = pbdv(v, x);
            let (w, dw) = pbvv(v, x);
            assert!((d * dw - dd * w - (2.0 / std::f64::consts::PI).sqrt()).abs() < 1e-9, "{v} {x}");
        }
        // W(a, x) W'(a, −x)… : the Wronskian of W(a, x) and W(a, −x) is 1.
        let (w1, dw1) = pbwa(0.7, 1.3);
        let (w2, dw2) = pbwa(0.7, -1.3);
        assert!((w1 * (-dw2) - dw1 * w2 - 1.0).abs() < 1e-10);
    }
}
