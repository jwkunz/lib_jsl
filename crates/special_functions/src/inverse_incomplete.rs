//! Inverses of the regularized incomplete gamma and beta functions (SciPy's `gammaincinv`,
//! `gammainccinv` and `betaincinv`): an initial approximation, then Halley's method on the
//! function itself, whose first two derivatives are known in closed form.
//!
//! See the book entry [Log-domain functions and inverse incomplete integrals](https://jwkunz.github.io/lib_jsl/book/special_functions/log-domain.html#inverting-the-incomplete-gamma-and-beta-functions).

use crate::gamma::ln_gamma;
use crate::incomplete_beta::incomplete_beta;
use crate::incomplete_gamma::{incomplete_gamma, incomplete_gamma_upper};

/// The Abramowitz-Stegun 26.2.22 approximation to the normal quantile of `1 − pp`, `pp ≤ 1/2`.
fn normal_guess(pp: f64) -> f64 {
    let t = (-2.0 * pp.ln()).sqrt();
    t - (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481))
}

// ANCHOR: gammaincinv
/// The `x ≥ 0` with `P(a, x) = p` (SciPy's `gammaincinv`). The start is the Wilson-Hilferty
/// approximation for `a > 1` and the small-`x` series for `a ≤ 1`; then Halley steps
/// `x ← x − u / (1 − ½ u ((a−1)/x − 1))` with `u = (P − p)/P'` and `P' = x^{a−1} e^{−x} / Γ(a)`.
pub fn gammaincinv(a: f64, p: f64) -> f64 {
    invert_gamma(a, p, false)
}
// ANCHOR_END: gammaincinv

/// The `x ≥ 0` with `Q(a, x) = 1 − P(a, x) = q` (SciPy's `gammainccinv`). Both inverses iterate
/// on the smaller of the two tails, `P` or `Q`, which is computed with full relative accuracy;
/// the other target is formed exactly, since `1 − t` is exact for `t ∈ [½, 1]`.
pub fn gammainccinv(a: f64, q: f64) -> f64 {
    invert_gamma(a, q, true)
}

fn invert_gamma(a: f64, target: f64, upper: bool) -> f64 {
    if !(a > 0.0) || !(0.0..=1.0).contains(&target) {
        return f64::NAN;
    }
    if target > 0.5 {
        return invert_gamma(a, 1.0 - target, !upper);
    }
    let p = if upper { 1.0 - target } else { target };
    if (upper && target == 1.0) || (!upper && target == 0.0) {
        return 0.0;
    }
    if (upper && target == 0.0) || (!upper && target == 1.0) {
        return f64::INFINITY;
    }
    let gln = ln_gamma(a);
    let mut x = if a > 1.0 {
        let small = if upper { target } else { p }; // the tail probability nearer 0
        let z = if p < 0.5 { -normal_guess(small.min(1.0 - small).max(f64::MIN_POSITIVE)) } else { normal_guess(if upper { target } else { 1.0 - p }) };
        (a * (1.0 - 1.0 / (9.0 * a) + z / (3.0 * a.sqrt())).powi(3)).max(1e-3)
    } else {
        let t = 1.0 - a * (0.253 + a * 0.12);
        if p < t { (p / t).powf(1.0 / a) } else { 1.0 - (-(p - t) / (1.0 - t)).ln_1p() }
    };
    for _ in 0..100 {
        if x <= 0.0 {
            return 0.0;
        }
        let err = if upper { target - incomplete_gamma_upper(a, x) } else { incomplete_gamma(a, x) - target };
        let dp = ((a - 1.0) * x.ln() - x - gln).exp();
        if dp == 0.0 {
            break;
        }
        let u = err / dp;
        let step = u / (1.0 - 0.5 * (u * ((a - 1.0) / x - 1.0)).min(1.0));
        let next = x - step;
        x = if next <= 0.0 { 0.5 * x } else { next };
        if step.abs() <= 1e-15 * x {
            break;
        }
    }
    x
}

// ANCHOR: betaincinv
/// The `x ∈ [0, 1]` with `I_x(a, b) = p` (SciPy's `betaincinv`). The start is the normal-based
/// approximation of Abramowitz and Stegun 26.5.22 for `a, b ≥ 1` and the small-`x` and
/// small-`1 − x` power laws otherwise; then Halley steps with
/// `I' = x^{a−1}(1−x)^{b−1} / B(a, b)`.
pub fn betaincinv(a: f64, b: f64, p: f64) -> f64 {
    if !(a > 0.0) || !(b > 0.0) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 || p == 1.0 {
        return p;
    }
    let mut x = if a >= 1.0 && b >= 1.0 {
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let mut z = normal_guess(pp);
        if p < 0.5 {
            z = -z;
        }
        let al = (z * z - 3.0) / 6.0;
        let h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
        let w = z * (al + h).sqrt() / h - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h));
        a / (a + b * (2.0 * w).exp())
    } else {
        let lna = (a / (a + b)).ln();
        let lnb = (b / (a + b)).ln();
        let t = (a * lna).exp() / a;
        let u = (b * lnb).exp() / b;
        let w = t + u;
        if p < t / w { (a * w * p).powf(1.0 / a) } else { 1.0 - (b * w * (1.0 - p)).powf(1.0 / b) }
    };
    let afac = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    for j in 0..100 {
        if x <= 0.0 || x >= 1.0 {
            return x.clamp(0.0, 1.0);
        }
        let err = incomplete_beta(a, b, x) - p;
        let dp = ((a - 1.0) * x.ln() + (b - 1.0) * (-x).ln_1p() + afac).exp();
        if dp == 0.0 {
            break;
        }
        let u = err / dp;
        let t = u / (1.0 - 0.5 * (u * ((a - 1.0) / x - (b - 1.0) / (1.0 - x))).min(1.0));
        let mut next = x - t;
        if next <= 0.0 {
            next = 0.5 * x;
        }
        if next >= 1.0 {
            next = 0.5 * (x + 1.0);
        }
        x = next;
        if t.abs() <= 1e-15 * x && j > 0 {
            break;
        }
    }
    x
}
// ANCHOR_END: betaincinv

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_inverse_round_trips() {
        for a in [0.05, 0.3, 1.0, 2.5, 10.0, 150.0] {
            for p in [1e-12, 1e-4, 0.1, 0.5, 0.9, 0.9999] {
                let x = gammaincinv(a, p);
                assert!((incomplete_gamma(a, x) - p).abs() <= 1e-12 * p.max(1e-300) + 1e-15, "a {a} p {p}: x {x}");
                let x = gammainccinv(a, p);
                assert!((incomplete_gamma_upper(a, x) - p).abs() <= 1e-12 * p, "upper a {a} q {p}: x {x}");
            }
        }
    }

    #[test]
    fn beta_inverse_round_trips() {
        for (a, b) in [(0.2, 0.3), (0.5, 0.5), (1.0, 3.0), (2.0, 5.0), (30.0, 0.7), (100.0, 200.0)] {
            for p in [1e-10, 0.01, 0.3, 0.5, 0.8, 0.999] {
                let x = betaincinv(a, b, p);
                assert!((incomplete_beta(a, b, x) - p).abs() <= 1e-11 * p.max(1e-3), "a {a} b {b} p {p}: x {x}");
            }
        }
    }
}
