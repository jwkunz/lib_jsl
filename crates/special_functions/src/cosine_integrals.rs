//! Cosine integrals.

use std::f64::consts::FRAC_PI_2;

const EULER: f64 = 0.577_215_664_901_532_9;
const EPS: f64 = 1e-16;
const FPMIN: f64 = f64::MIN_POSITIVE / EPS;
const TMIN: f64 = 2.0;

/// Returns `(Si(|x|), Ci(|x|))` for finite `x`.
pub(crate) fn si_ci_positive(t: f64) -> (f64, f64) {
    if t == 0.0 {
        return (0.0, f64::NEG_INFINITY);
    }
    if t > TMIN {
        // Complex continued fraction (modified Lentz) for E1(it).
        let mut b = (1.0, t);
        let mut c = (1.0 / FPMIN, 0.0);
        let inv = |z: (f64, f64)| {
            let den = z.0 * z.0 + z.1 * z.1;
            (z.0 / den, -z.1 / den)
        };
        let mul = |a: (f64, f64), b: (f64, f64)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
        let mut d = inv(b);
        let mut h = d;
        for i in 2..100_000 {
            let a = -((i - 1) as f64) * ((i - 1) as f64);
            b.0 += 2.0;
            d = inv((a * d.0 + b.0, a * d.1 + b.1));
            let q = inv(c);
            c = (b.0 + a * q.0, b.1 + a * q.1);
            let del = mul(c, d);
            h = mul(h, del);
            if (del.0 - 1.0).abs() + del.1.abs() < EPS {
                break;
            }
        }
        h = mul((t.cos(), -t.sin()), h);
        (FRAC_PI_2 + h.1, -h.0)
    } else if t < FPMIN.sqrt() {
        (t, EULER + t.ln())
    } else {
        // Power series.
        let mut sum = 0.0;
        let mut sums = 0.0;
        let mut sumc = 0.0;
        let mut sign = 1.0;
        let mut fact = 1.0;
        let mut odd = true;
        for k in 1..1000 {
            fact *= t / k as f64;
            let term = fact / k as f64;
            sum += sign * term;
            let err = term / sum.abs();
            if odd {
                sign = -sign;
                sums = sum;
                sum = sumc;
            } else {
                sumc = sum;
                sum = sums;
            }
            if err < EPS {
                break;
            }
            odd = !odd;
        }
        (sums, sumc + t.ln() + EULER)
    }
}

/// The cosine integral Ci(x) = γ + ln x + ∫₀ˣ (cos t − 1)/t dt for `x > 0`.
///
/// Returns `-inf` at 0 and `NaN` for negative `x` (where Ci is complex).
pub fn cosine_integral(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x.is_infinite() {
        return 0.0;
    }
    si_ci_positive(x).1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values() {
        let cases = [
            (0.5, -0.17778407880661290),
            (1.0, 0.33740392290096813),
            (2.0, 0.42298082877486500),
            (5.0, -0.19002974965664388),
            (10.0, -0.045456433004455373),
            (100.0, -0.00514882514261049),
        ];
        for &(x, want) in &cases[..5] {
            assert!((cosine_integral(x) - want).abs() < 1e-14, "Ci({x})");
        }
        assert!((cosine_integral(cases[5].0) - cases[5].1).abs() < 1e-14);
    }

    #[test]
    fn edges() {
        assert_eq!(cosine_integral(0.0), f64::NEG_INFINITY);
        assert!(cosine_integral(-1.0).is_nan());
        assert_eq!(cosine_integral(f64::INFINITY), 0.0);
        // small-x behaviour Ci(x) ~ gamma + ln x
        let x = 1e-6f64;
        assert!((cosine_integral(x) - (0.5772156649015329 + x.ln())).abs() < 1e-12);
    }
}
