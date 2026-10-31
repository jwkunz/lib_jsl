//! Dawson's integral.

use std::f64::consts::PI;
use std::sync::OnceLock;

const H: f64 = 0.2;
const NMAX: usize = 20;

fn coefficients() -> &'static [f64; NMAX] {
    static C: OnceLock<[f64; NMAX]> = OnceLock::new();
    C.get_or_init(|| {
        let mut c = [0.0; NMAX];
        for (i, v) in c.iter_mut().enumerate() {
            let t = (2 * i + 1) as f64 * H;
            *v = (-t * t).exp();
        }
        c
    })
}

/// Dawson's integral F(x) = e^(-x²) ∫₀ˣ e^(t²) dt.
///
/// Evaluated with Rybicki's method (relative accuracy about 1e-15). Odd in `x`.
pub fn dawson_integral(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let xx = x.abs();
    if xx.is_infinite() {
        return 0.0;
    }
    if xx < 0.2 {
        // Taylor series: F(x) = x - 2x³/3 + 4x⁵/15 - 8x⁷/105 + ...
        let x2 = x * x;
        let mut term = x;
        let mut sum = x;
        for k in 1..30 {
            term *= -2.0 * x2 / (2 * k + 1) as f64;
            sum += term;
            if term.abs() < 1e-18 * sum.abs() {
                break;
            }
        }
        return sum;
    }
    if xx > 1e8 {
        // F(x) ≈ 1/(2x) (1 + 1/(2x²))
        return (0.5 / x) * (1.0 + 0.5 / (x * x));
    }
    let c = coefficients();
    let n0 = 2.0 * (0.5 * xx / H + 0.5).floor();
    let xp = xx - n0 * H;
    let mut e1 = (2.0 * xp * H).exp();
    let e2 = e1 * e1;
    let mut d1 = n0 + 1.0;
    let mut d2 = d1 - 2.0;
    let mut sum = 0.0;
    for &ci in c.iter() {
        sum += ci * (e1 / d1 + 1.0 / (d2 * e1));
        d1 += 2.0;
        d2 -= 2.0;
        e1 *= e2;
    }
    let ans = (1.0 / PI.sqrt()) * (-xp * xp).exp() * sum;
    if x < 0.0 { -ans } else { ans }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values() {
        let cases = [
            (0.1, 0.09933599239785486),
            (0.5, 0.424_436_383_502_022_3),
            (1.0, 0.538_079_506_912_768_4),
            (2.0, 0.30134038892379197),
            (5.0, 0.10213407442427684),
            (10.0, 0.050_253_847_187_598_53),
            (-0.5, -0.424_436_383_502_022_3),
        ];
        for &(x, want) in &cases {
            assert!((dawson_integral(x) - want).abs() < 1e-14 * want.abs().max(1.0), "F({x})");
        }
    }

    #[test]
    fn asymptotics_and_edges() {
        assert_eq!(dawson_integral(0.0), 0.0);
        let x = 1e3;
        assert!((dawson_integral(x) / (0.5 / x * (1.0 + 0.5 / (x * x))) - 1.0).abs() < 1e-5);
        assert_eq!(dawson_integral(f64::INFINITY), 0.0);
    }

    #[test]
    fn branch_continuity() {
        let e = 1e-9;
        assert!((dawson_integral(0.2 - e) - dawson_integral(0.2 + e)).abs() < 1e-8);
    }
}
