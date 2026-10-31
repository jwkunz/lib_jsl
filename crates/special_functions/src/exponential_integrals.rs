//! Exponential integrals.

const EULER: f64 = 0.577_215_664_901_532_9;
const EPS: f64 = 1e-16;
const FPMIN: f64 = f64::MIN_POSITIVE / EPS;

/// The exponential integral E₁(x) = ∫ₓ^∞ e^(-t)/t dt for `x > 0`.
///
/// Returns `+inf` at 0 and `NaN` for negative `x` (where the function is complex).
pub fn exponential_integral(x: f64) -> f64 {
    exponential_integral_en(1, x)
}

/// The generalized exponential integral Eₙ(x) = ∫₁^∞ e^(-xt)/tⁿ dt, for `x >= 0`.
pub fn exponential_integral_en(n: u32, x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if n == 0 {
        return (-x).exp() / x;
    }
    let nm1 = (n - 1) as i64;
    if x == 0.0 {
        return if nm1 > 0 { 1.0 / nm1 as f64 } else { f64::INFINITY };
    }
    if x > 1.0 {
        // Continued fraction (modified Lentz).
        let mut b = x + n as f64;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=10_000 {
            let a = -(i as f64) * (nm1 as f64 + i as f64);
            b += 2.0;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            let del = c * d;
            h *= del;
            if (del - 1.0).abs() < EPS {
                break;
            }
        }
        h * (-x).exp()
    } else {
        // Power series.
        let mut ans = if nm1 != 0 { 1.0 / nm1 as f64 } else { -x.ln() - EULER };
        let mut fact = 1.0;
        for i in 1..=1000i64 {
            fact *= -x / i as f64;
            let del = if i != nm1 {
                -fact / (i - nm1) as f64
            } else {
                let mut psi = -EULER;
                for ii in 1..=nm1 {
                    psi += 1.0 / ii as f64;
                }
                fact * (-x.ln() + psi)
            };
            ans += del;
            if del.abs() < ans.abs() * EPS {
                break;
            }
        }
        ans
    }
}

/// The exponential integral Ei(x) = -PV ∫₋ₓ^∞ e^(-t)/t dt, defined for all real `x`
/// (principal value at 0, where it tends to -infinity).
pub fn exponential_integral_ei(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 0.0 {
        return -exponential_integral_en(1, -x);
    }
    if x < FPMIN {
        return x.ln() + EULER;
    }
    if x <= -EPS.ln() {
        let mut sum = 0.0;
        let mut fact = 1.0;
        for k in 1..=1000 {
            fact *= x / k as f64;
            let term = fact / k as f64;
            sum += term;
            if term < EPS * sum {
                break;
            }
        }
        sum + x.ln() + EULER
    } else {
        // Asymptotic series.
        let mut sum = 0.0;
        let mut term = 1.0;
        for k in 1..=1000 {
            let prev = term;
            term *= k as f64 / x;
            if term < EPS {
                break;
            }
            if term < prev {
                sum += term;
            } else {
                sum -= prev;
                break;
            }
        }
        x.exp() * (1.0 + sum) / x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs()
    }

    #[test]
    fn e1_values() {
        assert!(close(exponential_integral(1.0), 0.21938393439552029, 1e-14));
        assert!(close(exponential_integral(0.1), 1.8229239584193906, 1e-14));
        assert!(close(exponential_integral(5.0), 0.001148295591275326, 1e-13));
        assert!(exponential_integral(-1.0).is_nan());
        assert!(exponential_integral(0.0).is_infinite());
    }

    #[test]
    fn en_recurrence() {
        // n E_{n+1}(x) = e^-x - x E_n(x)
        for &x in &[0.3, 1.0, 2.5, 8.0] {
            for n in 1..5u32 {
                let lhs = n as f64 * exponential_integral_en(n + 1, x);
                let rhs = (-x).exp() - x * exponential_integral_en(n, x);
                assert!((lhs - rhs).abs() < 1e-14, "n={n} x={x}");
            }
        }
        assert!(close(exponential_integral_en(0, 2.0), (-2.0f64).exp() / 2.0, 1e-15));
    }

    #[test]
    fn ei_values() {
        assert!(close(exponential_integral_ei(1.0), 1.8951178163559368, 1e-14));
        assert!(close(exponential_integral_ei(10.0), 2492.228976241877, 1e-13));
        assert!(close(exponential_integral_ei(50.0), 1.058_563_689_713_169e20, 1e-12));
        assert!(close(exponential_integral_ei(-1.0), -0.21938393439552029, 1e-14));
        assert_eq!(exponential_integral_ei(0.0), f64::NEG_INFINITY);
    }
}
