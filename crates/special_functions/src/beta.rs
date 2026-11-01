//! Beta function.
//!
//! See the book entry [The gamma function and related functions](https://jwkunz.github.io/lib_jsl/book/special_functions/gamma.html#the-beta-function).

use crate::gamma::{gamma, ln_gamma};

/// The beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b).
///
/// For positive arguments the result is accurate across the whole range (it switches to
/// logarithms to avoid overflow). For other real arguments the gamma ratio is used directly.
pub fn beta(a: f64, b: f64) -> f64 {
    if a > 0.0 && b > 0.0 {
        if a + b < 100.0 {
            gamma(a) * gamma(b) / gamma(a + b)
        } else {
            ln_beta(a, b).exp()
        }
    } else {
        gamma(a) * gamma(b) / gamma(a + b)
    }
}

/// The natural logarithm of B(a, b) for positive `a` and `b`.
pub fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert!((beta(2.0, 3.0) - 1.0 / 12.0).abs() < 1e-15);
        assert!((beta(0.5, 0.5) - std::f64::consts::PI).abs() < 1e-14);
        assert!((beta(1.0, 7.0) - 1.0 / 7.0).abs() < 1e-15);
    }

    #[test]
    fn symmetric_and_large() {
        assert!((beta(3.3, 8.1) - beta(8.1, 3.3)).abs() < 1e-18);
        let v = beta(80.0, 90.0);
        assert!((v.ln() - ln_beta(80.0, 90.0)).abs() < 1e-10);
    }
}
