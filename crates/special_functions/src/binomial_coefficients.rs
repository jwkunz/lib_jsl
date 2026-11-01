//! Binomial coefficients.
//!
//! See the book entry [The gamma function and related functions](https://jwkunz.github.io/lib_jsl/book/special_functions/gamma.html#factorials-and-binomial-coefficients).

use crate::gamma::ln_gamma;

/// Returns C(n, k) if it fits in a `u64`, otherwise `None`. C(n, k) is 0 when `k > n`.
pub fn binomial_coefficient_checked(n: u64, k: u64) -> Option<u64> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut result: u128 = 1;
    for i in 0..k {
        // result = result * (n - i) / (i + 1) is always an exact integer division.
        result = result.checked_mul((n - i) as u128)? / (i + 1) as u128;
        if result > u64::MAX as u128 {
            return None;
        }
    }
    Some(result as u64)
}

/// Returns the binomial coefficient C(n, k).
///
/// # Panics
/// Panics if the result does not fit in a `u64`; see [`binomial_coefficient_checked`] and
/// [`ln_binomial_coefficient`] for alternatives.
pub fn binomial_coefficient(n: u64, k: u64) -> u64 {
    binomial_coefficient_checked(n, k)
        .unwrap_or_else(|| panic!("binomial_coefficient({n}, {k}) overflows u64"))
}

/// Returns ln C(n, k) (negative infinity when `k > n`).
pub fn ln_binomial_coefficient(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let (n, k) = (n as f64, k as f64);
    ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small() {
        assert_eq!(binomial_coefficient(5, 2), 10);
        assert_eq!(binomial_coefficient(10, 0), 1);
        assert_eq!(binomial_coefficient(10, 10), 1);
        assert_eq!(binomial_coefficient(3, 5), 0);
        assert_eq!(binomial_coefficient(52, 5), 2_598_960);
    }

    #[test]
    fn large_exact() {
        assert_eq!(binomial_coefficient(60, 30), 118_264_581_564_861_424);
        assert_eq!(binomial_coefficient(64, 32), 1_832_624_140_942_590_534);
    }

    #[test]
    fn overflow() {
        assert_eq!(binomial_coefficient_checked(100, 50), None);
    }

    #[test]
    fn logarithm() {
        let want = 100891344545564193334812497256f64.ln();
        assert!((ln_binomial_coefficient(100, 50) - want).abs() < 1e-11);
        assert_eq!(ln_binomial_coefficient(3, 5), f64::NEG_INFINITY);
    }
}
