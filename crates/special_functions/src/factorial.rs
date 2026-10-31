//! Factorial.

use crate::gamma::{gamma, ln_gamma};

/// Largest `n` for which `n!` fits in a `u64`.
pub const MAX_U64_FACTORIAL: u64 = 20;

/// Returns `n!` if it fits in a `u64`, otherwise `None`.
pub fn factorial_checked(n: u64) -> Option<u64> {
    (2..=n).try_fold(1u64, |acc, k| acc.checked_mul(k))
}

/// Returns `n!`.
///
/// # Panics
/// Panics if `n > 20`, since `21!` overflows a `u64`. Use [`factorial_f64`] or
/// [`ln_factorial`] for larger arguments.
pub fn factorial(n: u64) -> u64 {
    factorial_checked(n).unwrap_or_else(|| panic!("factorial({n}) overflows u64 (max n is 20)"))
}

/// Returns `n!` as an `f64` (infinite for `n > 170`).
pub fn factorial_f64(n: u64) -> f64 {
    if n > 170 {
        f64::INFINITY
    } else {
        gamma(n as f64 + 1.0)
    }
}

/// Returns `ln(n!)`.
pub fn ln_factorial(n: u64) -> f64 {
    ln_gamma(n as f64 + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_values() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(10), 3_628_800);
        assert_eq!(factorial(20), 2_432_902_008_176_640_000);
    }

    #[test]
    fn overflow() {
        assert_eq!(factorial_checked(21), None);
        assert!(std::panic::catch_unwind(|| factorial(21)).is_err());
    }

    #[test]
    fn float_and_log() {
        assert_eq!(factorial_f64(5), 120.0);
        assert!((factorial_f64(25) / 1.5511210043330986e25 - 1.0).abs() < 1e-14);
        assert!(factorial_f64(171).is_infinite());
        assert!((ln_factorial(100) - 363.73937555556347).abs() < 1e-11);
    }
}
