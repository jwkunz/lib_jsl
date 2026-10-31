//! Bessel functions of integer order.

use crate::bessel_functions_fractional::{bessel_fractional, bessel_i_fractional, bessel_jy, bessel_k_fractional, bessel_y_fractional};

/// The Bessel function of the first kind J_n(x) for integer order `n`.
///
/// Defined for all real `x` and all integers `n`, with `J_n(-x) = (-1)^n J_n(x)`.
pub fn bessel_j_integer(n: i32, x: f64) -> f64 {
    bessel_fractional(n as f64, x)
}

/// The Bessel function of the second kind Y_n(x) for integer order `n` and `x > 0`
/// (`-inf` at 0, `NaN` for negative `x`).
pub fn bessel_y_integer(n: i32, x: f64) -> f64 {
    bessel_y_fractional(n as f64, x)
}

/// Returns `(J_n(x), Y_n(x))` for integer `n` and `x > 0`, sharing the underlying computation.
pub fn bessel_jy_integer(n: i32, x: f64) -> (f64, f64) {
    bessel_jy(n as f64, x)
}

/// The modified Bessel function of the first kind I_n(x) for integer order `n`
/// (`I_n(-x) = (-1)^n I_n(x)`).
pub fn bessel_i_integer(n: i32, x: f64) -> f64 {
    if x < 0.0 {
        let r = bessel_i_fractional(n as f64, -x);
        return if n % 2 == 0 { r } else { -r };
    }
    bessel_i_fractional(n as f64, x)
}

/// The modified Bessel function of the second kind K_n(x) for integer order `n` and `x > 0`
/// (`+inf` at 0, `NaN` for negative `x`).
pub fn bessel_k_integer(n: i32, x: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    bessel_k_fractional(n as f64, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs()
    }

    #[test]
    fn j_values_and_symmetry() {
        assert!(close(bessel_j_integer(0, 1.0), 0.7651976865579666, 1e-14));
        assert!(close(bessel_j_integer(3, 4.0), 0.430171473875622, 1e-13));
        assert!(close(bessel_j_integer(-3, 4.0), -0.430171473875622, 1e-13));
        assert!(close(bessel_j_integer(3, -4.0), -0.430171473875622, 1e-13));
        assert_eq!(bessel_j_integer(0, 0.0), 1.0);
        assert_eq!(bessel_j_integer(5, 0.0), 0.0);
    }

    #[test]
    fn recurrence() {
        // J_{n-1} + J_{n+1} = (2n/x) J_n
        let x = 3.7;
        for n in 1..8 {
            let lhs = bessel_j_integer(n - 1, x) + bessel_j_integer(n + 1, x);
            let rhs = 2.0 * n as f64 / x * bessel_j_integer(n, x);
            assert!((lhs - rhs).abs() < 1e-13);
        }
    }

    #[test]
    fn other_kinds() {
        assert!(close(bessel_y_integer(0, 1.0), 0.08825696421567697, 1e-13));
        assert!(close(bessel_y_integer(2, 3.0), -0.16040039348492374, 1e-12));
        assert!(close(bessel_i_integer(0, 2.0), 2.279585302336067, 1e-14));
        assert!(close(bessel_i_integer(-1, 2.0), 1.590_636_854_637_329, 1e-13));
        assert!(close(bessel_i_integer(1, -2.0), -1.590_636_854_637_329, 1e-13));
        assert!(close(bessel_k_integer(2, 1.5), 0.583655963256651, 1e-12));
        assert!(bessel_k_integer(0, -1.0).is_nan());
        let (j, y) = bessel_jy_integer(1, 2.0);
        assert!(close(j, 0.5767248077568734, 1e-14) && close(y, -0.10703243154093754, 1e-12));
    }
}
