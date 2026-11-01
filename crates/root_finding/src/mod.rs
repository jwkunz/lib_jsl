//! Root finding algorithms.

pub mod bracketing;
pub mod bisection;
pub mod secant_method;
pub mod false_position_method;
pub mod ridders_method;
pub mod van_wijngaarden_dekker_brent;
pub mod newton_raphson;
pub mod polynomial_roots;
pub mod nonlinear_systems;

/// `x * y > 0` without the product, which underflows to zero (or overflows) for extreme values.
pub(crate) fn same_sign(x: f64, y: f64) -> bool {
    (x > 0.0 && y > 0.0) || (x < 0.0 && y < 0.0)
}

pub use bracketing::{bracket_root, bracket_roots_scan};
pub use bisection::bisection;
pub use secant_method::secant_method;
pub use false_position_method::false_position;
pub use ridders_method::ridders_method;
pub use van_wijngaarden_dekker_brent::brent_method;
pub use newton_raphson::newton_raphson;
pub use polynomial_roots::{polynomial_roots, polynomial_roots_complex};
pub use nonlinear_systems::{SystemOptions, SystemRootResult, broyden_system, newton_system};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bracketing_methods_survive_tiny_and_huge_function_values() {
        // The products f(a)·f(m) once underflowed to zero here, so bisection always kept the
        // left half and converged to the right end of the bracket.
        for scale in [1e-200, 1.0, 1e200] {
            let f = |x: f64| scale * (x - 1.0 / 3.0);
            for (name, r) in [
                ("bisection", bisection(f, 0.0, 1.0, 1e-13)),
                ("brent", brent_method(f, 0.0, 1.0, 1e-13)),
                ("false position", false_position(f, 0.0, 1.0, 1e-13)),
                ("ridders", ridders_method(f, 0.0, 1.0, 1e-13)),
            ] {
                let r = r.unwrap();
                assert!((r - 1.0 / 3.0).abs() < 1e-12, "{name} at scale {scale:e}: {r}");
            }
            let (lo, hi) = bracket_root(f, 2.0, 3.0).unwrap();
            assert!(lo <= 1.0 / 3.0 && 1.0 / 3.0 <= hi, "{scale:e}: {lo} {hi}");
            assert_eq!(bracket_roots_scan(f, 0.0, 1.0, 10).len(), 1);
        }
        assert!(bisection(|_| f64::NAN, 0.0, 1.0, 1e-9).is_err());
    }
}
