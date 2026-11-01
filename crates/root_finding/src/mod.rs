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

pub use bracketing::{bracket_root, bracket_roots_scan};
pub use bisection::bisection;
pub use secant_method::secant_method;
pub use false_position_method::false_position;
pub use ridders_method::ridders_method;
pub use van_wijngaarden_dekker_brent::brent_method;
pub use newton_raphson::newton_raphson;
pub use polynomial_roots::{polynomial_roots, polynomial_roots_complex};
pub use nonlinear_systems::{SystemOptions, SystemRootResult, broyden_system, newton_system};
