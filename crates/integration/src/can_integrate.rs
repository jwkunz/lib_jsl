//! The common interface of the one-dimensional quadrature engines.

use lib_jsl_core::ErrorsJSL;

/// A one-dimensional quadrature rule on an interval `[a, b]`, approximating `∫ₐᵇ f(x) dx`.
///
/// Engines are configured at construction (number of points, tolerance, …); the bounds can be
/// changed afterwards, so one engine can be reused across many integrals. See the book's
/// [numerical integration chapter](https://jwkunz.github.io/lib_jsl/book/integration/index.html)
/// for how to choose an engine.
pub trait CanIntegrate1D {
    /// Sets the lower bound `a`.
    fn set_lower_bound(&mut self, x: f64);
    /// Sets the upper bound `b`.
    fn set_upper_bound(&mut self, x: f64);
    /// Approximates `∫ₐᵇ f(x) dx`.
    fn integrate<F>(&self, f: F) -> Result<f64, ErrorsJSL>
    where
        F: Fn(f64) -> f64;
}
