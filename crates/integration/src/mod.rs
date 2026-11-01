//! Numerical integration: quadrature in one and many dimensions, ordinary differential
//! equations, boundary value problems and partial differential equations.
//!
//! The companion book explains how and why each method works, with derivations, history and
//! figures: <https://jwkunz.github.io/lib_jsl/book/integration/index.html>.
//!
//! # Quadrature in one dimension
//!
//! Most engines implement [`can_integrate::CanIntegrate1D`].
//!
//! | Method | Module | Best for |
//! |--------|--------|----------|
//! | Trapezoid, Simpson, midpoint | [`trapezoidal_integration_engine`], [`simpsons_1_3_integration_engine`], [`midpoint_integration_engine`] | teaching; periodic integrands (trapezoid) |
//! | Romberg | [`romberg`] | smooth integrands, cheap function evaluations |
//! | Gauss-Legendre, -Hermite, -Laguerre | [`gaussian_quadrature`] | smooth integrands, expensive evaluations, weighted and infinite intervals |
//! | Adaptive Gauss-Kronrod, adaptive trapezoid | [`adaptive_gl7k15_integration_engine`], [`adaptive_trapezoid_integration_engine`] | integrands with localised features |
//! | Tanh-sinh | [`tanh_sinh`] | endpoint singularities |
//! | Complex contour integrals | [`complex_path_integrals`] | analytic functions along paths |
//!
//! # Many dimensions
//!
//! [`cubature`] (adaptive Genz-Malik and tensor Gauss rules) for up to about ten dimensions;
//! [`monte_carlo_integration`] and [`quasi_monte_carlo`] beyond.
//!
//! # Differential equations
//!
//! [`runge_kutta_4::rk4_solve`] (fixed step) and [`dormand_prince::dormand_prince_solve`]
//! (adaptive) for non-stiff initial value problems `y' = f(t, y)`,
//! [`stiff_ode`] (Rosenbrock and BDF) for stiff ones, [`boundary_value_problems`] (shooting and
//! collocation), and [`pde`] (finite difference, finite element and spectral methods).

pub mod adaptive_gl7k15_integration_engine;
pub mod adaptive_trapezoid_integration_engine;
pub mod boundary_value_problems;
pub mod calculate_midpoints;
pub mod can_integrate;
pub mod complex_path_integrals;
pub mod cubature;
pub mod dormand_prince;
pub mod gaussian_quadrature;
pub mod midpoint_integration_engine;
pub mod monte_carlo_integration;
pub mod pde;
pub mod quasi_monte_carlo;
pub mod romberg;
pub mod runge_kutta_4;
pub mod simpsons_1_3_integration_engine;
pub mod stiff_ode;
pub mod tanh_sinh;
pub mod trapezoidal_integration_engine;
