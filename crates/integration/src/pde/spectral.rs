//! Spectral methods: Chebyshev collocation and Fourier differentiation.
//!
//! For smooth solutions these converge exponentially fast in the number of points, far beyond
//! the algebraic rates of finite differences and finite elements.
//!
//! * [`chebyshev_differentiation_matrix`] on the Chebyshev-Gauss-Lobatto points (Trefethen,
//!   *Spectral Methods in MATLAB*), and [`chebyshev_bvp`] solving
//!   `u'' + p(x) u' + q(x) u = f(x)` with Dirichlet ends by collocation.
//! * [`fourier_derivative`] of periodic samples by the FFT, and [`periodic_heat`], which advances
//!   the periodic heat equation exactly in Fourier space.
//!
//! See the book entry [Spectral methods](https://jwkunz.github.io/lib_jsl/book/integration/spectral-methods.html).

use crate::stiff_ode::Lu;
use lib_jsl_core::ErrorsJSL;
use lib_jsl_ffts::{Fft, FftDirection, FftEngine1D, FftOrdering, FftScaleFactor};
use num_complex::Complex;
use std::f64::consts::{PI, TAU};

/// The `n + 1` Chebyshev-Gauss-Lobatto points `x_j = cos(πj/n)`, from 1 down to −1.
pub fn chebyshev_points(n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![1.0];
    }
    (0..=n).map(|j| (PI * j as f64 / n as f64).cos()).collect()
}

/// The `(n+1) × (n+1)` differentiation matrix on [`chebyshev_points`]: `(D u)_i ≈ u'(x_i)`,
/// exact for polynomials of degree `n`. The diagonal uses the negative-row-sum trick for
/// accuracy.
pub fn chebyshev_differentiation_matrix(n: usize) -> Vec<Vec<f64>> {
    let x = chebyshev_points(n);
    if n == 0 {
        return vec![vec![0.0]];
    }
    let c: Vec<f64> = (0..=n).map(|i| if i == 0 || i == n { 2.0 } else { 1.0 } * if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let mut d = vec![vec![0.0; n + 1]; n + 1];
    for i in 0..=n {
        for j in 0..=n {
            if i != j {
                d[i][j] = c[i] / c[j] / (x[i] - x[j]);
            }
        }
        d[i][i] = -d[i].iter().sum::<f64>();
    }
    d
}

/// Solves `u'' + p(x) u' + q(x) u = f(x)` on `[a, b]` with `u(a) = ua`, `u(b) = ub` by Chebyshev
/// collocation at `n + 1` points. Returns the points (from `b` down to `a`) and the solution.
#[allow(clippy::too_many_arguments)]
pub fn chebyshev_bvp<P, Q, F>(p: P, q: Q, f: F, a: f64, b: f64, ua: f64, ub: f64, n: usize) -> Result<(Vec<f64>, Vec<f64>), ErrorsJSL>
where
    P: Fn(f64) -> f64,
    Q: Fn(f64) -> f64,
    F: Fn(f64) -> f64,
{
    if n < 2 || !(b > a) {
        return Err(ErrorsJSL::InvalidInputRange("Need n >= 2 and a < b."));
    }
    let scale = 2.0 / (b - a);
    let xs: Vec<f64> = chebyshev_points(n).iter().map(|&x| a + (b - a) * (x + 1.0) / 2.0).collect();
    let d = chebyshev_differentiation_matrix(n);
    let size = n + 1;
    let d2: Vec<Vec<f64>> = (0..size).map(|i| (0..size).map(|j| (0..size).map(|k| d[i][k] * d[k][j]).sum::<f64>()).collect()).collect();
    let mut matrix = vec![vec![0.0; size]; size];
    let mut rhs = vec![0.0; size];
    for i in 0..size {
        if i == 0 || i == n {
            matrix[i][i] = 1.0;
            rhs[i] = if i == 0 { ub } else { ua };
            continue;
        }
        let (pi, qi) = (p(xs[i]), q(xs[i]));
        for j in 0..size {
            matrix[i][j] = scale * scale * d2[i][j] + pi * scale * d[i][j] + if i == j { qi } else { 0.0 };
        }
        rhs[i] = f(xs[i]);
    }
    let u = Lu::new(matrix)?.solve(&rhs);
    Ok((xs, u))
}

fn wavenumbers(n: usize, length: f64) -> Vec<f64> {
    (0..n).map(|m| { let k = if m <= n / 2 { m as f64 } else { m as f64 - n as f64 }; TAU * k / length }).collect()
}

fn transform(u: &[Complex<f64>], direction: FftDirection) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
    let scale = if matches!(direction, FftDirection::Forward) { FftScaleFactor::None } else { FftScaleFactor::N };
    Fft::with_plan(u.len(), scale, direction, FftOrdering::Standard)?.execute(u)
}

/// The `order`-th derivative of the periodic function sampled at `u.len()` equispaced points over
/// one period `length`, by multiplying its spectrum by `(ik)^order`. For odd orders and even
/// lengths the Nyquist mode is zeroed (its derivative is not representable).
pub fn fourier_derivative(u: &[f64], order: u32, length: f64) -> Result<Vec<f64>, ErrorsJSL> {
    let n = u.len();
    if n == 0 || !(length > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Need samples and a positive period."));
    }
    let input: Vec<Complex<f64>> = u.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut spectrum = transform(&input, FftDirection::Forward)?;
    for (m, (s, k)) in spectrum.iter_mut().zip(wavenumbers(n, length)).enumerate() {
        if order % 2 == 1 && n.is_multiple_of(2) && m == n / 2 {
            *s = Complex::new(0.0, 0.0);
        } else {
            *s *= Complex::new(0.0, k).powu(order);
        }
    }
    Ok(transform(&spectrum, FftDirection::Inverse)?.iter().map(|c| c.re).collect())
}

/// Advances `u_t = α u_xx` with periodic boundaries over `length` exactly in Fourier space:
/// each mode decays by `e^{−α k² t}`. Exact in time and spectrally accurate in space.
pub fn periodic_heat(u0: &[f64], alpha: f64, t: f64, length: f64) -> Result<Vec<f64>, ErrorsJSL> {
    if u0.is_empty() || !(length > 0.0) || !(alpha >= 0.0) || !(t >= 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Need samples, a positive period and non-negative α and t."));
    }
    let input: Vec<Complex<f64>> = u0.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut spectrum = transform(&input, FftDirection::Forward)?;
    for (s, k) in spectrum.iter_mut().zip(wavenumbers(u0.len(), length)) {
        *s *= (-alpha * k * k * t).exp();
    }
    Ok(transform(&spectrum, FftDirection::Inverse)?.iter().map(|c| c.re).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chebyshev_differentiation_converges_spectrally() {
        let error = |n: usize| {
            let x = chebyshev_points(n);
            let d = chebyshev_differentiation_matrix(n);
            let u: Vec<f64> = x.iter().map(|v| (v * 2.0).exp() * v.sin()).collect();
            (0..=n).map(|i| {
                let du: f64 = (0..=n).map(|j| d[i][j] * u[j]).sum();
                (du - (2.0 * x[i]).exp() * (2.0 * x[i].sin() + x[i].cos())).abs()
            }).fold(0.0, f64::max)
        };
        assert!(error(8) > 1e-6 && error(24) < 1e-11, "{} {}", error(8), error(24));
        // Exact for polynomials of degree n.
        let d = chebyshev_differentiation_matrix(4);
        let x = chebyshev_points(4);
        assert!((0..5).all(|i| ((0..5).map(|j| d[i][j] * x[j].powi(4)).sum::<f64>() - 4.0 * x[i].powi(3)).abs() < 1e-12));
    }

    #[test]
    fn chebyshev_bvp_reaches_machine_precision() {
        // Trefethen's program 13: u'' = e^{4x}, u(±1) = 0.
        let exact = |x: f64| ((4.0 * x).exp() - x * 4f64.sinh() - 4f64.cosh()) / 16.0;
        let (xs, u) = chebyshev_bvp(|_| 0.0, |_| 0.0, |x| (4.0 * x).exp(), -1.0, 1.0, 0.0, 0.0, 24).unwrap();
        assert!(xs.iter().zip(&u).all(|(x, v)| (v - exact(*x)).abs() < 1e-12));
        // Variable coefficients on another interval: u'' + u' / x − u / x² = 0 has u = x on [1, 3]
        // (written with p = 1/x, q = −1/x²), plus Dirichlet data u(1) = 1, u(3) = 3.
        let (xs, u) = chebyshev_bvp(|x| 1.0 / x, |x| -1.0 / (x * x), |_| 0.0, 1.0, 3.0, 1.0, 3.0, 12).unwrap();
        assert!(xs.iter().zip(&u).all(|(x, v)| (v - x).abs() < 1e-12));
        assert!(chebyshev_bvp(|_| 0.0, |_| 0.0, |_| 0.0, 1.0, 0.0, 0.0, 0.0, 8).is_err());
    }

    #[test]
    fn fourier_methods_are_spectrally_accurate() {
        let n = 32;
        let x: Vec<f64> = (0..n).map(|i| TAU * i as f64 / n as f64).collect();
        let u: Vec<f64> = x.iter().map(|v| v.sin().exp()).collect();
        let du = fourier_derivative(&u, 1, TAU).unwrap();
        assert!(x.iter().zip(&du).all(|(v, d)| (d - v.cos() * v.sin().exp()).abs() < 1e-12));
        let d2 = fourier_derivative(&u, 2, TAU).unwrap();
        assert!(x.iter().zip(&d2).all(|(v, d)| (d - (v.cos().powi(2) - v.sin()) * v.sin().exp()).abs() < 1e-11));
        // Heat: sin(3x) decays by e^{−9αt}.
        let u0: Vec<f64> = x.iter().map(|v| (3.0 * v).sin() + 2.0).collect();
        let u = periodic_heat(&u0, 0.1, 0.5, TAU).unwrap();
        assert!(x.iter().zip(&u).all(|(v, w)| (w - (2.0 + (-0.9f64 * 0.5).exp() * (3.0 * v).sin())).abs() < 1e-13));
        assert!(fourier_derivative(&[], 1, 1.0).is_err());
    }
}
