//! Finite difference solvers for model partial differential equations.
//!
//! * The heat equation `u_t = α u_xx` on `[0, L]` with Dirichlet ends: explicit FTCS (stable for
//!   `r = α Δt / Δx² ≤ ½`) and Crank-Nicolson (unconditionally stable, second order in time and
//!   space), the latter solving a tridiagonal system per step by the Thomas algorithm.
//! * The wave equation `u_tt = c² u_xx` with fixed ends by the explicit leapfrog scheme (stable
//!   for Courant number `c Δt / Δx ≤ 1`, exact on the grid when it equals one).
//! * The Poisson equation `−∇²u = f` on a rectangle with Dirichlet data, by the five-point stencil
//!   solved matrix-free with conjugate gradients.
//!
//! See the book entry [Finite difference methods](https://jwkunz.github.io/lib_jsl/book/integration/finite-differences.html).

use lib_jsl_core::ErrorsJSL;

/// Solves the tridiagonal system with sub-diagonal `a`, diagonal `b`, super-diagonal `c`
/// (`a[0]` and `c[n−1]` unused) by the Thomas algorithm.
pub fn solve_tridiagonal(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Result<Vec<f64>, ErrorsJSL> {
    let n = d.len();
    if a.len() != n || b.len() != n || c.len() != n || n == 0 {
        return Err(ErrorsJSL::IncompatibleArraySizes((a.len(), n)));
    }
    let mut cp = vec![0.0; n];
    let mut dp = vec![0.0; n];
    let mut denominator = b[0];
    if denominator == 0.0 {
        return Err(ErrorsJSL::RuntimeError("Zero pivot in the tridiagonal solve."));
    }
    cp[0] = c[0] / denominator;
    dp[0] = d[0] / denominator;
    for i in 1..n {
        denominator = b[i] - a[i] * cp[i - 1];
        if denominator == 0.0 {
            return Err(ErrorsJSL::RuntimeError("Zero pivot in the tridiagonal solve."));
        }
        cp[i] = c[i] / denominator;
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denominator;
    }
    let mut x = dp;
    for i in (0..n - 1).rev() {
        x[i] -= cp[i] * x[i + 1];
    }
    Ok(x)
}

/// The time-stepping scheme for [`heat_equation_1d`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeatScheme {
    /// Forward Euler in time, central in space.
    Explicit,
    /// The trapezoidal (Crank-Nicolson) average of explicit and implicit steps.
    CrankNicolson,
}

/// Advances `u_t = α u_xx` on a uniform grid of `u0.len()` points spanning `[0, length]`
/// (endpoints included and held at `left` and `right`) through `steps` steps of size `dt`.
#[allow(clippy::too_many_arguments)]
pub fn heat_equation_1d(u0: &[f64], length: f64, alpha: f64, dt: f64, steps: usize, left: f64, right: f64, scheme: HeatScheme) -> Result<Vec<f64>, ErrorsJSL> {
    let n = u0.len();
    if n < 3 || !(length > 0.0) || !(alpha > 0.0) || !(dt > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Need at least 3 points and positive length, diffusivity and step."));
    }
    let dx = length / (n - 1) as f64;
    let r = alpha * dt / (dx * dx);
    if scheme == HeatScheme::Explicit && r > 0.5 {
        return Err(ErrorsJSL::InvalidInputRange("The explicit scheme is unstable for α Δt / Δx² > 1/2."));
    }
    let mut u = u0.to_vec();
    u[0] = left;
    u[n - 1] = right;
    let m = n - 2;
    let (a, b, c) = (vec![-r / 2.0; m], vec![1.0 + r; m], vec![-r / 2.0; m]);
    for _ in 0..steps {
        let next: Vec<f64> = match scheme {
            HeatScheme::Explicit => (1..n - 1).map(|i| u[i] + r * (u[i - 1] - 2.0 * u[i] + u[i + 1])).collect(),
            HeatScheme::CrankNicolson => {
                let mut rhs: Vec<f64> = (1..n - 1).map(|i| u[i] + 0.5 * r * (u[i - 1] - 2.0 * u[i] + u[i + 1])).collect();
                rhs[0] += 0.5 * r * left;
                rhs[m - 1] += 0.5 * r * right;
                solve_tridiagonal(&a, &b, &c, &rhs)?
            }
        };
        u[1..n - 1].copy_from_slice(&next);
    }
    Ok(u)
}

/// Advances `u_tt = c² u_xx` with fixed ends `u = 0` from displacement `u0` and velocity `v0` on
/// a uniform grid spanning `[0, length]`, by leapfrog with a Taylor first step; returns the
/// displacement after `steps` steps of size `dt`.
pub fn wave_equation_1d(u0: &[f64], v0: &[f64], length: f64, speed: f64, dt: f64, steps: usize) -> Result<Vec<f64>, ErrorsJSL> {
    let n = u0.len();
    if n < 3 || v0.len() != n || !(length > 0.0) || !(speed > 0.0) || !(dt > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Need matching grids of at least 3 points and positive parameters."));
    }
    let dx = length / (n - 1) as f64;
    let courant2 = (speed * dt / dx).powi(2);
    if courant2 > 1.0 + 1e-12 {
        return Err(ErrorsJSL::InvalidInputRange("Leapfrog is unstable for a Courant number above one."));
    }
    let laplacian = |u: &[f64], i: usize| u[i - 1] - 2.0 * u[i] + u[i + 1];
    let mut previous = u0.to_vec();
    previous[0] = 0.0;
    previous[n - 1] = 0.0;
    if steps == 0 {
        return Ok(previous);
    }
    let mut current = previous.clone();
    for i in 1..n - 1 {
        current[i] = previous[i] + dt * v0[i] + 0.5 * courant2 * laplacian(&previous, i);
    }
    for _ in 1..steps {
        let mut next = vec![0.0; n];
        for i in 1..n - 1 {
            next[i] = 2.0 * current[i] - previous[i] + courant2 * laplacian(&current, i);
        }
        previous = current;
        current = next;
    }
    Ok(current)
}

/// Solves `−∇²u = f` on `[0, width] × [0, height]` with `u = g` on the boundary, on an
/// `nx × ny` grid of points including the boundary. Returns `u[j][i]` at `(i Δx, j Δy)`. The
/// five-point system is symmetric positive definite and is solved by conjugate gradients to a
/// relative residual of `1e-12`.
pub fn poisson_2d<F, G>(f: F, boundary: G, width: f64, height: f64, nx: usize, ny: usize) -> Result<Vec<Vec<f64>>, ErrorsJSL>
where
    F: Fn(f64, f64) -> f64,
    G: Fn(f64, f64) -> f64,
{
    if nx < 3 || ny < 3 || !(width > 0.0) || !(height > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Need at least 3 × 3 points on a positive rectangle."));
    }
    let (dx, dy) = (width / (nx - 1) as f64, height / (ny - 1) as f64);
    let (ix, iy) = (nx - 2, ny - 2);
    let (cx, cy) = (1.0 / (dx * dx), 1.0 / (dy * dy));
    let point = |i: usize, j: usize| (i as f64 * dx, j as f64 * dy);
    let mut u = vec![vec![0.0; nx]; ny];
    for (j, row) in u.iter_mut().enumerate() {
        for (i, v) in row.iter_mut().enumerate() {
            if i == 0 || j == 0 || i == nx - 1 || j == ny - 1 {
                let (x, y) = point(i, j);
                *v = boundary(x, y);
            }
        }
    }
    // Right-hand side with the boundary values moved over.
    let index = |i: usize, j: usize| (j - 1) * ix + (i - 1);
    let mut b = vec![0.0; ix * iy];
    for j in 1..ny - 1 {
        for i in 1..nx - 1 {
            let (x, y) = point(i, j);
            let mut value = f(x, y);
            if i == 1 {
                value += cx * u[j][0];
            }
            if i == nx - 2 {
                value += cx * u[j][nx - 1];
            }
            if j == 1 {
                value += cy * u[0][i];
            }
            if j == ny - 2 {
                value += cy * u[ny - 1][i];
            }
            b[index(i, j)] = value;
        }
    }
    let apply = |v: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; v.len()];
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = index(i, j);
                let mut s = 2.0 * (cx + cy) * v[k];
                if i > 1 {
                    s -= cx * v[k - 1];
                }
                if i < nx - 2 {
                    s -= cx * v[k + 1];
                }
                if j > 1 {
                    s -= cy * v[k - ix];
                }
                if j < ny - 2 {
                    s -= cy * v[k + ix];
                }
                out[k] = s;
            }
        }
        out
    };
    let x = conjugate_gradient(apply, &b, 1e-12, 10 * (ix * iy).max(100))?;
    for j in 1..ny - 1 {
        for i in 1..nx - 1 {
            u[j][i] = x[index(i, j)];
        }
    }
    Ok(u)
}

/// Conjugate gradients for a symmetric positive-definite operator `apply`, from zero, to the
/// relative residual `tolerance`. If `max_iterations` is reached first, a solution whose residual
/// is within a factor 1000 of the target is still returned (rounding often stalls the last digits);
/// otherwise the result is an error.
pub fn conjugate_gradient<A: Fn(&[f64]) -> Vec<f64>>(apply: A, b: &[f64], tolerance: f64, max_iterations: usize) -> Result<Vec<f64>, ErrorsJSL> {
    let dot = |a: &[f64], c: &[f64]| a.iter().zip(c).map(|(p, q)| p * q).sum::<f64>();
    let mut x = vec![0.0; b.len()];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rr = dot(&r, &r);
    let target = tolerance * tolerance * rr.max(1e-300);
    for _ in 0..max_iterations {
        if rr <= target {
            return Ok(x);
        }
        let ap = apply(&p);
        let pap = dot(&p, &ap);
        if pap <= 0.0 {
            return Err(ErrorsJSL::RuntimeError("The operator is not positive definite."));
        }
        let alpha = rr / pap;
        x.iter_mut().zip(&p).for_each(|(xi, pi)| *xi += alpha * pi);
        r.iter_mut().zip(&ap).for_each(|(ri, api)| *ri -= alpha * api);
        let rr_new = dot(&r, &r);
        let beta = rr_new / rr;
        p.iter_mut().zip(&r).for_each(|(pi, ri)| *pi = ri + beta * *pi);
        rr = rr_new;
    }
    if rr <= target * 1e6 { Ok(x) } else { Err(ErrorsJSL::RuntimeError("Conjugate gradients did not converge.")) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn heat_equation_decays_a_sine_mode_at_the_exact_rate() {
        let (n, alpha, t) = (41, 0.5, 0.2);
        let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let u0: Vec<f64> = x.iter().map(|&v| (PI * v).sin()).collect();
        let exact = |v: f64| (-alpha * PI * PI * t).exp() * (PI * v).sin();
        let dx = 1.0 / (n - 1) as f64;
        let dt = 0.4 * dx * dx / alpha;
        let steps = (t / dt).round() as usize;
        let explicit = heat_equation_1d(&u0, 1.0, alpha, t / steps as f64, steps, 0.0, 0.0, HeatScheme::Explicit).unwrap();
        let cn = heat_equation_1d(&u0, 1.0, alpha, 0.01, 20, 0.0, 0.0, HeatScheme::CrankNicolson).unwrap();
        for (i, &v) in x.iter().enumerate() {
            // The spatial truncation error at Δx = 1/40 is about 2e-4.
            assert!((explicit[i] - exact(v)).abs() < 5e-4 && (cn[i] - exact(v)).abs() < 5e-4, "x={v}");
        }
        // Crank-Nicolson is second order in time: halving Δt quarters the temporal error (on a
        // fine grid where the spatial error is negligible).
        let fine: Vec<f64> = (0..401).map(|i| (PI * i as f64 / 400.0).sin()).collect();
        let error = |steps: usize| {
            let u = heat_equation_1d(&fine, 1.0, alpha, t / steps as f64, steps, 0.0, 0.0, HeatScheme::CrankNicolson).unwrap();
            (u[200] - exact(0.5)).abs()
        };
        let ratio = error(5) / error(10);
        assert!(ratio > 3.5 && ratio < 4.5, "{ratio}");
        assert!(heat_equation_1d(&u0, 1.0, alpha, 1.0, 1, 0.0, 0.0, HeatScheme::Explicit).is_err());
    }

    #[test]
    fn wave_equation_is_exact_at_unit_courant_number() {
        // A standing wave sin(πx) cos(πct).
        let (n, c) = (51, 1.0);
        let dx = 1.0 / (n - 1) as f64;
        let u0: Vec<f64> = (0..n).map(|i| (PI * i as f64 * dx).sin()).collect();
        let u = wave_equation_1d(&u0, &vec![0.0; n], 1.0, c, dx, 75).unwrap();
        let t = 75.0 * dx;
        assert!(u.iter().enumerate().all(|(i, v)| (v - (PI * i as f64 * dx).sin() * (PI * c * t).cos()).abs() < 2e-4));
        assert!(wave_equation_1d(&u0, &vec![0.0; n], 1.0, c, 1.5 * dx, 1).is_err());
    }

    #[test]
    fn poisson_converges_at_second_order() {
        // −∇²u = 2π² sin(πx) sin(πy) has u = sin(πx) sin(πy); plus a harmonic boundary part x²−y².
        let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin() + x * x - y * y;
        let f = |x: f64, y: f64| 2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();
        let error = |n: usize| {
            let u = poisson_2d(f, exact, 1.0, 1.0, n, n).unwrap();
            let h = 1.0 / (n - 1) as f64;
            (0..n).flat_map(|j| (0..n).map(move |i| (i, j))).map(|(i, j)| (u[j][i] - exact(i as f64 * h, j as f64 * h)).abs()).fold(0.0, f64::max)
        };
        let (coarse, fine) = (error(17), error(33));
        assert!(fine < 1e-3 && coarse / fine > 3.6 && coarse / fine < 4.4, "{coarse} {fine}");
        assert!(solve_tridiagonal(&[0.0, 1.0], &[2.0, 2.0], &[1.0, 0.0], &[3.0, 3.0]).unwrap().iter().all(|v| (v - 1.0).abs() < 1e-15));
    }
}
