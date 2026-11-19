//! Zeros of the integer-order Bessel functions and their derivatives (SciPy's `jn_zeros`,
//! `jnp_zeros`, `yn_zeros` and `ynp_zeros`).
//!
//! See the book entry [Faddeeva, Struve, Kelvin and Hankel functions](https://jwkunz.github.io/lib_jsl/book/special_functions/faddeeva-struve-kelvin.html#hankel-functions-and-bessel-zeros).

use crate::bessel_functions_fractional::bessel_jy;

/// Which function's zeros to find.
#[derive(Clone, Copy)]
enum Kind {
    J,
    Jp,
    Y,
    Yp,
}

fn evaluate(kind: Kind, n: u32, x: f64) -> f64 {
    let v = n as f64;
    match kind {
        Kind::J => bessel_jy(v, x).0,
        Kind::Y => bessel_jy(v, x).1,
        Kind::Jp | Kind::Yp => {
            // F'_n = (F_{n−1} − F_{n+1})/2, and F'_0 = −F_1.
            let (j1, y1) = bessel_jy(v + 1.0, x);
            let (a, b) = if n == 0 { (-j1, -y1) } else {
                let (j0, y0) = bessel_jy(v - 1.0, x);
                ((j0 - j1) / 2.0, (y0 - y1) / 2.0)
            };
            if matches!(kind, Kind::Jp) { a } else { b }
        }
    }
}

/// The Illinois variant of regula falsi on a bracket with a sign change, to full precision.
fn refine(f: impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
    let (mut fa, mut fb) = (f(a), f(b));
    let mut side = 0;
    for _ in 0..200 {
        let c = (a * fb - b * fa) / (fb - fa);
        let c = if c > a.min(b) && c < a.max(b) { c } else { 0.5 * (a + b) };
        let fc = f(c);
        if fc == 0.0 || (b - a).abs() <= 4.0 * f64::EPSILON * c.abs() {
            return c;
        }
        if fc.signum() == fb.signum() {
            b = c;
            fb = fc;
            if side == -1 {
                fa /= 2.0;
            }
            side = -1;
        } else {
            a = c;
            fa = fc;
            if side == 1 {
                fb /= 2.0;
            }
            side = 1;
        }
        if (b - a).abs() <= 4.0 * f64::EPSILON * b.abs() {
            break;
        }
    }
    let (fa, fb) = (f(a).abs(), f(b).abs());
    if fa < fb { a } else { b }
}

/// The first `nt` positive zeros, scanning from `0.99 n` (every zero of these functions lies
/// above `n`) in steps of ¼ — the zeros are at least 2.5 apart — and refining each sign change.
fn zeros(kind: Kind, n: u32, nt: usize) -> Vec<f64> {
    let f = |x: f64| evaluate(kind, n, x);
    let mut out = Vec::with_capacity(nt);
    let step = 0.25;
    let mut a = (0.99 * n as f64).max(1e-3);
    let mut fa = f(a);
    while out.len() < nt {
        let b = a + step;
        let fb = f(b);
        if fb == 0.0 {
            out.push(b);
        } else if fa.signum() != fb.signum() && fa != 0.0 {
            out.push(refine(f, a, b));
        }
        a = b;
        fa = fb;
    }
    out
}

// ANCHOR: zeros
/// The first `nt` positive zeros `j_{n,k}` of `J_n` (SciPy's `jn_zeros`).
pub fn jn_zeros(n: u32, nt: usize) -> Vec<f64> {
    zeros(Kind::J, n, nt)
}

/// The first `nt` positive zeros `j'_{n,k}` of `J_n'` (SciPy's `jnp_zeros`); for `n = 0` the
/// zero at the origin is not counted, so these are the zeros of `J₁`.
pub fn jnp_zeros(n: u32, nt: usize) -> Vec<f64> {
    zeros(Kind::Jp, n, nt)
}

/// The first `nt` positive zeros `y_{n,k}` of `Y_n` (SciPy's `yn_zeros`).
pub fn yn_zeros(n: u32, nt: usize) -> Vec<f64> {
    zeros(Kind::Y, n, nt)
}

/// The first `nt` positive zeros `y'_{n,k}` of `Y_n'` (SciPy's `ynp_zeros`).
pub fn ynp_zeros(n: u32, nt: usize) -> Vec<f64> {
    zeros(Kind::Yp, n, nt)
}
// ANCHOR_END: zeros

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn tabulated_zeros_and_mcmahon() {
        // Abramowitz and Stegun, table 9.5.
        let j0 = jn_zeros(0, 3);
        for (z, want) in j0.iter().zip([2.404825557695773, 5.520078110286311, 8.653727912911013]) {
            assert!((z - want).abs() < 1e-14);
        }
        assert!((jnp_zeros(0, 1)[0] - 3.831705970207512).abs() < 1e-14);
        assert!((jnp_zeros(1, 1)[0] - 1.841183781340659).abs() < 1e-14);
        assert!((yn_zeros(0, 1)[0] - 0.8935769662791675).abs() < 1e-14);
        assert!((ynp_zeros(0, 1)[0] - 2.197141326031017).abs() < 1e-14);
        // McMahon: j_{0,k} ≈ (k − ¼)π + 1/(8(k − ¼)π) for large k.
        let z = jn_zeros(0, 40);
        let b = (40.0 - 0.25) * PI;
        assert!((z[39] - (b + 1.0 / (8.0 * b))).abs() < 1e-6);
        // Interlacing j_{n,1} < j_{n+1,1} < j_{n,2}.
        let (a, c) = (jn_zeros(5, 2), jn_zeros(6, 1));
        assert!(a[0] < c[0] && c[0] < a[1]);
    }
}
