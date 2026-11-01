//! Polynomial roots as the eigenvalues of the companion matrix: all (complex) roots, or the
//! real ones, polished and with repeated roots merged.
//!
//! See the book entry [Polynomial roots](https://jwkunz.github.io/lib_jsl/book/root_finding/polynomial-roots.html).

use lib_jsl_linalg::EigVals;
use ndarray::Array2;
use num_complex::Complex;

fn eval(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().fold(0.0, |acc, &c| acc * x + c)
}

/// `Σ |aᵢ| |x|^(n−i)`, the scale of rounding errors in evaluating the polynomial at `x`.
fn eval_abs(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().fold(0.0, |acc, &c| acc * x.abs() + c.abs())
}

fn derivative(coeffs: &[f64]) -> Vec<f64> {
    let n = coeffs.len() - 1;
    coeffs[..n].iter().enumerate().map(|(i, &c)| c * (n - i) as f64).collect()
}

/// Strips leading zero coefficients and factors out roots at zero; returns the reduced
/// coefficients and the multiplicity of the zero root.
fn normalise(coeffs: &[f64]) -> Option<(&[f64], usize)> {
    let start = coeffs.iter().position(|&c| c != 0.0)?;
    let mut c = &coeffs[start..];
    let mut zeros = 0;
    while c.len() > 1 && *c.last().unwrap() == 0.0 {
        c = &c[..c.len() - 1];
        zeros += 1;
    }
    Some((c, zeros))
}

/// Returns the real roots of a polynomial in ascending order.
///
/// `coeffs` are ordered from the highest degree to the constant term, e.g. `[1, 0, -2]` is
/// `x^2 - 2`. The roots are the real eigenvalues of the companion matrix (see
/// [`polynomial_roots_complex`]), each polished by Newton's method; an eigenvalue with an imaginary
/// part below `10⁻³ |z|` counts as real when the polynomial vanishes at its real part to within
/// rounding error. A repeated root, which appears as a cluster of nearby eigenvalues, is reported once, at
/// the cluster's mean (accurate to about `ε^(1/m)` for multiplicity `m`, often much better).
/// Distinct roots closer together than the rounding error of the polynomial can resolve are merged.
pub fn polynomial_roots(coeffs: &[f64]) -> Vec<f64> {
    let Some((reduced, zeros)) = normalise(coeffs) else { return Vec::new() };
    // The QR iteration has never been observed to fail on a companion matrix; if it did, there
    // would be no roots to report.
    let mut r = real_roots(reduced).unwrap_or_default();
    if zeros > 0 {
        r.push(0.0);
    }
    r.sort_by(f64::total_cmp);
    r.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * (1.0 + a.abs()));
    r
}

// ANCHOR: real_roots
fn real_roots(coeffs: &[f64]) -> Result<Vec<f64>, String> {
    let deg = coeffs.len() - 1;
    match deg {
        0 => return Ok(Vec::new()),
        1 => return Ok(vec![-coeffs[1] / coeffs[0]]),
        _ => {}
    }
    // The rounding error in evaluating p at x: below it, the value of p(x) means nothing.
    let noise = |x: f64| 8.0 * deg as f64 * f64::EPSILON * eval_abs(coeffs, x);
    let mut eig = companion_eigenvalues(coeffs)?;
    // A real eigenvalue is a real root; a nearly real pair is a multiple real root perturbed by
    // rounding (by about ε^(1/m)), if p vanishes at its real part to within rounding error.
    eig.retain(|z| z.im == 0.0 || (z.im.abs() <= 1e-3 * z.norm().max(1.0) && eval(coeffs, z.re).abs() <= noise(z.re)));
    eig.sort_by(|a, b| a.re.total_cmp(&b.re));
    // Group eigenvalues that p cannot tell apart: close together, with p in the noise between.
    let mut out = Vec::new();
    let mut i = 0;
    while i < eig.len() {
        let mut j = i + 1;
        while j < eig.len() {
            let (a, b) = (eig[j - 1].re, eig[j].re);
            let mid = 0.5 * (a + b);
            if b - a > 1e-3 * mid.abs().max(1.0) || eval(coeffs, mid).abs() > noise(mid) {
                break;
            }
            j += 1;
        }
        let mean = eig[i..j].iter().map(|z| z.re).sum::<f64>() / (j - i) as f64;
        out.push(if j - i == 1 { polish(coeffs, mean) } else { mean });
        i = j;
    }
    Ok(out)
}

/// A few Newton steps, each accepted only if it reduces |p|.
fn polish(coeffs: &[f64], mut x: f64) -> f64 {
    let d = derivative(coeffs);
    let mut fx = eval(coeffs, x).abs();
    for _ in 0..4 {
        let step = eval(coeffs, x) / eval(&d, x);
        let y = x - step;
        let fy = eval(coeffs, y).abs();
        if fy.is_nan() || fy >= fx {
            break;
        }
        (x, fx) = (y, fy);
    }
    x
}
// ANCHOR_END: real_roots

fn companion_eigenvalues(coeffs: &[f64]) -> Result<Vec<Complex<f64>>, String> {
    let n = coeffs.len() - 1;
    // ANCHOR: companion
    // p(z)/a₀ = zⁿ + c₁ zⁿ⁻¹ + … + cₙ is the characteristic polynomial of this matrix.
    let mut companion = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        companion[[0, j]] = -coeffs[j + 1] / coeffs[0];
    }
    for i in 1..n {
        companion[[i, i - 1]] = 1.0;
    }
    let values = companion.eigvals().map_err(|e| format!("polynomial_roots_complex: {e:?}"))?;
    // ANCHOR_END: companion
    Ok(values.to_vec())
}

/// All roots of a polynomial, complex ones included, as the eigenvalues of its companion matrix
/// (the method of MATLAB's `roots`), sorted by real part and then imaginary part. Repeated roots
/// appear with their multiplicity (and, being ill-conditioned, are accurate only to about
/// `ε^(1/m)` for multiplicity `m`).
pub fn polynomial_roots_complex(coeffs: &[f64]) -> Result<Vec<Complex<f64>>, String> {
    let Some((coeffs, zeros)) = normalise(coeffs) else { return Ok(Vec::new()) };
    let mut roots = vec![Complex::new(0.0, 0.0); zeros];
    if coeffs.len() > 1 {
        roots.extend(companion_eigenvalues(coeffs)?);
    }
    roots.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
    Ok(roots)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic() {
        let r = polynomial_roots(&[1.0, 0.0, -2.0]);
        assert_eq!(r.len(), 2);
        assert!((r[0] + 2f64.sqrt()).abs() < 1e-10);
        assert!((r[1] - 2f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn cubic_three_real() {
        // (x-1)(x-2)(x-3)
        let r = polynomial_roots(&[1.0, -6.0, 11.0, -6.0]);
        assert_eq!(r.len(), 3);
        for (got, want) in r.iter().zip([1.0, 2.0, 3.0]) {
            assert!((got - want).abs() < 1e-9);
        }
    }

    #[test]
    fn no_real_roots() {
        assert!(polynomial_roots(&[1.0, 0.0, 1.0]).is_empty());
    }

    #[test]
    fn zero_root_and_leading_zero() {
        // 0*x^3 + x^2 - x
        let r = polynomial_roots(&[0.0, 1.0, -1.0, 0.0]);
        assert_eq!(r.len(), 2);
        assert!(r[0].abs() < 1e-12 && (r[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn repeated_root() {
        // (x-1)^2
        let r = polynomial_roots(&[1.0, -2.0, 1.0]);
        assert_eq!(r.len(), 1);
        assert!((r[0] - 1.0).abs() < 1e-6);
    }

    fn poly_from_roots(r: &[f64]) -> Vec<f64> {
        let mut c = vec![1.0];
        for &x in r {
            let mut n = vec![0.0; c.len() + 1];
            for (i, &a) in c.iter().enumerate() {
                n[i] += a;
                n[i + 1] -= a * x;
            }
            c = n;
        }
        c
    }

    #[test]
    fn wilkinson_polynomials_give_exactly_their_roots() {
        // The old isolation between critical points accepted critical points as roots (25 roots at
        // degree 10) and its bound overflowed at degree 20; without balancing, the companion
        // eigenvalues of the degree-20 polynomial were garbage.
        for (n, tol) in [(10usize, 1e-9), (15, 1e-3), (20, 1e-2)] {
            let roots: Vec<f64> = (1..=n).map(|k| k as f64).collect();
            let found = polynomial_roots(&poly_from_roots(&roots));
            assert_eq!(found.len(), n, "degree {n}: {found:?}");
            let err = found.iter().zip(&roots).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            // Wilkinson's roots are ill-conditioned: errors grow quickly with the degree.
            assert!(err < tol, "degree {n}: error {err:e}");
        }
        // Wilkinson's perturbation: 2⁻²³ off the x¹⁹ coefficient leaves ten real roots.
        let mut w = poly_from_roots(&(1..=20).map(|k| k as f64).collect::<Vec<_>>());
        w[1] -= 2f64.powi(-23);
        let found = polynomial_roots(&w);
        assert_eq!(found.len(), 10, "{found:?}");
        assert!((found[9] - 20.8469).abs() < 1e-3, "{found:?}");
        let r = polynomial_roots(&poly_from_roots(&[1.0, 1.0, 1.0]));
        assert!(r.len() == 1 && (r[0] - 1.0).abs() < 1e-12, "{r:?}");
    }

    #[test]
    fn complex_roots_from_the_companion_matrix() {
        let r = polynomial_roots_complex(&[1.0, 0.0, 1.0]).unwrap();
        assert!((r[0] - Complex::new(0.0, -1.0)).norm() < 1e-14 && (r[1] - Complex::new(0.0, 1.0)).norm() < 1e-14);
        // x^4 + 1: the four primitive eighth roots of unity.
        let r = polynomial_roots_complex(&[1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        assert!(r.iter().all(|z| (z.norm() - 1.0).abs() < 1e-14 && (z.powu(4) + 1.0).norm() < 1e-13));
        // Leading and trailing zeros: 0·x³ + x² − x = x(x − 1).
        let r = polynomial_roots_complex(&[0.0, 1.0, -1.0, 0.0]).unwrap();
        assert_eq!(r.len(), 2);
        assert!(r[0].norm() < 1e-15 && (r[1] - 1.0).norm() < 1e-14);
    }
}
