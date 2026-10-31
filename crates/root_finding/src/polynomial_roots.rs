//! Polynomial roots.

use crate::van_wijngaarden_dekker_brent::brent_method;

fn eval(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().fold(0.0, |acc, &c| acc * x + c)
}

fn derivative(coeffs: &[f64]) -> Vec<f64> {
    let n = coeffs.len() - 1;
    coeffs[..n]
        .iter()
        .enumerate()
        .map(|(i, &c)| c * (n - i) as f64)
        .collect()
}

/// Returns the real roots of a polynomial in ascending order.
///
/// `coeffs` are ordered from the highest degree to the constant term, e.g. `[1, 0, -2]` is
/// `x^2 - 2`. Roots are isolated recursively using the critical points (roots of the derivative),
/// then polished with Brent's method. Complex roots are not returned, and repeated roots are
/// reported once.
pub fn polynomial_roots(coeffs: &[f64]) -> Vec<f64> {
    let start = coeffs.iter().position(|&c| c != 0.0);
    let Some(start) = start else { return Vec::new() };
    let mut coeffs = &coeffs[start..];
    let mut roots = Vec::new();
    // Strip zero roots at the low end.
    let mut zeros = 0;
    while coeffs.len() > 1 && *coeffs.last().unwrap() == 0.0 {
        coeffs = &coeffs[..coeffs.len() - 1];
        zeros += 1;
    }
    let mut r = real_roots(coeffs);
    if zeros > 0 {
        r.push(0.0);
    }
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    r.dedup_by(|a, b| (*a - *b).abs() <= 1e-9 * (1.0 + a.abs()));
    roots.extend(r);
    roots
}

fn real_roots(coeffs: &[f64]) -> Vec<f64> {
    let deg = coeffs.len() - 1;
    if deg == 0 {
        return Vec::new();
    }
    if deg == 1 {
        return vec![-coeffs[1] / coeffs[0]];
    }
    // Cauchy bound on root magnitude.
    let lead = coeffs[0].abs();
    let bound = 1.0 + coeffs[1..].iter().map(|c| c.abs() / lead).fold(0.0, f64::max);
    let mut crit = real_roots(&derivative(coeffs));
    crit.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut pts = vec![-bound];
    pts.extend(crit.into_iter().filter(|c| c.abs() < bound));
    pts.push(bound);
    let scale = coeffs.iter().map(|c| c.abs()).fold(0.0, f64::max);
    let mut out = Vec::new();
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (fa, fb) = (eval(coeffs, a), eval(coeffs, b));
        if fa == 0.0 {
            out.push(a);
        } else if fa * fb < 0.0
            && let Ok(x) = brent_method(|x| eval(coeffs, x), a, b, 1e-14) {
                out.push(x);
            }
    }
    // Interior critical points that are (numerically) roots: repeated roots.
    for &c in pts[1..pts.len() - 1].iter() {
        if eval(coeffs, c).abs() <= 1e-10 * scale * (1.0 + c.abs()).powi(deg as i32) {
            out.push(c);
        }
    }
    if eval(coeffs, bound) == 0.0 {
        out.push(bound);
    }
    out
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
}
