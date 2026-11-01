//! Bisection method.
//!
//! See the book entry [Bracketing methods](https://jwkunz.github.io/lib_jsl/book/root_finding/bracketing.html#bisection).

use crate::same_sign;

/// Finds a root of `f` in `[a, b]` by bisection. Requires `f(a)` and `f(b)` to have opposite signs
/// (a NaN at either end is an error); stops when the bracket is narrower than `2 tol`.
pub fn bisection<F>(f: F, a: f64, b: f64, tol: f64) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
{
    let (mut a, mut b) = (a, b);
    let (mut fa, fb) = (f(a), f(b));
    if fa == 0.0 {
        return Ok(a);
    }
    if fb == 0.0 {
        return Ok(b);
    }
    if !(fa < 0.0 && fb > 0.0 || fa > 0.0 && fb < 0.0) {
        return Err("bisection: f(a) and f(b) must have opposite signs".to_string());
    }
    // ANCHOR: bisection
    for _ in 0..200 {
        let m = 0.5 * (a + b);
        let fm = f(m);
        if fm == 0.0 || 0.5 * (b - a).abs() < tol {
            return Ok(m);
        }
        if !same_sign(fa, fm) {
            b = m;
        } else {
            a = m;
            fa = fm;
        }
    }
    // ANCHOR_END: bisection
    Ok(0.5 * (a + b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqrt2() {
        let r = bisection(|x| x * x - 2.0, 0.0, 2.0, 1e-12).unwrap();
        assert!((r - 2f64.sqrt()).abs() < 1e-11);
    }

    #[test]
    fn rejects_same_sign() {
        assert!(bisection(|x| x * x + 1.0, -1.0, 1.0, 1e-9).is_err());
    }
}
