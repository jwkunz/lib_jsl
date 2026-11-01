//! False position method (Illinois variant).

use crate::same_sign;

/// Finds a root of `f` in `[a, b]` by false position, using the Illinois modification to
/// avoid stagnation of one endpoint. Requires a sign change over the interval.
///
/// Stops when the bracket is narrower than `tol` or `f` is exactly zero at the estimate; errors
/// after 500 iterations.
pub fn false_position<F>(f: F, a: f64, b: f64, tol: f64) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
{
    let (mut a, mut b) = (a, b);
    let (mut fa, mut fb) = (f(a), f(b));
    if fa == 0.0 {
        return Ok(a);
    }
    if fb == 0.0 {
        return Ok(b);
    }
    if !(fa < 0.0 && fb > 0.0 || fa > 0.0 && fb < 0.0) {
        return Err("false_position: f(a) and f(b) must have opposite signs".to_string());
    }
    let mut side = 0i8;
    for _ in 0..500 {
        let c = (a * fb - b * fa) / (fb - fa);
        let fc = f(c);
        if fc == 0.0 || (b - a).abs() < tol {
            return Ok(c);
        }
        if same_sign(fc, fb) {
            b = c;
            fb = fc;
            if side == -1 {
                fa *= 0.5;
            }
            side = -1;
        } else {
            a = c;
            fa = fc;
            if side == 1 {
                fb *= 0.5;
            }
            side = 1;
        }
        if (b - a).abs() < tol {
            return Ok(c);
        }
    }
    Err("false_position: failed to converge".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic() {
        let r = false_position(|x| x * x * x - x - 2.0, 1.0, 2.0, 1e-12).unwrap();
        assert!((r * r * r - r - 2.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_same_sign() {
        assert!(false_position(|x| x * x + 1.0, -1.0, 1.0, 1e-9).is_err());
    }
}
