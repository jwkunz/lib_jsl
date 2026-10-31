//! Bisection method.

/// Finds a root of `f` in `[a, b]` by bisection. Requires `f(a)` and `f(b)` to have opposite signs.
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
    if fa * fb > 0.0 {
        return Err("bisection: f(a) and f(b) must have opposite signs".to_string());
    }
    for _ in 0..200 {
        let m = 0.5 * (a + b);
        let fm = f(m);
        if fm == 0.0 || 0.5 * (b - a).abs() < tol {
            return Ok(m);
        }
        if fa * fm < 0.0 {
            b = m;
        } else {
            a = m;
            fa = fm;
        }
    }
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
