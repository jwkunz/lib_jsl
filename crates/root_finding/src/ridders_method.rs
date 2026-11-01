//! Ridders' method.
//!
//! See the book entry [Bracketing methods](https://jwkunz.github.io/lib_jsl/book/root_finding/bracketing.html#ridders-method).

/// Finds a root of `f` in `[a, b]` using Ridders' method (C. Ridders, 1979). Requires a sign
/// change over the interval.
///
/// Each iteration evaluates `f` at the midpoint and at an exponentially-corrected false-position
/// point, keeping a bracket; it stops when successive estimates or the bracket width fall below
/// `tol`, and errors after 100 iterations.
pub fn ridders_method<F>(f: F, a: f64, b: f64, tol: f64) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
{
    let (mut x1, mut x2) = (a, b);
    let (mut f1, mut f2) = (f(x1), f(x2));
    if f1 == 0.0 {
        return Ok(x1);
    }
    if f2 == 0.0 {
        return Ok(x2);
    }
    if !(f1 < 0.0 && f2 > 0.0 || f1 > 0.0 && f2 < 0.0) {
        return Err("ridders_method: f(a) and f(b) must have opposite signs".to_string());
    }
    let mut ans = f64::NAN;
    for _ in 0..100 {
        let xm = 0.5 * (x1 + x2);
        let fm = f(xm);
        // √(fm² − f1 f2), scaled to avoid overflow and underflow.
        let scale = fm.abs().max(f1.abs()).max(f2.abs());
        let s = scale * ((fm / scale).powi(2) - (f1 / scale) * (f2 / scale)).sqrt();
        if s == 0.0 {
            return Ok(xm);
        }
        let sign = if f1 >= f2 { 1.0 } else { -1.0 };
        let xn = xm + (xm - x1) * sign * fm / s;
        if ans.is_finite() && (xn - ans).abs() <= tol {
            return Ok(xn);
        }
        ans = xn;
        let fnew = f(xn);
        if fnew == 0.0 {
            return Ok(xn);
        }
        if fm.signum() != fnew.signum() {
            x1 = xm;
            f1 = fm;
            x2 = xn;
            f2 = fnew;
        } else if f1.signum() != fnew.signum() {
            x2 = xn;
            f2 = fnew;
        } else {
            x1 = xn;
            f1 = fnew;
        }
        if (x2 - x1).abs() <= tol {
            return Ok(ans);
        }
    }
    Err("ridders_method: failed to converge".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_root() {
        let r = ridders_method(|x| x.exp() - 3.0, 0.0, 2.0, 1e-12).unwrap();
        assert!((r - 3f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn rejects_same_sign() {
        assert!(ridders_method(|x| x * x + 1.0, -1.0, 1.0, 1e-9).is_err());
    }
}
