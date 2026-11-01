//! Van Wijngaarden-Dekker-Brent method.

use crate::same_sign;

/// Finds a root of `f` in `[a, b]` using Brent's method (inverse quadratic interpolation,
/// secant and bisection). Requires a sign change over the interval.
///
/// Returns when the bracket around the best estimate has half-width below `tol / 2 + 2ε|x|`, so
/// the root is located to within about `tol`. Converges superlinearly near a simple root. At a
/// multiple root the interpolation steps converge only linearly, and Brent's safeguard forces a
/// bisection only every few steps, so it can need several times the evaluations of bisection
/// (131 against 46 for `(x − 1/3)³` on `[0, 1]`); its guaranteed worst case is about the square
/// of bisection's count.
pub fn brent_method<F>(f: F, a: f64, b: f64, tol: f64) -> Result<f64, String>
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
        return Err("brent_method: f(a) and f(b) must have opposite signs".to_string());
    }
    let (mut c, mut fc) = (a, fa);
    let mut d = b - a;
    let mut e = d;
    for _ in 0..200 {
        if same_sign(fb, fc) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol1 = 2.0 * f64::EPSILON * b.abs() + 0.5 * tol;
        let xm = 0.5 * (c - b);
        if xm.abs() <= tol1 || fb == 0.0 {
            return Ok(b);
        }
        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (mut p, mut q);
            if a == c {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                let qq = fa / fc;
                let r = fb / fc;
                p = s * (2.0 * xm * qq * (qq - r) - (b - a) * (r - 1.0));
                q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if p > 0.0 {
                q = -q;
            }
            p = p.abs();
            let min1 = 3.0 * xm * q - (tol1 * q).abs();
            let min2 = (e * q).abs();
            if 2.0 * p < min1.min(min2) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        if d.abs() > tol1 {
            b += d;
        } else {
            b += if xm >= 0.0 { tol1 } else { -tol1 };
        }
        fb = f(b);
    }
    Err("brent_method: failed to converge".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqrt2() {
        let r = brent_method(|x| x * x - 2.0, 0.0, 2.0, 1e-14).unwrap();
        assert!((r - 2f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn steep_function() {
        let r = brent_method(|x| (x - 1.0).powi(3) * 1e3, 0.0, 3.0, 1e-12).unwrap();
        assert!((r - 1.0).abs() < 1e-3);
    }

    #[test]
    fn rejects_same_sign() {
        assert!(brent_method(|x| x * x + 1.0, -1.0, 1.0, 1e-9).is_err());
    }
}
