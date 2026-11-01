//! Root bracketing.

use crate::same_sign;

/// Searches outward from the interval `[a, b]` for a sign change of `f`.
///
/// The interval is expanded geometrically (by a factor of 1.6 per step, on the side with the
/// smaller `|f|`) for up to 50 iterations. Returns `(lo, hi)` with `f(lo) * f(hi) <= 0`.
pub fn bracket_root<F>(f: F, a: f64, b: f64) -> Result<(f64, f64), String>
where
    F: Fn(f64) -> f64,
{
    const FACTOR: f64 = 1.6;
    const MAX_ITER: usize = 50;
    if a == b || !a.is_finite() || !b.is_finite() {
        return Err("bracket_root requires two distinct finite points".to_string());
    }
    let (mut a, mut b) = if a < b { (a, b) } else { (b, a) };
    let (mut fa, mut fb) = (f(a), f(b));
    for _ in 0..MAX_ITER {
        if !same_sign(fa, fb) {
            return Ok((a, b));
        }
        if fa.abs() < fb.abs() {
            a += FACTOR * (a - b);
            fa = f(a);
        } else {
            b += FACTOR * (b - a);
            fb = f(b);
        }
    }
    Err("bracket_root failed to find a sign change".to_string())
}

/// Divides `[a, b]` into `n` subintervals and returns every one containing a sign change (or a
/// zero at an end, so a root exactly on a grid point is reported by both neighbouring intervals).
/// Pairs of roots closer together than the spacing can cancel and go unnoticed.
pub fn bracket_roots_scan<F>(f: F, a: f64, b: f64, n: usize) -> Vec<(f64, f64)>
where
    F: Fn(f64) -> f64,
{
    let mut out = Vec::new();
    if n == 0 {
        return out;
    }
    let dx = (b - a) / n as f64;
    let mut x0 = a;
    let mut f0 = f(x0);
    for i in 1..=n {
        let x1 = if i == n { b } else { a + dx * i as f64 };
        let f1 = f(x1);
        if !same_sign(f0, f1) {
            out.push((x0, x1));
        }
        x0 = x1;
        f0 = f1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_to_bracket() {
        let (lo, hi) = bracket_root(|x| x - 10.0, 0.0, 1.0).unwrap();
        assert!((lo - 10.0) * (hi - 10.0) <= 0.0);
    }

    #[test]
    fn fails_without_root() {
        assert!(bracket_root(|x| x * x + 1.0, -1.0, 1.0).is_err());
    }

    #[test]
    fn scan_finds_all() {
        let r = bracket_roots_scan(|x: f64| x.sin(), 1.0, 10.0, 90);
        assert_eq!(r.len(), 3); // pi, 2pi, 3pi
    }
}
