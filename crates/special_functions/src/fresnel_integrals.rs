//! Fresnel integrals.

use std::f64::consts::{FRAC_PI_2, PI};

const EPS: f64 = 1e-16;
const FPMIN: f64 = f64::MIN_POSITIVE / EPS;
const XMIN: f64 = 1.5;
const MAXIT: usize = 100_000;

// Minimal complex arithmetic on (re, im) pairs, local to the continued fraction.
type C = (f64, f64);
fn cmul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn cdiv(a: C, b: C) -> C {
    let den = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / den, (a.1 * b.0 - a.0 * b.1) / den)
}

/// The Fresnel integrals `(S(x), C(x))` with
/// `S(x) = ∫₀ˣ sin(πt²/2) dt` and `C(x) = ∫₀ˣ cos(πt²/2) dt`.
///
/// Both are odd in `x`; the limits as `x → ±∞` are `±1/2`.
pub fn fresnel_integrals(x: f64) -> (f64, f64) {
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    let ax = x.abs();
    let (s, c);
    if ax.is_infinite() {
        s = 0.5;
        c = 0.5;
    } else if ax < FPMIN.sqrt() {
        s = 0.0;
        c = ax;
    } else if ax <= XMIN {
        // Power series.
        let mut sum = 0.0;
        let mut sums = 0.0;
        let mut sumc = ax;
        let mut sign = 1.0;
        let fact = FRAC_PI_2 * ax * ax;
        let mut odd = true;
        let mut term = ax;
        let mut n = 3.0;
        for k in 1..MAXIT {
            term *= fact / k as f64;
            sum += sign * term / n;
            let test = sum.abs() * EPS;
            if odd {
                sign = -sign;
                sums = sum;
                sum = sumc;
            } else {
                sumc = sum;
                sum = sums;
            }
            if term < test {
                break;
            }
            odd = !odd;
            n += 2.0;
        }
        s = sums;
        c = sumc;
    } else {
        // Complex continued fraction (modified Lentz).
        let pix2 = PI * ax * ax;
        let mut b: C = (1.0, -pix2);
        let mut cc: C = (1.0 / FPMIN, 0.0);
        let mut d = cdiv((1.0, 0.0), b);
        let mut h = d;
        let mut n = -1.0;
        for _ in 2..MAXIT {
            n += 2.0;
            let a = -n * (n + 1.0);
            b.0 += 4.0;
            let t = cmul((a, 0.0), d);
            d = cdiv((1.0, 0.0), (t.0 + b.0, t.1 + b.1));
            let q = cdiv((a, 0.0), cc);
            cc = (b.0 + q.0, b.1 + q.1);
            let del = cmul(cc, d);
            h = cmul(h, del);
            if (del.0 - 1.0).abs() + del.1.abs() < EPS {
                break;
            }
        }
        h = cmul((ax, -ax), h);
        let e = cmul(((0.5 * pix2).cos(), (0.5 * pix2).sin()), h);
        let cs = cmul((0.5, 0.5), (1.0 - e.0, -e.1));
        c = cs.0;
        s = cs.1;
    }
    if x < 0.0 { (-s, -c) } else { (s, c) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        let cases = [
            (0.5, 0.06473243285999927, 0.49234422587144639),
            (1.0, 0.43825914739035476, 0.77989340037682283),
            (1.5, 0.69750496008209301, 0.44526117603982154),
            (2.0, 0.34341567836369824, 0.48825340607534075),
            (5.0, 0.49919138191711689, 0.56363118870401223),
            (10.0, 0.46816997858488224, 0.49989869420551572),
        ];
        for &(x, s, c) in &cases {
            let (sx, cx) = fresnel_integrals(x);
            assert!((sx - s).abs() < 1e-14, "S({x}) = {sx} vs {s}");
            assert!((cx - c).abs() < 1e-14, "C({x}) = {cx} vs {c}");
        }
    }

    #[test]
    fn odd_and_limits() {
        let (s, c) = fresnel_integrals(-1.5);
        assert!((s + 0.6975049600820930).abs() < 1e-14 && (c + 0.4452611760398215).abs() < 1e-14);
        assert_eq!(fresnel_integrals(0.0), (0.0, 0.0));
        let (s, c) = fresnel_integrals(1e6);
        assert!((s - 0.5).abs() < 1e-6 && (c - 0.5).abs() < 1e-6);
        assert_eq!(fresnel_integrals(f64::INFINITY), (0.5, 0.5));
    }

    #[test]
    fn continuity_at_branch() {
        let e = 1e-9;
        let (s1, c1) = fresnel_integrals(XMIN - e);
        let (s2, c2) = fresnel_integrals(XMIN + e);
        assert!((s1 - s2).abs() < 1e-8 && (c1 - c2).abs() < 1e-8);
    }
}
