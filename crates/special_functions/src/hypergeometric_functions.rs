//! Hypergeometric functions.

use crate::bessel_functions_fractional::bessel_fractional;
use crate::gamma::gamma;

const EPS: f64 = 1e-17;

fn is_nonpositive_integer(x: f64) -> bool {
    x <= 0.0 && x == x.round()
}

/// Reciprocal gamma function, defined (as zero) at the poles.
fn rgamma(x: f64) -> f64 {
    if is_nonpositive_integer(x) { 0.0 } else { 1.0 / gamma(x) }
}

/// Direct summation of the Gauss series; converges for `|z| < 1` (or terminates when `a` or `b`
/// is a non-positive integer).
fn series_2f1(a: f64, b: f64, c: f64, z: f64, max_terms: usize) -> f64 {
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 0..max_terms {
        let nf = n as f64;
        term *= (a + nf) * (b + nf) / ((c + nf) * (nf + 1.0)) * z;
        sum += term;
        if term == 0.0 || (term.abs() < EPS * sum.abs() && n > 2) {
            break;
        }
    }
    sum
}

/// Gauss's hypergeometric function ₂F₁(a, b; c; z) for real arguments and `z <= 1`.
///
/// For `z < -0.5` the Pfaff transformation maps the argument into `(0, 1)`. For `z >= 0.5` the
/// `1 - z` connection formula is used unless `c - a - b` is an integer, in which case the series
/// is summed directly (slow, but correct, very near `z = 1`). At `z = 1` Gauss's summation
/// theorem gives the value (`+inf` if `c - a - b <= 0`). Returns `NaN` for `z > 1` (except when
/// `a` or `b` is a non-positive integer, giving a polynomial) and for `c` a non-positive integer
/// (unless the series terminates earlier).
pub fn hypergeometric(a: f64, b: f64, c: f64, z: f64) -> f64 {
    if a.is_nan() || b.is_nan() || c.is_nan() || z.is_nan() {
        return f64::NAN;
    }
    let terminating = is_nonpositive_integer(a) || is_nonpositive_integer(b);
    if z > 1.0 && !terminating {
        return f64::NAN;
    }
    if is_nonpositive_integer(c) {
        // Well defined only when the series stops before the zero denominator.
        let stop = [a, b]
            .iter()
            .filter(|&&v| is_nonpositive_integer(v))
            .map(|&v| -v)
            .fold(f64::INFINITY, f64::min);
        if !(stop < -c) {
            return f64::NAN;
        }
    }
    if z == 0.0 {
        return 1.0;
    }
    if terminating {
        return series_2f1(a, b, c, z, 100_000);
    }
    if z == 1.0 {
        let s = c - a - b;
        if s <= 0.0 {
            return f64::INFINITY;
        }
        return gamma(c) * gamma(s) * rgamma(c - a) * rgamma(c - b);
    }
    if z < -0.5 {
        // Pfaff: F(a,b;c;z) = (1-z)^-a F(a, c-b; c; z/(z-1))
        let zp = z / (z - 1.0);
        return (1.0 - z).powf(-a) * hypergeometric(a, c - b, c, zp);
    }
    if z <= 0.5 {
        return series_2f1(a, b, c, z, 100_000);
    }
    // 0.5 < z < 1
    let s = c - a - b;
    if (s - s.round()).abs() > 1e-9 {
        let w = 1.0 - z;
        let t1 = gamma(c) * gamma(s) * rgamma(c - a) * rgamma(c - b) * series_2f1(a, b, a + b - c + 1.0, w, 100_000);
        let t2 = w.powf(s) * gamma(c) * gamma(-s) * rgamma(a) * rgamma(b) * series_2f1(c - a, c - b, s + 1.0, w, 100_000);
        return t1 + t2;
    }
    // Integer c - a - b: direct summation (may need many terms very close to 1).
    series_2f1(a, b, c, z, 5_000_000)
}

/// Kummer's confluent hypergeometric function ₁F₁(a; b; x) = M(a, b, x), for real arguments.
///
/// Uses Kummer's transformation `M(a,b,x) = e^x M(b-a,b,-x)` for `x < 0` to avoid cancellation.
/// Returns `NaN` if `b` is a non-positive integer (unless the series terminates first).
pub fn hypergeometric_1f1(a: f64, b: f64, x: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if is_nonpositive_integer(b) && !(is_nonpositive_integer(a) && -a < -b) {
        return f64::NAN;
    }
    if x < 0.0 && !is_nonpositive_integer(a) {
        return x.exp() * hypergeometric_1f1(b - a, b, -x);
    }
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 0..1_000_000usize {
        let nf = n as f64;
        term *= (a + nf) / ((b + nf) * (nf + 1.0)) * x;
        sum += term;
        if term == 0.0 || (term.abs() < EPS * sum.abs() && n as f64 > x.abs()) {
            break;
        }
    }
    sum
}

/// The confluent hypergeometric limit function ₀F₁(; b; x).
///
/// Summed as a series, except for `x < −10` (with `b ≤ 100`), where the alternating series loses
/// digits to cancellation (about 1e-9 relative at `x = −100`) and the Bessel function relation
/// `₀F₁(; b; −z²/4) = Γ(b) (z/2)^{1−b} J_{b−1}(z)` is used instead.
pub fn hypergeometric_0f1(b: f64, x: f64) -> f64 {
    if b.is_nan() || x.is_nan() || is_nonpositive_integer(b) {
        return f64::NAN;
    }
    if x < -10.0 && b <= 100.0 {
        let half_z = (-x).sqrt();
        return gamma(b) * half_z.powf(1.0 - b) * bessel_fractional(b - 1.0, 2.0 * half_z);
    }
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 0..100_000usize {
        let nf = n as f64;
        term *= x / ((b + nf) * (nf + 1.0));
        sum += term;
        if term == 0.0 || (term.abs() < EPS * sum.abs() && n as f64 > x.abs().sqrt()) {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs()
    }

    #[test]
    fn closed_forms() {
        // 2F1(1,1;2;z) = -ln(1-z)/z
        for &z in &[-100.0f64, -3.0, -0.4, 0.3, 0.5, 0.9, 0.999] {
            assert!(close(hypergeometric(1.0, 1.0, 2.0, z), -(-z).ln_1p() / z, 1e-12), "z={z}");
        }
        // 2F1(1/2,1/2;3/2;z²) = asin(z)/z
        assert!(close(hypergeometric(0.5, 0.5, 1.5, 0.49), 0.7f64.asin() / 0.7, 1e-14));
        // 2F1(a,b;b;z) = (1-z)^-a
        assert!(close(hypergeometric(1.3, 2.2, 2.2, 0.8), 0.2f64.powf(-1.3), 1e-12));
    }

    #[test]
    fn reference_values() {
        assert!(close(hypergeometric(1.0, 1.0, 2.0, 0.5), 1.3862943611198906, 1e-14));
        assert!(close(hypergeometric(0.5, 0.5, 1.0, 0.9), 1.6412644143423708, 1e-12));
        assert!(close(hypergeometric(0.5, 0.5, 1.0, -3.0), 0.686_440_250_309_175_1, 1e-13));
        assert!(close(hypergeometric(1.5, 2.5, 4.2, 0.7), 2.5577444106945205, 1e-13));
        assert!(close(hypergeometric(1.0, 2.0, 3.5, -10.0), 0.178_018_528_819_947_1, 1e-13));
        assert!(close(hypergeometric(2.1, 0.3, 1.7, 0.95), 3.971_073_611_225_176, 1e-12));
        assert!(close(hypergeometric(0.3, 0.4, 1.2, 0.99), 1.2619447158474417, 1e-12));
        assert!(close(hypergeometric(1.0, 1.0, 2.0, -100.0), 0.046_151_205_168_412_6, 1e-12));
    }

    #[test]
    fn terminating_and_special_points() {
        assert!(close(hypergeometric(-3.0, 1.2, 2.1, 2.0), -0.094_301_449_926_941_6, 1e-12));
        // Gauss: 2F1(a,b;c;1) = G(c)G(c-a-b)/(G(c-a)G(c-b))
        let want = gamma(3.0) * gamma(2.0) / (gamma(2.5) * gamma(2.5));
        assert!(close(hypergeometric(0.5, 0.5, 3.0, 1.0), want, 1e-14));
        assert!(hypergeometric(0.5, 0.5, 1.0, 1.0).is_infinite());
        assert!(hypergeometric(1.0, 1.0, 2.0, 1.5).is_nan());
        assert_eq!(hypergeometric(1.0, 2.0, 3.0, 0.0), 1.0);
        assert!(hypergeometric(1.0, 2.0, -2.0, 0.3).is_nan());
    }

    #[test]
    fn complete_elliptic_k() {
        // K(m) = (π/2) 2F1(1/2, 1/2; 1; m)
        let m: f64 = 0.64;
        let k = crate::elliptic_integrals::elliptic_integral(m.sqrt());
        assert!(close(std::f64::consts::FRAC_PI_2 * hypergeometric(0.5, 0.5, 1.0, m), k, 1e-12));
    }

    #[test]
    fn confluent() {
        assert!(close(hypergeometric_1f1(1.5, 2.5, 3.0), 7.931_662_465_149_578, 1e-13));
        assert!(close(hypergeometric_1f1(1.5, 2.5, -30.0), 0.008_090_107_968_977_325, 1e-11));
        assert!(close(hypergeometric_1f1(0.5, 1.5, -3.0), 0.504_343_560_231_438_8, 1e-13));
        // M(a, a, x) = e^x
        assert!(close(hypergeometric_1f1(2.0, 2.0, 1.7), 1.7f64.exp(), 1e-14));
        // 0F1(;3/2; -x²/4) = sin(x)/x
        let x: f64 = 2.0;
        assert!(close(hypergeometric_0f1(1.5, -x * x / 4.0), x.sin() / x, 1e-14));
        assert!(hypergeometric_1f1(1.0, 0.0, 1.0).is_nan());
    }

    #[test]
    fn zero_f_one_is_accurate_for_large_negative_arguments() {
        // mpmath values; the series lost digits to cancellation here (1.4e-9 relative at b = 0.5).
        for (b, x, want) in [(0.5, -100.0, 0.40808206181339196), (1.0, -100.0, 0.16702466434058316), (10.0, -100.0, 4.540581528666412e-5), (2.5, -100.0, -0.0027182609945775797)] {
            let got = hypergeometric_0f1(b, x);
            assert!(((got - want) / want).abs() < 1e-12, "b = {b}: {got} vs {want}");
        }
    }
}
