//! Statistical functions: probability densities, cumulative distribution functions and quantiles
//! built on the error function and the incomplete gamma and beta functions.
//!
//! Invalid parameters (non-positive scales, degrees of freedom, probabilities outside `[0, 1]`)
//! yield `NaN`.

use crate::error_function::erfc;
use crate::gamma::ln_gamma;
use crate::incomplete_beta::incomplete_beta;
use crate::incomplete_gamma::{incomplete_gamma, incomplete_gamma_upper};
use std::f64::consts::{PI, SQRT_2};

// ---------------------------------------------------------------- normal

/// The probability density of a normal distribution with the given mean and standard deviation.
pub fn normal_pdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    if !(std_dev > 0.0) {
        return f64::NAN;
    }
    let z = (x - mean) / std_dev;
    (-0.5 * z * z).exp() / (std_dev * (2.0 * PI).sqrt())
}

/// The cumulative distribution function of a normal distribution, accurate in both tails.
pub fn normal_cdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    if !(std_dev > 0.0) {
        return f64::NAN;
    }
    0.5 * erfc(-(x - mean) / (std_dev * SQRT_2))
}

/// The quantile (inverse CDF) of a normal distribution, for `p` in `[0, 1]`.
pub fn normal_quantile(p: f64, mean: f64, std_dev: f64) -> f64 {
    if !(std_dev > 0.0) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    mean + std_dev * standard_normal_quantile(p)
}

fn standard_normal_quantile(p: f64) -> f64 {
    // Acklam's rational approximation followed by Halley refinement.
    const A: [f64; 6] = [
        -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
    ];
    const D: [f64; 4] = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    let plow = 0.02425;
    let mut x = if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - plow {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    for _ in 0..3 {
        let e = normal_cdf(x, 0.0, 1.0) - p;
        let u = e * (2.0 * PI).sqrt() * (0.5 * x * x).exp();
        x -= u / (1.0 + 0.5 * x * u);
    }
    x
}

// ------------------------------------------------------- gamma-based laws

/// The CDF of a gamma distribution with the given shape and scale.
pub fn gamma_cdf(x: f64, shape: f64, scale: f64) -> f64 {
    if !(shape > 0.0) || !(scale > 0.0) {
        return f64::NAN;
    }
    if x <= 0.0 { 0.0 } else { incomplete_gamma(shape, x / scale) }
}

/// The CDF of a chi-square distribution with `k` degrees of freedom.
pub fn chi_square_cdf(x: f64, k: f64) -> f64 {
    gamma_cdf(x, 0.5 * k, 2.0)
}

/// The upper-tail probability `P(X > x)` of a chi-square distribution (the p-value of a statistic).
pub fn chi_square_survival(x: f64, k: f64) -> f64 {
    if !(k > 0.0) {
        return f64::NAN;
    }
    if x <= 0.0 { 1.0 } else { incomplete_gamma_upper(0.5 * k, 0.5 * x) }
}

/// The quantile of a chi-square distribution with `k` degrees of freedom, for `p` in `[0, 1]`.
pub fn chi_square_quantile(p: f64, k: f64) -> f64 {
    if !(k > 0.0) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    invert_monotone(|x| chi_square_cdf(x, k), p, 0.0, f64::INFINITY, k.max(1.0))
}

// -------------------------------------------------------- beta-based laws

/// The CDF of a beta distribution on `[0, 1]`.
pub fn beta_cdf(x: f64, a: f64, b: f64) -> f64 {
    if !(a > 0.0) || !(b > 0.0) {
        return f64::NAN;
    }
    incomplete_beta(a, b, x.clamp(0.0, 1.0))
}

/// The CDF of Student's t distribution with `nu` degrees of freedom.
pub fn student_t_cdf(t: f64, nu: f64) -> f64 {
    if !(nu > 0.0) || t.is_nan() {
        return f64::NAN;
    }
    if t.is_infinite() {
        return if t > 0.0 { 1.0 } else { 0.0 };
    }
    let tail = 0.5 * incomplete_beta(0.5 * nu, 0.5, nu / (nu + t * t));
    if t > 0.0 { 1.0 - tail } else { tail }
}

/// The quantile of Student's t distribution with `nu` degrees of freedom, for `p` in `[0, 1]`.
pub fn student_t_quantile(p: f64, nu: f64) -> f64 {
    if !(nu > 0.0) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }
    if p < 0.5 {
        return -student_t_quantile(1.0 - p, nu);
    }
    invert_monotone(|t| student_t_cdf(t, nu), p, 0.0, f64::INFINITY, 1.0)
}

/// The CDF of the F distribution with `d1` and `d2` degrees of freedom.
pub fn f_cdf(x: f64, d1: f64, d2: f64) -> f64 {
    if !(d1 > 0.0) || !(d2 > 0.0) {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x.is_infinite() {
        return 1.0;
    }
    incomplete_beta(0.5 * d1, 0.5 * d2, d1 * x / (d1 * x + d2))
}

// -------------------------------------------------------- discrete laws

/// The Poisson probability mass `P(X = k)` with mean `lambda`.
pub fn poisson_pmf(k: u64, lambda: f64) -> f64 {
    if !(lambda >= 0.0) {
        return f64::NAN;
    }
    if lambda == 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    (k as f64 * lambda.ln() - lambda - ln_gamma(k as f64 + 1.0)).exp()
}

/// The Poisson CDF `P(X <= k)` with mean `lambda`.
pub fn poisson_cdf(k: u64, lambda: f64) -> f64 {
    if !(lambda >= 0.0) {
        return f64::NAN;
    }
    if lambda == 0.0 {
        return 1.0;
    }
    incomplete_gamma_upper(k as f64 + 1.0, lambda)
}

/// The binomial probability mass `P(X = k)` for `n` trials with success probability `p`.
pub fn binomial_pmf(k: u64, n: u64, p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if k > n {
        return 0.0;
    }
    if p == 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    if p == 1.0 {
        return if k == n { 1.0 } else { 0.0 };
    }
    let (kf, nf) = (k as f64, n as f64);
    (ln_gamma(nf + 1.0) - ln_gamma(kf + 1.0) - ln_gamma(nf - kf + 1.0) + kf * p.ln() + (nf - kf) * (-p).ln_1p()).exp()
}

/// The binomial CDF `P(X <= k)` for `n` trials with success probability `p`.
pub fn binomial_cdf(k: u64, n: u64, p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if k >= n {
        return 1.0;
    }
    incomplete_beta((n - k) as f64, k as f64 + 1.0, 1.0 - p)
}

// -------------------------------------------------------------- helpers

/// Inverts an increasing function on `[lo, hi)` by bracket expansion followed by bisection.
fn invert_monotone<F: Fn(f64) -> f64>(f: F, target: f64, lo: f64, hi: f64, start: f64) -> f64 {
    let mut a = lo;
    let mut b = start;
    while f(b) < target && b < hi {
        a = b;
        b *= 2.0;
        if b > 1e300 {
            return f64::INFINITY;
        }
    }
    for _ in 0..300 {
        let m = 0.5 * (a + b);
        if f(m) < target {
            a = m;
        } else {
            b = m;
        }
        if (b - a) <= 4.0 * f64::EPSILON * b.abs() {
            break;
        }
    }
    0.5 * (a + b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs()
    }

    #[test]
    fn normal() {
        assert!(close(normal_cdf(1.5, 0.0, 1.0), 0.9331927987311419, 1e-14));
        assert!(close(normal_cdf(-6.0, 0.0, 1.0), 9.865876450376981e-10, 1e-12));
        assert!(close(normal_cdf(0.7, 1.0, 1.0), 0.3820885778110474, 1e-14));
        assert!(close(normal_pdf(0.0, 0.0, 1.0), 0.3989422804014327, 1e-15));
        assert!(close(normal_pdf(3.0, 1.0, 2.0), 0.12098536225957168, 1e-14));
        assert!(close(normal_quantile(0.975, 0.0, 1.0), 1.959963984540054, 1e-14));
        assert!(close(normal_quantile(1e-10, 0.0, 1.0), -6.361340902404056, 1e-13));
        assert_eq!(normal_quantile(0.5, 3.0, 2.0), 3.0);
        assert!(close(normal_quantile(0.975, 10.0, 2.0), 10.0 + 2.0 * 1.959963984540054, 1e-14));
        assert_eq!(normal_quantile(0.0, 0.0, 1.0), f64::NEG_INFINITY);
        assert!(normal_cdf(0.0, 0.0, -1.0).is_nan());
        for &p in &[1e-300, 1e-12, 0.01, 0.3, 0.9, 1.0 - 1e-12] {
            let x = normal_quantile(p, 0.0, 1.0);
            assert!(close(normal_cdf(x, 0.0, 1.0), p, 1e-12), "p={p}");
        }
    }

    #[test]
    fn chi_square() {
        assert!(close(chi_square_cdf(3.84, 1.0), 0.9499564787512949, 1e-13));
        assert!(close(chi_square_cdf(10.0, 4.0), 0.9595723180054872, 1e-13));
        assert!(close(chi_square_cdf(0.5, 7.0), 0.0005535186095750345, 1e-12));
        assert!((chi_square_cdf(3.0, 5.0) + chi_square_survival(3.0, 5.0) - 1.0).abs() < 1e-14);
        assert!(close(chi_square_quantile(0.95, 3.0), 7.814727903251178, 1e-12));
        assert_eq!(chi_square_cdf(-1.0, 3.0), 0.0);
    }

    #[test]
    fn student_and_f() {
        assert!(close(student_t_cdf(2.0, 5.0), 0.9490302605850708, 1e-13));
        assert!(close(student_t_cdf(-1.3, 3.0), 0.14223375436394869, 1e-13));
        assert!(close(student_t_cdf(0.4, 30.0), 0.6540047417429040, 1e-13));
        assert_eq!(student_t_cdf(0.0, 4.0), 0.5);
        assert!(close(student_t_quantile(0.975, 5.0), 2.5705818356363148, 1e-12));
        assert!(close(student_t_quantile(0.025, 5.0), -2.5705818356363148, 1e-12));
        assert!(close(f_cdf(2.5, 4.0, 9.0), 0.88328505728343125, 1e-13));
        assert!(close(f_cdf(0.7, 10.0, 3.0), 0.29180564480614543, 1e-13));
    }

    #[test]
    fn gamma_and_beta() {
        assert!(close(gamma_cdf(4.0, 2.5, 1.5), 0.6234323552910231, 1e-13));
        assert!(close(beta_cdf(0.4, 2.0, 3.0), 0.5248, 1e-13));
        assert_eq!(beta_cdf(-1.0, 2.0, 3.0), 0.0);
        assert_eq!(beta_cdf(2.0, 2.0, 3.0), 1.0);
    }

    #[test]
    fn discrete() {
        assert!(close(poisson_pmf(3, 3.2), 0.22261598332718404, 1e-13));
        assert!(close(poisson_cdf(3, 3.2), 0.6025197244055571, 1e-12));
        let sum: f64 = (0..=3).map(|k| poisson_pmf(k, 3.2)).sum();
        assert!(close(poisson_cdf(3, 3.2), sum, 1e-13));
        assert!(close(binomial_pmf(3, 10, 0.3), 0.266827932, 1e-13));
        assert!(close(binomial_cdf(4, 10, 0.3), 0.8497316674, 1e-12));
        assert_eq!(binomial_cdf(10, 10, 0.3), 1.0);
        assert_eq!(binomial_pmf(11, 10, 0.3), 0.0);
        assert!(binomial_pmf(1, 10, 1.5).is_nan());
        assert_eq!(poisson_pmf(0, 0.0), 1.0);
    }
}
