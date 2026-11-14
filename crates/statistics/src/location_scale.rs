//! More location, scale and shape summaries, following `scipy.stats`: trimmed means (`trim_mean`),
//! standard scores (`zscore`), the `mode`, circular statistics (`circmean`, `circvar`,
//! `circstd`), and the maximum-likelihood Box-Cox transform (`boxcox`, `boxcox_llf`,
//! `boxcox_normmax`).
//!
//! See the book entry [More descriptive statistics and robust regression](https://jwkunz.github.io/lib_jsl/book/statistics/robust-regression.html).

use crate::descriptive::{mean, standard_deviation};
use lib_jsl_core::ErrorsJSL;
use std::f64::consts::TAU;

fn sorted(x: &[f64]) -> Vec<f64> {
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    s
}

// ANCHOR: trim_mean
/// The mean after removing the `⌊proportion · n⌋` smallest and largest values (SciPy's
/// `trim_mean`). With `proportion = 0` it is the mean; as it approaches ½ it approaches the median.
pub fn trim_mean(x: &[f64], proportion: f64) -> Result<f64, ErrorsJSL> {
    if x.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("need at least one value"));
    }
    if !(0.0..0.5).contains(&proportion) {
        return Err(ErrorsJSL::InvalidInputRange("the proportion to cut must be in [0, 0.5)"));
    }
    let s = sorted(x);
    let cut = (proportion * s.len() as f64).floor() as usize;
    mean(&s[cut..s.len() - cut])
}
// ANCHOR_END: trim_mean

/// The standard scores `(xᵢ − x̄)/s` with `s` computed with `ddof` degrees of freedom removed
/// (SciPy's `zscore`, whose default is `ddof = 0`).
pub fn zscore(x: &[f64], ddof: usize) -> Result<Vec<f64>, ErrorsJSL> {
    let m = mean(x)?;
    let s = standard_deviation(x, ddof)?;
    Ok(x.iter().map(|v| (v - m) / s).collect())
}

/// The most frequent value and its count; ties go to the smallest value (SciPy's `mode`).
pub fn mode(x: &[f64]) -> Result<(f64, usize), ErrorsJSL> {
    if x.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("need at least one value"));
    }
    let s = sorted(x);
    let (mut best, mut best_count) = (s[0], 0);
    let mut i = 0;
    while i < s.len() {
        let j = s[i..].iter().take_while(|&&v| v == s[i]).count();
        if j > best_count {
            best = s[i];
            best_count = j;
        }
        i += j;
    }
    Ok((best, best_count))
}

// ANCHOR: circular
/// The mean resultant vector of angles in `[low, high)` (one period): its length `R ∈ [0, 1]`
/// and direction, as an angle in radians.
fn resultant(angles: &[f64], low: f64, high: f64) -> Result<(f64, f64), ErrorsJSL> {
    if angles.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("need at least one angle"));
    }
    if !(high > low) {
        return Err(ErrorsJSL::InvalidInputRange("need high > low"));
    }
    let k = TAU / (high - low);
    let (s, c) = angles.iter().fold((0.0, 0.0), |(s, c), &a| {
        let t = (a - low) * k;
        (s + t.sin(), c + t.cos())
    });
    let n = angles.len() as f64;
    Ok(((s / n).hypot(c / n), s.atan2(c)))
}

/// The circular mean of angles with period `high − low` (SciPy's `circmean`; use `0` and `2π`
/// for radians): the direction of the mean resultant vector, mapped into `[low, high)`.
pub fn circmean(angles: &[f64], low: f64, high: f64) -> Result<f64, ErrorsJSL> {
    let (_, theta) = resultant(angles, low, high)?;
    Ok(low + theta.rem_euclid(TAU) * (high - low) / TAU)
}

/// The circular variance `1 − R` (SciPy's `circvar`), between 0 (all angles equal) and 1.
pub fn circvar(angles: &[f64], low: f64, high: f64) -> Result<f64, ErrorsJSL> {
    Ok(1.0 - resultant(angles, low, high)?.0)
}

/// The circular standard deviation `√(−2 ln R)`, in the units of the angles (SciPy's `circstd`).
/// It approaches the ordinary standard deviation for concentrated angles.
pub fn circstd(angles: &[f64], low: f64, high: f64) -> Result<f64, ErrorsJSL> {
    let (r, _) = resultant(angles, low, high)?;
    Ok((-2.0 * r.ln()).sqrt() * (high - low) / TAU)
}
// ANCHOR_END: circular

/// The Box-Cox profile log-likelihood of `λ` for positive data (SciPy's `boxcox_llf`):
/// `(λ − 1) Σ ln xᵢ − (n/2) ln σ̂²(λ)`, where `σ̂²` is the variance (`ddof = 0`) of the
/// transformed data.
pub fn boxcox_llf(lambda: f64, x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let logs: Vec<f64> = x.iter().map(|v| v.ln()).collect();
    let y: Vec<f64> = logs.iter().map(|&l| if lambda == 0.0 { l } else { (lambda * l).exp_m1() / lambda }).collect();
    let m = y.iter().sum::<f64>() / n;
    let var = y.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
    (lambda - 1.0) * logs.iter().sum::<f64>() - 0.5 * n * var.ln()
}

// ANCHOR: boxcox_normmax
/// The `λ` that maximises [`boxcox_llf`] (SciPy's `boxcox_normmax` with `method='mle'`): the
/// power that makes the data most nearly normal. A golden-section search on `[−2, 2]`, widened
/// while the maximum sits at an end of the interval.
pub fn boxcox_normmax(x: &[f64]) -> Result<f64, ErrorsJSL> {
    if x.len() < 2 || x.iter().any(|&v| !(v > 0.0)) {
        return Err(ErrorsJSL::InvalidInputRange("Box-Cox needs at least two positive values"));
    }
    if x.iter().all(|&v| v == x[0]) {
        return Err(ErrorsJSL::InvalidInputRange("Box-Cox needs data that are not all equal"));
    }
    let f = |l: f64| -boxcox_llf(l, x);
    let (mut lo, mut hi) = (-2.0, 2.0);
    loop {
        let g = 0.5 * (5f64.sqrt() - 1.0);
        let (mut a, mut b) = (lo, hi);
        let (mut c, mut d) = (b - g * (b - a), a + g * (b - a));
        let (mut fc, mut fd) = (f(c), f(d));
        while (b - a).abs() > 1e-12 * (1.0 + c.abs()) {
            if fc < fd {
                b = d;
                d = c;
                fd = fc;
                c = b - g * (b - a);
                fc = f(c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + g * (b - a);
                fd = f(d);
            }
        }
        let best = 0.5 * (a + b);
        let width = hi - lo;
        if best - lo < 1e-6 * width && lo > -50.0 {
            lo -= width;
        } else if hi - best < 1e-6 * width && hi < 50.0 {
            hi += width;
        } else {
            return Ok(best);
        }
    }
}
// ANCHOR_END: boxcox_normmax

/// The maximum-likelihood Box-Cox transform of positive data and its `λ` (SciPy's `stats.boxcox`).
pub fn boxcox(x: &[f64]) -> Result<(Vec<f64>, f64), ErrorsJSL> {
    let lambda = boxcox_normmax(x)?;
    Ok((x.iter().map(|&v| lib_jsl_special_functions::boxcox(v, lambda)).collect(), lambda))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trimmed_mean_and_mode() {
        let x = [1.0, 2.0, 3.0, 4.0, 100.0];
        assert_eq!(trim_mean(&x, 0.2).unwrap(), 3.0);
        assert_eq!(trim_mean(&x, 0.0).unwrap(), 22.0);
        assert_eq!(mode(&[3.0, 1.0, 3.0, 1.0, 2.0]).unwrap(), (1.0, 2));
        let z = zscore(&[1.0, 2.0, 3.0], 0).unwrap();
        assert!((z[2] - 1.224744871391589).abs() < 1e-15);
    }

    #[test]
    fn circular_statistics_wrap_around() {
        // 350° and 10° average to 0°, not 180°.
        let a = [350f64.to_radians(), 10f64.to_radians()];
        let m = circmean(&a, 0.0, TAU).unwrap();
        assert!(m.min(TAU - m) < 1e-12);
        let deg = circmean(&[350.0, 10.0, 20.0], 0.0, 360.0).unwrap();
        assert!((deg - 6.70495327058324).abs() < 1e-12);
        assert!(circvar(&[1.0, 1.0], 0.0, TAU).unwrap().abs() < 1e-15);
    }

    #[test]
    fn boxcox_recovers_a_log_normal() {
        // exp of evenly spread normal quantiles: λ should be near 0.
        let x: Vec<f64> = (1..200).map(|i| (lib_jsl_special_functions::normal_quantile(i as f64 / 200.0, 0.0, 1.0)).exp()).collect();
        let l = boxcox_normmax(&x).unwrap();
        assert!(l.abs() < 1e-6, "{l}");
        // Squares of uniform-ish positive data want λ near 1/2.
        let y: Vec<f64> = (1..200).map(|i| (5.0 + lib_jsl_special_functions::normal_quantile(i as f64 / 200.0, 0.0, 1.0)).powi(2)).collect();
        assert!((boxcox_normmax(&y).unwrap() - 0.5).abs() < 0.05);
    }
}
