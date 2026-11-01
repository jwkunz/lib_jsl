//! Descriptive statistics of a sample.

use lib_jsl_core::ErrorsJSL;

fn non_empty(x: &[f64]) -> Result<(), ErrorsJSL> {
    if x.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("The sample must not be empty."));
    }
    Ok(())
}

/// The arithmetic mean.
pub fn mean(x: &[f64]) -> Result<f64, ErrorsJSL> {
    non_empty(x)?;
    // Pairwise-free compensated (Neumaier) summation keeps the mean accurate for long samples.
    let (mut sum, mut compensation) = (0.0f64, 0.0f64);
    for &v in x {
        let t = sum + v;
        compensation += if sum.abs() >= v.abs() { (sum - t) + v } else { (v - t) + sum };
        sum = t;
    }
    Ok((sum + compensation) / x.len() as f64)
}

/// The variance with `ddof` delta degrees of freedom: `Σ(x − x̄)² / (n − ddof)` (1 for the
/// unbiased sample variance, 0 for the population variance). Two-pass for accuracy.
pub fn variance(x: &[f64], ddof: usize) -> Result<f64, ErrorsJSL> {
    if x.len() <= ddof {
        return Err(ErrorsJSL::InvalidInputRange("Need more samples than delta degrees of freedom."));
    }
    let m = mean(x)?;
    let (sum_sq, sum) = x.iter().fold((0.0, 0.0), |(s2, s1), &v| (s2 + (v - m) * (v - m), s1 + (v - m)));
    // The second term corrects the rounding left in the mean.
    Ok((sum_sq - sum * sum / x.len() as f64) / (x.len() - ddof) as f64)
}

/// The standard deviation with `ddof` delta degrees of freedom.
pub fn standard_deviation(x: &[f64], ddof: usize) -> Result<f64, ErrorsJSL> {
    Ok(variance(x, ddof)?.sqrt())
}

/// The standard error of the mean, `s / √n`.
pub fn standard_error(x: &[f64]) -> Result<f64, ErrorsJSL> {
    Ok(standard_deviation(x, 1)? / (x.len() as f64).sqrt())
}

/// The weighted mean `Σ wᵢxᵢ / Σ wᵢ`.
pub fn weighted_mean(x: &[f64], weights: &[f64]) -> Result<f64, ErrorsJSL> {
    if x.len() != weights.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((x.len(), weights.len())));
    }
    let total: f64 = weights.iter().sum();
    if !(total > 0.0) || weights.iter().any(|w| *w < 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Weights must be non-negative with a positive sum."));
    }
    Ok(x.iter().zip(weights).map(|(a, w)| a * w).sum::<f64>() / total)
}

/// The geometric mean (all values positive).
pub fn geometric_mean(x: &[f64]) -> Result<f64, ErrorsJSL> {
    non_empty(x)?;
    if x.iter().any(|v| !(*v > 0.0)) {
        return Err(ErrorsJSL::InvalidInputRange("The geometric mean needs positive values."));
    }
    Ok((x.iter().map(|v| v.ln()).sum::<f64>() / x.len() as f64).exp())
}

/// The harmonic mean (all values positive).
pub fn harmonic_mean(x: &[f64]) -> Result<f64, ErrorsJSL> {
    non_empty(x)?;
    if x.iter().any(|v| !(*v > 0.0)) {
        return Err(ErrorsJSL::InvalidInputRange("The harmonic mean needs positive values."));
    }
    Ok(x.len() as f64 / x.iter().map(|v| 1.0 / v).sum::<f64>())
}

fn sorted(x: &[f64]) -> Result<Vec<f64>, ErrorsJSL> {
    non_empty(x)?;
    if x.iter().any(|v| v.is_nan()) {
        return Err(ErrorsJSL::InvalidInputRange("The sample contains NaN."));
    }
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    Ok(s)
}

fn quantile_sorted(s: &[f64], p: f64) -> f64 {
    let h = (s.len() - 1) as f64 * p;
    let (lo, frac) = (h.floor() as usize, h - h.floor());
    if lo + 1 < s.len() { s[lo] + frac * (s[lo + 1] - s[lo]) } else { s[lo] }
}

/// The `p`-quantile (`0 ≤ p ≤ 1`) by linear interpolation between order statistics at
/// `h = (n − 1)p` (Hyndman and Fan type 7, the default of R and NumPy).
pub fn quantile(x: &[f64], p: f64) -> Result<f64, ErrorsJSL> {
    if !(0.0..=1.0).contains(&p) {
        return Err(ErrorsJSL::InvalidInputRange("The quantile probability must lie in [0, 1]."));
    }
    Ok(quantile_sorted(&sorted(x)?, p))
}

/// The median.
pub fn median(x: &[f64]) -> Result<f64, ErrorsJSL> {
    quantile(x, 0.5)
}

/// The interquartile range `Q3 − Q1`.
pub fn interquartile_range(x: &[f64]) -> Result<f64, ErrorsJSL> {
    let s = sorted(x)?;
    Ok(quantile_sorted(&s, 0.75) - quantile_sorted(&s, 0.25))
}

/// The median absolute deviation `median(|x − median(x)|)`, scaled by `scale` (1.4826 makes it a
/// consistent estimator of σ for normal data).
pub fn median_absolute_deviation(x: &[f64], scale: f64) -> Result<f64, ErrorsJSL> {
    let m = median(x)?;
    Ok(scale * median(&x.iter().map(|v| (v - m).abs()).collect::<Vec<_>>())?)
}

/// The sample skewness `g₁ = m₃ / m₂^{3/2}` (biased, as `scipy.stats.skew` by default).
pub fn skewness(x: &[f64]) -> Result<f64, ErrorsJSL> {
    let m = mean(x)?;
    let n = x.len() as f64;
    let m2 = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
    let m3 = x.iter().map(|v| (v - m).powi(3)).sum::<f64>() / n;
    Ok(m3 / m2.powf(1.5))
}

/// The excess kurtosis `g₂ = m₄ / m₂² − 3` (biased; 0 for a normal distribution).
pub fn excess_kurtosis(x: &[f64]) -> Result<f64, ErrorsJSL> {
    let m = mean(x)?;
    let n = x.len() as f64;
    let m2 = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
    let m4 = x.iter().map(|v| (v - m).powi(4)).sum::<f64>() / n;
    Ok(m4 / (m2 * m2) - 3.0)
}

/// A summary of a sample.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub count: usize,
    pub mean: f64,
    /// Sample standard deviation (`ddof = 1`; NaN for a single value).
    pub standard_deviation: f64,
    pub minimum: f64,
    pub first_quartile: f64,
    pub median: f64,
    pub third_quartile: f64,
    pub maximum: f64,
}

/// Count, mean, standard deviation and the five-number summary.
pub fn summary(x: &[f64]) -> Result<Summary, ErrorsJSL> {
    let s = sorted(x)?;
    Ok(Summary {
        count: s.len(),
        mean: mean(&s)?,
        standard_deviation: if s.len() > 1 { standard_deviation(&s, 1)? } else { f64::NAN },
        minimum: s[0],
        first_quartile: quantile_sorted(&s, 0.25),
        median: quantile_sorted(&s, 0.5),
        third_quartile: quantile_sorted(&s, 0.75),
        maximum: s[s.len() - 1],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn location_and_spread() {
        let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert_eq!(mean(&x).unwrap(), 5.0);
        assert_eq!(variance(&x, 0).unwrap(), 4.0);
        assert!((variance(&x, 1).unwrap() - 32.0 / 7.0).abs() < 1e-15);
        assert_eq!(standard_deviation(&x, 0).unwrap(), 2.0);
        assert_eq!(median(&x).unwrap(), 4.5);
        assert_eq!(quantile(&x, 0.25).unwrap(), 4.0);
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0], 0.1).unwrap(), 1.3); // type 7
        assert_eq!(interquartile_range(&x).unwrap(), 1.5);
        assert_eq!(median_absolute_deviation(&x, 1.0).unwrap(), 0.5);
        assert!((geometric_mean(&[1.0, 4.0, 16.0]).unwrap() - 4.0).abs() < 1e-14);
        assert!((harmonic_mean(&[1.0, 2.0, 4.0]).unwrap() - 12.0 / 7.0).abs() < 1e-15);
        assert_eq!(weighted_mean(&[1.0, 3.0], &[3.0, 1.0]).unwrap(), 1.5);
        assert!(mean(&[]).is_err() && variance(&[1.0], 1).is_err() && geometric_mean(&[1.0, -1.0]).is_err());
    }

    #[test]
    fn variance_is_accurate_with_a_large_offset_and_shape_statistics() {
        // A naive Σx² − n x̄² formula loses every digit here.
        let x: Vec<f64> = (0..1000).map(|i| 1e9 + (i % 10) as f64).collect();
        assert!((variance(&x, 0).unwrap() - 8.25).abs() < 1e-6);
        // A symmetric sample has zero skewness; uniform-ish data has negative excess kurtosis.
        let y: Vec<f64> = (0..101).map(|i| i as f64).collect();
        assert!(skewness(&y).unwrap().abs() < 1e-12);
        assert!((excess_kurtosis(&y).unwrap() + 1.2).abs() < 0.01);
        assert!(skewness(&[1.0, 1.0, 1.0, 10.0]).unwrap() > 1.0);
        let s = summary(&[3.0, 1.0, 2.0]).unwrap();
        assert_eq!((s.count, s.minimum, s.median, s.maximum, s.standard_deviation), (3, 1.0, 2.0, 3.0, 1.0));
    }
}
