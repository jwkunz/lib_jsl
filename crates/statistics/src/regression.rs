//! Straight-line fits: least squares with its standard errors and test (SciPy's `linregress`), and
//! the robust median-of-slopes estimators of Theil and Sen (`theilslopes`) and Siegel
//! (`siegelslopes`).
//!
//! See the book entry [More descriptive statistics and robust regression](https://jwkunz.github.io/lib_jsl/book/statistics/robust-regression.html#robust-lines).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_special_functions::{normal_quantile, student_t_cdf};

/// The result of [`linregress`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinregressResult {
    pub slope: f64,
    pub intercept: f64,
    /// Pearson's correlation coefficient.
    pub rvalue: f64,
    /// Two-sided p-value for the hypothesis that the slope is zero (t-test, `n − 2` degrees of
    /// freedom).
    pub pvalue: f64,
    /// Standard error of the slope.
    pub stderr: f64,
    pub intercept_stderr: f64,
}

fn check(x: &[f64], y: &[f64], min: usize) -> Result<(), ErrorsJSL> {
    if x.len() != y.len() {
        return Err(ErrorsJSL::InvalidInputRange("x and y must have the same length"));
    }
    if x.len() < min {
        return Err(ErrorsJSL::InvalidInputRange("too few points"));
    }
    Ok(())
}

// ANCHOR: linregress
/// The least-squares line `y ≈ slope · x + intercept` (SciPy's `linregress`), from the centred
/// sums `Sxx`, `Sxy`, `Syy`: `slope = Sxy/Sxx`, `r = Sxy/√(Sxx Syy)`, and the slope's standard
/// error `√((1 − r²) Syy / ((n − 2) Sxx))`.
pub fn linregress(x: &[f64], y: &[f64]) -> Result<LinregressResult, ErrorsJSL> {
    check(x, y, 2)?;
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        let (dx, dy) = (a - mx, b - my);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if sxx == 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("all x values are identical"));
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let r = if syy == 0.0 { 0.0 } else { (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0) };
    if x.len() == 2 {
        return Ok(LinregressResult { slope, intercept, rvalue: r, pvalue: 1.0, stderr: 0.0, intercept_stderr: 0.0 });
    }
    let df = n - 2.0;
    let stderr = ((1.0 - r * r) * syy / sxx / df).sqrt();
    let t = r * (df / ((1.0 - r) * (1.0 + r))).sqrt();
    let pvalue = if t.is_infinite() { 0.0 } else { 2.0 * student_t_cdf(-t.abs(), df) };
    let intercept_stderr = stderr * (sxx / n + mx * mx).sqrt();
    Ok(LinregressResult { slope, intercept, rvalue: r, pvalue, stderr, intercept_stderr })
}
// ANCHOR_END: linregress

/// The result of [`theilslopes`] and [`siegelslopes`] (the latter has no confidence interval).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobustLine {
    pub slope: f64,
    pub intercept: f64,
    /// Lower and upper confidence limits on the slope (Theil-Sen only; `NaN` for Siegel).
    pub low_slope: f64,
    pub high_slope: f64,
}

fn median_of(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// Sizes of the groups of tied values.
fn tie_groups(x: &[f64]) -> Vec<usize> {
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let mut groups = Vec::new();
    let mut i = 0;
    while i < s.len() {
        let j = s[i..].iter().take_while(|&&v| v == s[i]).count();
        if j > 1 {
            groups.push(j);
        }
        i += j;
    }
    groups
}

// ANCHOR: theilslopes
/// The Theil-Sen line (SciPy's `theilslopes` with `method='separate'`): the slope is the median
/// of the pairwise slopes `(yⱼ − yᵢ)/(xⱼ − xᵢ)` over pairs with `xⱼ > xᵢ`, the intercept
/// `median(y) − slope · median(x)`. The confidence interval at level `alpha` (SciPy's default
/// 0.95) is Sen's: order statistics of the sorted slopes at ranks `(N ± z σ)/2`, where `σ²` is the
/// variance of Kendall's statistic corrected for ties. Breakdown point 29 %.
pub fn theilslopes(y: &[f64], x: &[f64], alpha: f64) -> Result<RobustLine, ErrorsJSL> {
    check(x, y, 2)?;
    let n = x.len();
    let mut slopes = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in 0..n {
            if x[j] > x[i] {
                slopes.push((y[j] - y[i]) / (x[j] - x[i]));
            }
        }
    }
    if slopes.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("all x values are identical"));
    }
    let slope = median_of(&mut slopes);
    let intercept = median_of(&mut y.to_vec()) - slope * median_of(&mut x.to_vec());
    let alpha = if alpha > 0.5 { 1.0 - alpha } else { alpha };
    let z = normal_quantile(alpha / 2.0, 0.0, 1.0);
    let term = |k: usize| (k * (k - 1) * (2 * k + 5)) as f64;
    let sigsq = (term(n) - tie_groups(x).into_iter().map(term).sum::<f64>() - tie_groups(y).into_iter().map(term).sum::<f64>()) / 18.0;
    let sigma = sigsq.sqrt();
    let nt = slopes.len() as f64;
    // NumPy rounds halves to even; so does `round_ties_even`.
    let ru = (((nt - z * sigma) / 2.0).round_ties_even() as i64).min(slopes.len() as i64 - 1).max(0) as usize;
    let rl = ((((nt + z * sigma) / 2.0).round_ties_even() as i64) - 1).max(0) as usize;
    Ok(RobustLine { slope, intercept, low_slope: slopes[rl.min(slopes.len() - 1)], high_slope: slopes[ru] })
}
// ANCHOR_END: theilslopes

/// Siegel's repeated-medians line (SciPy's `siegelslopes` with `method='hierarchical'`): for each
/// point the median slope to every other point with a different `x`, the slope is the median of
/// those, and the intercept the median of `yᵢ − slope · xᵢ`. Breakdown point 50 %.
pub fn siegelslopes(y: &[f64], x: &[f64]) -> Result<RobustLine, ErrorsJSL> {
    check(x, y, 2)?;
    let n = x.len();
    let mut medians = Vec::with_capacity(n);
    for j in 0..n {
        let mut s: Vec<f64> = (0..n).filter(|&i| x[i] != x[j]).map(|i| (y[j] - y[i]) / (x[j] - x[i])).collect();
        if !s.is_empty() {
            medians.push(median_of(&mut s));
        }
    }
    if medians.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("all x values are identical"));
    }
    let slope = median_of(&mut medians);
    let mut r: Vec<f64> = x.iter().zip(y).map(|(a, b)| b - slope * a).collect();
    Ok(RobustLine { slope, intercept: median_of(&mut r), low_slope: f64::NAN, high_slope: f64::NAN })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_line() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v - 1.0).collect();
        let r = linregress(&x, &y).unwrap();
        assert!((r.slope - 2.0).abs() < 1e-15 && (r.intercept + 1.0).abs() < 1e-15 && (r.rvalue - 1.0).abs() < 1e-15);
        let t = theilslopes(&y, &x, 0.95).unwrap();
        assert_eq!((t.slope, t.intercept), (2.0, -1.0));
        let s = siegelslopes(&y, &x).unwrap();
        assert_eq!((s.slope, s.intercept), (2.0, -1.0));
    }

    #[test]
    fn robust_lines_ignore_outliers() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let mut y: Vec<f64> = x.iter().map(|v| 0.5 * v + 3.0).collect();
        y[3] = 100.0;
        y[17] = -50.0;
        assert!((theilslopes(&y, &x, 0.95).unwrap().slope - 0.5).abs() < 1e-12);
        assert!((siegelslopes(&y, &x).unwrap().slope - 0.5).abs() < 1e-12);
        assert!((linregress(&x, &y).unwrap().slope - 0.5).abs() > 0.5);
    }
}
