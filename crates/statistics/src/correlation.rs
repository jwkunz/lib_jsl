//! Covariance and correlation: Pearson, Spearman and Kendall.

use crate::descriptive::mean;
use lib_jsl_core::ErrorsJSL;
use lib_jsl_special_functions::{normal_cdf, student_t_cdf};

fn paired(x: &[f64], y: &[f64], minimum: usize) -> Result<(), ErrorsJSL> {
    if x.len() != y.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((x.len(), y.len())));
    }
    if x.len() < minimum {
        return Err(ErrorsJSL::InvalidInputRange("Too few paired observations."));
    }
    Ok(())
}

/// The covariance `Σ(x − x̄)(y − ȳ) / (n − ddof)`.
pub fn covariance(x: &[f64], y: &[f64], ddof: usize) -> Result<f64, ErrorsJSL> {
    paired(x, y, ddof + 1)?;
    let (mx, my) = (mean(x)?, mean(y)?);
    Ok(x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / (x.len() - ddof) as f64)
}

/// A correlation coefficient with its two-sided p-value for the null hypothesis of no
/// association.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Correlation {
    pub coefficient: f64,
    pub p_value: f64,
}

fn t_test_of_r(r: f64, n: usize) -> f64 {
    if n <= 2 {
        return f64::NAN;
    }
    if r.abs() >= 1.0 {
        return 0.0;
    }
    let df = (n - 2) as f64;
    let t = r * (df / (1.0 - r * r)).sqrt();
    2.0 * student_t_cdf(-t.abs(), df)
}

/// Pearson's product-moment correlation, with the p-value from `t = r √((n−2)/(1−r²))` on
/// `n − 2` degrees of freedom (exact for bivariate normal data).
pub fn pearson(x: &[f64], y: &[f64]) -> Result<Correlation, ErrorsJSL> {
    paired(x, y, 2)?;
    let (mx, my) = (mean(x)?, mean(y)?);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
        syy += (b - my) * (b - my);
    }
    if sxx == 0.0 || syy == 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("Correlation is undefined for a constant sample."));
    }
    let r = (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0);
    Ok(Correlation { coefficient: r, p_value: t_test_of_r(r, x.len()) })
}

/// Ranks starting at 1, with tied values given the average of their ranks.
pub fn rank(x: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut ranks = vec![0.0; x.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && x[order[j + 1]] == x[order[i]] {
            j += 1;
        }
        let average = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[order[k]] = average;
        }
        i = j + 1;
    }
    ranks
}

/// Spearman's rank correlation (Pearson's on average ranks), with the same t approximation for
/// the p-value. Measures any monotone association.
pub fn spearman(x: &[f64], y: &[f64]) -> Result<Correlation, ErrorsJSL> {
    paired(x, y, 2)?;
    let r = pearson(&rank(x), &rank(y))?.coefficient;
    Ok(Correlation { coefficient: r, p_value: t_test_of_r(r, x.len()) })
}

/// Kendall's τ_b, `(C − D) / √((n₀ − n₁)(n₀ − n₂))` over concordant and discordant pairs with
/// tie corrections, and a two-sided p-value from the normal approximation to `S = C − D` with
/// the tie-corrected variance. `O(n²)`.
pub fn kendall(x: &[f64], y: &[f64]) -> Result<Correlation, ErrorsJSL> {
    paired(x, y, 2)?;
    let n = x.len();
    let mut s: i64 = 0;
    for i in 0..n {
        for j in i + 1..n {
            let product = (x[i] - x[j]).signum() * (y[i] - y[j]).signum();
            if x[i] != x[j] && y[i] != y[j] {
                s += product as i64;
            }
        }
    }
    // Tie groups of each variable.
    let groups = |v: &[f64]| -> Vec<f64> {
        let mut sorted = v.to_vec();
        sorted.sort_by(f64::total_cmp);
        let mut out = Vec::new();
        let mut i = 0;
        while i < sorted.len() {
            let mut j = i;
            while j + 1 < sorted.len() && sorted[j + 1] == sorted[i] {
                j += 1;
            }
            if j > i {
                out.push((j - i + 1) as f64);
            }
            i = j + 1;
        }
        out
    };
    let (tx, ty) = (groups(x), groups(y));
    let nf = n as f64;
    let n0 = nf * (nf - 1.0) / 2.0;
    let n1: f64 = tx.iter().map(|t| t * (t - 1.0) / 2.0).sum();
    let n2: f64 = ty.iter().map(|t| t * (t - 1.0) / 2.0).sum();
    let denominator = ((n0 - n1) * (n0 - n2)).sqrt();
    if denominator == 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("Correlation is undefined for a constant sample."));
    }
    let tau = s as f64 / denominator;
    let sum = |g: &[f64], f: &dyn Fn(f64) -> f64| g.iter().map(|&t| f(t)).sum::<f64>();
    let v0 = nf * (nf - 1.0) * (2.0 * nf + 5.0);
    let vt = sum(&tx, &|t| t * (t - 1.0) * (2.0 * t + 5.0));
    let vu = sum(&ty, &|t| t * (t - 1.0) * (2.0 * t + 5.0));
    let v1 = sum(&tx, &|t| t * (t - 1.0)) * sum(&ty, &|t| t * (t - 1.0)) / (2.0 * nf * (nf - 1.0));
    let v2 = if n > 2 { sum(&tx, &|t| t * (t - 1.0) * (t - 2.0)) * sum(&ty, &|t| t * (t - 1.0) * (t - 2.0)) / (9.0 * nf * (nf - 1.0) * (nf - 2.0)) } else { 0.0 };
    let variance = (v0 - vt - vu) / 18.0 + v1 + v2;
    let z = s as f64 / variance.sqrt();
    Ok(Correlation { coefficient: tau, p_value: 2.0 * normal_cdf(-z.abs(), 0.0, 1.0) })
}

/// The covariance matrix of the columns `variables[k]` (each a sample of one variable).
pub fn covariance_matrix(variables: &[Vec<f64>], ddof: usize) -> Result<Vec<Vec<f64>>, ErrorsJSL> {
    let k = variables.len();
    let mut out = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in i..k {
            let c = covariance(&variables[i], &variables[j], ddof)?;
            out[i][j] = c;
            out[j][i] = c;
        }
    }
    Ok(out)
}

/// The Pearson correlation matrix of the columns `variables[k]`.
pub fn correlation_matrix(variables: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, ErrorsJSL> {
    let c = covariance_matrix(variables, 1)?;
    let k = c.len();
    Ok((0..k).map(|i| (0..k).map(|j| if i == j { 1.0 } else { c[i][j] / (c[i][i] * c[j][j]).sqrt() }).collect()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_on_anscombes_first_set() {
        let x = [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0];
        let y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68];
        let r = pearson(&x, &y).unwrap();
        assert!((r.coefficient - 0.816_420_516).abs() < 1e-8);
        assert!((r.p_value - 0.002_169_6).abs() < 2e-6, "{}", r.p_value);
        assert!((covariance(&x, &y, 1).unwrap() - 5.501).abs() < 1e-3);
        assert_eq!(pearson(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).unwrap().coefficient, 1.0);
        assert!(pearson(&[1.0, 1.0], &[1.0, 2.0]).is_err() && pearson(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn rank_correlations() {
        assert_eq!(rank(&[3.0, 1.0, 4.0, 1.0, 5.0]), vec![3.0, 1.5, 4.0, 1.5, 5.0]);
        // Monotone but nonlinear: Spearman sees perfect association, Pearson does not.
        let x: Vec<f64> = (1..=10).map(|v| v as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v.powi(5)).collect();
        assert_eq!(spearman(&x, &y).unwrap().coefficient, 1.0);
        assert!(pearson(&x, &y).unwrap().coefficient < 0.9);
        // Kendall: C = 6, D = 4 of 10 pairs.
        let k = kendall(&[1.0, 2.0, 3.0, 4.0, 5.0], &[3.0, 4.0, 1.0, 2.0, 5.0]).unwrap();
        assert!((k.coefficient - 0.2).abs() < 1e-15);
        // Ties: τ_b uses the corrected denominator.
        let t = kendall(&[1.0, 1.0, 2.0, 3.0], &[1.0, 2.0, 2.0, 3.0]).unwrap();
        assert!((t.coefficient - 4.0 / 5.0).abs() < 1e-12, "{}", t.coefficient);
        let m = correlation_matrix(&[x.clone(), y.clone(), x.iter().map(|v| -v).collect()]).unwrap();
        assert!((m[0][2] + 1.0).abs() < 1e-12 && m[1][1] == 1.0 && (m[0][1] - m[1][0]).abs() < 1e-15);
    }
}
