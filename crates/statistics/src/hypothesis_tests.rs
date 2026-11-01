//! Classical hypothesis tests.

use crate::correlation::rank;
use crate::descriptive::{mean, variance};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_special_functions::{chi_square_survival, f_cdf, normal_cdf, student_t_cdf};

/// The alternative hypothesis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alternative {
    TwoSided,
    /// The location of the first sample is less (than the reference or the second sample).
    Less,
    Greater,
}

/// A test statistic with its p-value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
    /// Degrees of freedom of the reference distribution (NaN where not applicable).
    pub degrees_of_freedom: f64,
}

fn tail(cdf: impl Fn(f64) -> f64, statistic: f64, alternative: Alternative) -> f64 {
    // Symmetric reference distributions: use the lower tail to avoid cancellation.
    match alternative {
        Alternative::TwoSided => (2.0 * cdf(-statistic.abs())).min(1.0),
        Alternative::Less => cdf(statistic),
        Alternative::Greater => cdf(-statistic),
    }
}

/// One-sample Student t-test of the mean against `mu`.
pub fn t_test_one_sample(x: &[f64], mu: f64, alternative: Alternative) -> Result<TestResult, ErrorsJSL> {
    let n = x.len() as f64;
    let t = (mean(x)? - mu) / (variance(x, 1)? / n).sqrt();
    let df = n - 1.0;
    Ok(TestResult { statistic: t, p_value: tail(|v| student_t_cdf(v, df), t, alternative), degrees_of_freedom: df })
}

/// Two-sample t-test of equal means: Student's pooled-variance test with `equal_variance`,
/// otherwise Welch's test with the Welch-Satterthwaite degrees of freedom.
pub fn t_test_two_sample(x: &[f64], y: &[f64], equal_variance: bool, alternative: Alternative) -> Result<TestResult, ErrorsJSL> {
    let (nx, ny) = (x.len() as f64, y.len() as f64);
    let (vx, vy) = (variance(x, 1)?, variance(y, 1)?);
    let difference = mean(x)? - mean(y)?;
    let (t, df) = if equal_variance {
        let pooled = ((nx - 1.0) * vx + (ny - 1.0) * vy) / (nx + ny - 2.0);
        (difference / (pooled * (1.0 / nx + 1.0 / ny)).sqrt(), nx + ny - 2.0)
    } else {
        let (a, b) = (vx / nx, vy / ny);
        (difference / (a + b).sqrt(), (a + b).powi(2) / (a * a / (nx - 1.0) + b * b / (ny - 1.0)))
    };
    Ok(TestResult { statistic: t, p_value: tail(|v| student_t_cdf(v, df), t, alternative), degrees_of_freedom: df })
}

/// Paired t-test: the one-sample test on the differences `x − y`.
pub fn t_test_paired(x: &[f64], y: &[f64], alternative: Alternative) -> Result<TestResult, ErrorsJSL> {
    if x.len() != y.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((x.len(), y.len())));
    }
    let d: Vec<f64> = x.iter().zip(y).map(|(a, b)| a - b).collect();
    t_test_one_sample(&d, 0.0, alternative)
}

/// Pearson's chi-square goodness-of-fit test of observed counts against expected frequencies
/// (uniform when `None`; rescaled to the observed total), on `k − 1` degrees of freedom.
pub fn chi_square_goodness_of_fit(observed: &[f64], expected: Option<&[f64]>) -> Result<TestResult, ErrorsJSL> {
    let k = observed.len();
    if k < 2 {
        return Err(ErrorsJSL::InvalidInputRange("Need at least two categories."));
    }
    let total: f64 = observed.iter().sum();
    let expected: Vec<f64> = match expected {
        Some(e) if e.len() != k => return Err(ErrorsJSL::IncompatibleArraySizes((e.len(), k))),
        Some(e) => {
            let s: f64 = e.iter().sum();
            e.iter().map(|v| v * total / s).collect()
        }
        None => vec![total / k as f64; k],
    };
    if expected.iter().any(|e| !(*e > 0.0)) {
        return Err(ErrorsJSL::InvalidInputRange("Expected frequencies must be positive."));
    }
    let statistic: f64 = observed.iter().zip(&expected).map(|(o, e)| (o - e) * (o - e) / e).sum();
    let df = (k - 1) as f64;
    Ok(TestResult { statistic, p_value: chi_square_survival(statistic, df), degrees_of_freedom: df })
}

/// Pearson's chi-square test of independence for an `r × c` contingency table (no continuity
/// correction), on `(r − 1)(c − 1)` degrees of freedom.
pub fn chi_square_independence(table: &[Vec<f64>]) -> Result<TestResult, ErrorsJSL> {
    let r = table.len();
    let c = table.first().map_or(0, |row| row.len());
    if r < 2 || c < 2 || table.iter().any(|row| row.len() != c) {
        return Err(ErrorsJSL::InvalidInputRange("Need a rectangular table of at least 2 × 2."));
    }
    let rows: Vec<f64> = table.iter().map(|row| row.iter().sum()).collect();
    let columns: Vec<f64> = (0..c).map(|j| table.iter().map(|row| row[j]).sum()).collect();
    let total: f64 = rows.iter().sum();
    let mut statistic = 0.0;
    for i in 0..r {
        for j in 0..c {
            let e = rows[i] * columns[j] / total;
            if !(e > 0.0) {
                return Err(ErrorsJSL::InvalidInputRange("A row or column of the table is empty."));
            }
            statistic += (table[i][j] - e).powi(2) / e;
        }
    }
    let df = ((r - 1) * (c - 1)) as f64;
    Ok(TestResult { statistic, p_value: chi_square_survival(statistic, df), degrees_of_freedom: df })
}

/// One-way analysis of variance: the F test of equal group means, on `(k − 1, N − k)` degrees
/// of freedom (reported as the between-groups value; the within-groups one is `N − k`).
pub fn anova_one_way(groups: &[Vec<f64>]) -> Result<(TestResult, f64), ErrorsJSL> {
    let k = groups.len();
    let n: usize = groups.iter().map(|g| g.len()).sum();
    if k < 2 || n <= k || groups.iter().any(|g| g.is_empty()) {
        return Err(ErrorsJSL::InvalidInputRange("Need at least two non-empty groups and more observations than groups."));
    }
    let grand = groups.iter().flatten().sum::<f64>() / n as f64;
    let mut between = 0.0;
    let mut within = 0.0;
    for g in groups {
        let m = g.iter().sum::<f64>() / g.len() as f64;
        between += g.len() as f64 * (m - grand).powi(2);
        within += g.iter().map(|v| (v - m).powi(2)).sum::<f64>();
    }
    let (df1, df2) = ((k - 1) as f64, (n - k) as f64);
    let f = (between / df1) / (within / df2);
    Ok((TestResult { statistic: f, p_value: 1.0 - f_cdf(f, df1, df2), degrees_of_freedom: df1 }, df2))
}

/// The Kolmogorov distribution's survival function `Q(λ) = 2 Σ (−1)^{k−1} e^{−2k²λ²}`.
pub fn kolmogorov_survival(lambda: f64) -> f64 {
    if lambda < 0.2 {
        return 1.0;
    }
    let mut sum = 0.0;
    for k in 1..=100 {
        let term = (-2.0 * (k * k) as f64 * lambda * lambda).exp();
        sum += if k % 2 == 1 { term } else { -term };
        if term < 1e-17 {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

/// The one-sample Kolmogorov-Smirnov test of `x` against a continuous `cdf`:
/// `D = sup |F_n − F|`, with the asymptotic p-value `Q((√n + 0.12 + 0.11/√n) D)` (Stephens).
pub fn ks_test_one_sample<F: Fn(f64) -> f64>(x: &[f64], cdf: F) -> Result<TestResult, ErrorsJSL> {
    if x.is_empty() || x.iter().any(|v| v.is_nan()) {
        return Err(ErrorsJSL::InvalidInputRange("The sample must be non-empty and free of NaN."));
    }
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len() as f64;
    let d = s.iter().enumerate().map(|(i, &v)| { let f = cdf(v); (f - i as f64 / n).max((i + 1) as f64 / n - f) }).fold(0.0, f64::max);
    let en = n.sqrt();
    Ok(TestResult { statistic: d, p_value: kolmogorov_survival((en + 0.12 + 0.11 / en) * d), degrees_of_freedom: f64::NAN })
}

/// The two-sample Kolmogorov-Smirnov test, with the asymptotic p-value using the effective
/// size `nm / (n + m)`.
pub fn ks_test_two_sample(x: &[f64], y: &[f64]) -> Result<TestResult, ErrorsJSL> {
    if x.is_empty() || y.is_empty() || x.iter().chain(y).any(|v| v.is_nan()) {
        return Err(ErrorsJSL::InvalidInputRange("Both samples must be non-empty and free of NaN."));
    }
    let (mut a, mut b) = (x.to_vec(), y.to_vec());
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    let (n, m) = (a.len(), b.len());
    let (mut i, mut j, mut d) = (0usize, 0usize, 0.0f64);
    while i < n && j < m {
        let v = a[i].min(b[j]);
        while i < n && a[i] <= v {
            i += 1;
        }
        while j < m && b[j] <= v {
            j += 1;
        }
        d = d.max((i as f64 / n as f64 - j as f64 / m as f64).abs());
    }
    let en = ((n * m) as f64 / (n + m) as f64).sqrt();
    Ok(TestResult { statistic: d, p_value: kolmogorov_survival((en + 0.12 + 0.11 / en) * d), degrees_of_freedom: f64::NAN })
}

/// The Mann-Whitney U test (Wilcoxon rank-sum) for a location shift between two samples. The
/// statistic is `U` of the first sample; the p-value uses the normal approximation with tie and
/// continuity corrections (adequate for samples above about 10 each).
pub fn mann_whitney_u(x: &[f64], y: &[f64], alternative: Alternative) -> Result<TestResult, ErrorsJSL> {
    if x.is_empty() || y.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("Both samples must be non-empty."));
    }
    let (n1, n2) = (x.len() as f64, y.len() as f64);
    let combined: Vec<f64> = x.iter().chain(y).copied().collect();
    let ranks = rank(&combined);
    let r1: f64 = ranks[..x.len()].iter().sum();
    let u = r1 - n1 * (n1 + 1.0) / 2.0;
    let n = n1 + n2;
    // Tie correction Σ(t³ − t).
    let mut sorted = combined.clone();
    sorted.sort_by(f64::total_cmp);
    let mut ties = 0.0;
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i;
        while j + 1 < sorted.len() && sorted[j + 1] == sorted[i] {
            j += 1;
        }
        let t = (j - i + 1) as f64;
        ties += t * t * t - t;
        i = j + 1;
    }
    let mu = n1 * n2 / 2.0;
    let sigma = (n1 * n2 / 12.0 * ((n + 1.0) - ties / (n * (n - 1.0)))).sqrt();
    let z = |shift: f64| (u - mu + shift) / sigma;
    let p = match alternative {
        Alternative::TwoSided => (2.0 * normal_cdf(-((u - mu).abs() - 0.5).max(0.0) / sigma, 0.0, 1.0)).min(1.0),
        Alternative::Less => normal_cdf(z(0.5), 0.0, 1.0),
        Alternative::Greater => normal_cdf(-z(-0.5), 0.0, 1.0),
    };
    Ok(TestResult { statistic: u, p_value: p, degrees_of_freedom: f64::NAN })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_tests_reach_textbook_critical_values() {
        // x = 2.228 + e with e = −5..5: mean offset 2.228 and standard error exactly 1, so
        // t = 2.228, the two-sided 5% critical value for 10 degrees of freedom.
        let x: Vec<f64> = (-5..=5).map(|e| 2.228 + e as f64).collect();
        let r = t_test_one_sample(&x, 0.0, Alternative::TwoSided).unwrap();
        assert!((r.statistic - 2.228).abs() < 1e-12 && r.degrees_of_freedom == 10.0);
        assert!((r.p_value - 0.05).abs() < 2e-4, "{}", r.p_value);
        let greater = t_test_one_sample(&x, 0.0, Alternative::Greater).unwrap().p_value;
        assert!((greater - r.p_value / 2.0).abs() < 1e-12 && (t_test_one_sample(&x, 0.0, Alternative::Less).unwrap().p_value - (1.0 - greater)).abs() < 1e-12);
        // Two samples: pooled versus Welch.
        let a = [19.8, 20.4, 19.6, 17.8, 18.5, 18.9, 18.3, 18.9, 19.5, 22.0];
        let b = [28.2, 26.6, 20.1, 23.3, 25.2, 22.1, 17.7, 27.6, 20.6, 13.7, 23.2, 17.5, 20.6, 18.0, 23.9, 21.6, 24.3, 20.4, 24.0, 13.2];
        let student = t_test_two_sample(&a, &b, true, Alternative::TwoSided).unwrap();
        let welch = t_test_two_sample(&a, &b, false, Alternative::TwoSided).unwrap();
        let (va, vb) = (variance(&a, 1).unwrap() / 10.0, variance(&b, 1).unwrap() / 20.0);
        assert!((welch.degrees_of_freedom - (va + vb).powi(2) / (va * va / 9.0 + vb * vb / 19.0)).abs() < 1e-9);
        let difference = mean(&a).unwrap() - mean(&b).unwrap();
        assert!((welch.statistic - difference / (va + vb).sqrt()).abs() < 1e-12);
        assert!(student.degrees_of_freedom == 28.0 && student.statistic < 0.0 && (0.0..1.0).contains(&welch.p_value));
        let paired = t_test_paired(&[1.0, 2.0, 3.0, 4.0], &[1.5, 2.5, 3.0, 5.0], Alternative::TwoSided).unwrap();
        assert!(paired.statistic < 0.0 && paired.degrees_of_freedom == 3.0);
    }

    #[test]
    fn chi_square_and_anova() {
        // 60/40 against 50/50: χ² = 4, p = 0.0455.
        let g = chi_square_goodness_of_fit(&[60.0, 40.0], None).unwrap();
        assert!((g.statistic - 4.0).abs() < 1e-12 && (g.p_value - 0.045_500_26).abs() < 1e-7, "{}", g.p_value);
        let weighted = chi_square_goodness_of_fit(&[30.0, 70.0], Some(&[1.0, 3.0])).unwrap();
        assert!((weighted.statistic - (25.0 / 25.0 + 25.0 / 75.0)).abs() < 1e-12);
        // 2 × 2 table with expected 15 in every cell: χ² = 20/3, p = 0.00982.
        let ind = chi_square_independence(&[vec![10.0, 20.0], vec![20.0, 10.0]]).unwrap();
        assert!((ind.statistic - 20.0 / 3.0).abs() < 1e-12 && (ind.p_value - 0.009_823).abs() < 1e-5, "{}", ind.p_value);
        // ANOVA: F = MS_between / MS_within computed by hand.
        let groups = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![3.0, 4.0, 5.0, 6.0, 7.0], vec![5.0, 6.0, 7.0, 8.0, 9.0]];
        let (f, df2) = anova_one_way(&groups).unwrap();
        assert!((f.statistic - (40.0 / 2.0) / (30.0 / 12.0)).abs() < 1e-12 && df2 == 12.0);
        // F(2, 12) = 3.885 is the 5% critical value.
        let critical: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![0.0, 1.0]];
        assert!(anova_one_way(&critical).unwrap().0.p_value > 0.99);
        assert!((1.0 - f_cdf(3.885, 2.0, 12.0) - 0.05).abs() < 1e-3);
        assert!(f.p_value < 0.01);
        assert!(chi_square_goodness_of_fit(&[1.0], None).is_err() && chi_square_independence(&[vec![1.0, 2.0]]).is_err());
    }

    #[test]
    fn rank_and_distribution_tests() {
        assert!((kolmogorov_survival(1.358) - 0.05).abs() < 2e-4);
        // A perfectly spaced uniform sample is accepted; a shifted one rejected.
        let u: Vec<f64> = (0..200).map(|i| (i as f64 + 0.5) / 200.0).collect();
        let accept = ks_test_one_sample(&u, |v| v.clamp(0.0, 1.0)).unwrap();
        assert!((accept.statistic - 0.0025).abs() < 1e-12 && accept.p_value > 0.99);
        let shifted: Vec<f64> = u.iter().map(|v| v * 0.8).collect();
        assert!(ks_test_one_sample(&shifted, |v| v.clamp(0.0, 1.0)).unwrap().p_value < 1e-6);
        let two = ks_test_two_sample(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0, 7.0]).unwrap();
        assert_eq!(two.statistic, 1.0);
        assert_eq!(ks_test_two_sample(&u, &u).unwrap().statistic, 0.0);
        // Mann-Whitney: complete separation of two samples of 10 gives U = 0 and z = −49.5/13.23.
        let x: Vec<f64> = (0..10).map(|v| v as f64).collect();
        let y: Vec<f64> = (10..20).map(|v| v as f64).collect();
        let mw = mann_whitney_u(&x, &y, Alternative::TwoSided).unwrap();
        let z = 49.5 / (100.0f64 * 21.0 / 12.0).sqrt();
        assert!(mw.statistic == 0.0 && (mw.p_value - 2.0 * normal_cdf(-z, 0.0, 1.0)).abs() < 1e-12);
        assert!(mann_whitney_u(&x, &y, Alternative::Less).unwrap().p_value < 1e-3 && mann_whitney_u(&x, &y, Alternative::Greater).unwrap().p_value > 0.99);
    }
}
