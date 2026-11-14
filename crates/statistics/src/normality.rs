//! Tests of normality, following `scipy.stats`: Shapiro-Wilk ([`shapiro`], Royston's algorithm
//! AS R94), Anderson-Darling ([`anderson_normal`]), Jarque-Bera ([`jarque_bera`]) and
//! D'Agostino and Pearson's omnibus test ([`normaltest`], from [`skewtest`] and [`kurtosistest`]).
//!
//! See the book entry [More hypothesis tests](https://jwkunz.github.io/lib_jsl/book/statistics/more-tests.html).

use crate::hypothesis_tests::{Alternative, TestResult};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_special_functions::{chi_square_survival, normal_cdf, normal_quantile};

fn sorted(x: &[f64]) -> Vec<f64> {
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    s
}

fn poly(c: &[f64], x: f64) -> f64 {
    c.iter().rev().fold(0.0, |acc, &ci| acc * x + ci)
}

// ANCHOR: shapiro
/// The Shapiro-Wilk test (SciPy's `shapiro`): `W` measures how well the ordered sample matches
/// the expected normal order statistics, `W = (Σ aᵢ x₍ᵢ₎)² / Σ (xᵢ − x̄)²`, with Royston's
/// approximations (AS R94) to the coefficients `aᵢ` and to the null distribution of `ln(1 − W)`.
/// Valid for 3 ≤ n ≤ 5000.
pub fn shapiro(x: &[f64]) -> Result<TestResult, ErrorsJSL> {
    let n = x.len();
    if !(3..=5000).contains(&n) || x.iter().any(|v| !v.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("Shapiro-Wilk needs 3 to 5000 finite values"));
    }
    let xs = sorted(x);
    let nn2 = n / 2;
    let an = n as f64;
    // Coefficients for the lower half; a[i] pairs x₍ₙ₋ᵢ₎ with x₍ᵢ₎.
    let mut a = vec![0.0; nn2];
    if n == 3 {
        a[0] = 0.5f64.sqrt();
    } else {
        let an25 = an + 0.25;
        let m: Vec<f64> = (1..=nn2).map(|i| normal_quantile((i as f64 - 0.375) / an25, 0.0, 1.0)).collect();
        let summ2 = 2.0 * m.iter().map(|v| v * v).sum::<f64>();
        let ssumm2 = summ2.sqrt();
        let rsn = 1.0 / an.sqrt();
        let c1 = [0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056];
        let c2 = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633];
        let a1 = poly(&c1, rsn) - m[0] / ssumm2;
        let (i1, fac) = if n > 5 {
            let a2 = -m[1] / ssumm2 + poly(&c2, rsn);
            a[1] = a2;
            (2, ((summ2 - 2.0 * m[0] * m[0] - 2.0 * m[1] * m[1]) / (1.0 - 2.0 * a1 * a1 - 2.0 * a2 * a2)).sqrt())
        } else {
            (1, ((summ2 - 2.0 * m[0] * m[0]) / (1.0 - 2.0 * a1 * a1)).sqrt())
        };
        a[0] = a1;
        for i in i1..nn2 {
            a[i] = -m[i] / fac;
        }
    }
    let mean = xs.iter().sum::<f64>() / an;
    let ssq: f64 = xs.iter().map(|v| (v - mean) * (v - mean)).sum();
    if ssq == 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("all values are equal"));
    }
    let num: f64 = (0..nn2).map(|i| a[i] * (xs[n - 1 - i] - xs[i])).sum();
    let w = (num * num / ssq).min(1.0);
    // The p-value.
    let pw = if n == 3 {
        let pi6 = 6.0 / std::f64::consts::PI;
        let stqr = (0.75f64).sqrt().asin();
        (pi6 * (w.sqrt().asin() - stqr)).max(0.0)
    } else {
        let w1 = (1.0 - w).ln();
        let (y, m, s) = if n <= 11 {
            let gamma = poly(&[-2.273, 0.459], an);
            if w1 >= gamma {
                return Ok(TestResult { statistic: w, p_value: 1e-99, degrees_of_freedom: f64::NAN });
            }
            (-(gamma - w1).ln(), poly(&[0.544, -0.39978, 0.025054, -6.714e-4], an), poly(&[1.3822, -0.77857, 0.062767, -0.0020322], an).exp())
        } else {
            let xx = an.ln();
            (w1, poly(&[-1.5861, -0.31082, -0.083751, 0.0038915], xx), poly(&[-0.4803, -0.082676, 0.0030302], xx).exp())
        };
        1.0 - normal_cdf((y - m) / s, 0.0, 1.0)
    };
    Ok(TestResult { statistic: w, p_value: pw, degrees_of_freedom: f64::NAN })
}
// ANCHOR_END: shapiro

/// The result of [`anderson_normal`].
#[derive(Clone, Debug)]
pub struct AndersonResult {
    pub statistic: f64,
    /// The critical values at the significance levels.
    pub critical_values: [f64; 5],
    /// 15%, 10%, 5%, 2.5%, 1%.
    pub significance_levels: [f64; 5],
}

/// The Anderson-Darling test of normality with estimated mean and standard deviation (SciPy's
/// `anderson(x, 'norm')`): `A² = −n − Σ (2i − 1)/n [ln Φ(zᵢ) + ln(1 − Φ(z₍ₙ₊₁₋ᵢ₎))]`, with the
/// Stephens critical values adjusted for `n`. The null hypothesis is rejected at a level if the
/// statistic exceeds its critical value.
pub fn anderson_normal(x: &[f64]) -> Result<AndersonResult, ErrorsJSL> {
    let n = x.len();
    if n < 2 {
        return Err(ErrorsJSL::InvalidInputRange("need at least two values"));
    }
    let xs = sorted(x);
    let an = n as f64;
    let mean = xs.iter().sum::<f64>() / an;
    let sd = (xs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (an - 1.0)).sqrt();
    let z: Vec<f64> = xs.iter().map(|v| (v - mean) / sd).collect();
    let logcdf = |v: f64| normal_cdf(v, 0.0, 1.0).ln();
    let logsf = |v: f64| normal_cdf(-v, 0.0, 1.0).ln();
    let s: f64 = (0..n).map(|i| (2 * i + 1) as f64 / an * (logcdf(z[i]) + logsf(z[n - 1 - i]))).sum();
    let a2 = -an - s;
    let div = 1.0 + 0.75 / an + 2.25 / (an * an);
    let base = [0.561, 0.631, 0.752, 0.873, 1.035];
    Ok(AndersonResult { statistic: a2, critical_values: base.map(|b| (b / div * 1000.0).round() / 1000.0), significance_levels: [15.0, 10.0, 5.0, 2.5, 1.0] })
}

fn moments(x: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let m2 = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n;
    let m3 = x.iter().map(|v| (v - m).powi(3)).sum::<f64>() / n;
    let m4 = x.iter().map(|v| (v - m).powi(4)).sum::<f64>() / n;
    (m2, m3, m4)
}

/// The Jarque-Bera test: `JB = n/6 (S² + K²/4)` from the sample skewness `S` and excess kurtosis
/// `K` (biased), asymptotically χ² with 2 degrees of freedom (SciPy's `jarque_bera`).
pub fn jarque_bera(x: &[f64]) -> Result<TestResult, ErrorsJSL> {
    if x.len() < 2 {
        return Err(ErrorsJSL::InvalidInputRange("need at least two values"));
    }
    let n = x.len() as f64;
    let (m2, m3, m4) = moments(x);
    let (s, k) = (m3 / m2.powf(1.5), m4 / (m2 * m2) - 3.0);
    let jb = n / 6.0 * (s * s + k * k / 4.0);
    Ok(TestResult { statistic: jb, p_value: chi_square_survival(jb, 2.0), degrees_of_freedom: 2.0 })
}

fn normal_tail(z: f64, alternative: Alternative) -> f64 {
    match alternative {
        Alternative::TwoSided => 2.0 * normal_cdf(-z.abs(), 0.0, 1.0),
        Alternative::Less => normal_cdf(z, 0.0, 1.0),
        Alternative::Greater => normal_cdf(-z, 0.0, 1.0),
    }
}

// ANCHOR: dagostino
/// D'Agostino's test that the skewness is that of a normal distribution (SciPy's `skewtest`,
/// n ≥ 8): the sample skewness is transformed to an approximately standard normal `Z`.
pub fn skewtest(x: &[f64], alternative: Alternative) -> Result<TestResult, ErrorsJSL> {
    let n = x.len() as f64;
    if n < 8.0 {
        return Err(ErrorsJSL::InvalidInputRange("skewtest needs at least 8 values"));
    }
    let (m2, m3, _) = moments(x);
    let b2 = m3 / m2.powf(1.5);
    let mut y = b2 * ((n + 1.0) * (n + 3.0) / (6.0 * (n - 2.0))).sqrt();
    let beta2 = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0) / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));
    let w2 = -1.0 + (2.0 * (beta2 - 1.0)).sqrt();
    let delta = 1.0 / (0.5 * w2.ln()).sqrt();
    let alpha = (2.0 / (w2 - 1.0)).sqrt();
    if y == 0.0 {
        y = 1.0;
    }
    let z = delta * (y / alpha + ((y / alpha).powi(2) + 1.0).sqrt()).ln();
    Ok(TestResult { statistic: z, p_value: normal_tail(z, alternative), degrees_of_freedom: f64::NAN })
}

/// Anscombe and Glynn's test that the kurtosis is that of a normal distribution (SciPy's
/// `kurtosistest`, n ≥ 5).
pub fn kurtosistest(x: &[f64], alternative: Alternative) -> Result<TestResult, ErrorsJSL> {
    let n = x.len() as f64;
    if n < 5.0 {
        return Err(ErrorsJSL::InvalidInputRange("kurtosistest needs at least 5 values"));
    }
    let (m2, _, m4) = moments(x);
    let b2 = m4 / (m2 * m2);
    let e = 3.0 * (n - 1.0) / (n + 1.0);
    let varb2 = 24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0).powi(2) * (n + 3.0) * (n + 5.0));
    let xx = (b2 - e) / varb2.sqrt();
    let sqrtbeta1 = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0)) * (6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0))).sqrt();
    let a = 6.0 + 8.0 / sqrtbeta1 * (2.0 / sqrtbeta1 + (1.0 + 4.0 / (sqrtbeta1 * sqrtbeta1)).sqrt());
    let term1 = 1.0 - 2.0 / (9.0 * a);
    let denom = 1.0 + xx * (2.0 / (a - 4.0)).sqrt();
    let term2 = if denom == 0.0 { f64::NAN } else { denom.signum() * ((1.0 - 2.0 / a) / denom.abs()).cbrt() };
    let z = (term1 - term2) / (2.0 / (9.0 * a)).sqrt();
    Ok(TestResult { statistic: z, p_value: normal_tail(z, alternative), degrees_of_freedom: f64::NAN })
}

/// D'Agostino and Pearson's omnibus test (SciPy's `normaltest`): `K² = Z_skew² + Z_kurt²`,
/// approximately χ² with 2 degrees of freedom.
pub fn normaltest(x: &[f64]) -> Result<TestResult, ErrorsJSL> {
    let (s, k) = (skewtest(x, Alternative::TwoSided)?.statistic, kurtosistest(x, Alternative::TwoSided)?.statistic);
    let k2 = s * s + k * k;
    Ok(TestResult { statistic: k2, p_value: chi_square_survival(k2, 2.0), degrees_of_freedom: 2.0 })
}
// ANCHOR_END: dagostino
