//! The studentized range distribution and Tukey's honestly significant difference test (SciPy's
//! `studentized_range` and `tukey_hsd`).
//!
//! See the book entry [Multivariate distributions and further tests](https://jwkunz.github.io/lib_jsl/book/statistics/multivariate-tests.html#tukeys-honestly-significant-difference).

use crate::distributions::ContinuousDistribution;
use lib_jsl_core::ErrorsJSL;
use lib_jsl_special_functions::gamma::ln_gamma;
use lib_jsl_special_functions::error_function::erfc;

/// The standard normal density.
fn normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// The standard normal distribution function, `½ erfc(−z/√2)`, accurate in both tails.
fn normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

/// 16-point Gauss-Legendre nodes and weights on [−1, 1] (positive half).
const GL_X: [f64; 8] = [
    0.095_012_509_837_637_44,
    0.281_603_550_779_258_9,
    0.458_016_777_657_227_4,
    0.617_876_244_402_643_7,
    0.755_404_408_355_003,
    0.865_631_202_387_831_8,
    0.944_575_023_073_232_6,
    0.989_400_934_991_649_9,
];
const GL_W: [f64; 8] = [
    0.189_450_610_455_068_5,
    0.182_603_415_044_923_6,
    0.169_156_519_395_002_5,
    0.149_595_988_816_576_7,
    0.124_628_971_255_533_9,
    0.095_158_511_682_492_78,
    0.062_253_523_938_647_89,
    0.027_152_459_411_754_09,
];

/// `∫_a^b f` by the 16-point Gauss-Legendre rule on `panels` equal panels.
fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, panels: usize) -> f64 {
    let h = (b - a) / panels as f64;
    let mut total = 0.0;
    for p in 0..panels {
        let (c, r) = (a + (p as f64 + 0.5) * h, 0.5 * h);
        for (x, w) in GL_X.iter().zip(&GL_W) {
            total += w * (f(c - r * x) + f(c + r * x));
        }
    }
    total * 0.5 * h
}

// ANCHOR: studentized_range
/// The distribution of `Q = (max − min)/S` for `k` independent standard normal deviates and an
/// independent `S = √(χ²_ν / ν)` (SciPy's `studentized_range(k, df)`):
/// `F(q) = ∫₀^∞ f_S(s) W(qs) ds`, with the distribution of the range
/// `W(w) = k ∫ φ(z) (Φ(z) − Φ(z − w))^{k−1} dz` and
/// `f_S(s) = ν^{ν/2} s^{ν−1} e^{−νs²/2} / (Γ(ν/2) 2^{ν/2−1})`. Both integrals are evaluated by
/// composite Gauss-Legendre quadrature (the inner over `|z| ≤ 8.5`, the outer over the part of
/// `s` holding all but 10⁻¹⁷ of its mass); `df = ∞` gives the range of standard normals itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StudentizedRange {
    pub k: f64,
    pub df: f64,
}

impl StudentizedRange {
    pub fn new(k: f64, df: f64) -> Result<Self, ErrorsJSL> {
        if !(k > 1.0) || !(df > 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("need k > 1 and df > 0"));
        }
        Ok(StudentizedRange { k, df })
    }

    /// `P(range of k standard normals ≤ w)`.
    fn range_cdf(&self, w: f64) -> f64 {
        if w <= 0.0 {
            return 0.0;
        }
        let k = self.k;
        let f = |z: f64| normal_pdf(z) * (normal_cdf(z) - normal_cdf(z - w)).max(0.0).powf(k - 1.0);
        (k * integrate(&f, -8.5, 8.5, 12)).min(1.0)
    }

    /// The density of the range, `k(k−1) ∫ φ(z) φ(z − w) (Φ(z) − Φ(z − w))^{k−2} dz`.
    fn range_pdf(&self, w: f64) -> f64 {
        if w <= 0.0 {
            return 0.0;
        }
        let k = self.k;
        let f = |z: f64| normal_pdf(z) * normal_pdf(z - w) * (normal_cdf(z) - normal_cdf(z - w)).max(0.0).powf(k - 2.0);
        k * (k - 1.0) * integrate(&f, -8.5, 8.5 + w, 12 + w as usize)
    }

    /// The integration range of `s` and the log of `f_S`'s normalising constant.
    fn scale_range(&self) -> (f64, f64, f64) {
        let nu = self.df;
        let log_c = 0.5 * nu * nu.ln() - ln_gamma(0.5 * nu) - (0.5 * nu - 1.0) * 2f64.ln();
        let hi = ((nu + 12.0 * (2.0 * nu).sqrt() + 45.0) / nu).sqrt();
        let lo = if nu > 50.0 { (1.0 - 12.0 / (2.0 * nu).sqrt()).max(0.0) } else { 0.0 };
        (lo, hi, log_c)
    }

    fn mix(&self, inner: &dyn Fn(f64) -> f64, weight_s: bool) -> f64 {
        let (lo, hi, log_c) = self.scale_range();
        let nu = self.df;
        let f = |s: f64| {
            if s <= 0.0 {
                return 0.0;
            }
            let dens = (log_c + (nu - 1.0) * s.ln() - 0.5 * nu * s * s).exp();
            dens * if weight_s { s } else { 1.0 } * inner(s)
        };
        integrate(&f, lo, hi, 24)
    }
}

impl ContinuousDistribution for StudentizedRange {
    fn pdf(&self, q: f64) -> f64 {
        if q <= 0.0 {
            return 0.0;
        }
        if self.df.is_infinite() {
            return self.range_pdf(q);
        }
        self.mix(&|s| self.range_pdf(q * s), true)
    }
    fn cdf(&self, q: f64) -> f64 {
        if q <= 0.0 {
            return 0.0;
        }
        if self.df.is_infinite() {
            return self.range_cdf(q);
        }
        self.mix(&|s| self.range_cdf(q * s), false).clamp(0.0, 1.0)
    }
    /// By bisection on the distribution function (the default would need the moments, which are
    /// themselves integrals).
    fn ppf(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) {
            return f64::NAN;
        }
        if p == 0.0 {
            return 0.0;
        }
        if p == 1.0 {
            return f64::INFINITY;
        }
        // Bracket, then regula falsi with the Illinois modification (superlinear, and safe).
        let (mut lo, mut hi) = (0.0, 4.0);
        let mut fhi = self.cdf(hi) - p;
        while fhi < 0.0 {
            lo = hi;
            hi *= 2.0;
            fhi = self.cdf(hi) - p;
        }
        let mut flo = self.cdf(lo) - p;
        let mut side = 0;
        for _ in 0..100 {
            let x = (lo * fhi - hi * flo) / (fhi - flo);
            let fx = self.cdf(x) - p;
            if fx == 0.0 || (hi - lo).abs() <= 1e-13 * hi {
                return x;
            }
            if (fx < 0.0) == (flo < 0.0) {
                lo = x;
                flo = fx;
                if side == -1 {
                    fhi *= 0.5;
                }
                side = -1;
            } else {
                hi = x;
                fhi = fx;
                if side == 1 {
                    flo *= 0.5;
                }
                side = 1;
            }
            if (hi - lo).abs() <= 1e-13 * hi.abs() {
                break;
            }
        }
        (lo * fhi - hi * flo) / (fhi - flo)
    }
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
    /// `∫ sf`, numerically.
    fn mean(&self) -> f64 {
        self.raw_moment(1)
    }
    fn variance(&self) -> f64 {
        let m1 = self.raw_moment(1);
        self.raw_moment(2) - m1 * m1
    }
    fn skewness(&self) -> f64 {
        let (m1, m2, m3) = (self.raw_moment(1), self.raw_moment(2), self.raw_moment(3));
        let v = m2 - m1 * m1;
        (m3 - 3.0 * m1 * m2 + 2.0 * m1.powi(3)) / v.powf(1.5)
    }
    fn excess_kurtosis(&self) -> f64 {
        let (m1, m2, m3, m4) = (self.raw_moment(1), self.raw_moment(2), self.raw_moment(3), self.raw_moment(4));
        let v = m2 - m1 * m1;
        (m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1.powi(4)) / (v * v) - 3.0
    }
}

impl StudentizedRange {
    /// `E[Qʲ] = ∫₀^∞ j q^{j−1} P(Q > q) dq` (infinite when `j ≥ df`).
    fn raw_moment(&self, j: i32) -> f64 {
        if self.df <= j as f64 {
            return f64::INFINITY;
        }
        let upper = self.ppf(1.0 - 1e-12) * 1.5;
        let f = |q: f64| j as f64 * q.powi(j - 1) * (1.0 - self.cdf(q));
        integrate(&f, 0.0, upper, 24)
    }
}
// ANCHOR_END: studentized_range

/// Lower and upper limits, as `k × k` matrices.
pub type IntervalMatrices = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// The result of [`tukey_hsd`]: entry `(i, j)` compares treatment `i` with treatment `j`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TukeyHsdResult {
    /// The differences of the means, `x̄_i − x̄_j`.
    pub statistic: Vec<Vec<f64>>,
    pub p_values: Vec<Vec<f64>>,
    /// The standard errors of the differences (divided by √2, as SciPy).
    pub std_err: Vec<Vec<f64>>,
    /// The degrees of freedom of each comparison.
    pub df: Vec<Vec<f64>>,
    pub ntreatments: usize,
}

impl TukeyHsdResult {
    /// Simultaneous confidence intervals `x̄_i − x̄_j ± q_{1−α}(k, ν) · se_ij`, as matrices of lower
    /// and upper limits.
    pub fn confidence_interval(&self, confidence_level: f64) -> Result<IntervalMatrices, ErrorsJSL> {
        if !(confidence_level > 0.0 && confidence_level < 1.0) {
            return Err(ErrorsJSL::InvalidInputRange("the confidence level must lie in (0, 1)"));
        }
        let k = self.ntreatments;
        let mut lo = vec![vec![0.0; k]; k];
        let mut hi = vec![vec![0.0; k]; k];
        let mut cache: Vec<(f64, f64)> = Vec::new();
        for i in 0..k {
            for j in 0..k {
                let df = self.df[i][j];
                let q = match cache.iter().find(|c| c.0 == df) {
                    Some(c) => c.1,
                    None => {
                        let q = StudentizedRange::new(k as f64, df)?.ppf(confidence_level);
                        cache.push((df, q));
                        q
                    }
                };
                let r = q * self.std_err[i][j];
                lo[i][j] = self.statistic[i][j] - r;
                hi[i][j] = self.statistic[i][j] + r;
            }
        }
        Ok((lo, hi))
    }
}

// ANCHOR: tukey_hsd
/// Tukey's honestly significant difference test of all pairwise differences of means (SciPy's
/// `tukey_hsd`): `|x̄_i − x̄_j| / se_ij` against the studentized range with `k` treatments, where
/// `se_ij = √(MSE (1/n_i + 1/n_j) / 2)` with the pooled variance and `N − k` degrees of freedom
/// (the Tukey-Kramer form), or, with `equal_var = false`, the Games-Howell form
/// `se_ij = √((s_i²/n_i + s_j²/n_j)/2)` with Welch's degrees of freedom.
pub fn tukey_hsd(groups: &[Vec<f64>], equal_var: bool) -> Result<TukeyHsdResult, ErrorsJSL> {
    let k = groups.len();
    if k < 2 || groups.iter().any(|g| g.len() < 2 || g.iter().any(|v| !v.is_finite())) {
        return Err(ErrorsJSL::InvalidInputRange("tukey_hsd: need at least two groups of at least two finite values"));
    }
    let n: Vec<f64> = groups.iter().map(|g| g.len() as f64).collect();
    let means: Vec<f64> = groups.iter().map(|g| g.iter().sum::<f64>() / g.len() as f64).collect();
    let vars: Vec<f64> = groups.iter().zip(&means).map(|(g, m)| g.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (g.len() - 1) as f64).collect();
    let nobs: f64 = n.iter().sum();
    let mut std_err = vec![vec![0.0; k]; k];
    let mut df = vec![vec![0.0; k]; k];
    if equal_var {
        let mse = vars.iter().zip(&n).map(|(v, ni)| v * (ni - 1.0)).sum::<f64>() / (nobs - k as f64);
        for i in 0..k {
            for j in 0..k {
                std_err[i][j] = ((1.0 / n[i] + 1.0 / n[j]) * mse / 2.0).sqrt();
                df[i][j] = nobs - k as f64;
            }
        }
    } else {
        for i in 0..k {
            for j in 0..k {
                let (a, b) = (vars[i] / n[i], vars[j] / n[j]);
                std_err[i][j] = (a + b).sqrt() / 2f64.sqrt();
                df[i][j] = (a + b).powi(2) / (a * a / (n[i] - 1.0) + b * b / (n[j] - 1.0));
            }
        }
    }
    let statistic: Vec<Vec<f64>> = (0..k).map(|i| (0..k).map(|j| means[i] - means[j]).collect()).collect();
    let mut p_values = vec![vec![1.0; k]; k];
    for i in 0..k {
        for j in i + 1..k {
            let t = statistic[i][j].abs() / std_err[i][j];
            let p = 1.0 - StudentizedRange::new(k as f64, df[i][j])?.cdf(t);
            p_values[i][j] = p;
            p_values[j][i] = p;
        }
    }
    Ok(TukeyHsdResult { statistic, p_values, std_err, df, ntreatments: k })
}
// ANCHOR_END: tukey_hsd

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn studentized_range_and_tukey() {
        // k = 2, ν = ∞: the range of two standard normals is |Z₁ − Z₂| ~ √2 |Z|.
        let d = StudentizedRange::new(2.0, f64::INFINITY).unwrap();
        for q in [0.5, 1.7, 3.0] {
            let want = 2.0 * normal_cdf(q / std::f64::consts::SQRT_2) - 1.0;
            assert!((d.cdf(q) - want).abs() < 1e-13, "{q}");
        }
        let d = StudentizedRange::new(4.0, 12.0).unwrap();
        let q = d.ppf(0.95);
        assert!((d.cdf(q) - 0.95).abs() < 1e-12);
        // Tables give q(0.95; 4, 12) = 4.199.
        assert!((q - 4.199).abs() < 1e-3, "{q}");
        let groups = vec![vec![1.0, 2.0, 1.5, 1.8], vec![1.1, 2.1, 1.6, 1.7], vec![5.0, 5.5, 4.8, 5.2]];
        let r = tukey_hsd(&groups, true).unwrap();
        assert!(r.p_values[0][1] > 0.5 && r.p_values[0][2] < 1e-5 && r.p_values[2][0] == r.p_values[0][2]);
        let (lo, hi) = r.confidence_interval(0.95).unwrap();
        assert!(lo[0][1] < 0.0 && hi[0][1] > 0.0 && hi[0][2] < 0.0);
    }
}
