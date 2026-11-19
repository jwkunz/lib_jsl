//! Multivariate distributions (SciPy's `multivariate_normal`, `multivariate_t`, `dirichlet`,
//! `wishart`, `invwishart` and `multinomial`): densities, moments, entropies and sampling.
//! Matrices are `Vec<Vec<f64>>` of rows.
//!
//! See the book entry [Multivariate distributions and further tests](https://jwkunz.github.io/lib_jsl/book/statistics/multivariate-tests.html).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_random::uniform_generator::UniformGenerator;
use lib_jsl_special_functions::gamma::{digamma, ln_gamma};
use lib_jsl_special_functions::polygamma::multigammaln;
use std::f64::consts::PI;

type Matrix = Vec<Vec<f64>>;

/// The lower Cholesky factor of a symmetric positive definite matrix.
fn cholesky(a: &Matrix) -> Result<Matrix, ErrorsJSL> {
    let n = a.len();
    if a.iter().any(|r| r.len() != n) {
        return Err(ErrorsJSL::InvalidInputRange("the matrix must be square"));
    }
    let mut l = vec![vec![0.0; n]; n];
    for j in 0..n {
        let d = a[j][j] - (0..j).map(|k| l[j][k] * l[j][k]).sum::<f64>();
        if !(d > 0.0) || !d.is_finite() {
            return Err(ErrorsJSL::InvalidInputRange("the matrix must be symmetric positive definite"));
        }
        l[j][j] = d.sqrt();
        for i in j + 1..n {
            if (a[i][j] - a[j][i]).abs() > 1e-10 * (a[i][j].abs() + a[j][i].abs()).max(1e-300) {
                return Err(ErrorsJSL::InvalidInputRange("the matrix must be symmetric"));
            }
            l[i][j] = (a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>()) / l[j][j];
        }
    }
    Ok(l)
}

fn log_det_from_cholesky(l: &Matrix) -> f64 {
    2.0 * l.iter().enumerate().map(|(i, r)| r[i].ln()).sum::<f64>()
}

/// `L⁻¹ b` by forward substitution.
fn forward(l: &Matrix, b: &[f64]) -> Vec<f64> {
    let mut y = b.to_vec();
    for i in 0..y.len() {
        y[i] = (y[i] - (0..i).map(|k| l[i][k] * y[k]).sum::<f64>()) / l[i][i];
    }
    y
}

/// `A⁻¹` for symmetric positive definite `A`, from its Cholesky factor.
fn inverse_from_cholesky(l: &Matrix) -> Matrix {
    let n = l.len();
    let mut inv = vec![vec![0.0; n]; n];
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let y = forward(l, &e);
        // Back substitution with Lᵀ.
        let mut x = y;
        for i in (0..n).rev() {
            x[i] = (x[i] - (i + 1..n).map(|k| l[k][i] * x[k]).sum::<f64>()) / l[i][i];
        }
        for i in 0..n {
            inv[i][j] = x[i];
        }
    }
    inv
}

/// A standard normal deviate by the Box-Muller transform.
pub(crate) fn standard_normal<U: UniformGenerator>(rng: &mut U) -> f64 {
    let mut u = rng.next_f64();
    while u <= 0.0 {
        u = rng.next_f64();
    }
    let v = rng.next_f64();
    (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
}

/// A Gamma(shape, 1) deviate (Marsaglia and Tsang, 2000).
pub(crate) fn standard_gamma<U: UniformGenerator>(shape: f64, rng: &mut U) -> f64 {
    if shape < 1.0 {
        let u = rng.next_f64().max(f64::MIN_POSITIVE);
        return standard_gamma(shape + 1.0, rng) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = standard_normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = rng.next_f64();
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

// ANCHOR: mvn
/// The multivariate normal distribution `N(μ, Σ)` (SciPy's `multivariate_normal`), for a
/// positive definite `Σ`: `log f(x) = −½ (d log 2π + log|Σ| + (x − μ)ᵀ Σ⁻¹ (x − μ))`, with the
/// quadratic form computed as `‖L⁻¹(x − μ)‖²` from the Cholesky factor `Σ = L Lᵀ`.
#[derive(Clone, Debug)]
pub struct MultivariateNormal {
    pub mean: Vec<f64>,
    pub cov: Matrix,
    chol: Matrix,
    log_det: f64,
}

impl MultivariateNormal {
    pub fn new(mean: Vec<f64>, cov: Matrix) -> Result<Self, ErrorsJSL> {
        if cov.len() != mean.len() || mean.is_empty() {
            return Err(ErrorsJSL::InvalidInputRange("the mean and covariance must have one dimension"));
        }
        let chol = cholesky(&cov)?;
        let log_det = log_det_from_cholesky(&chol);
        Ok(MultivariateNormal { mean, cov, chol, log_det })
    }
    pub fn dim(&self) -> usize {
        self.mean.len()
    }
    /// The squared Mahalanobis distance `(x − μ)ᵀ Σ⁻¹ (x − μ)`.
    pub fn mahalanobis_sq(&self, x: &[f64]) -> f64 {
        let d: Vec<f64> = x.iter().zip(&self.mean).map(|(a, b)| a - b).collect();
        forward(&self.chol, &d).iter().map(|v| v * v).sum()
    }
    pub fn logpdf(&self, x: &[f64]) -> f64 {
        let d = self.dim() as f64;
        -0.5 * (d * (2.0 * PI).ln() + self.log_det + self.mahalanobis_sq(x))
    }
    pub fn pdf(&self, x: &[f64]) -> f64 {
        self.logpdf(x).exp()
    }
    /// `½ log((2πe)^d |Σ|)`.
    pub fn entropy(&self) -> f64 {
        0.5 * (self.dim() as f64 * (2.0 * PI * std::f64::consts::E).ln() + self.log_det)
    }
    /// `μ + L z` for standard normal `z`.
    pub fn sample<U: UniformGenerator>(&self, rng: &mut U) -> Vec<f64> {
        let z: Vec<f64> = (0..self.dim()).map(|_| standard_normal(rng)).collect();
        (0..self.dim()).map(|i| self.mean[i] + (0..=i).map(|k| self.chol[i][k] * z[k]).sum::<f64>()).collect()
    }
}
// ANCHOR_END: mvn

/// The multivariate Student t distribution with location `μ`, shape `Σ` and `ν` degrees of
/// freedom (SciPy's `multivariate_t`): `X = μ + Z/√(W/ν)`, `Z ~ N(0, Σ)`, `W ~ χ²_ν`.
#[derive(Clone, Debug)]
pub struct MultivariateT {
    pub loc: Vec<f64>,
    pub shape: Matrix,
    pub df: f64,
    normal: MultivariateNormal,
}

impl MultivariateT {
    pub fn new(loc: Vec<f64>, shape: Matrix, df: f64) -> Result<Self, ErrorsJSL> {
        if !(df > 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("the degrees of freedom must be positive"));
        }
        let normal = MultivariateNormal::new(loc.clone(), shape.clone())?;
        Ok(MultivariateT { loc, shape, df, normal })
    }
    /// `log Γ((ν+d)/2) − log Γ(ν/2) − (d/2) log(νπ) − ½ log|Σ| − ((ν+d)/2) log(1 + δ²/ν)`.
    pub fn logpdf(&self, x: &[f64]) -> f64 {
        let (d, nu) = (self.loc.len() as f64, self.df);
        let m = self.normal.mahalanobis_sq(x);
        ln_gamma((nu + d) / 2.0) - ln_gamma(nu / 2.0) - 0.5 * d * (nu * PI).ln() - 0.5 * self.normal.log_det - 0.5 * (nu + d) * (m / nu).ln_1p()
    }
    pub fn pdf(&self, x: &[f64]) -> f64 {
        self.logpdf(x).exp()
    }
    /// The covariance `ν/(ν − 2) Σ` (infinite for `ν ≤ 2`).
    pub fn cov(&self) -> Matrix {
        let f = if self.df > 2.0 { self.df / (self.df - 2.0) } else { f64::INFINITY };
        self.shape.iter().map(|r| r.iter().map(|v| v * f).collect()).collect()
    }
    pub fn entropy(&self) -> f64 {
        let (d, nu) = (self.loc.len() as f64, self.df);
        -ln_gamma((nu + d) / 2.0) + ln_gamma(nu / 2.0) + 0.5 * d * (nu * PI).ln() + 0.5 * self.normal.log_det + 0.5 * (nu + d) * (digamma((nu + d) / 2.0) - digamma(nu / 2.0))
    }
    pub fn sample<U: UniformGenerator>(&self, rng: &mut U) -> Vec<f64> {
        let z = MultivariateNormal { mean: vec![0.0; self.loc.len()], ..self.normal.clone() }.sample(rng);
        let w = 2.0 * standard_gamma(self.df / 2.0, rng) / self.df;
        z.iter().zip(&self.loc).map(|(zi, m)| m + zi / w.sqrt()).collect()
    }
}

/// The Dirichlet distribution on the simplex (SciPy's `dirichlet(α)`):
/// `f(x) = Π x_i^{α_i − 1} / B(α)`, `B(α) = Π Γ(α_i) / Γ(α₀)`, `α₀ = Σ α_i`.
#[derive(Clone, Debug)]
pub struct Dirichlet {
    pub alpha: Vec<f64>,
}

impl Dirichlet {
    pub fn new(alpha: Vec<f64>) -> Result<Self, ErrorsJSL> {
        if alpha.len() < 2 || alpha.iter().any(|a| !(*a > 0.0) || !a.is_finite()) {
            return Err(ErrorsJSL::InvalidInputRange("α needs at least two positive entries"));
        }
        Ok(Dirichlet { alpha })
    }
    fn alpha0(&self) -> f64 {
        self.alpha.iter().sum()
    }
    fn ln_beta(&self) -> f64 {
        self.alpha.iter().map(|&a| ln_gamma(a)).sum::<f64>() - ln_gamma(self.alpha0())
    }
    /// The log density at a point of the simplex (components positive, summing to 1 within 10⁻¹⁰).
    pub fn logpdf(&self, x: &[f64]) -> Result<f64, ErrorsJSL> {
        if x.len() != self.alpha.len() || x.iter().any(|v| !(*v > 0.0)) || (x.iter().sum::<f64>() - 1.0).abs() > 1e-10 {
            return Err(ErrorsJSL::InvalidInputRange("x must lie in the open simplex"));
        }
        Ok(x.iter().zip(&self.alpha).map(|(xi, a)| (a - 1.0) * xi.ln()).sum::<f64>() - self.ln_beta())
    }
    pub fn pdf(&self, x: &[f64]) -> Result<f64, ErrorsJSL> {
        Ok(self.logpdf(x)?.exp())
    }
    pub fn mean(&self) -> Vec<f64> {
        let a0 = self.alpha0();
        self.alpha.iter().map(|a| a / a0).collect()
    }
    /// The marginal variances `α_i (α₀ − α_i) / (α₀² (α₀ + 1))`.
    pub fn var(&self) -> Vec<f64> {
        let a0 = self.alpha0();
        self.alpha.iter().map(|a| a * (a0 - a) / (a0 * a0 * (a0 + 1.0))).collect()
    }
    /// The covariance matrix, `−α_i α_j / (α₀² (α₀ + 1))` off the diagonal.
    pub fn cov(&self) -> Matrix {
        let a0 = self.alpha0();
        let v = self.var();
        (0..self.alpha.len()).map(|i| (0..self.alpha.len()).map(|j| if i == j { v[i] } else { -self.alpha[i] * self.alpha[j] / (a0 * a0 * (a0 + 1.0)) }).collect()).collect()
    }
    /// `log B(α) + (α₀ − K) ψ(α₀) − Σ (α_i − 1) ψ(α_i)`.
    pub fn entropy(&self) -> f64 {
        let a0 = self.alpha0();
        let k = self.alpha.len() as f64;
        self.ln_beta() + (a0 - k) * digamma(a0) - self.alpha.iter().map(|&a| (a - 1.0) * digamma(a)).sum::<f64>()
    }
    /// Normalised independent Gamma(α_i) deviates.
    pub fn sample<U: UniformGenerator>(&self, rng: &mut U) -> Vec<f64> {
        let g: Vec<f64> = self.alpha.iter().map(|&a| standard_gamma(a, rng)).collect();
        let s: f64 = g.iter().sum();
        g.iter().map(|v| v / s).collect()
    }
}

// ANCHOR: wishart
/// The Wishart distribution of `X = Σ_k z_k z_kᵀ`, `z_k ~ N(0, S)`, with `ν > p − 1` degrees of
/// freedom (SciPy's `wishart(df, scale)`): `log f(X) = ((ν − p − 1)/2) log|X| − tr(S⁻¹X)/2 −
/// (νp/2) log 2 − (ν/2) log|S| − log Γ_p(ν/2)`. Sampling uses Bartlett's decomposition
/// `X = L A Aᵀ Lᵀ`, `A` lower triangular with `A_ii = √χ²_{ν−i}` and standard normal entries
/// below the diagonal.
#[derive(Clone, Debug)]
pub struct Wishart {
    pub df: f64,
    pub scale: Matrix,
    chol: Matrix,
    log_det: f64,
}

impl Wishart {
    pub fn new(df: f64, scale: Matrix) -> Result<Self, ErrorsJSL> {
        let p = scale.len() as f64;
        if !(df > p - 1.0) {
            return Err(ErrorsJSL::InvalidInputRange("the degrees of freedom must exceed p − 1"));
        }
        let chol = cholesky(&scale)?;
        let log_det = log_det_from_cholesky(&chol);
        Ok(Wishart { df, scale, chol, log_det })
    }
    pub fn logpdf(&self, x: &Matrix) -> Result<f64, ErrorsJSL> {
        let p = self.scale.len();
        let lx = cholesky(x)?;
        let inv_s = inverse_from_cholesky(&self.chol);
        let trace: f64 = (0..p).map(|i| (0..p).map(|j| inv_s[i][j] * x[j][i]).sum::<f64>()).sum();
        let (nu, pf) = (self.df, p as f64);
        Ok(0.5 * (nu - pf - 1.0) * log_det_from_cholesky(&lx) - 0.5 * trace - 0.5 * nu * pf * 2f64.ln() - 0.5 * nu * self.log_det - multigammaln(nu / 2.0, p as u32))
    }
    pub fn pdf(&self, x: &Matrix) -> Result<f64, ErrorsJSL> {
        Ok(self.logpdf(x)?.exp())
    }
    /// `ν S`.
    pub fn mean(&self) -> Matrix {
        self.scale.iter().map(|r| r.iter().map(|v| v * self.df).collect()).collect()
    }
    /// The variances of the entries, `ν (S_ij² + S_ii S_jj)`.
    pub fn var(&self) -> Matrix {
        let s = &self.scale;
        (0..s.len()).map(|i| (0..s.len()).map(|j| self.df * (s[i][j] * s[i][j] + s[i][i] * s[j][j])).collect()).collect()
    }
    /// `(ν − p − 1) S`, for `ν ≥ p + 1`.
    pub fn mode(&self) -> Option<Matrix> {
        let p = self.scale.len() as f64;
        (self.df >= p + 1.0).then(|| self.scale.iter().map(|r| r.iter().map(|v| v * (self.df - p - 1.0)).collect()).collect())
    }
    /// `((p + 1)/2) log|S| + (p(p + 1)/2) log 2 + log Γ_p(ν/2) − ((ν − p − 1)/2) ψ_p(ν/2) + νp/2`.
    pub fn entropy(&self) -> f64 {
        let p = self.scale.len();
        let (nu, pf) = (self.df, p as f64);
        let psi_p: f64 = (0..p).map(|i| digamma((nu - i as f64) / 2.0)).sum();
        0.5 * (pf + 1.0) * self.log_det + 0.5 * pf * (pf + 1.0) * 2f64.ln() + multigammaln(nu / 2.0, p as u32) - 0.5 * (nu - pf - 1.0) * psi_p + 0.5 * nu * pf
    }
    pub fn sample<U: UniformGenerator>(&self, rng: &mut U) -> Matrix {
        let p = self.scale.len();
        let mut a = vec![vec![0.0; p]; p];
        for (i, row) in a.iter_mut().enumerate() {
            row[i] = (2.0 * standard_gamma((self.df - i as f64) / 2.0, rng)).sqrt();
            for v in row.iter_mut().take(i) {
                *v = standard_normal(rng);
            }
        }
        // B = L A, X = B Bᵀ.
        let b: Matrix = (0..p).map(|i| (0..p).map(|j| (0..p).map(|k| self.chol[i][k] * a[k][j]).sum()).collect()).collect();
        (0..p).map(|i| (0..p).map(|j| (0..p).map(|k| b[i][k] * b[j][k]).sum()).collect()).collect()
    }
}
// ANCHOR_END: wishart

/// The inverse Wishart distribution of `X⁻¹` for `X ~ Wishart(ν, Ψ⁻¹)` (SciPy's
/// `invwishart(df, scale = Ψ)`): `log f(X) = (ν/2) log|Ψ| − (νp/2) log 2 − log Γ_p(ν/2) −
/// ((ν + p + 1)/2) log|X| − tr(Ψ X⁻¹)/2`.
#[derive(Clone, Debug)]
pub struct InvWishart {
    pub df: f64,
    pub scale: Matrix,
    log_det: f64,
}

impl InvWishart {
    pub fn new(df: f64, scale: Matrix) -> Result<Self, ErrorsJSL> {
        let p = scale.len() as f64;
        if !(df > p - 1.0) {
            return Err(ErrorsJSL::InvalidInputRange("the degrees of freedom must exceed p − 1"));
        }
        let log_det = log_det_from_cholesky(&cholesky(&scale)?);
        Ok(InvWishart { df, scale, log_det })
    }
    pub fn logpdf(&self, x: &Matrix) -> Result<f64, ErrorsJSL> {
        let p = self.scale.len();
        let lx = cholesky(x)?;
        let inv_x = inverse_from_cholesky(&lx);
        let trace: f64 = (0..p).map(|i| (0..p).map(|j| self.scale[i][j] * inv_x[j][i]).sum::<f64>()).sum();
        let (nu, pf) = (self.df, p as f64);
        Ok(0.5 * nu * self.log_det - 0.5 * nu * pf * 2f64.ln() - multigammaln(nu / 2.0, p as u32) - 0.5 * (nu + pf + 1.0) * log_det_from_cholesky(&lx) - 0.5 * trace)
    }
    pub fn pdf(&self, x: &Matrix) -> Result<f64, ErrorsJSL> {
        Ok(self.logpdf(x)?.exp())
    }
    /// `Ψ / (ν − p − 1)`, for `ν > p + 1`.
    pub fn mean(&self) -> Option<Matrix> {
        let p = self.scale.len() as f64;
        (self.df > p + 1.0).then(|| self.scale.iter().map(|r| r.iter().map(|v| v / (self.df - p - 1.0)).collect()).collect())
    }
    /// `Ψ / (ν + p + 1)`.
    pub fn mode(&self) -> Matrix {
        let p = self.scale.len() as f64;
        self.scale.iter().map(|r| r.iter().map(|v| v / (self.df + p + 1.0)).collect()).collect()
    }
    pub fn sample<U: UniformGenerator>(&self, rng: &mut U) -> Result<Matrix, ErrorsJSL> {
        let inv_scale = inverse_from_cholesky(&cholesky(&self.scale)?);
        let w = Wishart::new(self.df, inv_scale)?.sample(rng);
        Ok(inverse_from_cholesky(&cholesky(&w)?))
    }
}

/// The multinomial distribution of the counts of `n` independent draws from `p.len()` categories
/// with probabilities `p` (SciPy's `multinomial(n, p)`).
#[derive(Clone, Debug)]
pub struct Multinomial {
    pub n: u64,
    pub p: Vec<f64>,
}

impl Multinomial {
    pub fn new(n: u64, p: Vec<f64>) -> Result<Self, ErrorsJSL> {
        if p.is_empty() || p.iter().any(|v| !(0.0..=1.0).contains(v)) || (p.iter().sum::<f64>() - 1.0).abs() > 1e-10 {
            return Err(ErrorsJSL::InvalidInputRange("p must be probabilities summing to 1"));
        }
        Ok(Multinomial { n, p })
    }
    /// `log n! − Σ log x_i! + Σ x_i log p_i`; −∞ unless the counts sum to `n`.
    pub fn logpmf(&self, x: &[u64]) -> f64 {
        if x.len() != self.p.len() || x.iter().sum::<u64>() != self.n {
            return f64::NEG_INFINITY;
        }
        let mut s = ln_gamma(self.n as f64 + 1.0);
        for (&xi, &pi) in x.iter().zip(&self.p) {
            s -= ln_gamma(xi as f64 + 1.0);
            if xi > 0 {
                if pi == 0.0 {
                    return f64::NEG_INFINITY;
                }
                s += xi as f64 * pi.ln();
            }
        }
        s
    }
    pub fn pmf(&self, x: &[u64]) -> f64 {
        self.logpmf(x).exp()
    }
    pub fn mean(&self) -> Vec<f64> {
        self.p.iter().map(|p| self.n as f64 * p).collect()
    }
    /// `n (diag(p) − p pᵀ)`.
    pub fn cov(&self) -> Matrix {
        let n = self.n as f64;
        (0..self.p.len()).map(|i| (0..self.p.len()).map(|j| n * (if i == j { self.p[i] } else { 0.0 } - self.p[i] * self.p[j])).collect()).collect()
    }
    /// One draw, category by category: `x_i ~ Binomial(n − Σ_{j<i} x_j, p_i / (1 − Σ_{j<i} p_j))`.
    pub fn sample<U: UniformGenerator>(&self, rng: &mut U) -> Vec<u64> {
        let mut left = self.n;
        let mut mass = 1.0;
        let mut out = Vec::with_capacity(self.p.len());
        for (i, &pi) in self.p.iter().enumerate() {
            if i + 1 == self.p.len() {
                out.push(left);
                break;
            }
            let q = if mass > 0.0 { (pi / mass).clamp(0.0, 1.0) } else { 0.0 };
            let mut k = 0;
            for _ in 0..left {
                if rng.next_f64() < q {
                    k += 1;
                }
            }
            out.push(k);
            left -= k;
            mass -= pi;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lib_jsl_random::uniform_generator::UniformRNG;

    #[test]
    fn densities_and_sampling() {
        // A diagonal covariance: the log density is a sum of univariate ones.
        let mvn = MultivariateNormal::new(vec![1.0, -2.0], vec![vec![4.0, 0.0], vec![0.0, 0.25]]).unwrap();
        let x = [2.0, -1.5];
        let uni = |z: f64, m: f64, s: f64| -0.5 * ((z - m) / s).powi(2) - (s * (2.0 * PI).sqrt()).ln();
        assert!((mvn.logpdf(&x) - (uni(2.0, 1.0, 2.0) + uni(-1.5, -2.0, 0.5))).abs() < 1e-14);
        assert!(MultivariateNormal::new(vec![0.0, 0.0], vec![vec![1.0, 2.0], vec![2.0, 1.0]]).is_err(), "not positive definite");
        let mut rng = UniformRNG::from_seed(5);
        let n = 4000;
        let mut m = [0.0; 2];
        for _ in 0..n {
            let s = mvn.sample(&mut rng);
            m[0] += s[0] / n as f64;
            m[1] += s[1] / n as f64;
        }
        assert!((m[0] - 1.0).abs() < 0.1 && (m[1] + 2.0).abs() < 0.03, "{m:?}");
        // As ν → ∞ the multivariate t tends to the normal.
        let t = MultivariateT::new(vec![1.0, -2.0], vec![vec![4.0, 0.0], vec![0.0, 0.25]], 1e9).unwrap();
        assert!((t.logpdf(&x) - mvn.logpdf(&x)).abs() < 1e-6);
        let d = Dirichlet::new(vec![2.0, 3.0, 5.0]).unwrap();
        let s = d.sample(&mut rng);
        assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-14 && s.iter().all(|v| *v > 0.0));
        // Dirichlet(1, 1) is uniform on the simplex: density 1.
        assert!((Dirichlet::new(vec![1.0, 1.0]).unwrap().pdf(&[0.3, 0.7]).unwrap() - 1.0).abs() < 1e-14);
        // Wishart samples average to ν S.
        let w = Wishart::new(6.0, vec![vec![1.0, 0.3], vec![0.3, 2.0]]).unwrap();
        let mut mean = [[0.0; 2]; 2];
        for _ in 0..2000 {
            let x = w.sample(&mut rng);
            for i in 0..2 {
                for j in 0..2 {
                    mean[i][j] += x[i][j] / 2000.0;
                }
            }
        }
        let want = w.mean();
        assert!((0..2).all(|i| (0..2).all(|j| (mean[i][j] - want[i][j]).abs() < 0.08 * want[i][i].max(want[j][j]))), "{mean:?}");
        // The multinomial probabilities of all outcomes sum to 1.
        let m3 = Multinomial::new(4, vec![0.2, 0.5, 0.3]).unwrap();
        let total: f64 = (0..=4u64).flat_map(|a| (0..=4 - a).map(move |b| [a, b, 4 - a - b])).map(|x| m3.pmf(&x)).sum();
        assert!((total - 1.0).abs() < 1e-14);
        assert_eq!(m3.sample(&mut rng).iter().sum::<u64>(), 4);
    }
}
