//! Probability distributions as objects, following `scipy.stats`: each continuous distribution
//! provides its density, distribution and survival functions, quantiles, moments, sampling and
//! maximum-likelihood fitting through [`ContinuousDistribution`] and [`Fit`]; the discrete ones
//! through [`DiscreteDistribution`].
//!
//! Quantiles have closed forms where they exist; otherwise the distribution function is inverted
//! by Brent's method. Fitting uses the closed-form estimators or score equations where they exist
//! (normal, exponential, log-normal, gamma, beta, Weibull, Gumbel, Laplace, Pareto, Rayleigh,
//! von Mises, Poisson, negative binomial…) and Nelder-Mead on the log-likelihood otherwise.
//!
//! See the book entry [Probability distributions](https://jwkunz.github.io/lib_jsl/book/statistics/distributions.html).

// Guards such as `!(x > 0.0)` are deliberate: unlike `x <= 0.0` they also reject NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use lib_jsl_core::ErrorsJSL;
use lib_jsl_random::uniform_generator::UniformGenerator;
use lib_jsl_special_functions::bessel_functions_fractional::bessel_i_fractional;
use lib_jsl_special_functions::beta::ln_beta;
use lib_jsl_special_functions::error_function::erfc;
use lib_jsl_special_functions::gamma::{digamma, ln_gamma};
use lib_jsl_special_functions::incomplete_beta::incomplete_beta;
use lib_jsl_special_functions::incomplete_gamma::{incomplete_gamma, incomplete_gamma_upper};
use std::f64::consts::{PI, SQRT_2};

// ----------------------------------------------------------------------------------- helpers

fn std_normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / SQRT_2)
}

fn std_normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
}

/// The trigamma function ψ'(x), by recurrence to `x ≥ 6` and the asymptotic series.
pub fn trigamma(mut x: f64) -> f64 {
    let mut acc = 0.0;
    while x < 6.0 {
        acc += 1.0 / (x * x);
        x += 1.0;
    }
    let x2 = 1.0 / (x * x);
    acc + 1.0 / x + x2 / 2.0 + (1.0 / x) * x2 * (1.0 / 6.0 - x2 * (1.0 / 30.0 - x2 * (1.0 / 42.0 - x2 / 30.0)))
}

/// Brent's root finder on a bracket `[a, b]` with `f(a) f(b) ≤ 0`.
fn brent(f: impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
    let (mut fa, mut fb) = (f(a), f(b));
    if fa == 0.0 {
        return a;
    }
    if fb == 0.0 {
        return b;
    }
    let (mut c, mut fc) = (a, fa);
    let (mut d, mut e) = (b - a, b - a);
    for _ in 0..200 {
        if (fb > 0.0) == (fc > 0.0) {
            (c, fc) = (a, fa);
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            (a, b, c) = (b, c, b);
            (fa, fb, fc) = (fb, fc, fb);
        }
        let tol = 2.0 * f64::EPSILON * b.abs() + 1e-300;
        let m = 0.5 * (c - b);
        if m.abs() <= tol || fb == 0.0 {
            return b;
        }
        if e.abs() >= tol && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (mut p, mut q);
            if a == c {
                p = 2.0 * m * s;
                q = 1.0 - s;
            } else {
                let qq = fa / fc;
                let r = fb / fc;
                p = s * (2.0 * m * qq * (qq - r) - (b - a) * (r - 1.0));
                q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            if 2.0 * p < (3.0 * m * q - (tol * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = m;
            }
        } else {
            d = m;
            e = m;
        }
        (a, fa) = (b, fb);
        b += if d.abs() > tol { d } else { tol.copysign(m) };
        fb = f(b);
    }
    b
}

/// Inverts a continuous distribution function by bracketing and Brent's method.
fn invert_cdf(cdf: impl Fn(f64) -> f64, p: f64, support: (f64, f64), guess: f64, scale: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return support.0;
    }
    if p == 1.0 {
        return support.1;
    }
    let scale = if scale > 0.0 && scale.is_finite() { scale } else { 1.0 };
    let (mut lo, mut hi) = (support.0, support.1);
    if !lo.is_finite() {
        let mut step = scale;
        lo = guess.min(hi) - step;
        while cdf(lo) > p {
            step *= 2.0;
            lo = guess - step;
        }
    }
    if !hi.is_finite() {
        let mut step = scale;
        hi = guess.max(lo) + step;
        while cdf(hi) < p {
            step *= 2.0;
            hi = guess.max(lo) + step;
        }
    }
    brent(|x| cdf(x) - p, lo, hi)
}

/// Nelder-Mead minimisation (adaptive coefficients, tight tolerances), for likelihoods without
/// closed-form maximisers.
fn nelder_mead(f: impl Fn(&[f64]) -> f64, x0: &[f64]) -> Vec<f64> {
    let n = x0.len();
    let nf = n as f64;
    let (alpha, gamma, rho, sigma) = (1.0, 1.0 + 2.0 / nf, 0.75 - 1.0 / (2.0 * nf), 1.0 - 1.0 / nf);
    let mut simplex: Vec<Vec<f64>> = vec![x0.to_vec()];
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] += if v[i] != 0.0 { 0.05 * v[i] } else { 0.00025 };
        simplex.push(v);
    }
    let mut values: Vec<f64> = simplex.iter().map(|v| f(v)).collect();
    for _ in 0..20000 {
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        simplex = idx.iter().map(|&i| simplex[i].clone()).collect();
        values = idx.iter().map(|&i| values[i]).collect();
        let spread = values[n] - values[0];
        let size = simplex.iter().skip(1).map(|v| v.iter().zip(&simplex[0]).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
        if spread.abs() <= 1e-14 * values[0].abs().max(1e-300) && size <= 1e-11 * simplex[0].iter().map(|v| v.abs()).fold(1.0, f64::max) {
            break;
        }
        let centroid: Vec<f64> = (0..n).map(|j| simplex[..n].iter().map(|v| v[j]).sum::<f64>() / nf).collect();
        let point = |t: f64| -> Vec<f64> { (0..n).map(|j| centroid[j] + t * (simplex[n][j] - centroid[j])).collect() };
        let xr = point(-alpha);
        let fr = f(&xr);
        if fr < values[0] {
            let xe = point(-alpha * gamma);
            let fe = f(&xe);
            if fe < fr {
                (simplex[n], values[n]) = (xe, fe);
            } else {
                (simplex[n], values[n]) = (xr, fr);
            }
        } else if fr < values[n - 1] {
            (simplex[n], values[n]) = (xr, fr);
        } else {
            let (xc, fc) = if fr < values[n] {
                let xc = point(-alpha * rho);
                let fc = f(&xc);
                (xc, fc)
            } else {
                let xc = point(rho);
                let fc = f(&xc);
                (xc, fc)
            };
            if fc < values[n].min(fr) {
                (simplex[n], values[n]) = (xc, fc);
            } else {
                for i in 1..=n {
                    simplex[i] = (0..n).map(|j| simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j])).collect();
                    values[i] = f(&simplex[i]);
                }
            }
        }
    }
    let best = (0..=n).min_by(|&a, &b| values[a].total_cmp(&values[b])).expect("non-empty");
    simplex[best].clone()
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn check_data(x: &[f64], min_len: usize) -> Result<(), ErrorsJSL> {
    if x.len() < min_len || x.iter().any(|v| !v.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("fitting needs enough finite data"));
    }
    Ok(())
}

fn positive(v: f64, what: &'static str) -> Result<f64, ErrorsJSL> {
    if v > 0.0 && v.is_finite() { Ok(v) } else { Err(ErrorsJSL::InvalidInputRange(what)) }
}

// ------------------------------------------------------------------------------------ traits

// ANCHOR: traits
/// A continuous probability distribution on the real line.
pub trait ContinuousDistribution {
    /// The probability density.
    fn pdf(&self, x: f64) -> f64;
    /// The log density (more accurate than `pdf(x).ln()` in the tails where overridden).
    fn logpdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }
    /// The distribution function `P(X ≤ x)`.
    fn cdf(&self, x: f64) -> f64;
    /// The survival function `P(X > x)`.
    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }
    /// The quantile (percent-point) function, the inverse of the distribution function.
    fn ppf(&self, p: f64) -> f64 {
        let (lo, hi) = self.support();
        let m = self.mean();
        let guess = if m.is_finite() { m } else if lo.is_finite() { lo + 1.0 } else if hi.is_finite() { hi - 1.0 } else { 0.0 };
        let s = self.std();
        invert_cdf(|x| self.cdf(x), p, (lo, hi), guess, if s.is_finite() { s } else { 1.0 })
    }
    /// The inverse survival function.
    fn isf(&self, q: f64) -> f64 {
        self.ppf(1.0 - q)
    }
    /// The interval `[a, b]` outside which the density is zero.
    fn support(&self) -> (f64, f64);
    fn mean(&self) -> f64;
    fn variance(&self) -> f64;
    fn std(&self) -> f64 {
        self.variance().sqrt()
    }
    fn skewness(&self) -> f64;
    /// The excess kurtosis (zero for the normal distribution), as SciPy's `stats(moments='k')`.
    fn excess_kurtosis(&self) -> f64;
    fn median(&self) -> f64 {
        self.ppf(0.5)
    }
    /// The central interval containing a fraction `confidence` of the probability.
    fn interval(&self, confidence: f64) -> (f64, f64) {
        let a = 0.5 * (1.0 - confidence);
        (self.ppf(a), self.isf(a))
    }
    /// `Σ log f(xᵢ)`.
    fn log_likelihood(&self, data: &[f64]) -> f64 {
        data.iter().map(|&x| self.logpdf(x)).sum()
    }
    /// A sample by inversion, `ppf(U)`.
    fn sample<U: UniformGenerator>(&self, rng: &mut U) -> f64
    where
        Self: Sized,
    {
        let mut u = rng.next_f64();
        while u <= 0.0 || u >= 1.0 {
            u = rng.next_f64();
        }
        self.ppf(u)
    }
}

/// Maximum-likelihood estimation of a distribution's parameters (SciPy's `fit`, with the
/// location fixed where the distribution has no location parameter here).
pub trait Fit: Sized {
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL>;
}

/// A discrete distribution on the integers.
pub trait DiscreteDistribution {
    fn pmf(&self, k: i64) -> f64;
    fn logpmf(&self, k: i64) -> f64 {
        self.pmf(k).ln()
    }
    fn cdf(&self, k: i64) -> f64;
    fn sf(&self, k: i64) -> f64 {
        1.0 - self.cdf(k)
    }
    /// The smallest `k` with `cdf(k) ≥ p`.
    fn ppf(&self, p: f64) -> i64 {
        let (lo, hi) = self.support();
        if p <= 0.0 {
            return lo - 1;
        }
        // Find an upper bracket, then bisect.
        let mut b = lo.max((self.mean() + 3.0 * self.variance().sqrt()).ceil() as i64);
        let mut step = 1i64.max(self.variance().sqrt().ceil() as i64);
        while b < hi && self.cdf(b) < p * (1.0 - 1e-15) {
            b = b.saturating_add(step).min(hi);
            step = step.saturating_mul(2);
        }
        let mut a = lo - 1;
        while b - a > 1 {
            let mid = a + (b - a) / 2;
            if self.cdf(mid) >= p * (1.0 - 1e-15) { b = mid } else { a = mid }
        }
        b
    }
    fn support(&self) -> (i64, i64);
    fn mean(&self) -> f64;
    fn variance(&self) -> f64;
    fn skewness(&self) -> f64;
    fn excess_kurtosis(&self) -> f64;
    fn log_likelihood(&self, data: &[i64]) -> f64 {
        data.iter().map(|&k| self.logpmf(k)).sum()
    }
    fn sample<U: UniformGenerator>(&self, rng: &mut U) -> i64
    where
        Self: Sized,
    {
        self.ppf(rng.next_f64())
    }
}
// ANCHOR_END: traits

// -------------------------------------------------------------------------- continuous: normal

/// The normal distribution N(μ, σ²) (SciPy's `norm(loc=μ, scale=σ)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normal {
    pub mu: f64,
    pub sigma: f64,
}

impl Normal {
    pub fn new(mu: f64, sigma: f64) -> Result<Self, ErrorsJSL> {
        Ok(Normal { mu, sigma: positive(sigma, "σ must be positive")? })
    }
}

impl ContinuousDistribution for Normal {
    fn pdf(&self, x: f64) -> f64 {
        std_normal_pdf((x - self.mu) / self.sigma) / self.sigma
    }
    fn logpdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        -0.5 * z * z - self.sigma.ln() - 0.5 * (2.0 * PI).ln()
    }
    fn cdf(&self, x: f64) -> f64 {
        std_normal_cdf((x - self.mu) / self.sigma)
    }
    fn sf(&self, x: f64) -> f64 {
        std_normal_cdf(-(x - self.mu) / self.sigma)
    }
    fn ppf(&self, p: f64) -> f64 {
        lib_jsl_special_functions::statistical_functions::normal_quantile(p, self.mu, self.sigma)
    }
    fn isf(&self, q: f64) -> f64 {
        2.0 * self.mu - self.ppf(q)
    }
    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
    fn mean(&self) -> f64 {
        self.mu
    }
    fn variance(&self) -> f64 {
        self.sigma * self.sigma
    }
    fn skewness(&self) -> f64 {
        0.0
    }
    fn excess_kurtosis(&self) -> f64 {
        0.0
    }
}

impl Fit for Normal {
    /// The sample mean and the standard deviation with divisor n.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 2)?;
        let m = mean(data);
        let s = (data.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / data.len() as f64).sqrt();
        Normal::new(m, s)
    }
}

/// The continuous uniform distribution on `[a, b]` (SciPy's `uniform(loc=a, scale=b−a)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniform {
    pub a: f64,
    pub b: f64,
}

impl Uniform {
    pub fn new(a: f64, b: f64) -> Result<Self, ErrorsJSL> {
        if !(a < b) {
            return Err(ErrorsJSL::InvalidInputRange("need a < b"));
        }
        Ok(Uniform { a, b })
    }
}

impl ContinuousDistribution for Uniform {
    fn pdf(&self, x: f64) -> f64 {
        if x >= self.a && x <= self.b { 1.0 / (self.b - self.a) } else { 0.0 }
    }
    fn cdf(&self, x: f64) -> f64 {
        ((x - self.a) / (self.b - self.a)).clamp(0.0, 1.0)
    }
    fn ppf(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) { f64::NAN } else { self.a + p * (self.b - self.a) }
    }
    fn support(&self) -> (f64, f64) {
        (self.a, self.b)
    }
    fn mean(&self) -> f64 {
        0.5 * (self.a + self.b)
    }
    fn variance(&self) -> f64 {
        (self.b - self.a).powi(2) / 12.0
    }
    fn skewness(&self) -> f64 {
        0.0
    }
    fn excess_kurtosis(&self) -> f64 {
        -1.2
    }
}

impl Fit for Uniform {
    /// The sample minimum and maximum.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 2)?;
        Uniform::new(data.iter().copied().fold(f64::INFINITY, f64::min), data.iter().copied().fold(f64::NEG_INFINITY, f64::max))
    }
}

/// The exponential distribution with rate λ (SciPy's `expon(scale=1/λ)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exponential {
    pub lambda: f64,
}

impl Exponential {
    pub fn new(lambda: f64) -> Result<Self, ErrorsJSL> {
        Ok(Exponential { lambda: positive(lambda, "λ must be positive")? })
    }
}

impl ContinuousDistribution for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 { 0.0 } else { self.lambda * (-self.lambda * x).exp() }
    }
    fn logpdf(&self, x: f64) -> f64 {
        if x < 0.0 { f64::NEG_INFINITY } else { self.lambda.ln() - self.lambda * x }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 { 0.0 } else { -(-self.lambda * x).exp_m1() }
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 { 1.0 } else { (-self.lambda * x).exp() }
    }
    fn ppf(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) { f64::NAN } else { -(-p).ln_1p() / self.lambda }
    }
    fn isf(&self, q: f64) -> f64 {
        -q.ln() / self.lambda
    }
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
    fn mean(&self) -> f64 {
        1.0 / self.lambda
    }
    fn variance(&self) -> f64 {
        1.0 / (self.lambda * self.lambda)
    }
    fn skewness(&self) -> f64 {
        2.0
    }
    fn excess_kurtosis(&self) -> f64 {
        6.0
    }
}

impl Fit for Exponential {
    /// `λ = 1 / x̄`.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 1)?;
        Exponential::new(1.0 / mean(data))
    }
}

/// The gamma distribution with shape `k` and scale `θ` (SciPy's `gamma(a=k, scale=θ)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gamma {
    pub shape: f64,
    pub scale: f64,
}

impl Gamma {
    pub fn new(shape: f64, scale: f64) -> Result<Self, ErrorsJSL> {
        Ok(Gamma { shape: positive(shape, "the shape must be positive")?, scale: positive(scale, "the scale must be positive")? })
    }
}

impl ContinuousDistribution for Gamma {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 { 0.0 } else { self.logpdf(x).exp() }
    }
    fn logpdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return f64::NEG_INFINITY;
        }
        let z = x / self.scale;
        (self.shape - 1.0) * z.ln() - z - ln_gamma(self.shape) - self.scale.ln()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 { 0.0 } else { incomplete_gamma(self.shape, x / self.scale) }
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 { 1.0 } else { incomplete_gamma_upper(self.shape, x / self.scale) }
    }
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
    fn mean(&self) -> f64 {
        self.shape * self.scale
    }
    fn variance(&self) -> f64 {
        self.shape * self.scale * self.scale
    }
    fn skewness(&self) -> f64 {
        2.0 / self.shape.sqrt()
    }
    fn excess_kurtosis(&self) -> f64 {
        6.0 / self.shape
    }
}

// ANCHOR: gamma_fit
impl Fit for Gamma {
    /// The shape solves `ln k − ψ(k) = ln x̄ − mean(ln x)` (Newton's method from Minka's
    /// starting value), and `θ = x̄ / k`.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 2)?;
        if data.iter().any(|&v| v <= 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("gamma data must be positive"));
        }
        let m = mean(data);
        let s = m.ln() - data.iter().map(|v| v.ln()).sum::<f64>() / data.len() as f64;
        let mut k = (3.0 - s + ((s - 3.0).powi(2) + 24.0 * s).sqrt()) / (12.0 * s);
        for _ in 0..50 {
            let f = k.ln() - digamma(k) - s;
            let step = f / (1.0 / k - trigamma(k));
            let next = (k - step).max(k / 10.0);
            if (next - k).abs() <= 1e-15 * k {
                k = next;
                break;
            }
            k = next;
        }
        Gamma::new(k, m / k)
    }
}
// ANCHOR_END: gamma_fit

/// The beta distribution on `[0, 1]` (SciPy's `beta(a, b)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beta {
    pub a: f64,
    pub b: f64,
}

impl Beta {
    pub fn new(a: f64, b: f64) -> Result<Self, ErrorsJSL> {
        Ok(Beta { a: positive(a, "a must be positive")?, b: positive(b, "b must be positive")? })
    }
}

impl ContinuousDistribution for Beta {
    fn pdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) { 0.0 } else { self.logpdf(x).exp() }
    }
    fn logpdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            return f64::NEG_INFINITY;
        }
        (self.a - 1.0) * x.ln() + (self.b - 1.0) * (-x).ln_1p() - ln_beta(self.a, self.b)
    }
    fn cdf(&self, x: f64) -> f64 {
        incomplete_beta(self.a, self.b, x.clamp(0.0, 1.0))
    }
    fn sf(&self, x: f64) -> f64 {
        incomplete_beta(self.b, self.a, 1.0 - x.clamp(0.0, 1.0))
    }
    fn support(&self) -> (f64, f64) {
        (0.0, 1.0)
    }
    fn mean(&self) -> f64 {
        self.a / (self.a + self.b)
    }
    fn variance(&self) -> f64 {
        let s = self.a + self.b;
        self.a * self.b / (s * s * (s + 1.0))
    }
    fn skewness(&self) -> f64 {
        let (a, b) = (self.a, self.b);
        2.0 * (b - a) * (a + b + 1.0).sqrt() / ((a + b + 2.0) * (a * b).sqrt())
    }
    fn excess_kurtosis(&self) -> f64 {
        let (a, b) = (self.a, self.b);
        6.0 * ((a - b).powi(2) * (a + b + 1.0) - a * b * (a + b + 2.0)) / (a * b * (a + b + 2.0) * (a + b + 3.0))
    }
}

impl Fit for Beta {
    /// Newton's method on the score equations `ψ(a) − ψ(a+b) = mean(ln x)`,
    /// `ψ(b) − ψ(a+b) = mean(ln(1−x))`, from the method-of-moments estimates.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 2)?;
        if data.iter().any(|&v| !(v > 0.0 && v < 1.0)) {
            return Err(ErrorsJSL::InvalidInputRange("beta data must lie in (0, 1)"));
        }
        let n = data.len() as f64;
        let (g1, g2) = (data.iter().map(|v| v.ln()).sum::<f64>() / n, data.iter().map(|v| (-v).ln_1p()).sum::<f64>() / n);
        let m = mean(data);
        let v = data.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
        let common = (m * (1.0 - m) / v - 1.0).max(0.1);
        let (mut a, mut b) = (m * common, (1.0 - m) * common);
        for _ in 0..100 {
            let (f1, f2) = (digamma(a) - digamma(a + b) - g1, digamma(b) - digamma(a + b) - g2);
            let t = trigamma(a + b);
            let (j11, j12, j22) = (trigamma(a) - t, -t, trigamma(b) - t);
            let det = j11 * j22 - j12 * j12;
            let (da, db) = ((j22 * f1 - j12 * f2) / det, (j11 * f2 - j12 * f1) / det);
            let (na, nb) = ((a - da).max(a / 10.0), (b - db).max(b / 10.0));
            let done = (na - a).abs() <= 1e-15 * a && (nb - b).abs() <= 1e-15 * b;
            (a, b) = (na, nb);
            if done {
                break;
            }
        }
        Beta::new(a, b)
    }
}

/// The chi-squared distribution with `k` degrees of freedom (SciPy's `chi2(df=k)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChiSquared {
    pub k: f64,
}

impl ChiSquared {
    pub fn new(k: f64) -> Result<Self, ErrorsJSL> {
        Ok(ChiSquared { k: positive(k, "the degrees of freedom must be positive")? })
    }
    fn gamma(&self) -> Gamma {
        Gamma { shape: self.k / 2.0, scale: 2.0 }
    }
}

impl ContinuousDistribution for ChiSquared {
    fn pdf(&self, x: f64) -> f64 {
        self.gamma().pdf(x)
    }
    fn logpdf(&self, x: f64) -> f64 {
        self.gamma().logpdf(x)
    }
    fn cdf(&self, x: f64) -> f64 {
        self.gamma().cdf(x)
    }
    fn sf(&self, x: f64) -> f64 {
        self.gamma().sf(x)
    }
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
    fn mean(&self) -> f64 {
        self.k
    }
    fn variance(&self) -> f64 {
        2.0 * self.k
    }
    fn skewness(&self) -> f64 {
        (8.0 / self.k).sqrt()
    }
    fn excess_kurtosis(&self) -> f64 {
        12.0 / self.k
    }
}

impl Fit for ChiSquared {
    /// The degrees of freedom solve `ψ(k/2) = mean(ln x) − ln 2`.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 1)?;
        let target = data.iter().map(|v| v.ln()).sum::<f64>() / data.len() as f64 - 2f64.ln();
        let k = 2.0 * brent(|h| digamma(h) - target, 1e-8, 1e8);
        ChiSquared::new(k)
    }
}

/// Student's t distribution with `ν` degrees of freedom, location and scale (SciPy's
/// `t(df=ν, loc, scale)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StudentT {
    pub nu: f64,
    pub loc: f64,
    pub scale: f64,
}

impl StudentT {
    pub fn new(nu: f64, loc: f64, scale: f64) -> Result<Self, ErrorsJSL> {
        Ok(StudentT { nu: positive(nu, "ν must be positive")?, loc, scale: positive(scale, "the scale must be positive")? })
    }
}

impl ContinuousDistribution for StudentT {
    fn pdf(&self, x: f64) -> f64 {
        self.logpdf(x).exp()
    }
    fn logpdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        let nu = self.nu;
        ln_gamma((nu + 1.0) / 2.0) - ln_gamma(nu / 2.0) - 0.5 * (nu * PI).ln() - self.scale.ln() - (nu + 1.0) / 2.0 * (z * z / nu).ln_1p()
    }
    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        let tail = 0.5 * incomplete_beta(self.nu / 2.0, 0.5, self.nu / (self.nu + z * z));
        if z < 0.0 { tail } else { 1.0 - tail }
    }
    fn sf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        let tail = 0.5 * incomplete_beta(self.nu / 2.0, 0.5, self.nu / (self.nu + z * z));
        if z > 0.0 { tail } else { 1.0 - tail }
    }
    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
    fn mean(&self) -> f64 {
        if self.nu > 1.0 { self.loc } else { f64::NAN }
    }
    fn variance(&self) -> f64 {
        if self.nu > 2.0 {
            self.scale * self.scale * self.nu / (self.nu - 2.0)
        } else if self.nu > 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    }
    fn skewness(&self) -> f64 {
        if self.nu > 3.0 { 0.0 } else { f64::NAN }
    }
    fn excess_kurtosis(&self) -> f64 {
        if self.nu > 4.0 {
            6.0 / (self.nu - 4.0)
        } else if self.nu > 2.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    }
    fn ppf(&self, p: f64) -> f64 {
        let s = if self.nu > 2.0 { self.std() } else { self.scale };
        invert_cdf(|x| self.cdf(x), p, self.support(), self.loc, s)
    }
}

impl Fit for StudentT {
    /// Nelder-Mead on `(ln ν, μ, ln σ)` from the median and interquartile range.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 3)?;
        let mut sorted = data.to_vec();
        sorted.sort_by(f64::total_cmp);
        let med = sorted[sorted.len() / 2];
        let iqr = (sorted[3 * sorted.len() / 4] - sorted[sorted.len() / 4]).max(1e-12);
        let nll = |p: &[f64]| -> f64 {
            let d = StudentT { nu: p[0].exp(), loc: p[1], scale: p[2].exp() };
            -d.log_likelihood(data)
        };
        let p = nelder_mead(nll, &[5f64.ln(), med, (iqr / 1.35).ln()]);
        StudentT::new(p[0].exp(), p[1], p[2].exp())
    }
}

/// The F distribution with `d₁` and `d₂` degrees of freedom (SciPy's `f(dfn, dfd)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FDist {
    pub d1: f64,
    pub d2: f64,
}

impl FDist {
    pub fn new(d1: f64, d2: f64) -> Result<Self, ErrorsJSL> {
        Ok(FDist { d1: positive(d1, "d₁ must be positive")?, d2: positive(d2, "d₂ must be positive")? })
    }
}

impl ContinuousDistribution for FDist {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 { 0.0 } else { self.logpdf(x).exp() }
    }
    fn logpdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return f64::NEG_INFINITY;
        }
        let (d1, d2) = (self.d1, self.d2);
        0.5 * (d1 * (d1 * x).ln() + d2 * d2.ln() - (d1 + d2) * (d1 * x + d2).ln()) - x.ln() - ln_beta(d1 / 2.0, d2 / 2.0)
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 { 0.0 } else { incomplete_beta(self.d1 / 2.0, self.d2 / 2.0, self.d1 * x / (self.d1 * x + self.d2)) }
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 { 1.0 } else { incomplete_beta(self.d2 / 2.0, self.d1 / 2.0, self.d2 / (self.d1 * x + self.d2)) }
    }
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
    fn mean(&self) -> f64 {
        if self.d2 > 2.0 { self.d2 / (self.d2 - 2.0) } else { f64::INFINITY }
    }
    fn variance(&self) -> f64 {
        let (d1, d2) = (self.d1, self.d2);
        if d2 > 4.0 { 2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0).powi(2) * (d2 - 4.0)) } else if d2 > 2.0 { f64::INFINITY } else { f64::NAN }
    }
    fn skewness(&self) -> f64 {
        let (d1, d2) = (self.d1, self.d2);
        if d2 > 6.0 { (2.0 * d1 + d2 - 2.0) * (8.0 * (d2 - 4.0)).sqrt() / ((d2 - 6.0) * (d1 * (d1 + d2 - 2.0)).sqrt()) } else { f64::NAN }
    }
    fn excess_kurtosis(&self) -> f64 {
        let (d1, d2) = (self.d1, self.d2);
        if d2 > 8.0 { 12.0 * (d1 * (5.0 * d2 - 22.0) * (d1 + d2 - 2.0) + (d2 - 4.0) * (d2 - 2.0).powi(2)) / (d1 * (d2 - 6.0) * (d2 - 8.0) * (d1 + d2 - 2.0)) } else { f64::NAN }
    }
    fn ppf(&self, p: f64) -> f64 {
        let guess = if self.d2 > 2.0 { self.mean() } else { 1.0 };
        invert_cdf(|x| self.cdf(x), p, self.support(), guess, 1.0)
    }
}

impl Fit for FDist {
    /// Nelder-Mead on `(ln d₁, ln d₂)`.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 2)?;
        let m = mean(data);
        let d2 = if m > 1.0 { (2.0 * m / (m - 1.0)).max(2.5) } else { 10.0 };
        let nll = |p: &[f64]| -(FDist { d1: p[0].exp(), d2: p[1].exp() }).log_likelihood(data);
        let p = nelder_mead(nll, &[5f64.ln(), d2.ln()]);
        FDist::new(p[0].exp(), p[1].exp())
    }
}

/// The log-normal distribution: `ln X ~ N(μ, σ²)` (SciPy's `lognorm(s=σ, scale=e^μ)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogNormal {
    pub mu: f64,
    pub sigma: f64,
}

impl LogNormal {
    pub fn new(mu: f64, sigma: f64) -> Result<Self, ErrorsJSL> {
        Ok(LogNormal { mu, sigma: positive(sigma, "σ must be positive")? })
    }
}

impl ContinuousDistribution for LogNormal {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 { 0.0 } else { self.logpdf(x).exp() }
    }
    fn logpdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let z = (x.ln() - self.mu) / self.sigma;
        -0.5 * z * z - (x * self.sigma).ln() - 0.5 * (2.0 * PI).ln()
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 { 0.0 } else { std_normal_cdf((x.ln() - self.mu) / self.sigma) }
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 { 1.0 } else { std_normal_cdf(-(x.ln() - self.mu) / self.sigma) }
    }
    fn ppf(&self, p: f64) -> f64 {
        Normal { mu: self.mu, sigma: self.sigma }.ppf(p).exp()
    }
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
    fn mean(&self) -> f64 {
        (self.mu + 0.5 * self.sigma * self.sigma).exp()
    }
    fn variance(&self) -> f64 {
        let s2 = self.sigma * self.sigma;
        s2.exp_m1() * (2.0 * self.mu + s2).exp()
    }
    fn skewness(&self) -> f64 {
        let s2 = self.sigma * self.sigma;
        (s2.exp() + 2.0) * s2.exp_m1().sqrt()
    }
    fn excess_kurtosis(&self) -> f64 {
        let s2 = self.sigma * self.sigma;
        (4.0 * s2).exp() + 2.0 * (3.0 * s2).exp() + 3.0 * (2.0 * s2).exp() - 6.0
    }
}

impl Fit for LogNormal {
    /// The mean and standard deviation (divisor n) of `ln x`.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        if data.iter().any(|&v| v <= 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("log-normal data must be positive"));
        }
        let logs: Vec<f64> = data.iter().map(|v| v.ln()).collect();
        let n = Normal::fit(&logs)?;
        LogNormal::new(n.mu, n.sigma)
    }
}

/// The Weibull distribution with shape `k` and scale `λ` (SciPy's `weibull_min(c=k, scale=λ)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weibull {
    pub shape: f64,
    pub scale: f64,
}

impl Weibull {
    pub fn new(shape: f64, scale: f64) -> Result<Self, ErrorsJSL> {
        Ok(Weibull { shape: positive(shape, "the shape must be positive")?, scale: positive(scale, "the scale must be positive")? })
    }
    fn raw_moment(&self, r: f64) -> f64 {
        self.scale.powf(r) * ln_gamma(1.0 + r / self.shape).exp()
    }
}

impl ContinuousDistribution for Weibull {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 { 0.0 } else { self.logpdf(x).exp() }
    }
    fn logpdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return f64::NEG_INFINITY;
        }
        let z = x / self.scale;
        (self.shape / self.scale).ln() + (self.shape - 1.0) * z.ln() - z.powf(self.shape)
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 { 0.0 } else { -(-(x / self.scale).powf(self.shape)).exp_m1() }
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 { 1.0 } else { (-(x / self.scale).powf(self.shape)).exp() }
    }
    fn ppf(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) { f64::NAN } else { self.scale * (-(-p).ln_1p()).powf(1.0 / self.shape) }
    }
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
    fn mean(&self) -> f64 {
        self.raw_moment(1.0)
    }
    fn variance(&self) -> f64 {
        self.raw_moment(2.0) - self.mean().powi(2)
    }
    fn skewness(&self) -> f64 {
        let (m, v) = (self.mean(), self.variance());
        (self.raw_moment(3.0) - 3.0 * m * v - m.powi(3)) / v.powf(1.5)
    }
    fn excess_kurtosis(&self) -> f64 {
        let (m1, m2, m3, m4) = (self.raw_moment(1.0), self.raw_moment(2.0), self.raw_moment(3.0), self.raw_moment(4.0));
        let v = m2 - m1 * m1;
        (m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1.powi(4)) / (v * v) - 3.0
    }
}

impl Fit for Weibull {
    /// The shape solves `Σ xᵏ ln x / Σ xᵏ − 1/k − mean(ln x) = 0` (Brent's method), and
    /// `λ = (mean(xᵏ))^{1/k}`.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 2)?;
        if data.iter().any(|&v| v <= 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("Weibull data must be positive"));
        }
        let n = data.len() as f64;
        let mlog = data.iter().map(|v| v.ln()).sum::<f64>() / n;
        let xmax = data.iter().copied().fold(0.0, f64::max);
        // Work with x / max(x) to avoid overflow of xᵏ.
        let g = |k: f64| {
            let (mut s0, mut s1) = (0.0, 0.0);
            for &x in data {
                let t = (x / xmax).powf(k);
                s0 += t;
                s1 += t * x.ln();
            }
            s1 / s0 - 1.0 / k - mlog
        };
        let (mut lo, mut hi) = (0.01, 1.0);
        while g(hi) < 0.0 {
            hi *= 2.0;
        }
        while g(lo) > 0.0 {
            lo /= 2.0;
        }
        let k = brent(g, lo, hi);
        let lam = xmax * (data.iter().map(|x| (x / xmax).powf(k)).sum::<f64>() / n).powf(1.0 / k);
        Weibull::new(k, lam)
    }
}

/// The Cauchy distribution (SciPy's `cauchy(loc, scale)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cauchy {
    pub loc: f64,
    pub scale: f64,
}

impl Cauchy {
    pub fn new(loc: f64, scale: f64) -> Result<Self, ErrorsJSL> {
        Ok(Cauchy { loc, scale: positive(scale, "the scale must be positive")? })
    }
}

impl ContinuousDistribution for Cauchy {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        1.0 / (PI * self.scale * (1.0 + z * z))
    }
    fn cdf(&self, x: f64) -> f64 {
        0.5 + ((x - self.loc) / self.scale).atan() / PI
    }
    fn sf(&self, x: f64) -> f64 {
        0.5 - ((x - self.loc) / self.scale).atan() / PI
    }
    fn ppf(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) { f64::NAN } else { self.loc + self.scale * (PI * (p - 0.5)).tan() }
    }
    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
    fn mean(&self) -> f64 {
        f64::NAN
    }
    fn variance(&self) -> f64 {
        f64::NAN
    }
    fn skewness(&self) -> f64 {
        f64::NAN
    }
    fn excess_kurtosis(&self) -> f64 {
        f64::NAN
    }
    fn median(&self) -> f64 {
        self.loc
    }
}

impl Fit for Cauchy {
    /// Nelder-Mead on `(x₀, ln γ)` from the median and half the interquartile range.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 2)?;
        let mut s = data.to_vec();
        s.sort_by(f64::total_cmp);
        let (med, iqr) = (s[s.len() / 2], (s[3 * s.len() / 4] - s[s.len() / 4]).max(1e-12));
        let p = nelder_mead(|p| -(Cauchy { loc: p[0], scale: p[1].exp() }).log_likelihood(data), &[med, (iqr / 2.0).ln()]);
        Cauchy::new(p[0], p[1].exp())
    }
}

/// The Laplace (double exponential) distribution (SciPy's `laplace(loc=μ, scale=b)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Laplace {
    pub mu: f64,
    pub b: f64,
}

impl Laplace {
    pub fn new(mu: f64, b: f64) -> Result<Self, ErrorsJSL> {
        Ok(Laplace { mu, b: positive(b, "the scale must be positive")? })
    }
}

impl ContinuousDistribution for Laplace {
    fn pdf(&self, x: f64) -> f64 {
        (-(x - self.mu).abs() / self.b).exp() / (2.0 * self.b)
    }
    fn logpdf(&self, x: f64) -> f64 {
        -(x - self.mu).abs() / self.b - (2.0 * self.b).ln()
    }
    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.b;
        if z < 0.0 { 0.5 * z.exp() } else { 1.0 - 0.5 * (-z).exp() }
    }
    fn sf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.b;
        if z > 0.0 { 0.5 * (-z).exp() } else { 1.0 - 0.5 * z.exp() }
    }
    fn ppf(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) {
            f64::NAN
        } else if p < 0.5 {
            self.mu + self.b * (2.0 * p).ln()
        } else {
            self.mu - self.b * (2.0 - 2.0 * p).ln()
        }
    }
    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
    fn mean(&self) -> f64 {
        self.mu
    }
    fn variance(&self) -> f64 {
        2.0 * self.b * self.b
    }
    fn skewness(&self) -> f64 {
        0.0
    }
    fn excess_kurtosis(&self) -> f64 {
        3.0
    }
}

impl Fit for Laplace {
    /// The sample median and the mean absolute deviation from it.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 2)?;
        let mut s = data.to_vec();
        s.sort_by(f64::total_cmp);
        let n = s.len();
        let med = if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) };
        Laplace::new(med, data.iter().map(|x| (x - med).abs()).sum::<f64>() / n as f64)
    }
}

/// The logistic distribution (SciPy's `logistic(loc=μ, scale=s)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Logistic {
    pub mu: f64,
    pub s: f64,
}

impl Logistic {
    pub fn new(mu: f64, s: f64) -> Result<Self, ErrorsJSL> {
        Ok(Logistic { mu, s: positive(s, "the scale must be positive")? })
    }
}

impl ContinuousDistribution for Logistic {
    fn pdf(&self, x: f64) -> f64 {
        self.logpdf(x).exp()
    }
    fn logpdf(&self, x: f64) -> f64 {
        let z = -((x - self.mu) / self.s).abs();
        z - 2.0 * z.exp().ln_1p() - self.s.ln()
    }
    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.s;
        if z >= 0.0 { 1.0 / (1.0 + (-z).exp()) } else { z.exp() / (1.0 + z.exp()) }
    }
    fn sf(&self, x: f64) -> f64 {
        Logistic { mu: -self.mu, s: self.s }.cdf(-x)
    }
    fn ppf(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) { f64::NAN } else { self.mu + self.s * (p / (1.0 - p)).ln() }
    }
    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
    fn mean(&self) -> f64 {
        self.mu
    }
    fn variance(&self) -> f64 {
        self.s * self.s * PI * PI / 3.0
    }
    fn skewness(&self) -> f64 {
        0.0
    }
    fn excess_kurtosis(&self) -> f64 {
        1.2
    }
}

impl Fit for Logistic {
    /// Nelder-Mead on `(μ, ln s)` from the moment estimates.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 2)?;
        let n = Normal::fit(data)?;
        let p = nelder_mead(|p| -(Logistic { mu: p[0], s: p[1].exp() }).log_likelihood(data), &[n.mu, (n.sigma * 3f64.sqrt() / PI).ln()]);
        Logistic::new(p[0], p[1].exp())
    }
}

/// The Pareto distribution with shape `α` and scale `x_m` (SciPy's `pareto(b=α, scale=x_m)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pareto {
    pub alpha: f64,
    pub xm: f64,
}

impl Pareto {
    pub fn new(alpha: f64, xm: f64) -> Result<Self, ErrorsJSL> {
        Ok(Pareto { alpha: positive(alpha, "α must be positive")?, xm: positive(xm, "x_m must be positive")? })
    }
}

impl ContinuousDistribution for Pareto {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.xm { 0.0 } else { self.alpha * self.xm.powf(self.alpha) / x.powf(self.alpha + 1.0) }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.xm { 0.0 } else { -((self.alpha * (self.xm / x).ln()).exp_m1()) }
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= self.xm { 1.0 } else { (self.xm / x).powf(self.alpha) }
    }
    fn ppf(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) { f64::NAN } else { self.xm * (1.0 - p).powf(-1.0 / self.alpha) }
    }
    fn support(&self) -> (f64, f64) {
        (self.xm, f64::INFINITY)
    }
    fn mean(&self) -> f64 {
        if self.alpha > 1.0 { self.alpha * self.xm / (self.alpha - 1.0) } else { f64::INFINITY }
    }
    fn variance(&self) -> f64 {
        let a = self.alpha;
        if a > 2.0 { self.xm * self.xm * a / ((a - 1.0).powi(2) * (a - 2.0)) } else { f64::INFINITY }
    }
    fn skewness(&self) -> f64 {
        let a = self.alpha;
        if a > 3.0 { 2.0 * (1.0 + a) / (a - 3.0) * ((a - 2.0) / a).sqrt() } else { f64::NAN }
    }
    fn excess_kurtosis(&self) -> f64 {
        let a = self.alpha;
        if a > 4.0 { 6.0 * (a.powi(3) + a * a - 6.0 * a - 2.0) / (a * (a - 3.0) * (a - 4.0)) } else { f64::NAN }
    }
}

impl Fit for Pareto {
    /// `x_m = min x`, `α = n / Σ ln(x / x_m)`.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 2)?;
        let xm = data.iter().copied().fold(f64::INFINITY, f64::min);
        Pareto::new(data.len() as f64 / data.iter().map(|x| (x / xm).ln()).sum::<f64>(), xm)
    }
}

/// The Rayleigh distribution (SciPy's `rayleigh(scale=σ)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rayleigh {
    pub sigma: f64,
}

impl Rayleigh {
    pub fn new(sigma: f64) -> Result<Self, ErrorsJSL> {
        Ok(Rayleigh { sigma: positive(sigma, "σ must be positive")? })
    }
}

impl ContinuousDistribution for Rayleigh {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 { 0.0 } else { x / (self.sigma * self.sigma) * (-0.5 * (x / self.sigma).powi(2)).exp() }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 { 0.0 } else { -(-0.5 * (x / self.sigma).powi(2)).exp_m1() }
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 { 1.0 } else { (-0.5 * (x / self.sigma).powi(2)).exp() }
    }
    fn ppf(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) { f64::NAN } else { self.sigma * (-2.0 * (-p).ln_1p()).sqrt() }
    }
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }
    fn mean(&self) -> f64 {
        self.sigma * (PI / 2.0).sqrt()
    }
    fn variance(&self) -> f64 {
        (4.0 - PI) / 2.0 * self.sigma * self.sigma
    }
    fn skewness(&self) -> f64 {
        2.0 * PI.sqrt() * (PI - 3.0) / (4.0 - PI).powf(1.5)
    }
    fn excess_kurtosis(&self) -> f64 {
        -(6.0 * PI * PI - 24.0 * PI + 16.0) / (4.0 - PI).powi(2)
    }
}

impl Fit for Rayleigh {
    /// `σ² = Σ x² / (2n)`.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 1)?;
        Rayleigh::new((data.iter().map(|x| x * x).sum::<f64>() / (2.0 * data.len() as f64)).sqrt())
    }
}

/// The Gumbel distribution of maxima (SciPy's `gumbel_r(loc=μ, scale=β)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gumbel {
    pub mu: f64,
    pub beta: f64,
}

const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
/// ζ(3), for the Gumbel skewness.
const ZETA3: f64 = 1.202_056_903_159_594_2;

impl Gumbel {
    pub fn new(mu: f64, beta: f64) -> Result<Self, ErrorsJSL> {
        Ok(Gumbel { mu, beta: positive(beta, "β must be positive")? })
    }
}

impl ContinuousDistribution for Gumbel {
    fn pdf(&self, x: f64) -> f64 {
        self.logpdf(x).exp()
    }
    fn logpdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.beta;
        -z - (-z).exp() - self.beta.ln()
    }
    fn cdf(&self, x: f64) -> f64 {
        (-(-(x - self.mu) / self.beta).exp()).exp()
    }
    fn sf(&self, x: f64) -> f64 {
        -(-(-(x - self.mu) / self.beta).exp()).exp_m1()
    }
    fn ppf(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) { f64::NAN } else { self.mu - self.beta * (-p.ln()).ln() }
    }
    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
    fn mean(&self) -> f64 {
        self.mu + self.beta * EULER_GAMMA
    }
    fn variance(&self) -> f64 {
        PI * PI / 6.0 * self.beta * self.beta
    }
    fn skewness(&self) -> f64 {
        12.0 * 6f64.sqrt() * ZETA3 / PI.powi(3)
    }
    fn excess_kurtosis(&self) -> f64 {
        2.4
    }
}

impl Fit for Gumbel {
    /// `β` solves `β = x̄ − Σ x e^{−x/β} / Σ e^{−x/β}`, and `μ = −β ln mean(e^{−x/β})`.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 2)?;
        let m = mean(data);
        let xmin = data.iter().copied().fold(f64::INFINITY, f64::min);
        let weights = |b: f64| -> (f64, f64) {
            let (mut s0, mut s1) = (0.0, 0.0);
            for &x in data {
                let w = (-(x - xmin) / b).exp();
                s0 += w;
                s1 += w * x;
            }
            (s0, s1)
        };
        let g = |b: f64| {
            let (s0, s1) = weights(b);
            b - m + s1 / s0
        };
        let sd = Normal::fit(data)?.sigma;
        let (mut lo, mut hi) = (sd * 0.05, sd * 2.0);
        while g(lo) > 0.0 {
            lo /= 2.0;
        }
        while g(hi) < 0.0 {
            hi *= 2.0;
        }
        let b = brent(g, lo, hi);
        let (s0, _) = weights(b);
        Gumbel::new(xmin - b * (s0 / data.len() as f64).ln(), b)
    }
}

/// The generalised extreme value distribution with shape `ξ`, location and scale, in the
/// statistics convention: `ξ > 0` is heavy-tailed (Fréchet), `ξ < 0` bounded above (Weibull).
/// SciPy's `genextreme(c)` uses `c = −ξ`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeneralizedExtremeValue {
    pub xi: f64,
    pub mu: f64,
    pub sigma: f64,
}

impl GeneralizedExtremeValue {
    pub fn new(xi: f64, mu: f64, sigma: f64) -> Result<Self, ErrorsJSL> {
        if !xi.is_finite() {
            return Err(ErrorsJSL::InvalidInputRange("ξ must be finite"));
        }
        Ok(GeneralizedExtremeValue { xi, mu, sigma: positive(sigma, "σ must be positive")? })
    }
    /// `t(x) = (1 + ξ z)^{−1/ξ}` (or `e^{−z}` for ξ = 0), or `None` outside the support.
    fn t(&self, x: f64) -> Option<f64> {
        let z = (x - self.mu) / self.sigma;
        if self.xi.abs() < 1e-12 {
            return Some((-z).exp());
        }
        let u = 1.0 + self.xi * z;
        if u <= 0.0 { None } else { Some((-(u.ln()) / self.xi).exp()) }
    }
    fn g(&self, k: f64) -> f64 {
        ln_gamma(1.0 - k * self.xi).exp()
    }
}

impl ContinuousDistribution for GeneralizedExtremeValue {
    fn pdf(&self, x: f64) -> f64 {
        self.logpdf(x).exp()
    }
    fn logpdf(&self, x: f64) -> f64 {
        match self.t(x) {
            None => f64::NEG_INFINITY,
            Some(t) => (self.xi + 1.0) * t.ln() - t - self.sigma.ln(),
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        match self.t(x) {
            Some(t) => (-t).exp(),
            None => {
                if self.xi > 0.0 { 0.0 } else { 1.0 }
            }
        }
    }
    fn sf(&self, x: f64) -> f64 {
        match self.t(x) {
            Some(t) => -(-t).exp_m1(),
            None => {
                if self.xi > 0.0 { 1.0 } else { 0.0 }
            }
        }
    }
    fn ppf(&self, p: f64) -> f64 {
        if !(0.0..=1.0).contains(&p) {
            return f64::NAN;
        }
        let y = -p.ln();
        if self.xi.abs() < 1e-12 { self.mu - self.sigma * y.ln() } else { self.mu + self.sigma * (y.powf(-self.xi) - 1.0) / self.xi }
    }
    fn support(&self) -> (f64, f64) {
        let edge = self.mu - self.sigma / self.xi;
        if self.xi > 0.0 {
            (edge, f64::INFINITY)
        } else if self.xi < 0.0 {
            (f64::NEG_INFINITY, edge)
        } else {
            (f64::NEG_INFINITY, f64::INFINITY)
        }
    }
    fn mean(&self) -> f64 {
        if self.xi.abs() < 1e-12 {
            self.mu + self.sigma * EULER_GAMMA
        } else if self.xi < 1.0 {
            self.mu + self.sigma * (self.g(1.0) - 1.0) / self.xi
        } else {
            f64::INFINITY
        }
    }
    fn variance(&self) -> f64 {
        if self.xi.abs() < 1e-12 {
            self.sigma * self.sigma * PI * PI / 6.0
        } else if self.xi < 0.5 {
            self.sigma * self.sigma * (self.g(2.0) - self.g(1.0).powi(2)) / (self.xi * self.xi)
        } else {
            f64::INFINITY
        }
    }
    fn skewness(&self) -> f64 {
        if self.xi.abs() < 1e-12 {
            return 12.0 * 6f64.sqrt() * ZETA3 / PI.powi(3);
        }
        if self.xi >= 1.0 / 3.0 {
            return f64::NAN;
        }
        let (g1, g2, g3) = (self.g(1.0), self.g(2.0), self.g(3.0));
        self.xi.signum() * (g3 - 3.0 * g2 * g1 + 2.0 * g1.powi(3)) / (g2 - g1 * g1).powf(1.5)
    }
    fn excess_kurtosis(&self) -> f64 {
        if self.xi.abs() < 1e-12 {
            return 2.4;
        }
        if self.xi >= 0.25 {
            return f64::NAN;
        }
        let (g1, g2, g3, g4) = (self.g(1.0), self.g(2.0), self.g(3.0), self.g(4.0));
        (g4 - 4.0 * g3 * g1 + 6.0 * g2 * g1 * g1 - 3.0 * g1.powi(4)) / (g2 - g1 * g1).powi(2) - 3.0
    }
}

impl Fit for GeneralizedExtremeValue {
    /// Nelder-Mead on `(ξ, μ, ln σ)` from the Gumbel fit.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 3)?;
        let g = Gumbel::fit(data)?;
        let nll = |p: &[f64]| {
            let d = GeneralizedExtremeValue { xi: p[0], mu: p[1], sigma: p[2].exp() };
            let l = d.log_likelihood(data);
            if l.is_finite() { -l } else { f64::INFINITY }
        };
        let p = nelder_mead(nll, &[0.05, g.mu, g.beta.ln()]);
        GeneralizedExtremeValue::new(p[0], p[1], p[2].exp())
    }
}

/// The normal distribution `N(μ, σ²)` truncated to `[a, b]` (SciPy's
/// `truncnorm((a−μ)/σ, (b−μ)/σ, loc=μ, scale=σ)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TruncatedNormal {
    pub mu: f64,
    pub sigma: f64,
    pub a: f64,
    pub b: f64,
}

impl TruncatedNormal {
    pub fn new(mu: f64, sigma: f64, a: f64, b: f64) -> Result<Self, ErrorsJSL> {
        if !(a < b) {
            return Err(ErrorsJSL::InvalidInputRange("need a < b"));
        }
        Ok(TruncatedNormal { mu, sigma: positive(sigma, "σ must be positive")?, a, b })
    }
    fn alpha_beta(&self) -> (f64, f64) {
        ((self.a - self.mu) / self.sigma, (self.b - self.mu) / self.sigma)
    }
    /// The probability mass `Φ(β) − Φ(α)`, from whichever tail is accurate.
    fn mass(&self) -> f64 {
        let (al, be) = self.alpha_beta();
        if al > 0.0 { std_normal_cdf(-al) - std_normal_cdf(-be) } else { std_normal_cdf(be) - std_normal_cdf(al) }
    }
    /// The raw moments of the standardised variable, `E Zᵏ` for k = 1..4.
    fn std_moments(&self) -> [f64; 4] {
        let (al, be) = self.alpha_beta();
        let z = self.mass();
        let (pa, pb) = (std_normal_pdf(al), std_normal_pdf(be));
        let f = |x: f64, p: f64| if x.is_finite() { x * p } else { 0.0 };
        let m1 = (pa - pb) / z;
        let m2 = 1.0 + (f(al, pa) - f(be, pb)) / z;
        let m3 = 2.0 * m1 + (f(al * al, pa) - f(be * be, pb)) / z;
        let m4 = 3.0 * m2 + (f(al.powi(3), pa) - f(be.powi(3), pb)) / z;
        [m1, m2, m3, m4]
    }
}

impl ContinuousDistribution for TruncatedNormal {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.a || x > self.b { 0.0 } else { std_normal_pdf((x - self.mu) / self.sigma) / (self.sigma * self.mass()) }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.a {
            return 0.0;
        }
        if x >= self.b {
            return 1.0;
        }
        let (al, _) = self.alpha_beta();
        let z = (x - self.mu) / self.sigma;
        let num = if al > 0.0 { std_normal_cdf(-al) - std_normal_cdf(-z) } else { std_normal_cdf(z) - std_normal_cdf(al) };
        (num / self.mass()).clamp(0.0, 1.0)
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= self.a {
            return 1.0;
        }
        if x >= self.b {
            return 0.0;
        }
        let (_, be) = self.alpha_beta();
        let z = (x - self.mu) / self.sigma;
        let num = if be < 0.0 { std_normal_cdf(be) - std_normal_cdf(z) } else { std_normal_cdf(-z) - std_normal_cdf(-be) };
        (num / self.mass()).clamp(0.0, 1.0)
    }
    fn support(&self) -> (f64, f64) {
        (self.a, self.b)
    }
    fn mean(&self) -> f64 {
        self.mu + self.sigma * self.std_moments()[0]
    }
    fn variance(&self) -> f64 {
        let m = self.std_moments();
        self.sigma * self.sigma * (m[1] - m[0] * m[0])
    }
    fn skewness(&self) -> f64 {
        let m = self.std_moments();
        let v = m[1] - m[0] * m[0];
        (m[2] - 3.0 * m[0] * m[1] + 2.0 * m[0].powi(3)) / v.powf(1.5)
    }
    fn excess_kurtosis(&self) -> f64 {
        let m = self.std_moments();
        let v = m[1] - m[0] * m[0];
        (m[3] - 4.0 * m[0] * m[2] + 6.0 * m[0] * m[0] * m[1] - 3.0 * m[0].powi(4)) / (v * v) - 3.0
    }
    fn ppf(&self, p: f64) -> f64 {
        invert_cdf(|x| self.cdf(x), p, self.support(), self.mean(), self.std())
    }
}

/// The von Mises distribution on the circle, `f(x) ∝ e^{κ cos(x − μ)}` on `[μ − π, μ + π]`
/// (SciPy's `vonmises(kappa=κ, loc=μ)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VonMises {
    pub mu: f64,
    pub kappa: f64,
}

impl VonMises {
    pub fn new(mu: f64, kappa: f64) -> Result<Self, ErrorsJSL> {
        Ok(VonMises { mu, kappa: positive(kappa, "κ must be positive")? })
    }
    /// `I_j(κ) / I_0(κ)` for j = 1..terms, by Miller's backward recurrence.
    fn bessel_ratios(&self, terms: usize) -> Vec<f64> {
        let n = terms + 40;
        let mut r = vec![0.0; n + 1];
        let mut acc = 0.0;
        for j in (1..=n).rev() {
            acc = 1.0 / (2.0 * j as f64 / self.kappa + acc);
            r[j] = acc;
        }
        let mut out = Vec::with_capacity(terms);
        let mut prod = 1.0;
        for rj in r.iter().take(terms + 1).skip(1) {
            prod *= rj;
            out.push(prod);
        }
        out
    }
    /// The mean resultant length `A(κ) = I₁(κ)/I₀(κ)`.
    pub fn mean_resultant_length(&self) -> f64 {
        self.bessel_ratios(1)[0]
    }
}

impl ContinuousDistribution for VonMises {
    fn pdf(&self, x: f64) -> f64 {
        if (x - self.mu).abs() > PI {
            return 0.0;
        }
        // e^{κ cos} / (2π I₀(κ)), with the exponential scaled to avoid overflow.
        let i0_scaled = bessel_i_fractional(0.0, self.kappa) * (-self.kappa).exp();
        (self.kappa * ((x - self.mu).cos() - 1.0)).exp() / (2.0 * PI * i0_scaled)
    }
    /// The Fourier series `F(x) = (x − μ + π + 2 Σ_j A_j(κ) sin(j(x − μ)) / j) / (2π)` with
    /// `A_j = I_j(κ)/I₀(κ)`.
    fn cdf(&self, x: f64) -> f64 {
        let y = x - self.mu;
        if y <= -PI {
            return 0.0;
        }
        if y >= PI {
            return 1.0;
        }
        let terms = (60.0 + 3.0 * self.kappa).min(4000.0) as usize;
        let s: f64 = self.bessel_ratios(terms).iter().enumerate().map(|(j, a)| a * ((j + 1) as f64 * y).sin() / (j + 1) as f64).sum();
        ((y + PI + 2.0 * s) / (2.0 * PI)).clamp(0.0, 1.0)
    }
    fn support(&self) -> (f64, f64) {
        (self.mu - PI, self.mu + PI)
    }
    fn mean(&self) -> f64 {
        self.mu
    }
    /// The variance of the linear variable on `[μ − π, μ + π]`: `π²/3 + 4 Σ (−1)ʲ A_j / j²`.
    fn variance(&self) -> f64 {
        let terms = (60.0 + 3.0 * self.kappa).min(4000.0) as usize;
        PI * PI / 3.0 + 4.0 * self.bessel_ratios(terms).iter().enumerate().map(|(j, a)| if j % 2 == 0 { -a } else { *a } / ((j + 1) as f64).powi(2)).sum::<f64>()
    }
    fn skewness(&self) -> f64 {
        0.0
    }
    /// `E(x−μ)⁴ / σ⁴ − 3`, with `E(x−μ)⁴ = π⁴/5 + Σ (−1)ʲ A_j (8π²/j² − 48/j⁴)`.
    fn excess_kurtosis(&self) -> f64 {
        let terms = (60.0 + 3.0 * self.kappa).min(4000.0) as usize;
        let m4 = PI.powi(4) / 5.0
            + self
                .bessel_ratios(terms)
                .iter()
                .enumerate()
                .map(|(j, a)| {
                    let jj = (j + 1) as f64;
                    let sign = if j % 2 == 0 { -1.0 } else { 1.0 };
                    sign * a * (8.0 * PI * PI / (jj * jj) - 48.0 / jj.powi(4))
                })
                .sum::<f64>();
        m4 / self.variance().powi(2) - 3.0
    }
}

impl Fit for VonMises {
    /// `μ = atan2(Σ sin x, Σ cos x)` and `κ` solves `A(κ) = R̄`, the mean resultant length.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 2)?;
        let n = data.len() as f64;
        let (s, c) = (data.iter().map(|x| x.sin()).sum::<f64>(), data.iter().map(|x| x.cos()).sum::<f64>());
        let r = (s * s + c * c).sqrt() / n;
        if !(r > 0.0 && r < 1.0) {
            return Err(ErrorsJSL::InvalidInputRange("the mean resultant length must lie in (0, 1)"));
        }
        let kappa = brent(|k| VonMises { mu: 0.0, kappa: k }.mean_resultant_length() - r, 1e-8, 1e5);
        VonMises::new(s.atan2(c), kappa)
    }
}

/// Owen's T function `T(h, a) = (1/2π) ∫₀ᵃ e^{−h²(1+x²)/2} / (1 + x²) dx`.
pub fn owens_t(h: f64, a: f64) -> f64 {
    // 40-point Gauss-Legendre on [0, a] for |a| ≤ 1; the reflection T(h,a) + T(ah, 1/a) for |a| > 1.
    fn small(h: f64, a: f64) -> f64 {
        let (x, w) = gauss_legendre_40();
        let mut s = 0.0;
        for i in 0..40 {
            let t = 0.5 * a * (x[i] + 1.0);
            s += w[i] * (-0.5 * h * h * (1.0 + t * t)).exp() / (1.0 + t * t);
        }
        0.5 * a * s / (2.0 * PI)
    }
    if a < 0.0 {
        return -owens_t(h, -a);
    }
    let h = h.abs();
    if a <= 1.0 {
        return small(h, a);
    }
    let ah = a * h;
    let (ph, pah) = (std_normal_cdf(h), std_normal_cdf(ah));
    0.5 * ph + 0.5 * pah - ph * pah - small(ah, 1.0 / a)
}

/// Nodes and weights of 40-point Gauss-Legendre quadrature on [−1, 1] (by Newton's method on
/// the Legendre polynomial).
fn gauss_legendre_40() -> ([f64; 40], [f64; 40]) {
    let n = 40;
    let (mut x, mut w) = ([0.0; 40], [0.0; 40]);
    for i in 0..n / 2 {
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
            let dz = p1 / dp;
            z -= dz;
            if dz.abs() < 1e-16 {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
        w[n - 1 - i] = w[i];
    }
    (x, w)
}

/// The skew-normal distribution with shape `α` (SciPy's `skewnorm(a=α, loc, scale)`):
/// density `2 φ(z) Φ(αz) / scale`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkewNormal {
    pub alpha: f64,
    pub loc: f64,
    pub scale: f64,
}

impl SkewNormal {
    pub fn new(alpha: f64, loc: f64, scale: f64) -> Result<Self, ErrorsJSL> {
        if !alpha.is_finite() {
            return Err(ErrorsJSL::InvalidInputRange("α must be finite"));
        }
        Ok(SkewNormal { alpha, loc, scale: positive(scale, "the scale must be positive")? })
    }
    fn delta(&self) -> f64 {
        self.alpha / (1.0 + self.alpha * self.alpha).sqrt()
    }
}

impl ContinuousDistribution for SkewNormal {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        2.0 * std_normal_pdf(z) * std_normal_cdf(self.alpha * z) / self.scale
    }
    /// `Φ(z) − 2 T(z, α)`.
    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        (std_normal_cdf(z) - 2.0 * owens_t(z, self.alpha)).clamp(0.0, 1.0)
    }
    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
    fn mean(&self) -> f64 {
        self.loc + self.scale * self.delta() * (2.0 / PI).sqrt()
    }
    fn variance(&self) -> f64 {
        self.scale * self.scale * (1.0 - 2.0 * self.delta().powi(2) / PI)
    }
    fn skewness(&self) -> f64 {
        let m = self.delta() * (2.0 / PI).sqrt();
        (4.0 - PI) / 2.0 * m.powi(3) / (1.0 - m * m).powf(1.5)
    }
    fn excess_kurtosis(&self) -> f64 {
        let m = self.delta() * (2.0 / PI).sqrt();
        2.0 * (PI - 3.0) * m.powi(4) / (1.0 - m * m).powi(2)
    }
}

impl Fit for SkewNormal {
    /// Nelder-Mead on `(α, loc, ln scale)` from the normal fit, shifted by the sample skewness.
    fn fit(data: &[f64]) -> Result<Self, ErrorsJSL> {
        check_data(data, 3)?;
        let n = Normal::fit(data)?;
        let g = data.iter().map(|x| ((x - n.mu) / n.sigma).powi(3)).sum::<f64>() / data.len() as f64;
        let alpha0 = (3.0 * g).clamp(-5.0, 5.0);
        let nll = |p: &[f64]| -(SkewNormal { alpha: p[0], loc: p[1], scale: p[2].exp() }).log_likelihood(data);
        let p = nelder_mead(nll, &[alpha0, n.mu, n.sigma.ln()]);
        SkewNormal::new(p[0], p[1], p[2].exp())
    }
}

// -------------------------------------------------------------------------------- discrete

fn ln_choose(n: f64, k: f64) -> f64 {
    ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
}

/// The binomial distribution `B(n, p)` (SciPy's `binom(n, p)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Binomial {
    pub n: u64,
    pub p: f64,
}

impl Binomial {
    pub fn new(n: u64, p: f64) -> Result<Self, ErrorsJSL> {
        if !(0.0..=1.0).contains(&p) {
            return Err(ErrorsJSL::InvalidInputRange("p must lie in [0, 1]"));
        }
        Ok(Binomial { n, p })
    }
    /// The maximum-likelihood `p = x̄ / n` for a known `n`.
    pub fn fit(data: &[i64], n: u64) -> Result<Self, ErrorsJSL> {
        if data.is_empty() {
            return Err(ErrorsJSL::InvalidInputRange("no data"));
        }
        Binomial::new(n, data.iter().sum::<i64>() as f64 / (data.len() as f64 * n as f64))
    }
}

impl DiscreteDistribution for Binomial {
    fn pmf(&self, k: i64) -> f64 {
        if k < 0 || k as u64 > self.n { 0.0 } else { self.logpmf(k).exp() }
    }
    fn logpmf(&self, k: i64) -> f64 {
        if k < 0 || k as u64 > self.n {
            return f64::NEG_INFINITY;
        }
        let (n, kf) = (self.n as f64, k as f64);
        let term = |c: f64, q: f64| if c == 0.0 { 0.0 } else { c * q.ln() };
        ln_choose(n, kf) + term(kf, self.p) + term(n - kf, 1.0 - self.p)
    }
    /// `I_{1−p}(n − k, k + 1)`.
    fn cdf(&self, k: i64) -> f64 {
        if k < 0 {
            0.0
        } else if k as u64 >= self.n {
            1.0
        } else {
            incomplete_beta((self.n - k as u64) as f64, k as f64 + 1.0, 1.0 - self.p)
        }
    }
    fn sf(&self, k: i64) -> f64 {
        if k < 0 {
            1.0
        } else if k as u64 >= self.n {
            0.0
        } else {
            incomplete_beta(k as f64 + 1.0, (self.n - k as u64) as f64, self.p)
        }
    }
    fn support(&self) -> (i64, i64) {
        (0, self.n as i64)
    }
    fn mean(&self) -> f64 {
        self.n as f64 * self.p
    }
    fn variance(&self) -> f64 {
        self.n as f64 * self.p * (1.0 - self.p)
    }
    fn skewness(&self) -> f64 {
        (1.0 - 2.0 * self.p) / self.variance().sqrt()
    }
    fn excess_kurtosis(&self) -> f64 {
        (1.0 - 6.0 * self.p * (1.0 - self.p)) / self.variance()
    }
}

/// The Poisson distribution (SciPy's `poisson(mu=λ)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Poisson {
    pub lambda: f64,
}

impl Poisson {
    pub fn new(lambda: f64) -> Result<Self, ErrorsJSL> {
        Ok(Poisson { lambda: positive(lambda, "λ must be positive")? })
    }
    /// The maximum-likelihood `λ = x̄`.
    pub fn fit(data: &[i64]) -> Result<Self, ErrorsJSL> {
        if data.is_empty() {
            return Err(ErrorsJSL::InvalidInputRange("no data"));
        }
        Poisson::new(data.iter().sum::<i64>() as f64 / data.len() as f64)
    }
}

impl DiscreteDistribution for Poisson {
    fn pmf(&self, k: i64) -> f64 {
        if k < 0 { 0.0 } else { self.logpmf(k).exp() }
    }
    fn logpmf(&self, k: i64) -> f64 {
        if k < 0 { f64::NEG_INFINITY } else { k as f64 * self.lambda.ln() - self.lambda - ln_gamma(k as f64 + 1.0) }
    }
    /// `Q(k + 1, λ)`, the regularised upper incomplete gamma function.
    fn cdf(&self, k: i64) -> f64 {
        if k < 0 { 0.0 } else { incomplete_gamma_upper(k as f64 + 1.0, self.lambda) }
    }
    fn sf(&self, k: i64) -> f64 {
        if k < 0 { 1.0 } else { incomplete_gamma(k as f64 + 1.0, self.lambda) }
    }
    fn support(&self) -> (i64, i64) {
        (0, i64::MAX)
    }
    fn mean(&self) -> f64 {
        self.lambda
    }
    fn variance(&self) -> f64 {
        self.lambda
    }
    fn skewness(&self) -> f64 {
        1.0 / self.lambda.sqrt()
    }
    fn excess_kurtosis(&self) -> f64 {
        1.0 / self.lambda
    }
}

/// The geometric distribution of the number of trials up to the first success, `k ≥ 1`
/// (SciPy's `geom(p)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometric {
    pub p: f64,
}

impl Geometric {
    pub fn new(p: f64) -> Result<Self, ErrorsJSL> {
        if !(p > 0.0 && p <= 1.0) {
            return Err(ErrorsJSL::InvalidInputRange("p must lie in (0, 1]"));
        }
        Ok(Geometric { p })
    }
    /// The maximum-likelihood `p = 1 / x̄`.
    pub fn fit(data: &[i64]) -> Result<Self, ErrorsJSL> {
        if data.is_empty() {
            return Err(ErrorsJSL::InvalidInputRange("no data"));
        }
        Geometric::new(data.len() as f64 / data.iter().sum::<i64>() as f64)
    }
}

impl DiscreteDistribution for Geometric {
    fn pmf(&self, k: i64) -> f64 {
        if k < 1 { 0.0 } else { self.p * ((k - 1) as f64 * (-self.p).ln_1p()).exp() }
    }
    fn cdf(&self, k: i64) -> f64 {
        if k < 1 { 0.0 } else { -(k as f64 * (-self.p).ln_1p()).exp_m1() }
    }
    fn sf(&self, k: i64) -> f64 {
        if k < 1 { 1.0 } else { (k as f64 * (-self.p).ln_1p()).exp() }
    }
    fn ppf(&self, q: f64) -> i64 {
        if q <= 0.0 { 0 } else { ((-q).ln_1p() / (-self.p).ln_1p()).ceil().max(1.0) as i64 }
    }
    fn support(&self) -> (i64, i64) {
        (1, i64::MAX)
    }
    fn mean(&self) -> f64 {
        1.0 / self.p
    }
    fn variance(&self) -> f64 {
        (1.0 - self.p) / (self.p * self.p)
    }
    fn skewness(&self) -> f64 {
        (2.0 - self.p) / (1.0 - self.p).sqrt()
    }
    fn excess_kurtosis(&self) -> f64 {
        6.0 + self.p * self.p / (1.0 - self.p)
    }
}

/// The negative binomial distribution of the number of failures before the `r`-th success
/// (SciPy's `nbinom(n=r, p)`); `r` need not be an integer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NegativeBinomial {
    pub r: f64,
    pub p: f64,
}

impl NegativeBinomial {
    pub fn new(r: f64, p: f64) -> Result<Self, ErrorsJSL> {
        if !(p > 0.0 && p <= 1.0) {
            return Err(ErrorsJSL::InvalidInputRange("p must lie in (0, 1]"));
        }
        Ok(NegativeBinomial { r: positive(r, "r must be positive")?, p })
    }
    // ANCHOR: nbinom_fit
    /// Maximum likelihood: for each `r` the best `p` is `r / (r + x̄)`, and `r` solves the profile
    /// score `Σ ψ(xᵢ + r) − n ψ(r) + n ln(r / (r + x̄)) = 0`. Requires over-dispersed data
    /// (sample variance above the mean).
    pub fn fit(data: &[i64]) -> Result<Self, ErrorsJSL> {
        if data.len() < 2 || data.iter().any(|&k| k < 0) {
            return Err(ErrorsJSL::InvalidInputRange("need at least two non-negative counts"));
        }
        let n = data.len() as f64;
        let m = data.iter().sum::<i64>() as f64 / n;
        let v = data.iter().map(|&k| (k as f64 - m).powi(2)).sum::<f64>() / n;
        if !(v > m) {
            return Err(ErrorsJSL::InvalidInputRange("the data are not over-dispersed; a Poisson fit is better"));
        }
        let score = |r: f64| data.iter().map(|&k| digamma(k as f64 + r)).sum::<f64>() - n * digamma(r) + n * (r / (r + m)).ln();
        let r0 = m * m / (v - m);
        let (mut lo, mut hi) = (r0 / 2.0, r0 * 2.0);
        while score(lo) < 0.0 && lo > 1e-12 {
            lo /= 2.0;
        }
        while score(hi) > 0.0 && hi < 1e12 {
            hi *= 2.0;
        }
        let r = brent(score, lo, hi);
        NegativeBinomial::new(r, r / (r + m))
    }
    // ANCHOR_END: nbinom_fit
}

impl DiscreteDistribution for NegativeBinomial {
    fn pmf(&self, k: i64) -> f64 {
        if k < 0 { 0.0 } else { self.logpmf(k).exp() }
    }
    fn logpmf(&self, k: i64) -> f64 {
        if k < 0 {
            return f64::NEG_INFINITY;
        }
        let kf = k as f64;
        ln_gamma(kf + self.r) - ln_gamma(self.r) - ln_gamma(kf + 1.0) + self.r * self.p.ln() + if k > 0 { kf * (-self.p).ln_1p() } else { 0.0 }
    }
    /// `I_p(r, k + 1)`.
    fn cdf(&self, k: i64) -> f64 {
        if k < 0 { 0.0 } else { incomplete_beta(self.r, k as f64 + 1.0, self.p) }
    }
    fn sf(&self, k: i64) -> f64 {
        if k < 0 { 1.0 } else { incomplete_beta(k as f64 + 1.0, self.r, 1.0 - self.p) }
    }
    fn support(&self) -> (i64, i64) {
        (0, i64::MAX)
    }
    fn mean(&self) -> f64 {
        self.r * (1.0 - self.p) / self.p
    }
    fn variance(&self) -> f64 {
        self.r * (1.0 - self.p) / (self.p * self.p)
    }
    fn skewness(&self) -> f64 {
        (2.0 - self.p) / (self.r * (1.0 - self.p)).sqrt()
    }
    fn excess_kurtosis(&self) -> f64 {
        6.0 / self.r + self.p * self.p / (self.r * (1.0 - self.p))
    }
}

/// The hypergeometric distribution of the successes in `draws` draws without replacement from a
/// population of `total` items of which `successes` are successes (SciPy's
/// `hypergeom(M=total, n=successes, N=draws)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hypergeometric {
    pub total: u64,
    pub successes: u64,
    pub draws: u64,
}

impl Hypergeometric {
    pub fn new(total: u64, successes: u64, draws: u64) -> Result<Self, ErrorsJSL> {
        if successes > total || draws > total {
            return Err(ErrorsJSL::InvalidInputRange("successes and draws cannot exceed the population"));
        }
        Ok(Hypergeometric { total, successes, draws })
    }
}

impl DiscreteDistribution for Hypergeometric {
    fn pmf(&self, k: i64) -> f64 {
        let (lo, hi) = self.support();
        if k < lo || k > hi { 0.0 } else { self.logpmf(k).exp() }
    }
    fn logpmf(&self, k: i64) -> f64 {
        let (lo, hi) = self.support();
        if k < lo || k > hi {
            return f64::NEG_INFINITY;
        }
        let (m, n, d, kf) = (self.total as f64, self.successes as f64, self.draws as f64, k as f64);
        ln_choose(n, kf) + ln_choose(m - n, d - kf) - ln_choose(m, d)
    }
    /// By summing the probability mass from the nearer end of the support.
    fn cdf(&self, k: i64) -> f64 {
        let (lo, hi) = self.support();
        if k < lo {
            return 0.0;
        }
        if k >= hi {
            return 1.0;
        }
        if (k as f64) < self.mean() { (lo..=k).map(|j| self.pmf(j)).sum::<f64>().min(1.0) } else { (1.0 - (k + 1..=hi).map(|j| self.pmf(j)).sum::<f64>()).max(0.0) }
    }
    fn sf(&self, k: i64) -> f64 {
        let (lo, hi) = self.support();
        if k < lo {
            return 1.0;
        }
        if k >= hi {
            return 0.0;
        }
        if (k as f64) >= self.mean() { (k + 1..=hi).map(|j| self.pmf(j)).sum::<f64>().min(1.0) } else { (1.0 - (lo..=k).map(|j| self.pmf(j)).sum::<f64>()).max(0.0) }
    }
    fn support(&self) -> (i64, i64) {
        ((self.draws as i64 + self.successes as i64 - self.total as i64).max(0), self.draws.min(self.successes) as i64)
    }
    fn mean(&self) -> f64 {
        self.draws as f64 * self.successes as f64 / self.total as f64
    }
    fn variance(&self) -> f64 {
        let (m, n, d) = (self.total as f64, self.successes as f64, self.draws as f64);
        d * n / m * (m - n) / m * (m - d) / (m - 1.0)
    }
    fn skewness(&self) -> f64 {
        let (m, n, d) = (self.total as f64, self.successes as f64, self.draws as f64);
        (m - 2.0 * n) * (m - 1.0).sqrt() * (m - 2.0 * d) / ((d * n * (m - n) * (m - d)).sqrt() * (m - 2.0))
    }
    fn excess_kurtosis(&self) -> f64 {
        let (m, n, d) = (self.total as f64, self.successes as f64, self.draws as f64);
        let num = (m - 1.0) * m * m * (m * (m + 1.0) - 6.0 * n * (m - n) - 6.0 * d * (m - d)) + 6.0 * d * n * (m - n) * (m - d) * (5.0 * m - 6.0);
        num / (d * n * (m - n) * (m - d) * (m - 2.0) * (m - 3.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lib_jsl_random::xoshiro256plusplus::Xoshiro256PlusPlus;

    fn check_continuous<D: ContinuousDistribution>(d: &D, points: &[f64]) {
        for &x in points {
            let (c, s) = (d.cdf(x), d.sf(x));
            assert!((c + s - 1.0).abs() < 1e-12, "cdf + sf at {x}: {c} {s}");
            if c > 1e-10 && c < 1.0 - 1e-10 {
                let back = d.ppf(c);
                assert!((back - x).abs() < 1e-7 * (1.0 + x.abs()), "ppf(cdf({x})) = {back}");
            }
        }
        // The density integrates to the distribution function (Simpson between two points).
        let (a, b) = (points[0], points[points.len() - 1]);
        let n = 2000;
        let h = (b - a) / n as f64;
        let integral: f64 = (0..=n).map(|i| d.pdf(a + i as f64 * h) * if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 }).sum::<f64>() * h / 3.0;
        assert!((integral - (d.cdf(b) - d.cdf(a))).abs() < 1e-6, "∫pdf = {integral}");
    }

    #[test]
    fn every_continuous_distribution_is_consistent() {
        check_continuous(&Normal::new(1.0, 2.0).unwrap(), &[-3.0, 0.0, 1.5, 4.0]);
        check_continuous(&Gamma::new(2.5, 1.5).unwrap(), &[0.2, 1.0, 3.0, 9.0]);
        check_continuous(&Beta::new(2.0, 3.5).unwrap(), &[0.05, 0.3, 0.6, 0.95]);
        check_continuous(&StudentT::new(4.5, 1.0, 2.0).unwrap(), &[-5.0, 0.0, 1.0, 6.0]);
        check_continuous(&FDist::new(5.0, 12.0).unwrap(), &[0.1, 0.8, 2.0, 5.0]);
        check_continuous(&LogNormal::new(0.3, 0.6).unwrap(), &[0.2, 1.0, 2.0, 5.0]);
        check_continuous(&Weibull::new(1.7, 2.0).unwrap(), &[0.1, 1.0, 2.5, 6.0]);
        check_continuous(&Gumbel::new(1.0, 0.8).unwrap(), &[-1.0, 0.5, 2.0, 5.0]);
        check_continuous(&GeneralizedExtremeValue::new(0.2, 1.0, 0.8).unwrap(), &[-1.0, 0.5, 2.0, 5.0]);
        check_continuous(&TruncatedNormal::new(0.5, 1.2, -1.0, 2.5).unwrap(), &[-0.9, 0.0, 1.0, 2.4]);
        check_continuous(&VonMises::new(0.4, 2.5).unwrap(), &[-2.5, -0.5, 0.4, 3.0]);
        check_continuous(&SkewNormal::new(3.0, 0.5, 1.5).unwrap(), &[-2.0, 0.5, 1.5, 5.0]);
        check_continuous(&Laplace::new(0.5, 1.2).unwrap(), &[-3.0, 0.0, 1.0, 4.0]);
        check_continuous(&Logistic::new(0.5, 1.2).unwrap(), &[-3.0, 0.0, 1.0, 4.0]);
    }

    #[test]
    fn fits_recover_the_parameters_of_large_samples() {
        let mut rng = Xoshiro256PlusPlus::from_seed(5);
        let g = Gamma::new(2.5, 1.5).unwrap();
        let data: Vec<f64> = (0..20000).map(|_| g.sample(&mut rng)).collect();
        let f = Gamma::fit(&data).unwrap();
        assert!((f.shape - 2.5).abs() < 0.08 && (f.scale - 1.5).abs() < 0.06, "{f:?}");
        let w = Weibull::new(1.7, 2.0).unwrap();
        let data: Vec<f64> = (0..20000).map(|_| w.sample(&mut rng)).collect();
        let f = Weibull::fit(&data).unwrap();
        assert!((f.shape - 1.7).abs() < 0.04 && (f.scale - 2.0).abs() < 0.03, "{f:?}");
    }

    #[test]
    fn discrete_cdfs_sum_the_pmf() {
        type PmfCdf = Box<dyn Fn(i64) -> (f64, f64)>;
        let dists: Vec<PmfCdf> = vec![
            Box::new(|k| (Binomial::new(20, 0.3).unwrap().pmf(k), Binomial::new(20, 0.3).unwrap().cdf(k))),
            Box::new(|k| (Poisson::new(4.2).unwrap().pmf(k), Poisson::new(4.2).unwrap().cdf(k))),
            Box::new(|k| (NegativeBinomial::new(3.5, 0.4).unwrap().pmf(k), NegativeBinomial::new(3.5, 0.4).unwrap().cdf(k))),
            Box::new(|k| (Hypergeometric::new(50, 12, 15).unwrap().pmf(k), Hypergeometric::new(50, 12, 15).unwrap().cdf(k))),
        ];
        for d in dists {
            let mut acc = 0.0;
            for k in 0..30 {
                let (p, c) = d(k);
                acc += p;
                assert!((acc - c).abs() < 1e-12, "k = {k}: {acc} vs {c}");
            }
        }
    }
}
