//! Figures and tables for "Probability distributions" (the statistics chapter): every
//! distribution's functions and moments against SciPy, maximum-likelihood fits, and model choice
//! by likelihood on a skewed sample.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_random::xoshiro256plusplus::Xoshiro256PlusPlus;
use lib_jsl_statistics::distributions::*;

const CHAPTER: &str = "statistics";

pub fn generate() {
    scipy_reference();
    model_choice();
}

fn build(name: &str, p: &[f64]) -> Box<dyn ContinuousDistribution> {
    match name {
        "normal" => Box::new(Normal::new(p[0], p[1]).unwrap()),
        "uniform" => Box::new(Uniform::new(p[0], p[1]).unwrap()),
        "exponential" => Box::new(Exponential::new(p[0]).unwrap()),
        "gamma" => Box::new(Gamma::new(p[0], p[1]).unwrap()),
        "beta" => Box::new(Beta::new(p[0], p[1]).unwrap()),
        "chi2" => Box::new(ChiSquared::new(p[0]).unwrap()),
        "t" => Box::new(StudentT::new(p[0], p[1], p[2]).unwrap()),
        "f" => Box::new(FDist::new(p[0], p[1]).unwrap()),
        "lognormal" => Box::new(LogNormal::new(p[0], p[1]).unwrap()),
        "weibull" => Box::new(Weibull::new(p[0], p[1]).unwrap()),
        "cauchy" => Box::new(Cauchy::new(p[0], p[1]).unwrap()),
        "laplace" => Box::new(Laplace::new(p[0], p[1]).unwrap()),
        "logistic" => Box::new(Logistic::new(p[0], p[1]).unwrap()),
        "pareto" => Box::new(Pareto::new(p[0], p[1]).unwrap()),
        "rayleigh" => Box::new(Rayleigh::new(p[0]).unwrap()),
        "gumbel" => Box::new(Gumbel::new(p[0], p[1]).unwrap()),
        "gev" | "gev_bounded" => Box::new(GeneralizedExtremeValue::new(p[0], p[1], p[2]).unwrap()),
        "truncnorm" => Box::new(TruncatedNormal::new(p[0], p[1], p[2], p[3]).unwrap()),
        "vonmises" => Box::new(VonMises::new(p[0], p[1]).unwrap()),
        "skewnorm" => Box::new(SkewNormal::new(p[0], p[1], p[2]).unwrap()),
        _ => panic!("unknown distribution {name}"),
    }
}

fn fit(name: &str, x: &[f64]) -> Vec<f64> {
    match name {
        "normal" => Normal::fit(x).map(|d| vec![d.mu, d.sigma]),
        "exponential" => Exponential::fit(x).map(|d| vec![d.lambda]),
        "gamma" => Gamma::fit(x).map(|d| vec![d.shape, d.scale]),
        "beta" => Beta::fit(x).map(|d| vec![d.a, d.b]),
        "chi2" => ChiSquared::fit(x).map(|d| vec![d.k]),
        "t" => StudentT::fit(x).map(|d| vec![d.nu, d.loc, d.scale]),
        "f" => FDist::fit(x).map(|d| vec![d.d1, d.d2]),
        "lognormal" => LogNormal::fit(x).map(|d| vec![d.mu, d.sigma]),
        "weibull" => Weibull::fit(x).map(|d| vec![d.shape, d.scale]),
        "cauchy" => Cauchy::fit(x).map(|d| vec![d.loc, d.scale]),
        "laplace" => Laplace::fit(x).map(|d| vec![d.mu, d.b]),
        "logistic" => Logistic::fit(x).map(|d| vec![d.mu, d.s]),
        "pareto" => Pareto::fit(x).map(|d| vec![d.alpha, d.xm]),
        "rayleigh" => Rayleigh::fit(x).map(|d| vec![d.sigma]),
        "gumbel" => Gumbel::fit(x).map(|d| vec![d.mu, d.beta]),
        "gev" => GeneralizedExtremeValue::fit(x).map(|d| vec![d.xi, d.mu, d.sigma]),
        "vonmises" => VonMises::fit(x).map(|d| vec![d.mu, d.kappa]),
        "skewnorm" => SkewNormal::fit(x).map(|d| vec![d.alpha, d.loc, d.scale]),
        _ => panic!("no fit for {name}"),
    }
    .unwrap()
}

fn rel(a: f64, b: f64) -> f64 {
    if a == b || (a.is_nan() && b.is_nan()) || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum()) {
        0.0
    } else {
        (a - b).abs() / b.abs().max(1e-300)
    }
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max(rel(*x, *y)))
}

/// Moments are compared relative to max(|reference|, 1): a skewness of exactly zero against a
/// numerically computed 5·10⁻¹⁵ is agreement, not a relative error of 1.
fn moment_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| if x.is_finite() && y.is_finite() { m.max((x - y).abs() / y.abs().max(1.0)) } else { m.max(rel(*x, *y)) })
}

fn scipy_reference() {
    let refs = Refs::parse(include_str!("../data/distributions_reference.txt"));
    let names = [
        ("normal", "Normal"), ("uniform", "Uniform"), ("exponential", "Exponential"), ("gamma", "Gamma"), ("beta", "Beta"), ("chi2", "ChiSquared"),
        ("t", "StudentT"), ("f", "FDist"), ("lognormal", "LogNormal"), ("weibull", "Weibull"), ("cauchy", "Cauchy"), ("laplace", "Laplace"),
        ("logistic", "Logistic"), ("pareto", "Pareto"), ("rayleigh", "Rayleigh"), ("gumbel", "Gumbel"), ("gev", "GeneralizedExtremeValue, ξ > 0"),
        ("gev_bounded", "GeneralizedExtremeValue, ξ < 0"), ("truncnorm", "TruncatedNormal"), ("vonmises", "VonMises"), ("skewnorm", "SkewNormal"),
    ];
    let mut rows = Vec::new();
    for (key, label) in names {
        let r = refs.get(&format!("eval {key}"));
        let d = build(key, &r[0]);
        let xs = &r[1];
        let pdf: Vec<f64> = xs.iter().map(|&x| d.pdf(x)).collect();
        let cdf: Vec<f64> = xs.iter().map(|&x| d.cdf(x)).collect();
        let sf: Vec<f64> = xs.iter().map(|&x| d.sf(x)).collect();
        let ppf: Vec<f64> = r[5].iter().map(|&q| d.ppf(q)).collect();
        let moments = [d.mean(), d.variance(), d.skewness(), d.excess_kurtosis()];
        let fitted = if key == "uniform" || key.starts_with("gev_") || key == "truncnorm" {
            String::new()
        } else {
            let sample = refs.values(&format!("sample {key}"));
            fmt_diff(max_rel(&fit(key, sample), refs.values(&format!("fit {key}"))))
        };
        rows.push(vec![format!("`{label}`"), fmt_diff(max_rel(&pdf, &r[2])), fmt_diff(max_rel(&cdf, &r[3]).max(max_rel(&sf, &r[4]))), fmt_diff(max_rel(&ppf, &r[6])), fmt_diff(moment_diff(&moments, &r[7])), fitted]);
    }
    // Discrete.
    for (key, label) in [("binomial", "Binomial"), ("poisson", "Poisson"), ("geometric", "Geometric"), ("nbinom", "NegativeBinomial"), ("hypergeom", "Hypergeometric")] {
        let r = refs.get(&format!("discrete {key}"));
        let p = &r[0];
        let d: Box<dyn Fn(i64) -> (f64, f64, f64)> = match key {
            "binomial" => {
                let d = Binomial::new(p[0] as u64, p[1]).unwrap();
                Box::new(move |k| (d.pmf(k), d.cdf(k), d.sf(k)))
            }
            "poisson" => {
                let d = Poisson::new(p[0]).unwrap();
                Box::new(move |k| (d.pmf(k), d.cdf(k), d.sf(k)))
            }
            "geometric" => {
                let d = Geometric::new(p[0]).unwrap();
                Box::new(move |k| (d.pmf(k), d.cdf(k), d.sf(k)))
            }
            "nbinom" => {
                let d = NegativeBinomial::new(p[0], p[1]).unwrap();
                Box::new(move |k| (d.pmf(k), d.cdf(k), d.sf(k)))
            }
            _ => {
                let d = Hypergeometric::new(p[0] as u64, p[1] as u64, p[2] as u64).unwrap();
                Box::new(move |k| (d.pmf(k), d.cdf(k), d.sf(k)))
            }
        };
        let (ppf, moments): (Vec<f64>, [f64; 4]) = match key {
            "binomial" => {
                let d = Binomial::new(p[0] as u64, p[1]).unwrap();
                (r[5].iter().map(|&q| d.ppf(q) as f64).collect(), [d.mean(), d.variance(), d.skewness(), d.excess_kurtosis()])
            }
            "poisson" => {
                let d = Poisson::new(p[0]).unwrap();
                (r[5].iter().map(|&q| d.ppf(q) as f64).collect(), [d.mean(), d.variance(), d.skewness(), d.excess_kurtosis()])
            }
            "geometric" => {
                let d = Geometric::new(p[0]).unwrap();
                (r[5].iter().map(|&q| d.ppf(q) as f64).collect(), [d.mean(), d.variance(), d.skewness(), d.excess_kurtosis()])
            }
            "nbinom" => {
                let d = NegativeBinomial::new(p[0], p[1]).unwrap();
                (r[5].iter().map(|&q| d.ppf(q) as f64).collect(), [d.mean(), d.variance(), d.skewness(), d.excess_kurtosis()])
            }
            _ => {
                let d = Hypergeometric::new(p[0] as u64, p[1] as u64, p[2] as u64).unwrap();
                (r[5].iter().map(|&q| d.ppf(q) as f64).collect(), [d.mean(), d.variance(), d.skewness(), d.excess_kurtosis()])
            }
        };
        let vals: Vec<(f64, f64, f64)> = r[1].iter().map(|&k| d(k as i64)).collect();
        let pmf: Vec<f64> = vals.iter().map(|v| v.0).collect();
        let cdf: Vec<f64> = vals.iter().map(|v| v.1).collect();
        let sf: Vec<f64> = vals.iter().map(|v| v.2).collect();
        let fitted = if key == "nbinom" {
            let sample: Vec<i64> = refs.values("sample nbinom").iter().map(|&v| v as i64).collect();
            let f = NegativeBinomial::fit(&sample).unwrap();
            fmt_diff(max_rel(&[f.r, f.p], refs.values("fit nbinom")))
        } else {
            String::new()
        };
        rows.push(vec![format!("`{label}`"), fmt_diff(max_rel(&pmf, &r[2])), fmt_diff(max_rel(&cdf, &r[3]).max(max_rel(&sf, &r[4]))), fmt_diff(max_rel(&ppf, &r[6])), fmt_diff(moment_diff(&moments, &r[7])), fitted]);
    }
    write(CHAPTER, "distributions_scipy.md", &md_table(&["distribution", "pdf / pmf", "cdf and sf", "ppf", "mean, variance, skewness, kurtosis", "maximum-likelihood fit"], &rows));
}

/// Fits four positive distributions to a sample of rainfall-like data drawn from a gamma
/// distribution, and compares them by log-likelihood and AIC.
fn model_choice() {
    let mut rng = Xoshiro256PlusPlus::from_seed(99);
    let truth = Gamma::new(1.8, 2.2).unwrap();
    let data: Vec<f64> = (0..300).map(|_| truth.sample(&mut rng)).collect();
    let candidates: Vec<(&str, Box<dyn ContinuousDistribution>, usize)> = vec![
        ("gamma", Box::new(Gamma::fit(&data).unwrap()), 2),
        ("Weibull", Box::new(Weibull::fit(&data).unwrap()), 2),
        ("log-normal", Box::new(LogNormal::fit(&data).unwrap()), 2),
        ("exponential", Box::new(Exponential::fit(&data).unwrap()), 1),
    ];
    let mut rows = Vec::new();
    let mut plot = Plot::new("Four distributions fitted to 300 gamma(1.8, 2.2) samples", "x", "density");
    plot.legend = Legend::TopRight;
    // Histogram as a step curve.
    let (bins, top) = (24usize, 16.0);
    let w = top / bins as f64;
    let mut hist = Vec::new();
    for b in 0..bins {
        let count = data.iter().filter(|&&x| x >= b as f64 * w && x < (b + 1) as f64 * w).count() as f64;
        let h = count / (data.len() as f64 * w);
        hist.push((b as f64 * w, h));
        hist.push(((b + 1) as f64 * w, h));
    }
    plot.series.push(Series::new("histogram of the sample", hist, Style::Line, 7));
    let xs: Vec<f64> = (1..=320).map(|i| i as f64 * top / 320.0).collect();
    for (i, (name, d, k)) in candidates.iter().enumerate() {
        let ll = d.log_likelihood(&data);
        let aic = 2.0 * *k as f64 - 2.0 * ll;
        rows.push(vec![name.to_string(), format!("{k}"), format!("{ll:.1}"), format!("{aic:.1}")]);
        plot.series.push(Series::new(name, xs.iter().map(|&x| (x, d.pdf(x))).collect(), Style::Line, i));
    }
    write(CHAPTER, "distributions_fit.svg", &plot.render());
    write(CHAPTER, "distributions_aic.md", &md_table(&["model", "parameters", "log-likelihood", "AIC"], &rows));
}
