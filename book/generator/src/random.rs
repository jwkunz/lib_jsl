//! Figures and tables for the random numbers chapter.

use crate::linear_algebra::time;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_random::distributions::*;
use lib_jsl_random::histogram::Histogram;
use lib_jsl_random::mcmc::{MarkovChain, MetropolisOptions, metropolis_hastings};
use lib_jsl_random::quasi_random::{HaltonSequence, SobolSequence};
use lib_jsl_random::split_mix_64::SplitMix64;
use lib_jsl_random::uniform_generator::{UniformGenerator, UniformRNG};
use lib_jsl_special_functions::{beta_cdf, binomial_cdf, chi_square_cdf, gamma, gamma_cdf, normal_cdf, poisson_cdf, student_t_cdf};
use std::f64::consts::PI;

const CHAPTER: &str = "random";
const N: usize = 100_000;

/// The Kolmogorov-Smirnov statistic of continuous samples against `cdf`.
fn ks_continuous(mut s: Vec<f64>, cdf: impl Fn(f64) -> f64) -> f64 {
    s.sort_by(f64::total_cmp);
    let n = s.len() as f64;
    s.iter().enumerate().map(|(i, &x)| {
        let f = cdf(x);
        (f - i as f64 / n).abs().max(((i + 1) as f64 / n - f).abs())
    }).fold(0.0, f64::max)
}

/// The largest difference between the empirical and exact CDFs at the integers.
fn ks_discrete(s: &[f64], cdf: impl Fn(u64) -> f64) -> f64 {
    let max = s.iter().fold(0.0f64, |m, &x| m.max(x)) as u64;
    let mut counts = vec![0usize; max as usize + 1];
    for &x in s {
        counts[x as usize] += 1;
    }
    let mut cum = 0usize;
    (0..=max).map(|k| {
        cum += counts[k as usize];
        (cum as f64 / s.len() as f64 - cdf(k)).abs()
    }).fold(0.0, f64::max)
}

fn moments(s: &[f64]) -> (f64, f64) {
    let m = s.iter().sum::<f64>() / s.len() as f64;
    (m, s.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (s.len() - 1) as f64)
}

fn audit_table() {
    let mut rng = UniformRNG::from_seed(2024);
    let mut table = String::from("| distribution | method | sample mean (exact) | sample variance (exact) | KS statistic $D$ |\n|---|---|---:|---:|---:|\n");
    let mut row = |name: &str, method: &str, s: Vec<f64>, mean: f64, var: f64, d: f64| {
        let (m, v) = moments(&s);
        if mean.is_nan() {
            // No mean or variance exists (Cauchy).
            table.push_str(&format!("| {name} | {method} | (none) | (none) | {d:.4} |\n"));
        } else {
            table.push_str(&format!("| {name} | {method} | {m:.4} ({mean:.4}) | {v:.4} ({var:.4}) | {d:.4} |\n"));
        }
    };
    let s: Vec<f64> = (0..N).map(|_| uniform_distribution(&mut rng, -1.0, 3.0)).collect();
    let d = ks_continuous(s.clone(), |x| ((x + 1.0) / 4.0).clamp(0.0, 1.0));
    row("uniform $U(-1, 3)$", "scaling", s, 1.0, 16.0 / 12.0, d);
    let s: Vec<f64> = (0..N).map(|_| exponential_distribution(&mut rng, 2.0)).collect();
    let d = ks_continuous(s.clone(), |x| 1.0 - (-2.0 * x).exp());
    row("exponential, rate 2", "inversion", s, 0.5, 0.25, d);
    let (k, l) = (1.5, 2.0);
    let s: Vec<f64> = (0..N).map(|_| weibull_distribution(&mut rng, k, l)).collect();
    let d = ks_continuous(s.clone(), |x| 1.0 - (-(x / l).powf(k)).exp());
    let wm = l * gamma(1.0 + 1.0 / k);
    row("Weibull, shape 1.5, scale 2", "inversion", s, wm, l * l * gamma(1.0 + 2.0 / k) - wm * wm, d);
    let s: Vec<f64> = (0..N).map(|_| gaussian_distribution_box_muller(&mut rng, 1.0, 2.0).0).collect();
    let d = ks_continuous(s.clone(), |x| normal_cdf(x, 1.0, 2.0));
    row("normal $N(1, 4)$", "Box-Muller", s, 1.0, 4.0, d);
    let s: Vec<f64> = (0..N).map(|_| log_normal_distribution(&mut rng, 0.0, 0.5).unwrap()).collect();
    let d = ks_continuous(s.clone(), |x| normal_cdf(x.ln(), 0.0, 0.5));
    row("log-normal, $\\mu = 0$, $\\sigma = 0.5$", "exp of a normal", s, (0.125f64).exp(), ((0.25f64).exp() - 1.0) * (0.25f64).exp(), d);
    let s: Vec<f64> = (0..N).map(|_| chi_square_distribution(&mut rng, 5, 0.0, 1.0)).collect();
    let d = ks_continuous(s.clone(), |x| chi_square_cdf(x, 5.0));
    row("chi-square, $k = 5$", "sum of squared normals", s, 5.0, 10.0, d);
    let s: Vec<f64> = (0..N).map(|_| students_t_distribution(&mut rng, 5, 0.0, 1.0)).collect();
    let d = ks_continuous(s.clone(), |x| student_t_cdf(x, 5.0));
    row("Student's $t$, $\\nu = 5$", "normal over $\\sqrt{\\chi^2/\\nu}$", s, 0.0, 5.0 / 3.0, d);
    let s: Vec<f64> = (0..N).map(|_| gamma_distribution(&mut rng, 2.5, 2.0).unwrap()).collect();
    let d = ks_continuous(s.clone(), |x| gamma_cdf(x, 2.5, 2.0));
    row("gamma, shape 2.5, scale 2", "Marsaglia-Tsang", s, 5.0, 10.0, d);
    let s: Vec<f64> = (0..N).map(|_| gamma_distribution(&mut rng, 0.3, 1.0).unwrap()).collect();
    let d = ks_continuous(s.clone(), |x| gamma_cdf(x, 0.3, 1.0));
    row("gamma, shape 0.3", "Marsaglia-Tsang with boosting", s, 0.3, 0.3, d);
    let s: Vec<f64> = (0..N).map(|_| beta_distribution(&mut rng, 2.0, 5.0).unwrap()).collect();
    let d = ks_continuous(s.clone(), |x| beta_cdf(x.clamp(0.0, 1.0), 2.0, 5.0));
    row("beta $(2, 5)$", "ratio of gammas", s, 2.0 / 7.0, 10.0 / (49.0 * 8.0), d);
    let s: Vec<f64> = (0..N).map(|_| cauchy_distribution(&mut rng, 0.0, 1.0).unwrap()).collect();
    let d = ks_continuous(s.clone(), |x| 0.5 + x.atan() / PI);
    row("Cauchy", "inversion", s, f64::NAN, f64::NAN, d);
    let s: Vec<f64> = (0..N).map(|_| bernoulli_distribution(&mut rng, 0.3)).collect();
    let d = ks_discrete(&s, |k| if k == 0 { 0.7 } else { 1.0 });
    row("Bernoulli, $p = 0.3$", "comparison", s, 0.3, 0.21, d);
    let s: Vec<f64> = (0..N).map(|_| geometric_distribution(&mut rng, 0.2)).collect();
    let d = ks_discrete(&s, |k| 1.0 - 0.8f64.powi(k as i32 + 1));
    row("geometric, $p = 0.2$ (failures)", "inversion", s, 4.0, 20.0, d);
    for lambda in [4.0, 300.0] {
        let s: Vec<f64> = (0..N).map(|_| poisson_distribution_knuth(&mut rng, lambda)).collect();
        let d = ks_discrete(&s, |k| poisson_cdf(k, lambda));
        let method = if lambda < 16.0 { "Knuth's products" } else { "Ahrens-Dieter recursion" };
        row(&format!("Poisson, $\\lambda = {lambda}$"), method, s, lambda, lambda, d);
    }
    for (n, p) in [(20u64, 0.3), (100_000, 0.4)] {
        let s: Vec<f64> = (0..N).map(|_| binomial_distribution(&mut rng, n, p).unwrap() as f64).collect();
        let d = ks_discrete(&s, |k| binomial_cdf(k, n, p));
        let method = if (n as f64) * p.min(1.0 - p) < 20.0 { "inversion" } else { "beta splitting" };
        row(&format!("binomial, $n = {n}$, $p = {p}$"), method, s, n as f64 * p, n as f64 * p * (1.0 - p), d);
    }
    table.push_str(&format!("\nWith $n = {N}$ samples the 5% critical value of $D$ is $1.36/\\sqrt n = {:.4}$.\n", 1.36 / (N as f64).sqrt()));
    write(CHAPTER, "audit.md", &table);
}

fn histogram_figure() {
    let mut rng = UniformRNG::from_seed(7);
    let mut plot = Plot::new("Histograms of 100 000 samples against the exact densities", "x", "density");
    plot.legend = Legend::TopRight;
    plot.y_range = Some((0.0, 0.85));
    type Sampler = Box<dyn Fn(&mut UniformRNG) -> f64>;
    type Density = Box<dyn Fn(f64) -> f64>;
    let cases: Vec<(&str, Sampler, Density, usize)> = vec![
        ("normal N(0, 1)", Box::new(|r: &mut UniformRNG| gaussian_distribution_box_muller(r, 0.0, 1.0).0), Box::new(|x: f64| (-x * x / 2.0).exp() / (2.0 * PI).sqrt()), 0),
        ("gamma, shape 2", Box::new(|r: &mut UniformRNG| gamma_distribution(r, 2.0, 1.0).unwrap()), Box::new(|x: f64| if x > 0.0 { x * (-x).exp() } else { 0.0 }), 1),
        ("exponential, rate 1.5", Box::new(|r: &mut UniformRNG| exponential_distribution(r, 1.5)), Box::new(|x: f64| if x >= 0.0 { 1.5 * (-1.5 * x).exp() } else { 0.0 }), 2),
    ];
    for (name, sample, pdf, color) in cases {
        let mut h = Histogram::new(-4.0, 8.0, 96);
        for _ in 0..N {
            let _ = h.count(sample(&mut rng));
        }
        let width = h.get_bin_width();
        let pts: Vec<(f64, f64)> = h.get_counts_raw_paired().iter().map(|&(x, c)| (x + width / 2.0, c as f64 / (N as f64 * width))).collect();
        plot.series.push(Series::new(&format!("{name}: histogram"), pts, Style::Markers, color));
        plot.series.push(Series::new("", (0..=600).map(|i| -4.0 + 12.0 * i as f64 / 600.0).map(|x| (x, pdf(x))).collect(), Style::Line, color));
    }
    write(CHAPTER, "histograms.svg", &plot.render());
}

fn halton_projection_figure() {
    let mut halton = HaltonSequence::new(30).unwrap();
    let mut sobol = SobolSequence::new(21).unwrap();
    let h: Vec<(f64, f64)> = (0..256).map(|_| { let p = halton.next_point(); (p[28], p[29]) }).collect();
    let s: Vec<(f64, f64)> = (0..256).map(|_| { let p = sobol.next_point(); (p[19], p[20]) }).collect();
    for (file, title, pts) in [
        ("halton_29_30.svg", "Halton, dimensions 29 and 30 (bases 109, 113)", h),
        ("sobol_20_21.svg", "Sobol, dimensions 20 and 21", s),
    ] {
        let mut plot = Plot::new(title, "", "");
        plot.x_range = Some((0.0, 1.0));
        plot.y_range = Some((0.0, 1.0));
        plot.series.push(Series::new("", pts, Style::Markers, 0));
        write(CHAPTER, file, &plot.render());
    }
}

fn qmc_error_figure() {
    // A smooth product with known integral 1: f(x) = Π_i (π/2) sin(π x_i), d = 6.
    let d = 6;
    let f = |x: &[f64]| x.iter().map(|&t| PI / 2.0 * (PI * t).sin()).product::<f64>();
    let ns: Vec<usize> = (6..=17).map(|p| 1usize << p).collect();
    let mut plot = Plot::new("Integrating Π (π/2) sin(π xᵢ) over [0, 1]⁶ (exact value 1)", "number of points N", "absolute error");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::BottomLeft;
    let mut rng = UniformRNG::from_seed(9);
    let (mut mc, mut halton, mut sobol) = (vec![], vec![], vec![]);
    for &n in &ns {
        // Pseudo-random: the root-mean-square error over 20 independent runs.
        let rms = ((0..20).map(|_| {
            let s: f64 = (0..n).map(|_| f(&(0..d).map(|_| rng.next_f64()).collect::<Vec<_>>())).sum::<f64>() / n as f64;
            (s - 1.0).powi(2)
        }).sum::<f64>() / 20.0).sqrt();
        mc.push((n as f64, rms));
        let mut h = HaltonSequence::new(d).unwrap();
        halton.push((n as f64, ((0..n).map(|_| f(&h.next_point())).sum::<f64>() / n as f64 - 1.0).abs()));
        let mut s = SobolSequence::new(d).unwrap();
        sobol.push((n as f64, ((0..n).map(|_| f(&s.next_point())).sum::<f64>() / n as f64 - 1.0).abs()));
    }
    plot.series.push(Series::new("pseudo-random (RMS of 20 runs)", mc, Style::LineMarkers, 0));
    plot.series.push(Series::new("Halton", halton, Style::LineMarkers, 1));
    plot.series.push(Series::new("Sobol", sobol, Style::LineMarkers, 2));
    plot.series.push(Series::new("N^(−1/2) and N^(−1)", vec![], Style::Dashed, 7));
    plot.series.push(Series::new("", ns.iter().map(|&n| (n as f64, 0.5 / (n as f64).sqrt())).collect(), Style::Dashed, 7));
    plot.series.push(Series::new("", ns.iter().map(|&n| (n as f64, 1.0 / n as f64)).collect(), Style::Dashed, 7));
    write(CHAPTER, "qmc_error.svg", &plot.render());
}

fn speed_table() {
    let reps = 10_000_000usize;
    let mut rng = UniformRNG::from_seed(1);
    let mut sm = SplitMix64::new(1);
    let ns = |t: f64| t / reps as f64 * 1e9;
    let mut table = String::from("| operation | time per sample |\n|---|---:|\n");
    let mut row = |name: &str, t: f64| table.push_str(&format!("| {name} | {:.1} ns |\n", ns(t)));
    row("xoshiro256++ `next_u64`", time(|| { let mut a = 0u64; for _ in 0..reps { a ^= rng.next_u64(); } std::hint::black_box(a); }));
    row("xoshiro256++ `next_f64`", time(|| { let mut a = 0.0; for _ in 0..reps { a += rng.next_f64(); } std::hint::black_box(a); }));
    row("SplitMix64 `next_u64`", time(|| { let mut a = 0u64; for _ in 0..reps { a ^= sm.next_u64(); } std::hint::black_box(a); }));
    row("exponential (inversion)", time(|| { let mut a = 0.0; for _ in 0..reps { a += exponential_distribution(&mut rng, 1.0); } std::hint::black_box(a); }));
    row("normal (Box-Muller, one pair per two samples)", time(|| { let mut a = 0.0; for _ in 0..reps / 2 { let (x, y) = gaussian_distribution_box_muller(&mut rng, 0.0, 1.0); a += x + y; } std::hint::black_box(a); }));
    row("gamma, shape 2.5 (Marsaglia-Tsang)", time(|| { let mut a = 0.0; for _ in 0..reps { a += gamma_distribution(&mut rng, 2.5, 1.0).unwrap(); } std::hint::black_box(a); }));
    row("Poisson, $\\lambda = 4$", time(|| { let mut a = 0.0; for _ in 0..reps { a += poisson_distribution_knuth(&mut rng, 4.0); } std::hint::black_box(a); }));
    write(CHAPTER, "speed.md", &table);
}

/// The integrated autocorrelation time τ = 1 + 2 Σ ρ(k), summed until ρ first drops below 0.05.
fn autocorrelation_time(chain: &MarkovChain) -> f64 {
    let mut tau = 1.0;
    for lag in 1..chain.samples.len() / 2 {
        let r = chain.autocorrelation(0, lag);
        if r < 0.05 {
            break;
        }
        tau += 2.0 * r;
    }
    tau
}

fn mcmc_figures() {
    // A standard normal target, sampled with three fixed random-walk steps.
    let log_density = |x: &[f64]| -0.5 * x[0] * x[0];
    let scales = [(0.1, "step 0.1 (too small)"), (2.4, "step 2.4 (about right)"), (25.0, "step 25 (too large)")];
    let mut trace = Plot::new("Random-walk Metropolis on N(0, 1): the first 400 states", "iteration", "x");
    trace.legend = Legend::BottomRight;
    trace.y_range = Some((-4.0, 6.0));
    let mut acf = Plot::new("Autocorrelation of the chains", "lag", "autocorrelation");
    acf.legend = Legend::TopRight;
    let mut table = String::from("| proposal step | acceptance rate | autocorrelation time $\\tau$ | effective samples (of 50 000) | sample mean (exact 0) |\n|---|---:|---:|---:|---:|\n");
    for (k, &(scale, name)) in scales.iter().enumerate() {
        let options = MetropolisOptions { proposal_scale: vec![scale], samples: 50_000, burn_in: 0, thin: 1, adapt_during_burn_in: false };
        let chain = metropolis_hastings(log_density, vec![4.0], options, &mut UniformRNG::from_seed(31)).unwrap();
        trace.series.push(Series::new(name, chain.samples.iter().take(400).enumerate().map(|(i, s)| (i as f64, s[0])).collect(), Style::Line, k));
        acf.series.push(Series::new(name, (0..=100).map(|lag| (lag as f64, chain.autocorrelation(0, lag))).collect(), Style::Line, k));
        let tau = autocorrelation_time(&chain);
        table.push_str(&format!("| {name} | {:.3} | {tau:.1} | {:.0} | {:+.3} |\n", chain.acceptance_rate, 50_000.0 / tau, chain.mean()[0]));
    }
    let options = MetropolisOptions { proposal_scale: vec![0.1], samples: 50_000, burn_in: 5_000, thin: 1, adapt_during_burn_in: true };
    let chain = metropolis_hastings(log_density, vec![4.0], options, &mut UniformRNG::from_seed(31)).unwrap();
    let tau = autocorrelation_time(&chain);
    table.push_str(&format!("| adapted from 0.1 during 5000 burn-in steps: {:.2} | {:.3} | {tau:.1} | {:.0} | {:+.3} |\n", chain.proposal_scale[0], chain.acceptance_rate, 50_000.0 / tau, chain.mean()[0]));
    write(CHAPTER, "mcmc_trace.svg", &trace.render());
    write(CHAPTER, "mcmc_acf.svg", &acf.render());
    write(CHAPTER, "mcmc_table.md", &table);

    // A curved "banana" target: x ~ N(0, 4), y | x ~ N(x²/4, 1).
    let banana = |p: &[f64]| -p[0] * p[0] / 8.0 - 0.5 * (p[1] - p[0] * p[0] / 4.0).powi(2);
    let options = MetropolisOptions { proposal_scale: vec![0.5, 0.5], samples: 4_000, burn_in: 5_000, thin: 10, adapt_during_burn_in: true };
    let chain = metropolis_hastings(banana, vec![0.0, 0.0], options, &mut UniformRNG::from_seed(41)).unwrap();
    let mut plot = Plot::new("4000 samples (thinned by 10) from a banana-shaped density", "x", "y");
    plot.legend = Legend::TopLeft;
    plot.x_range = Some((-7.0, 7.0));
    plot.y_range = Some((-3.0, 12.0));
    plot.series.push(Series::new("samples", chain.samples.iter().map(|s| (s[0], s[1])).collect(), Style::Markers, 0));
    plot.series.push(Series::new("ridge y = x²/4", (0..=100).map(|i| -7.0 + 14.0 * i as f64 / 100.0).map(|x| (x, x * x / 4.0)).collect(), Style::Line, 1));
    write(CHAPTER, "mcmc_banana.svg", &plot.render());
}

pub fn generate() {
    audit_table();
    histogram_figure();
    halton_projection_figure();
    qmc_error_figure();
    mcmc_figures();
    speed_table();
}
