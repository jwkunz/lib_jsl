//! Figures and tables for the statistics chapter.

use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_random::distributions::*;
use lib_jsl_random::uniform_generator::{UniformGenerator, UniformRNG};
use lib_jsl_special_functions::{normal_cdf, normal_pdf, student_t_quantile};
use lib_jsl_statistics::bootstrap::bootstrap;
use lib_jsl_statistics::correlation::{kendall, pearson, spearman};
use lib_jsl_statistics::descriptive::*;
use lib_jsl_statistics::hypothesis_tests::*;
use lib_jsl_statistics::kernel_density::{Bandwidth, GaussianKde};

const CHAPTER: &str = "statistics";

pub fn generate() {
    scipy_table();
    variance_accuracy_figure();
    robustness_figures();
    anscombe();
    size_table();
    power_figures();
    kde_figure();
    kde_error_table();
    bootstrap_coverage_table();
    bootstrap_figure();
}

fn parse(v: &str) -> Vec<f64> {
    if v.is_empty() { Vec::new() } else { v.split(',').map(|t| t.parse().unwrap()).collect() }
}

fn alternative(name: &str) -> Alternative {
    match name {
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        _ => Alternative::TwoSided,
    }
}

/// The library against `scipy.stats` on the cases of `data/stats_reference.txt`.
fn scipy_table() {
    let reference = include_str!("../data/stats_reference.txt");
    let mut tests = String::from("| case | statistic (SciPy) | p-value | SciPy p-value | relative difference of p |\n|---|---:|---:|---:|---:|\n");
    let mut descriptive = String::from("| quantity | library | SciPy / NumPy | relative difference |\n|---|---:|---:|---:|\n");
    for line in reference.lines() {
        let f: Vec<&str> = line.split('|').collect();
        let (case, function, options) = (f[0], f[1], f[2]);
        let x = parse(f[3]);
        let rows: Vec<Vec<f64>> = f[4].split(';').map(parse).collect();
        let y = &rows[0];
        let (statistic, p): (f64, f64) = match function {
            "t1" => { let r = t_test_one_sample(&x, 0.0, alternative(options)).unwrap(); (r.statistic, r.p_value) }
            "t2" => { let r = t_test_two_sample(&x, y, options == "pooled", Alternative::TwoSided).unwrap(); (r.statistic, r.p_value) }
            "tp" => { let r = t_test_paired(&x, y, Alternative::TwoSided).unwrap(); (r.statistic, r.p_value) }
            "gof" => { let r = chi_square_goodness_of_fit(&x, if y.is_empty() { None } else { Some(y) }).unwrap(); (r.statistic, r.p_value) }
            "ind" => { let r = chi_square_independence(&[x.clone(), y.clone()]).unwrap(); (r.statistic, r.p_value) }
            "anova" => { let mut groups = vec![x.clone()]; groups.extend(rows.iter().cloned()); let r = anova_one_way(&groups).unwrap().0; (r.statistic, r.p_value) }
            "ks1" => { let r = ks_test_one_sample(&x, |v| normal_cdf(v, 0.0, 1.0)).unwrap(); (r.statistic, r.p_value) }
            "ks2" => { let r = ks_test_two_sample(&x, y).unwrap(); (r.statistic, r.p_value) }
            "mw" => { let r = mann_whitney_u(&x, y, alternative(options)).unwrap(); (r.statistic, r.p_value) }
            "pearson" => { let r = pearson(&x, y).unwrap(); (r.coefficient, r.p_value) }
            "spearman" => { let r = spearman(&x, y).unwrap(); (r.coefficient, r.p_value) }
            "kendall" => { let r = kendall(&x, y).unwrap(); (r.coefficient, r.p_value) }
            "var" => (variance(&x, 1).unwrap(), f64::NAN),
            "skew" => (skewness(&x).unwrap(), f64::NAN),
            "kurt" => (excess_kurtosis(&x).unwrap(), f64::NAN),
            "quantile" => (quantile(&x, options.parse().unwrap()).unwrap(), f64::NAN),
            "mad" => (median_absolute_deviation(&x, 1.482_602_218_505_602).unwrap(), f64::NAN),
            "kde" => (GaussianKde::new(&x, Bandwidth::Scott).unwrap().pdf(options.parse().unwrap()), f64::NAN),
            other => panic!("unknown reference function {other}"),
        };
        let (ref_statistic, ref_p): (f64, f64) = (f[5].parse().unwrap(), f[6].parse().unwrap());
        let relative = |a: f64, b: f64| if a == b { "0".to_string() } else { format!("{:.1e}", ((a - b) / b).abs()) };
        if ref_p.is_nan() {
            descriptive.push_str(&format!("| {case} | {statistic:.6} | {ref_statistic:.6} | {} |\n", relative(statistic, ref_statistic)));
        } else {
            assert!(((statistic - ref_statistic) / ref_statistic).abs() < 1e-9, "{case}: statistic {statistic} vs {ref_statistic}");
            tests.push_str(&format!("| {case} | {statistic:.4} | {p:.5} | {ref_p:.5} | {} |\n", relative(p, ref_p)));
        }
    }
    write(CHAPTER, "scipy_tests.md", &tests);
    write(CHAPTER, "scipy_descriptive.md", &descriptive);
}

fn normal(rng: &mut UniformRNG) -> f64 {
    gaussian_distribution_box_muller(rng, 0.0, 1.0).0
}

fn normals(rng: &mut UniformRNG, n: usize, mean: f64, sd: f64) -> Vec<f64> {
    (0..n).map(|_| mean + sd * normal(rng)).collect()
}

/// Relative error of three variance algorithms for data `c + u` with `u` uniform on `[0, 1)`.
fn variance_accuracy_figure() {
    let mut rng = UniformRNG::from_seed(11);
    let u: Vec<f64> = (0..1000).map(|_| rng.next_f64()).collect();
    let mut series = [Vec::new(), Vec::new(), Vec::new()];
    for e in 0..=24 {
        let c = 10f64.powf(e as f64 / 2.0);
        let x: Vec<f64> = u.iter().map(|v| c + v).collect();
        // x − c is exact (Sterbenz), so the variance of the stored data is known to full precision.
        let d: Vec<f64> = x.iter().map(|v| v - c).collect();
        let exact = variance(&d, 0).unwrap();
        let n = x.len() as f64;
        let (s1, s2) = x.iter().fold((0.0, 0.0), |(a, b), v| (a + v, b + v * v));
        let naive = (s2 - s1 * s1 / n) / n;
        let (mut m, mut m2) = (0.0, 0.0);
        for (k, v) in x.iter().enumerate() {
            let delta = v - m;
            m += delta / (k + 1) as f64;
            m2 += delta * (v - m);
        }
        let welford = m2 / n;
        let library = variance(&x, 0).unwrap();
        for (s, value) in series.iter_mut().zip([naive, welford, library]) {
            s.push((c, ((value - exact) / exact).abs().clamp(1e-17, 1e3)));
        }
    }
    let mut plot = Plot::new("Variance of 1000 values c + U(0, 1): relative error", "offset c", "relative error");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::TopLeft;
    let [naive, welford, library] = series;
    plot.series.push(Series::new("one pass, Σx² − (Σx)²/n", naive, Style::LineMarkers, 1));
    plot.series.push(Series::new("Welford's update", welford, Style::LineMarkers, 2));
    plot.series.push(Series::new("library (corrected two-pass)", library, Style::LineMarkers, 0));
    write(CHAPTER, "variance_accuracy.svg", &plot.render());
}

/// Location and scale estimates of N(0, 1) samples as a fraction are replaced by outliers.
fn robustness_figures() {
    let mut rng = UniformRNG::from_seed(5);
    let reps = 200;
    let n = 200;
    let (mut means, mut medians, mut sds, mut mads, mut iqrs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for step in 0..=24 {
        let fraction = step as f64 * 0.02;
        let mut acc = [0.0; 5];
        for _ in 0..reps {
            let mut x = normals(&mut rng, n, 0.0, 1.0);
            let bad = (fraction * n as f64).round() as usize;
            for v in x.iter_mut().take(bad) {
                *v = 10.0 + normal(&mut rng);
            }
            let values = [mean(&x).unwrap(), median(&x).unwrap(), standard_deviation(&x, 1).unwrap(), median_absolute_deviation(&x, 1.4826).unwrap(), interquartile_range(&x).unwrap() / 1.349];
            for (a, v) in acc.iter_mut().zip(values) {
                *a += v / reps as f64;
            }
        }
        means.push((fraction, acc[0]));
        medians.push((fraction, acc[1]));
        sds.push((fraction, acc[2]));
        mads.push((fraction, acc[3]));
        iqrs.push((fraction, acc[4]));
    }
    let mut plot = Plot::new("Location of N(0, 1) data with outliers near 10", "fraction of outliers", "estimate (true value 0)");
    plot.legend = Legend::TopLeft;
    plot.series.push(Series::new("mean", means, Style::LineMarkers, 1));
    plot.series.push(Series::new("median", medians, Style::LineMarkers, 0));
    write(CHAPTER, "robust_location.svg", &plot.render());
    let mut plot = Plot::new("Scale of N(0, 1) data with outliers near 10", "fraction of outliers", "estimate (true value 1)");
    plot.legend = Legend::TopLeft;
    plot.series.push(Series::new("standard deviation", sds, Style::LineMarkers, 1));
    plot.series.push(Series::new("1.4826 × MAD", mads, Style::LineMarkers, 0));
    plot.series.push(Series::new("IQR / 1.349", iqrs, Style::LineMarkers, 2));
    write(CHAPTER, "robust_scale.svg", &plot.render());
}

fn anscombe() {
    let x123 = [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0];
    let x4 = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 19.0, 8.0, 8.0, 8.0];
    let sets: [(&str, &[f64; 11], [f64; 11]); 4] = [
        ("I", &x123, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
        ("II", &x123, [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
        ("III", &x123, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
        ("IV", &x4, [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
    ];
    let mut table = String::from("| set | mean of $y$ | variance of $y$ | Pearson $r$ | Spearman $\\rho$ | Kendall $\\tau_b$ |\n|---|---:|---:|---:|---:|---:|\n");
    for (name, x, y) in &sets {
        let (r, rho, tau) = (pearson(*x, y).unwrap(), spearman(*x, y).unwrap(), kendall(*x, y).unwrap());
        table.push_str(&format!(
            "| {name} | {:.2} | {:.2} | {:.3} (p = {:.4}) | {:.3} (p = {:.4}) | {:.3} (p = {:.4}) |\n",
            mean(y).unwrap(), variance(y, 1).unwrap(), r.coefficient, r.p_value, rho.coefficient, rho.p_value, tau.coefficient, tau.p_value
        ));
        let mut plot = Plot::new(&format!("Anscombe's set {name}"), "x", "y");
        plot.x_range = Some((2.0, 20.0));
        plot.y_range = Some((2.0, 14.0));
        plot.series.push(Series::new("", vec![(2.0, 4.0), (20.0, 13.0)], Style::Dashed, 7));
        plot.series.push(Series::new("", x.iter().copied().zip(y.iter().copied()).collect(), Style::Markers, 0));
        write(CHAPTER, &format!("anscombe_{}.svg", name.to_lowercase()), &plot.render());
    }
    write(CHAPTER, "anscombe.md", &table);
}

/// Rejection rates at the 5% level when the null hypothesis is true.
fn size_table() {
    let mut rng = UniformRNG::from_seed(2025);
    let reps = 20_000;
    let mut table = String::from("| test | data (null hypothesis true) | rejection rate at 5% |\n|---|---|---:|\n");
    let mut row = |name: &str, data: &str, rng: &mut UniformRNG, p_value: &mut dyn FnMut(&mut UniformRNG) -> f64| {
        let rejected = (0..reps).filter(|_| p_value(rng) < 0.05).count();
        table.push_str(&format!("| {name} | {data} | {:.4} |\n", rejected as f64 / reps as f64));
    };
    let two = Alternative::TwoSided;
    row("one-sample $t$", "normal, $n = 10$", &mut rng, &mut |r| t_test_one_sample(&normals(r, 10, 0.0, 1.0), 0.0, two).unwrap().p_value);
    row("one-sample $t$", "exponential, $n = 10$", &mut rng, &mut |r| {
        let x: Vec<f64> = (0..10).map(|_| exponential_distribution(r, 1.0)).collect();
        t_test_one_sample(&x, 1.0, two).unwrap().p_value
    });
    row("one-sample $t$", "exponential, $n = 100$", &mut rng, &mut |r| {
        let x: Vec<f64> = (0..100).map(|_| exponential_distribution(r, 1.0)).collect();
        t_test_one_sample(&x, 1.0, two).unwrap().p_value
    });
    row("Student's two-sample $t$", "normal, $n = 10, 10$, equal variances", &mut rng, &mut |r| t_test_two_sample(&normals(r, 10, 0.0, 1.0), &normals(r, 10, 0.0, 1.0), true, two).unwrap().p_value);
    row("Student's two-sample $t$", "normal, $n = 10, 40$, $\\sigma = 3, 1$", &mut rng, &mut |r| t_test_two_sample(&normals(r, 10, 0.0, 3.0), &normals(r, 40, 0.0, 1.0), true, two).unwrap().p_value);
    row("Welch's $t$", "normal, $n = 10, 40$, $\\sigma = 3, 1$", &mut rng, &mut |r| t_test_two_sample(&normals(r, 10, 0.0, 3.0), &normals(r, 40, 0.0, 1.0), false, two).unwrap().p_value);
    row("paired $t$", "normal pairs, correlation 0.8, $n = 15$", &mut rng, &mut |r| {
        let z: Vec<(f64, f64)> = (0..15).map(|_| { let (a, b) = (normal(r), normal(r)); (a, 0.8 * a + 0.6 * b) }).collect();
        let (x, y): (Vec<f64>, Vec<f64>) = z.into_iter().unzip();
        t_test_paired(&x, &y, two).unwrap().p_value
    });
    row("Mann-Whitney $U$", "normal, $n = 10, 10$", &mut rng, &mut |r| mann_whitney_u(&normals(r, 10, 0.0, 1.0), &normals(r, 10, 0.0, 1.0), two).unwrap().p_value);
    row("Mann-Whitney $U$", "Cauchy, $n = 20, 30$", &mut rng, &mut |r| {
        let x: Vec<f64> = (0..20).map(|_| cauchy_distribution(r, 0.0, 1.0).unwrap()).collect();
        let y: Vec<f64> = (0..30).map(|_| cauchy_distribution(r, 0.0, 1.0).unwrap()).collect();
        mann_whitney_u(&x, &y, two).unwrap().p_value
    });
    row("Kolmogorov-Smirnov, one sample", "uniform, $n = 50$", &mut rng, &mut |r| {
        let x: Vec<f64> = (0..50).map(|_| r.next_f64()).collect();
        ks_test_one_sample(&x, |v| v.clamp(0.0, 1.0)).unwrap().p_value
    });
    row("Kolmogorov-Smirnov, two samples", "normal, $n = 50, 50$", &mut rng, &mut |r| ks_test_two_sample(&normals(r, 50, 0.0, 1.0), &normals(r, 50, 0.0, 1.0)).unwrap().p_value);
    row("$\\chi^2$ goodness of fit", "60 rolls of a fair die", &mut rng, &mut |r| {
        let mut counts = [0.0; 6];
        for _ in 0..60 {
            counts[((r.next_f64() * 6.0) as usize).min(5)] += 1.0;
        }
        chi_square_goodness_of_fit(&counts, None).unwrap().p_value
    });
    row("$\\chi^2$ independence", "$2 \\times 3$ table, 200 independent cases", &mut rng, &mut |r| {
        let mut table = vec![vec![0.0; 3]; 2];
        for _ in 0..200 {
            let i = (r.next_f64() < 0.3) as usize;
            let u = r.next_f64();
            let j = if u < 0.2 { 0 } else if u < 0.7 { 1 } else { 2 };
            table[i][j] += 1.0;
        }
        chi_square_independence(&table).unwrap().p_value
    });
    row("one-way ANOVA", "three normal groups of 8", &mut rng, &mut |r| anova_one_way(&[normals(r, 8, 0.0, 1.0), normals(r, 8, 0.0, 1.0), normals(r, 8, 0.0, 1.0)]).unwrap().0.p_value);
    row("Pearson $r$", "independent normals, $n = 20$", &mut rng, &mut |r| pearson(&normals(r, 20, 0.0, 1.0), &normals(r, 20, 0.0, 1.0)).unwrap().p_value);
    row("Spearman $\\rho$", "independent normals, $n = 20$", &mut rng, &mut |r| spearman(&normals(r, 20, 0.0, 1.0), &normals(r, 20, 0.0, 1.0)).unwrap().p_value);
    row("Kendall $\\tau_b$", "independent normals, $n = 20$", &mut rng, &mut |r| kendall(&normals(r, 20, 0.0, 1.0), &normals(r, 20, 0.0, 1.0)).unwrap().p_value);
    table.push_str(&format!("\nEach rate is over {reps} simulated data sets; its Monte Carlo standard error is about 0.0015.\n"));
    write(CHAPTER, "size.md", &table);
}

/// Power of three two-sample tests against a location shift, for normal and Cauchy data.
fn power_figures() {
    let mut rng = UniformRNG::from_seed(99);
    let reps = 4000;
    let n = 20;
    for (file, title, cauchy, max_shift) in [("power_normal.svg", "Power against a shift: normal data, n = 20 + 20", false, 1.5), ("power_cauchy.svg", "Power against a shift: Cauchy data, n = 20 + 20", true, 3.0)] {
        let mut curves = [Vec::new(), Vec::new(), Vec::new()];
        for step in 0..=15 {
            let shift = max_shift * step as f64 / 15.0;
            let mut rejected = [0usize; 3];
            for _ in 0..reps {
                let draw = |r: &mut UniformRNG, m: f64| -> Vec<f64> {
                    (0..n).map(|_| if cauchy { cauchy_distribution(r, m, 1.0).unwrap() } else { m + normal(r) }).collect()
                };
                let (x, y) = (draw(&mut rng, 0.0), draw(&mut rng, shift));
                let p = [
                    t_test_two_sample(&x, &y, false, Alternative::TwoSided).unwrap().p_value,
                    mann_whitney_u(&x, &y, Alternative::TwoSided).unwrap().p_value,
                    ks_test_two_sample(&x, &y).unwrap().p_value,
                ];
                for (c, p) in rejected.iter_mut().zip(p) {
                    *c += (p < 0.05) as usize;
                }
            }
            for (curve, c) in curves.iter_mut().zip(rejected) {
                curve.push((shift, c as f64 / reps as f64));
            }
        }
        let mut plot = Plot::new(title, "shift (in units of the scale)", "probability of rejecting at 5%");
        plot.legend = Legend::BottomRight;
        plot.y_range = Some((0.0, 1.0));
        let [t, u, ks] = curves;
        plot.series.push(Series::new("Welch t", t, Style::LineMarkers, 0));
        plot.series.push(Series::new("Mann-Whitney U", u, Style::LineMarkers, 1));
        plot.series.push(Series::new("Kolmogorov-Smirnov", ks, Style::LineMarkers, 2));
        write(CHAPTER, file, &plot.render());
    }
}

/// A bimodal mixture `0.6 N(0, 1) + 0.4 N(3, 0.5²)`.
fn mixture_pdf(x: f64) -> f64 {
    0.6 * normal_pdf(x, 0.0, 1.0) + 0.4 * normal_pdf(x, 3.0, 0.5)
}

fn mixture_sample(rng: &mut UniformRNG, n: usize) -> Vec<f64> {
    (0..n).map(|_| if rng.next_f64() < 0.6 { normal(rng) } else { 3.0 + 0.5 * normal(rng) }).collect()
}

fn kde_figure() {
    let mut rng = UniformRNG::from_seed(3);
    let data = mixture_sample(&mut rng, 200);
    let grid: Vec<f64> = (0..=300).map(|i| -4.0 + 9.0 * i as f64 / 300.0).collect();
    let mut plot = Plot::new("Gaussian kernel density estimates from 200 samples", "x", "density");
    plot.series.push(Series::new("true density", grid.iter().map(|&x| (x, mixture_pdf(x))).collect(), Style::Line, 6));
    for (bandwidth, color) in [(Bandwidth::Fixed(0.05), 7), (Bandwidth::Silverman, 0), (Bandwidth::Scott, 1)] {
        let kde = GaussianKde::new(&data, bandwidth).unwrap();
        let name = match bandwidth {
            Bandwidth::Fixed(_) => "h = 0.05 (too small)".to_string(),
            Bandwidth::Silverman => format!("Silverman, h = {:.2}", kde.bandwidth()),
            Bandwidth::Scott => format!("Scott, h = {:.2}", kde.bandwidth()),
        };
        plot.series.push(Series::new(&name, grid.iter().map(|&x| (x, kde.pdf(x))).collect(), Style::Line, color));
    }
    write(CHAPTER, "kde.svg", &plot.render());
}

/// Mean integrated squared error of the bandwidth rules, and of the best fixed bandwidth.
fn kde_error_table() {
    let mut rng = UniformRNG::from_seed(8);
    let grid: Vec<f64> = (0..=400).map(|i| -6.0 + 12.0 * i as f64 / 400.0).collect();
    let dx = 12.0 / 400.0;
    let ise = |kde: &GaussianKde, pdf: &dyn Fn(f64) -> f64| grid.iter().map(|&x| (kde.pdf(x) - pdf(x)).powi(2)).sum::<f64>() * dx;
    let candidates: Vec<f64> = (0..40).map(|i| 0.03 * 1.1f64.powi(i)).collect();
    let mut table = String::from("| density | $n$ | Scott | Silverman | best fixed $h$ |\n|---|---:|---:|---:|---:|\n");
    let normal_pdf01 = |x: f64| normal_pdf(x, 0.0, 1.0);
    type Density<'a> = &'a dyn Fn(f64) -> f64;
    let cases: [(&str, Density, bool); 2] = [("normal", &normal_pdf01, false), ("bimodal mixture", &mixture_pdf, true)];
    for (name, pdf, bimodal) in cases {
        for n in [50, 500, 5000] {
            let reps = 20;
            let (mut scott, mut silverman) = (0.0, 0.0);
            let mut fixed = vec![0.0; candidates.len()];
            for _ in 0..reps {
                let data = if bimodal { mixture_sample(&mut rng, n) } else { normals(&mut rng, n, 0.0, 1.0) };
                scott += ise(&GaussianKde::new(&data, Bandwidth::Scott).unwrap(), pdf) / reps as f64;
                silverman += ise(&GaussianKde::new(&data, Bandwidth::Silverman).unwrap(), pdf) / reps as f64;
                for (f, &h) in fixed.iter_mut().zip(&candidates) {
                    *f += ise(&GaussianKde::new(&data, Bandwidth::Fixed(h)).unwrap(), pdf) / reps as f64;
                }
            }
            let (best, h) = fixed.iter().zip(&candidates).fold((f64::INFINITY, 0.0), |(b, bh), (&f, &h)| if f < b { (f, h) } else { (b, bh) });
            table.push_str(&format!("| {name} | {n} | {scott:.2e} | {silverman:.2e} | {best:.2e} ($h = {h:.2}$) |\n"));
        }
    }
    table.push_str("\nMean integrated squared error $\\int(\\hat f - f)^2\\,dx$ over 20 samples.\n");
    write(CHAPTER, "kde_error.md", &table);
}

fn sample_sd(x: &[f64]) -> f64 {
    standard_deviation(x, 1).unwrap()
}

fn sample_mean(x: &[f64]) -> f64 {
    mean(x).unwrap()
}

/// Coverage of nominal 90% intervals for the mean and standard deviation of exponential data.
fn bootstrap_coverage_table() {
    let mut rng = UniformRNG::from_seed(17);
    let trials = 1000;
    let resamples = 1000;
    let mut table = String::from("| statistic (true value) | $n$ | $t$ interval | percentile | BCa |\n|---|---:|---:|---:|---:|\n");
    for (name, statistic, truth) in [("mean (1)", sample_mean as fn(&[f64]) -> f64, 1.0), ("standard deviation (1)", sample_sd as fn(&[f64]) -> f64, 1.0)] {
        for n in [10, 30, 100] {
            let mut covered = [0usize; 3];
            for trial in 0..trials {
                let data: Vec<f64> = (0..n).map(|_| exponential_distribution(&mut rng, 1.0)).collect();
                let b = bootstrap(&data, statistic, resamples, trial as u64).unwrap();
                let (plo, phi) = b.percentile_interval(0.9);
                let (blo, bhi) = b.bca_interval(&data, statistic, 0.9);
                let half = student_t_quantile(0.95, (n - 1) as f64) * standard_error(&data).unwrap();
                let m = mean(&data).unwrap();
                covered[0] += (m - half <= truth && truth <= m + half) as usize;
                covered[1] += (plo <= truth && truth <= phi) as usize;
                covered[2] += (blo <= truth && truth <= bhi) as usize;
            }
            let t = if name.starts_with("mean") { format!("{:.3}", covered[0] as f64 / trials as f64) } else { "—".to_string() };
            table.push_str(&format!("| {name} | {n} | {t} | {:.3} | {:.3} |\n", covered[1] as f64 / trials as f64, covered[2] as f64 / trials as f64));
        }
    }
    table.push_str(&format!("\nFraction of {trials} exponential samples whose nominal 90% interval ({resamples} resamples) contains the true value; the Monte Carlo standard error is about 0.01.\n"));
    write(CHAPTER, "bootstrap_coverage.md", &table);
}

fn bootstrap_figure() {
    let mut rng = UniformRNG::from_seed(23);
    let data: Vec<f64> = (0..30).map(|_| exponential_distribution(&mut rng, 1.0)).collect();
    let b = bootstrap(&data, sample_sd, 10_000, 1).unwrap();
    let (plo, phi) = b.percentile_interval(0.9);
    let (blo, bhi) = b.bca_interval(&data, sample_sd, 0.9);
    let (lo, hi, bins) = (0.2, 2.2, 50);
    let width = (hi - lo) / bins as f64;
    let mut counts = vec![0.0; bins];
    for &r in &b.replicates {
        let k = ((r - lo) / width).floor();
        if k >= 0.0 && (k as usize) < bins {
            counts[k as usize] += 1.0;
        }
    }
    let scale = 1.0 / (b.replicates.len() as f64 * width);
    let mut hist = Vec::new();
    for (k, c) in counts.iter().enumerate() {
        let (a, z) = (lo + k as f64 * width, lo + (k + 1) as f64 * width);
        hist.extend([(a, 0.0), (a, c * scale), (z, c * scale), (z, 0.0)]);
    }
    let top = counts.iter().fold(0.0f64, |m, &c| m.max(c)) * scale * 1.1;
    let mut plot = Plot::new("Bootstrap replicates of the standard deviation (exponential, n = 30)", "standard deviation", "density");
    plot.y_range = Some((0.0, top * 1.15));
    plot.series.push(Series::new("10 000 replicates", hist, Style::Line, 7));
    let vline = |x: f64| vec![(x, 0.0), (x, top)];
    plot.series.push(Series::new(&format!("estimate {:.2}", b.estimate), vline(b.estimate), Style::Line, 6));
    plot.series.push(Series::new(&format!("percentile [{plo:.2}, {phi:.2}]"), vline(plo), Style::Dashed, 1));
    plot.series.push(Series::new("", vline(phi), Style::Dashed, 1));
    plot.series.push(Series::new(&format!("BCa [{blo:.2}, {bhi:.2}]"), vline(blo), Style::Line, 0));
    plot.series.push(Series::new("", vline(bhi), Style::Line, 0));
    plot.series.push(Series::new("true value 1", vline(1.0), Style::Dashed, 2));
    write(CHAPTER, "bootstrap_sd.svg", &plot.render());
}
