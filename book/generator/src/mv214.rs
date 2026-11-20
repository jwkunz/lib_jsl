//! Tables and figures for release 2.14.0 (statistics chapter): multivariate normal and t
//! distribution functions, singular covariances, Dunnett's test, quasi-Monte Carlo normal
//! samples and Poisson-disk sampling, compared with SciPy.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_random::qmc::{Hypersphere, PoissonDisk};
use lib_jsl_random::quasi_random::SobolSequence;
use lib_jsl_random::uniform_generator::{UniformGenerator, UniformRNG};
use lib_jsl_statistics::multivariate::{MultivariateNormal, MultivariateT};
use lib_jsl_statistics::mvn_cdf::{DunnettAlternative, MvCdfOptions, bivariate_normal_upper, dunnett};
use lib_jsl_statistics::qmc_normal::MultivariateNormalQmc;

const CHAPTER: &str = "statistics";

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/mv214_reference.txt"));
    table(&refs);
    figures(&refs);
}

fn square(v: &[f64]) -> Vec<Vec<f64>> {
    let d = (v.len() as f64).sqrt() as usize;
    v.chunks(d).map(|c| c.to_vec()).collect()
}

fn sci(x: f64) -> String {
    format!("{x:.7}")
}

fn table(refs: &Refs) {
    let mut rows = Vec::new();
    let bv = refs.values("bvn");
    let mut worst = 0.0f64;
    for c in bv.chunks(4) {
        // P(X ≤ h, Y ≤ k) = bvnu(−h, −k, r).
        let p = bivariate_normal_upper(-c[0], -c[1], c[2]);
        worst = worst.max((p - c[3]).abs());
    }
    rows.push(vec!["Bivariate normal (exact), 6 cases with r from −0.97 to 0.999".into(), "—".into(), fmt_diff(worst), "both exact".into()]);
    let mut rng = UniformRNG::from_seed(214);
    for (key, d) in [("mvn3", 3usize), ("mvn5", 5), ("mvn8", 8)] {
        let w = refs.get(key);
        let mvn = MultivariateNormal::new(vec![0.0; d], square(&w[0])).unwrap();
        let (p, e) = mvn.cdf_between(Some(&w[2]), &w[1], MvCdfOptions::default(), &mut rng).unwrap();
        let (scipy, truth) = (w[3][0], w[3][1]);
        rows.push(vec![
            format!("Normal, d = {d}, a box"),
            format!("{} (±{})", sci(p), fmt_diff(e)),
            fmt_diff((p - truth).abs()),
            fmt_diff((scipy - truth).abs()),
        ]);
    }
    for (key, d) in [("mvt2", 2usize), ("mvt3", 3), ("mvt5", 5)] {
        let w = refs.get(key);
        let t = MultivariateT::new(vec![0.0; d], square(&w[0]), w[2][0]).unwrap();
        let (p, e) = t.cdf_between(None, &w[1], MvCdfOptions::default(), &mut rng).unwrap();
        let (scipy, truth) = (w[2][1], w[2][2]);
        rows.push(vec![
            format!("t, d = {d}, ν = {}, 1000 d points", w[2][0]),
            format!("{} (±{})", sci(p), fmt_diff(e)),
            fmt_diff((p - truth).abs()),
            fmt_diff((scipy - truth).abs()),
        ]);
    }
    write(CHAPTER, "mv214_cdf.md", &md_table(&["Distribution function", "Library (error estimate)", "Library vs reference", "SciPy vs reference"], &rows));

    let mut rows = Vec::new();
    let w = refs.get("singular");
    let m = MultivariateNormal::new_allow_singular(vec![1.0, 0.0, -1.0, 0.5], square(&w[0])).unwrap();
    let x = &w[1];
    let off: Vec<f64> = x.iter().enumerate().map(|(i, v)| if i == 1 { v + 0.1 } else { *v }).collect();
    rows.push(vec![
        "Singular normal (rank 2 in 4-D): logpdf, entropy; a point off the support".into(),
        format!("{} / {}; {}", fmt_diff((m.logpdf(x) - w[2][0]).abs()), fmt_diff((m.entropy() - w[2][1]).abs()), if m.logpdf(&off) == f64::NEG_INFINITY && w[2][2] == f64::NEG_INFINITY { "−∞ in both" } else { "differ" }),
    ]);
    let control = [7.40, 8.50, 7.20, 8.24, 9.84, 8.32];
    let a = [9.76, 8.80, 7.68, 9.36];
    let b = [12.80, 9.68, 12.16, 9.20, 10.55];
    for (key, alt) in [("dunnett two-sided", DunnettAlternative::TwoSided), ("dunnett greater", DunnettAlternative::Greater), ("dunnett less", DunnettAlternative::Less)] {
        let r = dunnett(&[&a, &b], &control, alt, &mut rng).unwrap();
        let w = refs.get(key);
        let ds = r.statistic.iter().zip(&w[0]).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        let dp = r.pvalue.iter().zip(&w[1]).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        rows.push(vec![format!("`dunnett`, {}: statistics; p-values (QMC)", &key[8..]), format!("{}; {} (p = {:.4}, {:.4})", fmt_diff(ds), fmt_diff(dp), r.pvalue[0], r.pvalue[1])]);
    }
    let cq = vec![vec![2.0, 0.6, 0.1], vec![0.6, 1.0, -0.3], vec![0.1, -0.3, 0.5]];
    let mut g = MultivariateNormalQmc::new(vec![1.0, -1.0, 0.0], &cq, true, SobolSequence::new(3).unwrap()).unwrap();
    let s: Vec<f64> = g.random(16).into_iter().flatten().collect();
    let dq = s.iter().zip(refs.values("mvnqmc")).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
    let mut g = MultivariateNormalQmc::new(vec![1.0, -1.0, 0.0], &cq, false, SobolSequence::new(4).unwrap()).unwrap();
    let s: Vec<f64> = g.random(16).into_iter().skip(1).flatten().collect();
    let db = s.iter().zip(refs.values("mvnqmc bm")).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
    rows.push(vec!["`MultivariateNormalQmc`, unscrambled Sobol, 16 samples: inverse transform; Box-Muller".into(), format!("{}; {}", fmt_diff(dq), fmt_diff(db))]);
    let counts = refs.get("poisson counts");
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let lib_counts: Vec<f64> = (0..5)
        .map(|s| PoissonDisk::new(2, 0.05, Hypersphere::Volume, 30, None).unwrap().fill_space(&mut UniformRNG::from_seed(s)).len() as f64)
        .collect();
    let lib_surface: Vec<f64> = (0..5)
        .map(|s| PoissonDisk::new(2, 0.05, Hypersphere::Surface, 30, None).unwrap().fill_space(&mut UniformRNG::from_seed(s)).len() as f64)
        .collect();
    rows.push(vec![
        "`PoissonDisk`, r = 0.05, `fill_space`: points (mean of 5 seeds), volume; surface".into(),
        format!("{:.0} vs {:.0}; {:.0} vs {:.0}", mean(&lib_counts), mean(&counts[0]), mean(&lib_surface), mean(&counts[1])),
    ]);
    write(CHAPTER, "mv214_other.md", &md_table(&["Function", "Difference from SciPy"], &rows));
}

fn figures(refs: &Refs) {
    let mut pd = PoissonDisk::new(2, 0.05, Hypersphere::Volume, 30, None).unwrap();
    let mut rng = UniformRNG::from_seed(1);
    let pts = pd.fill_space(&mut rng);
    let n = pts.len();
    let mut plot = Plot::new(&format!("Poisson-disk sample (r = 0.05, {n} points) and {n} independent uniform points"), "x", "y");
    plot.legend = Legend::TopRight;
    plot.x_range = Some((0.0, 2.15));
    plot.y_range = Some((0.0, 1.22));
    plot.series.push(Series::new("Poisson disk", pts.iter().map(|p| (p[0], p[1])).collect(), Style::Markers, 0));
    let unif: Vec<(f64, f64)> = (0..n).map(|_| (1.15 + rng.next_f64(), rng.next_f64())).collect();
    plot.series.push(Series::new("uniform (shifted right)", unif, Style::Markers, 1));
    write(CHAPTER, "poisson_disk.svg", &plot.render());

    // Error of the 5-D normal box probability against the number of points: lattice QMC
    // versus plain Monte Carlo sampling.
    let w = refs.get("mvn5");
    let truth = w[3][1];
    let mvn = MultivariateNormal::new(vec![0.0; 5], square(&w[0])).unwrap();
    let mut qmc = Vec::new();
    let mut mc = Vec::new();
    for &m in &[1000usize, 3000, 10_000, 30_000, 100_000, 300_000] {
        let mut errs = Vec::new();
        let mut errs_mc = Vec::new();
        for s in 0..6u64 {
            let mut rng = UniformRNG::from_seed(100 + s);
            let (p, _) = mvn.cdf_between(Some(&w[2]), &w[1], MvCdfOptions { maxpts: Some(m), abseps: 0.0 }, &mut rng).unwrap();
            errs.push((p - truth).abs());
            let hits = (0..m)
                .filter(|_| {
                    let x = mvn.sample(&mut rng);
                    x.iter().zip(&w[1]).zip(&w[2]).all(|((v, hi), lo)| v <= hi && v > lo)
                })
                .count();
            errs_mc.push((hits as f64 / m as f64 - truth).abs());
        }
        let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt().max(1e-12);
        qmc.push((m as f64, rms(&errs)));
        mc.push((m as f64, rms(&errs_mc)));
    }
    let mut plot = Plot::new("A 5-D normal box probability: randomised lattice rules against Monte Carlo", "points", "RMS error (6 seeds)");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::BottomLeft;
    plot.series.push(Series::new("Genz's lattice QMC", qmc, Style::LineMarkers, 0));
    plot.series.push(Series::new("plain Monte Carlo", mc, Style::LineMarkers, 1));
    write(CHAPTER, "mvn_qmc_convergence.svg", &plot.render());
}
