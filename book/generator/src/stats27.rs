//! Tables for release 2.7.0 (statistics chapter): the Rice and multivariate distributions,
//! Cramér-von Mises, Brunner-Munzel, the studentized range and Tukey's HSD, odds ratios,
//! relative risk and association, compared with SciPy.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::write;
use lib_jsl_special_functions::normal_quantile;
use lib_jsl_statistics::comparisons::{BrunnerMunzelDistribution, brunnermunzel};
use lib_jsl_statistics::contingency::{AssociationMethod, OddsRatioKind, association, odds_ratio, relative_risk};
use lib_jsl_statistics::distributions::{ContinuousDistribution, Fit, Rice};
use lib_jsl_statistics::goodness_of_fit::{CvmMethod, cramervonmises, cramervonmises_2samp};
use lib_jsl_statistics::hypothesis_tests::Alternative;
use lib_jsl_statistics::multivariate::*;
use lib_jsl_statistics::posthoc::{StudentizedRange, tukey_hsd};

const CHAPTER: &str = "statistics";

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/stats27_reference.txt"));
    distributions_table(&refs);
    tests_table(&refs);
}

fn rel(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).fold(0.0f64, |m, (g, w)| {
        let d = if g.is_finite() || w.is_finite() { (g - w).abs() / w.abs().max(1e-300) } else { 0.0 };
        m.max(d)
    })
}

fn distributions_table(refs: &Refs) {
    let mut rows = Vec::new();
    let xs = [0.1, 0.8, 1.5, 2.6, 3.9, 6.0];
    let mut worst = [0.0f64; 4]; // density/cdf/sf, ppf, moments
    for (b, scale) in [(0.0, 1.3), (1.7, 0.8), (6.0, 0.5)] {
        let r = Rice::new(b * scale, scale).unwrap();
        let w = refs.get(&format!("rice {b} {scale}"));
        let pdf: Vec<f64> = xs.iter().map(|&x| r.pdf(x)).collect();
        let cdf: Vec<f64> = xs.iter().map(|&x| r.cdf(x)).collect();
        let sf: Vec<f64> = xs.iter().map(|&x| r.sf(x)).collect();
        // sf at x = 6 is compared with direct integration below.
        worst[0] = worst[0].max(rel(&pdf, &w[0])).max(rel(&cdf, &w[1])).max(rel(&sf[..5], &w[2][..5]));
        let ppf: Vec<f64> = [0.01, 0.5, 0.99].iter().map(|&p| r.ppf(p)).collect();
        worst[1] = worst[1].max(rel(&ppf, &w[3]));
        let m = [r.mean(), r.variance(), r.skewness(), r.excess_kurtosis()];
        worst[2] = worst[2].max(rel(&m, &w[4]));
    }
    let r = Rice::new(20.0, 0.5).unwrap();
    let w = refs.get("rice tails");
    let tails = rel(&[r.cdf(18.0), r.cdf(19.0)], &w[0]).max(rel(&[r.sf(21.5), r.sf(22.0)], &w[1]));
    // ν/σ = 40: SciPy's moments overflow to NaN; the library's approach the normal limit.
    let normal_gap = (r.mean() - (20.0 + 0.25 / 40.0)).abs() / 20.0;
    let sample: Vec<f64> = (0..60).map(|i| Rice::new(3.0, 1.5).unwrap().ppf((i as f64 + 0.5) / 60.0)).collect();
    let f = Rice::fit(&sample).unwrap();
    let wf = refs.get("rice fit");
    let fit = rel(&[f.nu, f.sigma], &wf[0]);
    let ll_gain = f.log_likelihood(&sample) - wf[1][0];
    assert!(ll_gain >= -1e-9, "the library's fit should be at least as likely as SciPy's");
    let wt = refs.get("rice tail quad");
    let lib_tail: Vec<f64> = [(0.0, 1.3), (1.7, 0.8), (6.0, 0.5)].iter().map(|&(b, s)| Rice::new(b * s, s).unwrap().sf(6.0)).collect();
    rows.push(vec!["`Rice`: pdf, cdf, sf (ν/σ = 0, 1.7, 6)".into(), fmt_diff(worst[0])]);
    rows.push(vec![format!("`Rice`: sf at x = 6 (10⁻⁵ to 10⁻⁹) against direct integration of the density (SciPy: {})", fmt_diff(rel(&wt[1], &wt[0]))), fmt_diff(rel(&lib_tail, &wt[0]))]);
    rows.push(vec!["`Rice`: ppf; mean, variance, skewness, kurtosis".into(), fmt_diff(worst[1].max(worst[2]))]);
    rows.push(vec!["`Rice`, ν/σ = 40: cdf and sf in the tails (SciPy's moments are NaN here)".into(), fmt_diff(tails)]);
    rows.push(vec![format!("`Rice::fit` (location 0): log-likelihood {ll_gain:.0e} above SciPy's, whose optimiser stops earlier"), fmt_diff(fit)]);
    rows.push(vec!["`Rice`, ν/σ = 40: mean against the limit ν + σ²/(2ν)".into(), fmt_diff(normal_gap)]);

    let mean = vec![1.0, -0.5, 2.0];
    let cov = vec![vec![2.0, 0.3, -0.4], vec![0.3, 1.5, 0.2], vec![-0.4, 0.2, 1.0]];
    let pts = [[0.0, 0.0, 0.0], [1.5, -1.0, 2.5], [3.0, 2.0, -1.0]];
    let mvn = MultivariateNormal::new(mean.clone(), cov.clone()).unwrap();
    let w = refs.get("mvn");
    let d1 = rel(&pts.iter().map(|p| mvn.logpdf(p)).collect::<Vec<_>>(), &w[0]).max(rel(&[mvn.entropy()], &w[1]));
    rows.push(vec!["`MultivariateNormal`: logpdf, entropy".into(), fmt_diff(d1)]);
    let mvt = MultivariateT::new(mean, cov, 4.5).unwrap();
    let w = refs.get("mvt");
    let d2 = rel(&pts.iter().map(|p| mvt.logpdf(p)).collect::<Vec<_>>(), &w[0]).max(rel(&[mvt.entropy()], &w[1]));
    rows.push(vec!["`MultivariateT`: logpdf, entropy".into(), fmt_diff(d2)]);
    let d = Dirichlet::new(vec![1.5, 2.0, 0.7, 3.2]).unwrap();
    let w = refs.get("dirichlet");
    let d3 = rel(&[d.logpdf(&[0.2, 0.3, 0.1, 0.4]).unwrap()], &w[0]).max(rel(&d.mean(), &w[1])).max(rel(&d.var(), &w[2])).max(rel(&[d.entropy()], &w[3]));
    rows.push(vec!["`Dirichlet`: logpdf, mean, variance, entropy".into(), fmt_diff(d3)]);
    let s = vec![vec![2.0, 0.5], vec![0.5, 1.0]];
    let x = vec![vec![3.0, 1.0], vec![1.0, 2.5]];
    let wi = Wishart::new(5.5, s.clone()).unwrap();
    let w = refs.get("wishart");
    let d4 = rel(&[wi.logpdf(&x).unwrap()], &w[0]).max(rel(&wi.var().concat(), &w[1])).max(rel(&[wi.entropy()], &w[2]));
    rows.push(vec!["`Wishart`: logpdf, variance, entropy".into(), fmt_diff(d4)]);
    let iw = InvWishart::new(6.0, s).unwrap();
    let w = refs.get("invwishart");
    let d5 = rel(&[iw.logpdf(&x).unwrap()], &w[0]).max(rel(&iw.mean().unwrap().concat(), &w[1])).max(rel(&iw.mode().concat(), &w[2]));
    rows.push(vec!["`InvWishart`: logpdf, mean, mode".into(), fmt_diff(d5)]);
    let m = Multinomial::new(10, vec![0.2, 0.5, 0.3]).unwrap();
    let w = refs.get("multinomial");
    let d6 = rel(&[m.logpmf(&[2, 5, 3]), m.logpmf(&[0, 10, 0])], &w[0]).max(rel(&m.cov().concat(), &w[1]));
    rows.push(vec!["`Multinomial`: logpmf, covariance".into(), fmt_diff(d6)]);
    write(CHAPTER, "stats27_distributions.md", &md_table(&["distribution and quantities", "largest relative difference from SciPy"], &rows));
}

fn tests_table(refs: &Refs) {
    let mut rows = Vec::new();
    let norm = |x: f64| 0.5 * lib_jsl_special_functions::error_function::erfc(-x / std::f64::consts::SQRT_2);
    let x10: Vec<f64> = (0..10).map(|i| normal_quantile((i as f64 + 0.3) / 10.0, 0.0, 1.0) * 1.1 + 0.2).collect();
    let x50: Vec<f64> = (0..50).map(|i| (i as f64 * 1.7).sin() * 1.5 + 0.1).collect();
    let mut cvm = 0.0f64;
    for (name, data) in [("10", &x10), ("50", &x50)] {
        let r = cramervonmises(data, norm).unwrap();
        cvm = cvm.max(rel(&[r.statistic, r.p_value], refs.values(&format!("cvm {name}"))));
    }
    rows.push(vec!["`cramervonmises` (n = 10, 50)".into(), fmt_diff(cvm)]);
    let a = [1.2, 3.4, 0.5, 2.2, 2.2, 4.1, 0.9];
    let b = [2.5, 3.9, 4.4, 1.8, 5.0, 3.3, 2.2, 6.1, 4.7];
    let r = cramervonmises_2samp(&a, &b, CvmMethod::Auto).unwrap();
    let e1 = rel(&[r.statistic, r.p_value], refs.values("cvm2 exact"));
    let r = cramervonmises_2samp(&a, &b, CvmMethod::Asymptotic).unwrap();
    let e2 = rel(&[r.statistic, r.p_value], refs.values("cvm2 asym small"));
    let ba: Vec<f64> = (0..25).map(|i| (i as f64 * 0.9).sin() * 2.0).collect();
    let bb: Vec<f64> = (0..30).map(|i| (i as f64 * 1.3).cos() * 2.0 + 0.6).collect();
    let r = cramervonmises_2samp(&ba, &bb, CvmMethod::Auto).unwrap();
    let e3 = rel(&[r.statistic, r.p_value], refs.values("cvm2 asym"));
    rows.push(vec!["`cramervonmises_2samp`: exact (7 and 9, with ties), asymptotic (7 and 9; 25 and 30)".into(), fmt_diff(e1.max(e2).max(e3))]);
    let x1 = [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 1.0, 1.0];
    let y1 = [3.0, 3.0, 4.0, 3.0, 1.0, 2.0, 3.0, 1.0, 1.0, 5.0, 4.0];
    let mut bm = 0.0f64;
    for (alt, name) in [(Alternative::TwoSided, "two-sided"), (Alternative::Less, "less"), (Alternative::Greater, "greater")] {
        let r = brunnermunzel(&x1, &y1, alt, BrunnerMunzelDistribution::T).unwrap();
        bm = bm.max(rel(&[r.statistic, r.p_value], refs.values(&format!("bm {name}"))));
    }
    let r = brunnermunzel(&x1, &y1, Alternative::TwoSided, BrunnerMunzelDistribution::Normal).unwrap();
    bm = bm.max(rel(&[r.statistic, r.p_value], refs.values("bm normal")));
    rows.push(vec!["`brunnermunzel` (three alternatives; t and normal)".into(), fmt_diff(bm)]);
    let cases = [(3.5, 3.0, 10.0), (2.0, 5.0, 20.0), (4.8, 4.0, 60.0), (1.2, 2.0, 5.0), (5.5, 6.0, f64::INFINITY), (3.0, 10.0, 200.0)];
    let dists: Vec<StudentizedRange> = cases.iter().map(|&(_, k, df)| StudentizedRange::new(k, df).unwrap()).collect();
    let cdf = rel(&cases.iter().zip(&dists).map(|(c, d)| d.cdf(c.0)).collect::<Vec<_>>(), refs.values("srange cdf"));
    let pdf = rel(&cases.iter().zip(&dists).map(|(c, d)| d.pdf(c.0)).collect::<Vec<_>>(), refs.values("srange pdf"));
    let ppf = rel(&dists.iter().map(|d| d.ppf(0.95)).collect::<Vec<_>>(), refs.values("srange ppf"));
    rows.push(vec!["`StudentizedRange`: cdf, pdf, 95 % point (k = 2–10, ν = 5–∞)".into(), fmt_diff(cdf.max(pdf).max(ppf))]);
    let groups = vec![vec![24.5, 23.5, 26.4, 27.1, 29.9], vec![28.4, 34.2, 29.5, 32.2, 30.1, 33.0], vec![26.1, 28.3, 24.3, 26.2, 27.8], vec![31.0, 29.8, 33.5, 32.6]];
    let mut tk = 0.0f64;
    let mut tk_p = 0.0f64;
    for eq in [true, false] {
        let r = tukey_hsd(&groups, eq).unwrap();
        let (lo, hi) = r.confidence_interval(0.95).unwrap();
        let w = refs.get(&format!("tukey {}", if eq { "True" } else { "False" }));
        tk = tk.max(rel(&r.statistic.concat(), &w[0]).max(rel(&lo.concat(), &w[2])).max(rel(&hi.concat(), &w[3])));
        // p-values: absolute differences (they are probabilities).
        tk_p = tk_p.max(r.p_values.concat().iter().zip(&w[1]).fold(0.0f64, |m, (g, v)| m.max((g - v).abs())));
    }
    rows.push(vec!["`tukey_hsd` (4 groups, unequal sizes; Tukey-Kramer and Games-Howell): differences, intervals".into(), fmt_diff(tk)]);
    rows.push(vec!["`tukey_hsd`: p-values (absolute difference)".into(), fmt_diff(tk_p)]);
    let tables = [("a", [[12, 5], [3, 15]]), ("b", [[7, 2], [9, 30]]), ("zero", [[0, 6], [5, 9]])];
    let mut ors = 0.0f64;
    for (name, t) in tables {
        for (kind, kname) in [(OddsRatioKind::Conditional, "conditional"), (OddsRatioKind::Sample, "sample")] {
            let r = odds_ratio(t, kind);
            let mut vals = vec![r.statistic];
            for alt in [Alternative::TwoSided, Alternative::Less, Alternative::Greater] {
                let (l, h) = r.confidence_interval(0.95, alt).unwrap();
                vals.push(l);
                vals.push(h);
            }
            let vals: Vec<f64> = vals.iter().map(|v| if v.is_nan() { -1.0 } else if v.is_infinite() { 1e300 } else { *v }).collect();
            ors = ors.max(rel(&vals, refs.values(&format!("or {name} {kname}"))));
        }
    }
    rows.push(vec!["`odds_ratio`, conditional and sample, with intervals for three alternatives (one table with a zero)".into(), fmt_diff(ors)]);
    let rr = relative_risk(27, 122, 44, 487).unwrap();
    let (l, h) = rr.confidence_interval(0.9).unwrap();
    rows.push(vec!["`relative_risk` with Katz's interval".into(), fmt_diff(rel(&[rr.relative_risk, l, h], refs.values("rr")))]);
    let t34 = vec![vec![10, 4, 6, 2], vec![3, 12, 5, 8], vec![6, 2, 9, 11]];
    let assoc: Vec<f64> = [AssociationMethod::Cramer, AssociationMethod::Tschuprow, AssociationMethod::Pearson].iter().map(|&m| association(&t34, m).unwrap()).collect();
    rows.push(vec!["`association`: Cramér's V, Tschuprow's T, Pearson's C (3 × 4 table)".into(), fmt_diff(rel(&assoc, refs.values("association")))]);
    write(CHAPTER, "stats27_tests.md", &md_table(&["test or estimate", "largest relative difference from SciPy"], &rows));
}
