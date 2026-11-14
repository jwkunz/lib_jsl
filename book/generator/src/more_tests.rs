//! Tables for "More hypothesis tests" (the statistics chapter): every test against SciPy, the
//! power of the normality tests, and false discoveries with and without adjustment.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::write;
use lib_jsl_random::xoshiro256plusplus::Xoshiro256PlusPlus;
use lib_jsl_statistics::comparisons::*;
use lib_jsl_statistics::distributions::{ContinuousDistribution, Gamma, Normal, StudentT, Uniform};
use lib_jsl_statistics::hypothesis_tests::{Alternative, TestResult, t_test_one_sample};
use lib_jsl_statistics::normality::*;

const CHAPTER: &str = "statistics";

pub fn generate() {
    scipy_reference();
    power_table();
    fdr_table();
}

fn rel(a: f64, b: f64) -> f64 {
    if a == b { 0.0 } else { (a - b).abs() / b.abs().max(1e-300) }
}

fn scipy_reference() {
    let refs = Refs::parse(include_str!("../data/more_tests_reference.txt"));
    let data = |name: &str| refs.values(&format!("data {name}")).to_vec();
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row = |label: String, got: TestResult, key: &str| {
        let want = refs.values(key);
        rows.push(vec![label, fmt_diff(rel(got.statistic, want[0])), fmt_diff(rel(got.p_value, want[1])), format!("{:.4}", want[1])]);
    };
    for name in ["normal", "skewed", "small", "tiny"] {
        row(format!("`shapiro`, {name} sample (n = {})", data(name).len()), shapiro(&data(name)).unwrap(), &format!("shapiro {name}"));
    }
    for name in ["normal", "skewed"] {
        row(format!("`jarque_bera`, {name}"), jarque_bera(&data(name)).unwrap(), &format!("jarque_bera {name}"));
        row(format!("`normaltest` (D'Agostino-Pearson), {name}"), normaltest(&data(name)).unwrap(), &format!("normaltest {name}"));
        row(format!("`skewtest`, {name}"), skewtest(&data(name), Alternative::TwoSided).unwrap(), &format!("skewtest {name}"));
        row(format!("`kurtosistest`, {name}"), kurtosistest(&data(name), Alternative::TwoSided).unwrap(), &format!("kurtosistest {name}"));
    }
    let groups = vec![data("g1"), data("g2"), data("g3")];
    row("`levene` (median centred), three groups".into(), levene(&groups).unwrap(), "levene groups");
    row("`bartlett`, three groups".into(), bartlett(&groups).unwrap(), "bartlett groups");
    row("`fligner`, three groups".into(), fligner(&groups).unwrap(), "fligner groups");
    row("`kruskal`, three groups".into(), kruskal(&groups).unwrap(), "kruskal groups");
    let b = data("blocks");
    let treatments: Vec<Vec<f64>> = (0..4).map(|j| (0..10).map(|i| b[i * 4 + j]).collect()).collect();
    row("`friedman`, 4 treatments × 10 blocks, with ties".into(), friedman(&treatments).unwrap(), "friedman blocks");
    for (alt, key) in [(Alternative::TwoSided, "two-sided"), (Alternative::Less, "less"), (Alternative::Greater, "greater")] {
        row(format!("`wilcoxon`, n = 20, exact, {key}"), wilcoxon(&data("diffs"), alt).unwrap(), &format!("wilcoxon exact {key}"));
    }
    row("`wilcoxon`, n = 12 with ties (all sign changes)".into(), wilcoxon(&data("diffs_ties"), Alternative::TwoSided).unwrap(), "wilcoxon ties");
    row("`wilcoxon`, n = 40 with ties (normal approximation)".into(), wilcoxon(&data("diffs_large_ties"), Alternative::TwoSided).unwrap(), "wilcoxon asymptotic");
    for (name, table) in [("a", [[8, 2], [1, 5]]), ("b", [[12, 5], [9, 14]]), ("c", [[0, 7], [6, 3]])] {
        for (alt, key) in [(Alternative::TwoSided, "two-sided"), (Alternative::Less, "less"), (Alternative::Greater, "greater")] {
            row(format!("`fisher_exact` {table:?}, {key}"), fisher_exact(table, alt).unwrap(), &format!("fisher {name} {key}"));
        }
    }
    for (k, n, p) in [(7u64, 20u64, 0.5), (3, 40, 0.2), (61, 100, 0.5)] {
        row(format!("`binomtest` k = {k}, n = {n}, p = {p}"), binomtest(k, n, p, Alternative::TwoSided).unwrap(), &format!("binomtest {k} {n} {p}"));
    }
    let (x, y) = (data("perm_x"), data("perm_y"));
    let diff = |a: &[f64], b: &[f64]| a.iter().sum::<f64>() / a.len() as f64 - b.iter().sum::<f64>() / b.len() as f64;
    for (alt, key) in [(Alternative::TwoSided, "two-sided"), (Alternative::Less, "less"), (Alternative::Greater, "greater")] {
        let r = permutation_test(&x, &y, diff, alt, 10_000, 1).unwrap();
        row(format!("`permutation_test`, difference of means, 6435 splits, {key}"), TestResult { statistic: r.statistic, p_value: r.p_value, degrees_of_freedom: f64::NAN }, &format!("permutation {key}"));
    }
    // Anderson-Darling and the FDR adjustments have no single p-value.
    let mut extra = Vec::new();
    for name in ["normal", "skewed"] {
        let a = anderson_normal(&data(name)).unwrap();
        let want = refs.get(&format!("anderson {name}"));
        let cv = a.critical_values.iter().zip(&want[1]).fold(0.0f64, |m, (g, w)| m.max(rel(*g, *w)));
        extra.push(vec![format!("`anderson_normal`, {name}: statistic, critical values"), fmt_diff(rel(a.statistic, want[0][0])), fmt_diff(cv), String::new()]);
    }
    let p = data("pvalues");
    for (m, key) in [(Adjustment::BenjaminiHochberg, "bh"), (Adjustment::BenjaminiYekutieli, "by")] {
        let a = adjust_p_values(&p, m).unwrap();
        let want = refs.values(&format!("fdr {key}"));
        extra.push(vec![format!("`adjust_p_values`, {m:?} (`false_discovery_control`)"), String::new(), fmt_diff(a.iter().zip(want).fold(0.0f64, |m, (g, w)| m.max(rel(*g, *w)))), String::new()]);
    }
    rows.extend(extra);
    write(CHAPTER, "more_tests_scipy.md", &md_table(&["test (SciPy reference)", "statistic", "p-value", "SciPy's p-value"], &rows));
}

/// Rejection rates at 5% of four normality tests on 2000 samples of size 50 from each
/// distribution.
fn power_table() {
    let mut rng = Xoshiro256PlusPlus::from_seed(31);
    let reps = 2000;
    let n = 50;
    let sources: Vec<(&str, Box<dyn ContinuousDistribution>)> = vec![
        ("normal (the null hypothesis)", Box::new(Normal::new(0.0, 1.0).unwrap())),
        ("Student t, 5 degrees of freedom (heavy tails)", Box::new(StudentT::new(5.0, 0.0, 1.0).unwrap())),
        ("gamma, shape 4 (skewed)", Box::new(Gamma::new(4.0, 1.0).unwrap())),
        ("uniform (light tails)", Box::new(Uniform::new(0.0, 1.0).unwrap())),
    ];
    let mut rows = Vec::new();
    for (name, d) in sources {
        let mut rejected = [0usize; 4];
        for _ in 0..reps {
            let x: Vec<f64> = (0..n).map(|_| {
                use lib_jsl_random::uniform_generator::UniformGenerator;
                let mut u = rng.next_f64();
                while u <= 0.0 || u >= 1.0 {
                    u = rng.next_f64();
                }
                d.ppf(u)
            }).collect();
            if shapiro(&x).unwrap().p_value < 0.05 {
                rejected[0] += 1;
            }
            if anderson_normal(&x).unwrap().statistic > anderson_normal(&x).unwrap().critical_values[2] {
                rejected[1] += 1;
            }
            if normaltest(&x).unwrap().p_value < 0.05 {
                rejected[2] += 1;
            }
            if jarque_bera(&x).unwrap().p_value < 0.05 {
                rejected[3] += 1;
            }
        }
        let pct = |c: usize| format!("{:.1}%", 100.0 * c as f64 / reps as f64);
        rows.push(vec![name.to_string(), pct(rejected[0]), pct(rejected[1]), pct(rejected[2]), pct(rejected[3])]);
    }
    write(CHAPTER, "more_tests_power.md", &md_table(&["samples of 50 from", "Shapiro-Wilk", "Anderson-Darling", "D'Agostino-Pearson", "Jarque-Bera"], &rows));
}

/// 1000 one-sample t-tests, 900 of true nulls and 100 of means shifted by 0.6, with n = 20:
/// discoveries at 5% with each adjustment.
fn fdr_table() {
    use lib_jsl_random::uniform_generator::UniformGenerator;
    let mut rng = Xoshiro256PlusPlus::from_seed(8);
    let std = Normal::new(0.0, 1.0).unwrap();
    let mut gauss = || {
        let mut u = rng.next_f64();
        while u <= 0.0 || u >= 1.0 {
            u = rng.next_f64();
        }
        std.ppf(u)
    };
    let mut p = Vec::new();
    let mut truth = Vec::new();
    for i in 0..1000 {
        let shift = if i < 100 { 0.6 } else { 0.0 };
        let x: Vec<f64> = (0..20).map(|_| gauss() + shift).collect();
        p.push(t_test_one_sample(&x, 0.0, Alternative::TwoSided).unwrap().p_value);
        truth.push(shift > 0.0);
    }
    let mut rows = Vec::new();
    for (label, adj) in [("none", None), ("Bonferroni", Some(Adjustment::Bonferroni)), ("Holm", Some(Adjustment::Holm)), ("Benjamini-Hochberg", Some(Adjustment::BenjaminiHochberg)), ("Benjamini-Yekutieli", Some(Adjustment::BenjaminiYekutieli))] {
        let q = match adj {
            None => p.clone(),
            Some(a) => adjust_p_values(&p, a).unwrap(),
        };
        let (mut tp, mut fp) = (0, 0);
        for (qi, &t) in q.iter().zip(&truth) {
            if *qi <= 0.05 {
                if t { tp += 1 } else { fp += 1 }
            }
        }
        let fdp = if tp + fp > 0 { 100.0 * fp as f64 / (tp + fp) as f64 } else { 0.0 };
        rows.push(vec![label.to_string(), format!("{}", tp + fp), format!("{tp}"), format!("{fp}"), format!("{fdp:.1}%")]);
    }
    write(CHAPTER, "more_tests_fdr.md", &md_table(&["adjustment", "rejections", "true effects found (of 100)", "false rejections (of 900 nulls)", "share of rejections that are false"], &rows));
}
