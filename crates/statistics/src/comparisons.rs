//! More tests comparing samples, following `scipy.stats`: equal variances (Levene, Bartlett,
//! Fligner-Killeen), rank tests (Kruskal-Wallis, Friedman, Wilcoxon signed-rank), exact tests
//! (Fisher's exact test for 2 × 2 tables, the binomial test), permutation tests, and the
//! adjustment of p-values for multiple comparisons (Benjamini-Hochberg, Benjamini-Yekutieli,
//! Holm, Bonferroni).
//!
//! See the book entry [More hypothesis tests](https://jwkunz.github.io/lib_jsl/book/statistics/more-tests.html).

use crate::correlation::rank;
use crate::distributions::{DiscreteDistribution, Hypergeometric, Binomial};
use crate::hypothesis_tests::{Alternative, TestResult};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_random::uniform_generator::UniformGenerator;
use lib_jsl_random::xoshiro256plusplus::Xoshiro256PlusPlus;
use lib_jsl_special_functions::{chi_square_survival, f_cdf, normal_cdf, normal_quantile};

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn median(x: &[f64]) -> f64 {
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
}

fn var(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() as f64 - 1.0)
}

fn check_groups(groups: &[Vec<f64>]) -> Result<(), ErrorsJSL> {
    if groups.len() < 2 || groups.iter().any(|g| g.len() < 2) {
        return Err(ErrorsJSL::InvalidInputRange("need at least two groups of at least two values"));
    }
    Ok(())
}

/// `Σ (t³ − t)` over the groups of tied values.
fn tie_sum(values: &[f64]) -> f64 {
    let mut s = values.to_vec();
    s.sort_by(f64::total_cmp);
    let mut total = 0.0;
    let mut i = 0;
    while i < s.len() {
        let mut j = i;
        while j + 1 < s.len() && s[j + 1] == s[i] {
            j += 1;
        }
        let t = (j - i + 1) as f64;
        total += t * t * t - t;
        i = j + 1;
    }
    total
}

// ------------------------------------------------------------------------------ equal variances

// ANCHOR: levene
/// Levene's test of equal variances with Brown and Forsythe's median centring (SciPy's `levene`
/// with `center='median'`): one-way ANOVA on the absolute deviations from each group's median,
/// referred to the F distribution with `k − 1` and `N − k` degrees of freedom.
pub fn levene(groups: &[Vec<f64>]) -> Result<TestResult, ErrorsJSL> {
    check_groups(groups)?;
    let z: Vec<Vec<f64>> = groups.iter().map(|g| {
        let m = median(g);
        g.iter().map(|v| (v - m).abs()).collect()
    }).collect();
    let k = groups.len() as f64;
    let n_total: f64 = z.iter().map(|g| g.len() as f64).sum();
    let grand = z.iter().flatten().sum::<f64>() / n_total;
    let between: f64 = z.iter().map(|g| g.len() as f64 * (mean(g) - grand).powi(2)).sum();
    let within: f64 = z.iter().map(|g| {
        let m = mean(g);
        g.iter().map(|v| (v - m).powi(2)).sum::<f64>()
    }).sum();
    let w = (n_total - k) / (k - 1.0) * between / within;
    Ok(TestResult { statistic: w, p_value: 1.0 - f_cdf(w, k - 1.0, n_total - k), degrees_of_freedom: k - 1.0 })
}
// ANCHOR_END: levene

/// Bartlett's test of equal variances for normal samples (SciPy's `bartlett`), χ² with `k − 1`
/// degrees of freedom.
pub fn bartlett(groups: &[Vec<f64>]) -> Result<TestResult, ErrorsJSL> {
    check_groups(groups)?;
    let k = groups.len() as f64;
    let ni: Vec<f64> = groups.iter().map(|g| g.len() as f64).collect();
    let vi: Vec<f64> = groups.iter().map(|g| var(g)).collect();
    let n_total: f64 = ni.iter().sum();
    let sp2 = ni.iter().zip(&vi).map(|(n, v)| (n - 1.0) * v).sum::<f64>() / (n_total - k);
    let num = (n_total - k) * sp2.ln() - ni.iter().zip(&vi).map(|(n, v)| (n - 1.0) * v.ln()).sum::<f64>();
    let denom = 1.0 + (ni.iter().map(|n| 1.0 / (n - 1.0)).sum::<f64>() - 1.0 / (n_total - k)) / (3.0 * (k - 1.0));
    let t = num / denom;
    Ok(TestResult { statistic: t, p_value: chi_square_survival(t, k - 1.0), degrees_of_freedom: k - 1.0 })
}

/// The Fligner-Killeen test of equal variances (SciPy's `fligner`, median centring): normal
/// scores of the ranks of the absolute deviations, compared across groups; robust to
/// non-normality.
pub fn fligner(groups: &[Vec<f64>]) -> Result<TestResult, ErrorsJSL> {
    check_groups(groups)?;
    let k = groups.len() as f64;
    let dev: Vec<f64> = groups.iter().flat_map(|g| {
        let m = median(g);
        g.iter().map(move |v| (v - m).abs()).collect::<Vec<_>>()
    }).collect();
    let n_total = dev.len() as f64;
    let ranks = rank(&dev);
    let a: Vec<f64> = ranks.iter().map(|r| normal_quantile(r / (2.0 * (n_total + 1.0)) + 0.5, 0.0, 1.0)).collect();
    let abar = mean(&a);
    let varsq = var(&a);
    let mut start = 0;
    let mut stat = 0.0;
    for g in groups {
        let seg = &a[start..start + g.len()];
        stat += g.len() as f64 * (mean(seg) - abar).powi(2);
        start += g.len();
    }
    let stat = stat / varsq;
    Ok(TestResult { statistic: stat, p_value: chi_square_survival(stat, k - 1.0), degrees_of_freedom: k - 1.0 })
}

// ----------------------------------------------------------------------------------- rank tests

/// The Kruskal-Wallis H test that `k` samples come from the same distribution (SciPy's `kruskal`),
/// with the tie correction, χ² with `k − 1` degrees of freedom.
pub fn kruskal(groups: &[Vec<f64>]) -> Result<TestResult, ErrorsJSL> {
    if groups.len() < 2 || groups.iter().any(|g| g.is_empty()) {
        return Err(ErrorsJSL::InvalidInputRange("need at least two non-empty groups"));
    }
    let all: Vec<f64> = groups.iter().flatten().copied().collect();
    let n = all.len() as f64;
    let ranks = rank(&all);
    let mut start = 0;
    let mut h = 0.0;
    for g in groups {
        let r: f64 = ranks[start..start + g.len()].iter().sum();
        h += r * r / g.len() as f64;
        start += g.len();
    }
    h = 12.0 / (n * (n + 1.0)) * h - 3.0 * (n + 1.0);
    let correction = 1.0 - tie_sum(&all) / (n * n * n - n);
    let h = h / correction;
    let df = groups.len() as f64 - 1.0;
    Ok(TestResult { statistic: h, p_value: chi_square_survival(h, df), degrees_of_freedom: df })
}

/// Friedman's test for `k` treatments measured on the same `n` blocks (SciPy's
/// `friedmanchisquare`): `treatments[j][i]` is treatment `j` on block `i`. Ranks within each
/// block, with the tie correction, χ² with `k − 1` degrees of freedom.
pub fn friedman(treatments: &[Vec<f64>]) -> Result<TestResult, ErrorsJSL> {
    let k = treatments.len();
    if k < 3 {
        return Err(ErrorsJSL::InvalidInputRange("Friedman's test needs at least three treatments"));
    }
    let n = treatments[0].len();
    if treatments.iter().any(|t| t.len() != n) || n < 2 {
        return Err(ErrorsJSL::InvalidInputRange("every treatment needs the same blocks"));
    }
    let mut rank_sums = vec![0.0; k];
    let mut ties = 0.0;
    for i in 0..n {
        let block: Vec<f64> = treatments.iter().map(|t| t[i]).collect();
        let r = rank(&block);
        for j in 0..k {
            rank_sums[j] += r[j];
        }
        ties += tie_sum(&block);
    }
    let (nf, kf) = (n as f64, k as f64);
    let c = 1.0 - ties / (nf * kf * (kf * kf - 1.0));
    let ssbn: f64 = rank_sums.iter().map(|r| r * r).sum();
    let chi2 = (12.0 / (nf * kf * (kf + 1.0)) * ssbn - 3.0 * nf * (kf + 1.0)) / c;
    Ok(TestResult { statistic: chi2, p_value: chi_square_survival(chi2, kf - 1.0), degrees_of_freedom: kf - 1.0 })
}

// ANCHOR: wilcoxon
/// The Wilcoxon signed-rank test that the differences `d` (or `x − y`) are symmetric about zero
/// (SciPy's `wilcoxon` with its defaults: zeros dropped, no continuity correction). Without ties
/// or zeros the p-value is exact, from the distribution of `W⁺` computed by dynamic programming
/// over the ranks; with ties, by enumerating all sign changes when `n ≤ 13`, and otherwise from the
/// normal approximation with the tie correction. For the two-sided test the statistic is
/// `min(W⁺, W⁻)`, otherwise `W⁺`.
pub fn wilcoxon(d: &[f64], alternative: Alternative) -> Result<TestResult, ErrorsJSL> {
    let n_all = d.len();
    let nonzero: Vec<f64> = d.iter().copied().filter(|v| *v != 0.0).collect();
    let n = nonzero.len();
    if n == 0 {
        return Err(ErrorsJSL::InvalidInputRange("all differences are zero"));
    }
    let abs: Vec<f64> = nonzero.iter().map(|v| v.abs()).collect();
    let r = rank(&abs);
    let r_plus: f64 = nonzero.iter().zip(&r).filter(|(v, _)| **v > 0.0).map(|(_, r)| r).sum();
    let r_minus: f64 = nonzero.iter().zip(&r).filter(|(v, _)| **v < 0.0).map(|(_, r)| r).sum();
    let has_ties = tie_sum(&abs) > 0.0;
    let has_zeros = n < n_all;
    let p = if !has_ties && !has_zeros {
        // Exact distribution of W⁺ = Σ of a random subset of 1..n.
        let max = n * (n + 1) / 2;
        let mut counts = vec![0.0f64; max + 1];
        counts[0] = 1.0;
        for k in 1..=n {
            for s in (k..=max).rev() {
                counts[s] += counts[s - k];
            }
        }
        let total = 2f64.powi(n as i32);
        let cdf = |w: usize| counts[..=w.min(max)].iter().sum::<f64>() / total;
        let sf = |w: usize| if w == 0 { 1.0 } else { 1.0 - cdf(w - 1) };
        let (lo, hi) = (r_plus.ceil() as usize, r_plus.floor() as usize);
        match alternative {
            Alternative::Less => cdf(lo),
            Alternative::Greater => sf(hi),
            Alternative::TwoSided => (2.0 * sf(hi).min(cdf(lo))).min(1.0),
        }
    } else if n_all <= 13 {
        // Permutation: every assignment of signs to the non-zero ranks.
        let eps = (100.0 * f64::EPSILON * r_plus).abs();
        let (mut le, mut ge) = (0usize, 0usize);
        for mask in 0..(1usize << n) {
            let w: f64 = (0..n).filter(|&i| mask >> i & 1 == 1).map(|i| r[i]).sum();
            if w <= r_plus + eps {
                le += 1;
            }
            if w >= r_plus - eps {
                ge += 1;
            }
        }
        let total = (1usize << n) as f64;
        let (pl, pg) = (le as f64 / total, ge as f64 / total);
        match alternative {
            Alternative::Less => pl,
            Alternative::Greater => pg,
            Alternative::TwoSided => (2.0 * pl.min(pg)).min(1.0),
        }
    } else {
        let nf = n as f64;
        let mn = nf * (nf + 1.0) * 0.25;
        let se = ((nf * (nf + 1.0) * (2.0 * nf + 1.0) - tie_sum(&abs) / 2.0) / 24.0).sqrt();
        let z = (r_plus - mn) / se;
        match alternative {
            Alternative::Less => normal_cdf(z, 0.0, 1.0),
            Alternative::Greater => normal_cdf(-z, 0.0, 1.0),
            Alternative::TwoSided => 2.0 * normal_cdf(-z.abs(), 0.0, 1.0),
        }
    };
    let statistic = if alternative == Alternative::TwoSided { r_plus.min(r_minus) } else { r_plus };
    Ok(TestResult { statistic, p_value: p, degrees_of_freedom: f64::NAN })
}
// ANCHOR_END: wilcoxon

// ---------------------------------------------------------------------------------- exact tests

/// Fisher's exact test for a 2 × 2 table `[[a, b], [c, d]]` (SciPy's `fisher_exact`): the
/// statistic is the sample odds ratio `ad / bc`, and the p-value comes from the hypergeometric
/// distribution of `a` with the margins fixed; the two-sided p-value sums the probabilities of all
/// tables no more likely than the observed one.
pub fn fisher_exact(table: [[u64; 2]; 2], alternative: Alternative) -> Result<TestResult, ErrorsJSL> {
    let [[a, b], [c, d]] = table;
    let odds = if b == 0 || c == 0 {
        if a == 0 || d == 0 { f64::NAN } else { f64::INFINITY }
    } else {
        (a * d) as f64 / (b * c) as f64
    };
    let total = a + b + c + d;
    if a + b == 0 || c + d == 0 || a + c == 0 || b + d == 0 {
        return Ok(TestResult { statistic: f64::NAN, p_value: 1.0, degrees_of_freedom: f64::NAN });
    }
    let h = Hypergeometric::new(total, a + b, a + c)?;
    let k = a as i64;
    let p = match alternative {
        Alternative::Less => h.cdf(k),
        Alternative::Greater => h.sf(k - 1),
        Alternative::TwoSided => {
            let p0 = h.pmf(k) * (1.0 + 1e-7);
            let (lo, hi) = h.support();
            (lo..=hi).map(|j| h.pmf(j)).filter(|&p| p <= p0).sum::<f64>().min(1.0)
        }
    };
    Ok(TestResult { statistic: odds, p_value: p, degrees_of_freedom: f64::NAN })
}

/// The exact binomial test of `k` successes in `n` trials against success probability `p`
/// (SciPy's `binomtest`); the two-sided p-value sums the probabilities of all outcomes no more
/// likely than `k`. The statistic is `k / n`.
pub fn binomtest(k: u64, n: u64, p: f64, alternative: Alternative) -> Result<TestResult, ErrorsJSL> {
    if k > n {
        return Err(ErrorsJSL::InvalidInputRange("k cannot exceed n"));
    }
    let b = Binomial::new(n, p)?;
    let ki = k as i64;
    let pv = match alternative {
        Alternative::Less => b.cdf(ki),
        Alternative::Greater => b.sf(ki - 1),
        Alternative::TwoSided => {
            let d = b.pmf(ki) * (1.0 + 1e-7);
            if (k as f64 - n as f64 * p).abs() < 1e-12 { 1.0 } else { (0..=n as i64).map(|j| b.pmf(j)).filter(|&q| q <= d).sum::<f64>().min(1.0) }
        }
    };
    Ok(TestResult { statistic: k as f64 / n as f64, p_value: pv, degrees_of_freedom: f64::NAN })
}

// --------------------------------------------------------------------------- permutation tests

/// The result of [`permutation_test`].
#[derive(Clone, Debug)]
pub struct PermutationResult {
    pub statistic: f64,
    pub p_value: f64,
    /// The statistic for each permutation examined.
    pub null_distribution: Vec<f64>,
    /// `true` when every partition was enumerated.
    pub exact: bool,
}

// ANCHOR: permutation
/// A two-sample permutation test (SciPy's `permutation_test` with `permutation_type='independent'`):
/// the statistic `f(x, y)` is recomputed for every way of splitting the pooled observations into
/// groups of the original sizes when there are at most `n_resamples` of them (exact), and otherwise
/// for `n_resamples` random splits, for which the p-value counts the observed split as well.
pub fn permutation_test(x: &[f64], y: &[f64], statistic: impl Fn(&[f64], &[f64]) -> f64, alternative: Alternative, n_resamples: usize, seed: u64) -> Result<PermutationResult, ErrorsJSL> {
    if x.is_empty() || y.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("both samples must be non-empty"));
    }
    let observed = statistic(x, y);
    let pooled: Vec<f64> = x.iter().chain(y).copied().collect();
    let (nx, n) = (x.len(), pooled.len());
    // C(n, nx), saturating.
    let mut combos = 1f64;
    for i in 0..nx {
        combos = combos * (n - i) as f64 / (i + 1) as f64;
    }
    let exact = combos <= n_resamples as f64;
    let mut null = Vec::new();
    if exact {
        // Enumerate the nx-subsets in lexicographic order.
        let mut idx: Vec<usize> = (0..nx).collect();
        loop {
            let mut in_x = vec![false; n];
            idx.iter().for_each(|&i| in_x[i] = true);
            let xs: Vec<f64> = (0..n).filter(|&i| in_x[i]).map(|i| pooled[i]).collect();
            let ys: Vec<f64> = (0..n).filter(|&i| !in_x[i]).map(|i| pooled[i]).collect();
            null.push(statistic(&xs, &ys));
            let mut i = nx;
            while i > 0 && idx[i - 1] == n - nx + i - 1 {
                i -= 1;
            }
            if i == 0 {
                break;
            }
            idx[i - 1] += 1;
            for j in i..nx {
                idx[j] = idx[j - 1] + 1;
            }
        }
    } else {
        let mut rng = Xoshiro256PlusPlus::from_seed(seed);
        let mut perm = pooled.clone();
        for _ in 0..n_resamples {
            for i in (1..n).rev() {
                let j = (rng.next_f64() * (i + 1) as f64) as usize;
                perm.swap(i, j.min(i));
            }
            null.push(statistic(&perm[..nx], &perm[nx..]));
        }
    }
    let gamma = (100.0 * f64::EPSILON * observed).abs();
    let le = null.iter().filter(|&&v| v <= observed + gamma).count() as f64;
    let ge = null.iter().filter(|&&v| v >= observed - gamma).count() as f64;
    let (pl, pg) = if exact { (le / null.len() as f64, ge / null.len() as f64) } else { ((le + 1.0) / (null.len() as f64 + 1.0), (ge + 1.0) / (null.len() as f64 + 1.0)) };
    let p = match alternative {
        Alternative::Less => pl,
        Alternative::Greater => pg,
        Alternative::TwoSided => (2.0 * pl.min(pg)).min(1.0),
    };
    Ok(PermutationResult { statistic: observed, p_value: p, null_distribution: null, exact })
}
// ANCHOR_END: permutation

// ------------------------------------------------------------------------ multiple comparisons

/// How [`adjust_p_values`] controls the error rate across many tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Adjustment {
    /// Family-wise error rate: `min(1, m p)`.
    Bonferroni,
    /// Family-wise error rate, step-down: Holm (1979).
    Holm,
    /// False discovery rate for independent or positively dependent tests: Benjamini and
    /// Hochberg (1995) (SciPy's `false_discovery_control(method='bh')`).
    BenjaminiHochberg,
    /// False discovery rate under any dependence: Benjamini and Yekutieli (2001) (`method='by'`).
    BenjaminiYekutieli,
}

// ANCHOR: adjust
/// Adjusted p-values: reject the hypotheses whose adjusted p-value is at most the target rate.
pub fn adjust_p_values(p: &[f64], method: Adjustment) -> Result<Vec<f64>, ErrorsJSL> {
    if p.iter().any(|v| !(0.0..=1.0).contains(v)) {
        return Err(ErrorsJSL::InvalidInputRange("p-values must lie in [0, 1]"));
    }
    let m = p.len();
    let mf = m as f64;
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let mut adj = vec![0.0; m];
    match method {
        Adjustment::Bonferroni => {
            for i in 0..m {
                adj[i] = (mf * p[i]).min(1.0);
            }
        }
        Adjustment::Holm => {
            let mut running: f64 = 0.0;
            for (rank, &i) in order.iter().enumerate() {
                running = running.max((mf - rank as f64) * p[i]);
                adj[i] = running.min(1.0);
            }
        }
        Adjustment::BenjaminiHochberg | Adjustment::BenjaminiYekutieli => {
            let c = if method == Adjustment::BenjaminiYekutieli { (1..=m).map(|i| 1.0 / i as f64).sum::<f64>() } else { 1.0 };
            let mut running: f64 = 1.0;
            for (rank, &i) in order.iter().enumerate().rev() {
                running = running.min(p[i] * mf * c / (rank + 1) as f64);
                adj[i] = running.min(1.0);
            }
        }
    }
    Ok(adj)
}
// ANCHOR_END: adjust

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilcoxon_exact_matches_a_hand_count() {
        // n = 5, all positive: W⁺ = 15, P(W⁺ ≥ 15) = 1/32; two-sided statistic min(15, 0) = 0.
        let r = wilcoxon(&[1.0, 2.0, 3.0, 4.0, 5.0], Alternative::Greater).unwrap();
        assert!((r.p_value - 1.0 / 32.0).abs() < 1e-15);
        let r = wilcoxon(&[1.0, 2.0, 3.0, 4.0, 5.0], Alternative::TwoSided).unwrap();
        assert_eq!(r.statistic, 0.0);
        assert!((r.p_value - 1.0 / 16.0).abs() < 1e-15);
    }

    #[test]
    fn fisher_and_binomial_known_values() {
        let r = fisher_exact([[8, 2], [1, 5]], Alternative::TwoSided).unwrap();
        assert!((r.statistic - 20.0).abs() < 1e-12);
        assert!((r.p_value - 0.034965034965034975).abs() < 1e-12, "{}", r.p_value);
        let r = binomtest(7, 20, 0.5, Alternative::TwoSided).unwrap();
        assert!((r.p_value - 0.26317596435546875).abs() < 1e-12, "{}", r.p_value);
    }

    #[test]
    fn fdr_adjustment_is_monotone() {
        let p = [0.01, 0.04, 0.03, 0.005, 0.2];
        let a = adjust_p_values(&p, Adjustment::BenjaminiHochberg).unwrap();
        // Sorted p: 0.005, 0.01, 0.03, 0.04, 0.2 → 0.025, 0.025, 0.05, 0.05, 0.2.
        let want = [0.025, 0.05, 0.05, 0.025, 0.2];
        for (g, w) in a.iter().zip(want) {
            assert!((g - w).abs() < 1e-15, "{a:?}");
        }
    }
}
