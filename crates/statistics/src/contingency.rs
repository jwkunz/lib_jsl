//! Effect sizes for contingency tables (SciPy's `odds_ratio`, `relative_risk` and
//! `contingency.association`), with confidence intervals.
//!
//! See the book entry [Multivariate distributions and further tests](https://jwkunz.github.io/lib_jsl/book/statistics/multivariate-tests.html#effect-sizes-for-contingency-tables).

use crate::hypothesis_tests::Alternative;
use lib_jsl_core::ErrorsJSL;
use lib_jsl_special_functions::gamma::ln_gamma;
use lib_jsl_special_functions::normal_quantile as nq;

/// The standard normal quantile.
fn normal_quantile(p: f64) -> f64 {
    nq(p, 0.0, 1.0)
}

fn ln_choose(n: f64, k: f64) -> f64 {
    ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
}

/// Fisher's noncentral hypergeometric distribution of `x = table[0][0]` given the margins:
/// `P(x) ∝ C(n, x) C(M − n, N − x) ωˣ` on `max(0, N − M + n) ≤ x ≤ min(n, N)`.
struct FisherNcHypergeom {
    lo: u64,
    hi: u64,
    log_base: Vec<f64>,
}

impl FisherNcHypergeom {
    fn new(m: u64, n: u64, nn: u64) -> Self {
        let lo = nn.saturating_sub(m - n);
        let hi = n.min(nn);
        let log_base = (lo..=hi).map(|x| ln_choose(n as f64, x as f64) + ln_choose((m - n) as f64, (nn - x) as f64)).collect();
        FisherNcHypergeom { lo, hi, log_base }
    }
    /// The probabilities for odds ratio `ω`, normalised in logarithms.
    fn probs(&self, omega: f64) -> Vec<f64> {
        let lw = omega.ln();
        let logs: Vec<f64> = self.log_base.iter().enumerate().map(|(i, b)| b + (self.lo + i as u64) as f64 * lw).collect();
        let top = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = logs.iter().map(|l| (l - top).exp()).collect();
        let s: f64 = w.iter().sum();
        w.iter().map(|v| v / s).collect()
    }
    fn mean(&self, omega: f64) -> f64 {
        self.probs(omega).iter().enumerate().map(|(i, p)| (self.lo + i as u64) as f64 * p).sum()
    }
    /// `P(X ≤ x)`.
    fn cdf(&self, x: i64, omega: f64) -> f64 {
        if x < self.lo as i64 {
            return 0.0;
        }
        self.probs(omega).iter().take((x - self.lo as i64 + 1) as usize).sum::<f64>().min(1.0)
    }
    /// `P(X > x)`, summed directly.
    fn sf(&self, x: i64, omega: f64) -> f64 {
        let start = (x + 1 - self.lo as i64).max(0) as usize;
        self.probs(omega).iter().skip(start).sum::<f64>().min(1.0)
    }
}

/// Brent's root finder on a bracket (as SciPy's `brentq`, with absolute tolerance `xtol` and
/// relative tolerance 4ε).
fn brentq(f: impl Fn(f64) -> f64, mut a: f64, mut b: f64, xtol: f64) -> f64 {
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
        if fb.signum() == fc.signum() {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol = 2.0 * 4.0 * f64::EPSILON * b.abs() + 0.5 * xtol;
        let m = 0.5 * (c - b);
        if m.abs() <= tol || fb == 0.0 {
            return b;
        }
        if e.abs() >= tol && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (p, q) = if a == c {
                (2.0 * m * s, 1.0 - s)
            } else {
                let q0 = fa / fc;
                let r = fb / fc;
                (s * (2.0 * m * q0 * (q0 - r) - (b - a) * (r - 1.0)), (q0 - 1.0) * (r - 1.0) * (s - 1.0))
            };
            let (p, q) = if p > 0.0 { (p, -q) } else { (-p, q) };
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
        a = b;
        fa = fb;
        b += if d.abs() > tol { d } else { tol.copysign(m) };
        fb = f(b);
    }
    b
}

/// SciPy's `_solve`: the root in `(0, ∞)` of an increasing or decreasing function, bracketed by
/// halving or doubling from 1.
fn solve_positive(f: impl Fn(f64) -> f64) -> f64 {
    let v = f(1.0);
    if v == 0.0 {
        return 1.0;
    }
    let (lo, hi) = if v > 0.0 {
        let mut nc = 0.5;
        while f(nc) > 0.0 {
            nc /= 2.0;
        }
        (nc, 2.0 * nc)
    } else {
        let mut nc = 2.0;
        while f(nc) < 0.0 {
            nc *= 2.0;
        }
        (nc / 2.0, nc)
    };
    brentq(f, lo, hi, 1e-13)
}

/// Which odds ratio [`odds_ratio`] estimates (SciPy's `kind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OddsRatioKind {
    /// The conditional maximum-likelihood estimate, given the margins (SciPy's default).
    Conditional,
    /// The sample odds ratio `ad/(bc)`.
    Sample,
}

/// The result of [`odds_ratio`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct OddsRatioResult {
    pub statistic: f64,
    pub kind: OddsRatioKind,
    table: [[u64; 2]; 2],
}

// ANCHOR: odds_ratio
/// The odds ratio of a 2 × 2 table `[[a, b], [c, d]]` (SciPy's `odds_ratio`). The conditional
/// estimate solves `E_ω[X] = a` for the odds ratio `ω` of Fisher's noncentral hypergeometric
/// distribution of `X = table[0][0]` given all margins; its confidence limits invert that
/// distribution's tails (the exact interval that matches Fisher's exact test). The sample odds
/// ratio is `ad/(bc)`, with Woolf's interval `exp(log OR ± z √(1/a + 1/b + 1/c + 1/d))`.
pub fn odds_ratio(table: [[u64; 2]; 2], kind: OddsRatioKind) -> OddsRatioResult {
    let [[a, b], [c, d]] = table;
    let statistic = if a + b == 0 || c + d == 0 || a + c == 0 || b + d == 0 {
        f64::NAN
    } else {
        match kind {
            OddsRatioKind::Sample => sample_odds_ratio(&table),
            OddsRatioKind::Conditional => {
                let h = FisherNcHypergeom::new(a + b + c + d, a + b, a + c);
                if a == h.lo {
                    0.0
                } else if a == h.hi {
                    f64::INFINITY
                } else {
                    solve_positive(|nc| h.mean(nc) - a as f64)
                }
            }
        }
    };
    OddsRatioResult { statistic, kind, table }
}
// ANCHOR_END: odds_ratio

fn sample_odds_ratio(t: &[[u64; 2]; 2]) -> f64 {
    let [[a, b], [c, d]] = *t;
    if c > 0 && b > 0 {
        (a * d) as f64 / (c * b) as f64
    } else if a == 0 || d == 0 {
        f64::NAN
    } else {
        f64::INFINITY
    }
}

impl OddsRatioResult {
    /// The confidence interval for the odds ratio (SciPy's `confidence_interval`).
    pub fn confidence_interval(&self, confidence_level: f64, alternative: Alternative) -> Result<(f64, f64), ErrorsJSL> {
        if !(0.0..=1.0).contains(&confidence_level) {
            return Err(ErrorsJSL::InvalidInputRange("the confidence level must lie in [0, 1]"));
        }
        let [[a, b], [c, d]] = self.table;
        if self.statistic.is_nan() && (a + b == 0 || c + d == 0 || a + c == 0 || b + d == 0) {
            return Ok((f64::NAN, f64::NAN));
        }
        Ok(match self.kind {
            OddsRatioKind::Sample => {
                let or = sample_odds_ratio(&self.table);
                let log_or = or.ln();
                let se = (1.0 / a as f64 + 1.0 / b as f64 + 1.0 / c as f64 + 1.0 / d as f64).sqrt();
                match alternative {
                    Alternative::Less => (0.0, (log_or + normal_quantile(confidence_level) * se).exp()),
                    Alternative::Greater => ((log_or - normal_quantile(confidence_level) * se).exp(), f64::INFINITY),
                    Alternative::TwoSided => {
                        let z = normal_quantile(0.5 * confidence_level + 0.5);
                        ((log_or - z * se).exp(), (log_or + z * se).exp())
                    }
                }
            }
            OddsRatioKind::Conditional => {
                let h = FisherNcHypergeom::new(a + b + c + d, a + b, a + c);
                let or = sample_odds_ratio(&self.table);
                let x = a as i64;
                let upper = |alpha: f64| if or == f64::INFINITY { f64::INFINITY } else { solve_positive(|nc| alpha - h.cdf(x, nc)) };
                let lower = |alpha: f64| if or == 0.0 { 0.0 } else { solve_positive(|nc| h.sf(x - 1, nc) - alpha) };
                match alternative {
                    Alternative::TwoSided => {
                        let alpha = 0.5 * (1.0 - confidence_level);
                        (lower(alpha), upper(alpha))
                    }
                    Alternative::Less => (0.0, upper(1.0 - confidence_level)),
                    Alternative::Greater => (lower(1.0 - confidence_level), f64::INFINITY),
                }
            }
        })
    }
}

/// The result of [`relative_risk`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct RelativeRiskResult {
    pub relative_risk: f64,
    counts: (u64, u64, u64, u64),
}

/// The relative risk `(a/n₁)/(c/n₂)` of `a` cases among `n₁` exposed and `c` among `n₂` controls
/// (SciPy's `relative_risk`).
pub fn relative_risk(exposed_cases: u64, exposed_total: u64, control_cases: u64, control_total: u64) -> Result<RelativeRiskResult, ErrorsJSL> {
    if exposed_total == 0 || control_total == 0 || exposed_cases > exposed_total || control_cases > control_total {
        return Err(ErrorsJSL::InvalidInputRange("relative_risk: need 0 ≤ cases ≤ total and totals ≥ 1"));
    }
    let rr = match (exposed_cases, control_cases) {
        (0, 0) => f64::NAN,
        (0, _) => 0.0,
        (_, 0) => f64::INFINITY,
        _ => (exposed_cases as f64 / exposed_total as f64) / (control_cases as f64 / control_total as f64),
    };
    Ok(RelativeRiskResult { relative_risk: rr, counts: (exposed_cases, exposed_total, control_cases, control_total) })
}

impl RelativeRiskResult {
    /// Katz's interval `RR exp(±z √(1/a − 1/n₁ + 1/c − 1/n₂))`.
    pub fn confidence_interval(&self, confidence_level: f64) -> Result<(f64, f64), ErrorsJSL> {
        if !(0.0..=1.0).contains(&confidence_level) {
            return Err(ErrorsJSL::InvalidInputRange("the confidence level must lie in [0, 1]"));
        }
        let (a, n1, c, n2) = self.counts;
        Ok(match (a, c) {
            (0, 0) => (f64::NAN, f64::NAN),
            (0, _) => (0.0, f64::NAN),
            (_, 0) => (f64::NAN, f64::INFINITY),
            _ => {
                let z = normal_quantile(1.0 - (1.0 - confidence_level) / 2.0);
                let se = (1.0 / a as f64 - 1.0 / n1 as f64 + 1.0 / c as f64 - 1.0 / n2 as f64).sqrt();
                (self.relative_risk * (-z * se).exp(), self.relative_risk * (z * se).exp())
            }
        })
    }
}

/// A measure of association for an `r × c` table (SciPy's `association`'s `method`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum AssociationMethod {
    /// Cramér's V, `√(φ² / min(r − 1, c − 1))`.
    Cramer,
    /// Tschuprow's T, `√(φ² / √((r − 1)(c − 1)))`.
    Tschuprow,
    /// Pearson's contingency coefficient, `√(φ² / (1 + φ²))`.
    Pearson,
}

/// The strength of association between the two variables of a contingency table (SciPy's
/// `contingency.association`), from `φ² = χ²/N` without continuity correction: 0 for
/// independence, 1 (for Cramér's V and Tschuprow's T on square tables) for perfect association.
pub fn association(table: &[Vec<u64>], method: AssociationMethod) -> Result<f64, ErrorsJSL> {
    let r = table.len();
    let c = table.first().map_or(0, |row| row.len());
    if r < 2 || c < 2 || table.iter().any(|row| row.len() != c) {
        return Err(ErrorsJSL::InvalidInputRange("association: need a rectangular table with at least 2 × 2 cells"));
    }
    let total: f64 = table.iter().flatten().map(|&v| v as f64).sum();
    let rows: Vec<f64> = table.iter().map(|row| row.iter().map(|&v| v as f64).sum()).collect();
    let cols: Vec<f64> = (0..c).map(|j| table.iter().map(|row| row[j] as f64).sum()).collect();
    if rows.iter().chain(&cols).any(|&s| s == 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("association: a row or column sums to zero"));
    }
    let mut chi2 = 0.0;
    for i in 0..r {
        for j in 0..c {
            let e = rows[i] * cols[j] / total;
            chi2 += (table[i][j] as f64 - e).powi(2) / e;
        }
    }
    let phi2 = chi2 / total;
    let (rf, cf) = (r as f64, c as f64);
    Ok(match method {
        AssociationMethod::Cramer => (phi2 / (cf - 1.0).min(rf - 1.0)).sqrt(),
        AssociationMethod::Tschuprow => (phi2 / ((rf - 1.0) * (cf - 1.0)).sqrt()).sqrt(),
        AssociationMethod::Pearson => (phi2 / (1.0 + phi2)).sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effect_sizes() {
        let t = [[12, 5], [3, 15]];
        assert!((odds_ratio(t, OddsRatioKind::Sample).statistic - 12.0).abs() < 1e-15);
        let c = odds_ratio(t, OddsRatioKind::Conditional);
        // The conditional MLE is shrunk towards 1 relative to the sample odds ratio.
        assert!(c.statistic > 1.0 && c.statistic < 12.0);
        let (lo, hi) = c.confidence_interval(0.95, Alternative::TwoSided).unwrap();
        assert!(lo < c.statistic && c.statistic < hi);
        assert_eq!(odds_ratio([[0, 6], [5, 9]], OddsRatioKind::Conditional).statistic, 0.0);
        let rr = relative_risk(10, 100, 5, 100).unwrap();
        assert!((rr.relative_risk - 2.0).abs() < 1e-15);
        // A perfectly associated square table.
        let perfect = vec![vec![10, 0, 0], vec![0, 7, 0], vec![0, 0, 4]];
        assert!((association(&perfect, AssociationMethod::Cramer).unwrap() - 1.0).abs() < 1e-15);
        let independent = vec![vec![10, 20], vec![30, 60]];
        assert!(association(&independent, AssociationMethod::Cramer).unwrap() < 1e-15);
    }
}
