//! The Cramér-von Mises tests (SciPy's `cramervonmises` and `cramervonmises_2samp`): the
//! integrated squared difference between a sample's empirical distribution function and a
//! hypothesised one, or between two samples' empirical distribution functions.
//!
//! See the book entry [Multivariate distributions and further tests](https://jwkunz.github.io/lib_jsl/book/statistics/multivariate-tests.html#the-cramér-von-mises-tests).

use crate::correlation::rank;
use crate::hypothesis_tests::TestResult;
use lib_jsl_core::ErrorsJSL;
use lib_jsl_special_functions::bessel_functions_fractional::bessel_k_fractional;
use lib_jsl_special_functions::gamma::{gamma, ln_gamma};
use std::collections::HashMap;
use std::f64::consts::PI;

// ANCHOR: cdf_cvm_inf
/// The limiting distribution of the statistic `ω²` (Anderson and Darling, 1952), as SciPy sums
/// it: `Σ_k Γ(k + ½)/(Γ(k + 1) π^{3/2} √x) √(4k+1) e^{−q} K_{1/4}(q)`, `q = (4k+1)²/(16x)`,
/// stopping after the first term below 10⁻⁷.
pub fn cramer_von_mises_cdf_limit(x: f64) -> f64 {
    if !(x > 0.0) {
        return 0.0;
    }
    let mut total = 0.0;
    let mut k = 0;
    loop {
        let u = (ln_gamma(k as f64 + 0.5) - ln_gamma(k as f64 + 1.0)).exp() / (PI.powf(1.5) * x.sqrt());
        let y = 4.0 * k as f64 + 1.0;
        let q = y * y / (16.0 * x);
        let term = u * y.sqrt() * (-q).exp() * bessel_k_fractional(0.25, q);
        total += term;
        if term.abs() < 1e-7 || k > 1000 {
            return total;
        }
        k += 1;
    }
}
// ANCHOR_END: cdf_cvm_inf

/// Csörgő and Faraway's (1996) correction term `ψ₁` for finite samples, as SciPy's `_psi1_mod`.
fn psi1_mod(x: f64) -> f64 {
    let ed2 = |y: f64| {
        let z = y * y / 4.0;
        (-z).exp() * (y / 2.0).powf(1.5) * (bessel_k_fractional(0.25, z) + bessel_k_fractional(0.75, z)) / PI.sqrt()
    };
    let ed3 = |y: f64| {
        let z = y * y / 4.0;
        (-z).exp() / PI.sqrt() * (y / 2.0).powf(2.5) * (2.0 * bessel_k_fractional(0.25, z) + 3.0 * bessel_k_fractional(0.75, z) - bessel_k_fractional(1.25, z))
    };
    let ak = |k: f64| {
        let m = 2.0 * k + 1.0;
        let sx = 2.0 * x.sqrt();
        let (y1, y2) = (x.powf(0.75), x.powf(1.25));
        let (g1, g3) = (gamma(k + 0.5), gamma(k + 1.5));
        m * g1 * ed2((4.0 * k + 3.0) / sx) / (9.0 * y1)
            + g1 * ed3((4.0 * k + 1.0) / sx) / (72.0 * y2)
            + 2.0 * (m + 2.0) * g3 * ed3((4.0 * k + 5.0) / sx) / (12.0 * y2)
            + 7.0 * m * g1 * ed2((4.0 * k + 1.0) / sx) / (144.0 * y1)
            + 7.0 * m * g1 * ed2((4.0 * k + 5.0) / sx) / (144.0 * y1)
    };
    let mut total = 0.0;
    let mut k = 0.0;
    loop {
        let z = -ak(k) / (PI * gamma(k + 1.0));
        total += z;
        if z.abs() < 1e-7 || k > 1000.0 {
            return total;
        }
        k += 1.0;
    }
}

/// The distribution function of `ω²` for samples of size `n` (Csörgő and Faraway's
/// approximation, as SciPy's `_cdf_cvm`).
pub fn cramer_von_mises_cdf(x: f64, n: usize) -> f64 {
    let n = n as f64;
    if x >= n / 3.0 {
        1.0
    } else if x > 1.0 / (12.0 * n) {
        cramer_von_mises_cdf_limit(x) * (1.0 + 1.0 / (12.0 * n)) + psi1_mod(x) / n
    } else {
        0.0
    }
}

// ANCHOR: cramervonmises
/// The one-sample Cramér-von Mises test of the hypothesis that `x` comes from the distribution
/// `cdf` (SciPy's `cramervonmises`): `ω² = 1/(12n) + Σ ((2i − 1)/(2n) − F(x_(i)))²`, with the
/// p-value from the finite-sample distribution of Csörgő and Faraway.
pub fn cramervonmises(x: &[f64], cdf: impl Fn(f64) -> f64) -> Result<TestResult, ErrorsJSL> {
    let n = x.len();
    if n < 2 || x.iter().any(|v| !v.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("cramervonmises: need at least two finite observations"));
    }
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let nf = n as f64;
    let w = 1.0 / (12.0 * nf) + s.iter().enumerate().map(|(i, &v)| ((2 * i + 1) as f64 / (2.0 * nf) - cdf(v)).powi(2)).sum::<f64>();
    let p = (1.0 - cramer_von_mises_cdf(w, n)).max(0.0);
    Ok(TestResult::new(w, p, f64::NAN))
}
// ANCHOR_END: cramervonmises

/// How [`cramervonmises_2samp`] computes its p-value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CvmMethod {
    /// Exact when both samples have at most 20 observations, otherwise asymptotic (SciPy's default).
    Auto,
    Exact,
    Asymptotic,
}

/// The exact upper tail `P(U ≥ u)` of the two-sample statistic, by Xiao, Gordon and Yakovlev's
/// (2006) recursion over the lattice paths that interleave the two samples: each path's
/// contribution `Σ (a v − b w)²`, `a = lcm/m`, `b = lcm/n`, is accumulated with counts.
fn cvm_2samp_exact(u: f64, m: usize, n: usize) -> f64 {
    let gcd = |mut a: usize, mut b: usize| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let lcm = m / gcd(m, n) * n;
    let (a, b) = ((lcm / m) as i64, (lcm / n) as i64);
    let mn = (m * n) as f64;
    let zeta = ((lcm * lcm * (m + n)) as f64 * (6.0 * u - mn * (4.0 * mn - 1.0)) / (6.0 * mn * mn)).floor();
    // gs[v]: path end-points with v steps of the first sample, value → count.
    let mut gs: Vec<HashMap<u64, u64>> = vec![HashMap::new(); m + 1];
    gs[0].insert(0, 1);
    for w in 0..=n {
        let mut next = Vec::with_capacity(m + 1);
        let mut tmp: HashMap<u64, u64> = HashMap::new();
        for (v, g) in gs.iter().enumerate() {
            for (&k, &c) in g {
                *tmp.entry(k).or_insert(0) += c;
            }
            let res = ((a * v as i64 - b * w as i64).pow(2)) as u64;
            tmp = tmp.into_iter().map(|(k, c)| (k + res, c)).collect();
            next.push(tmp.clone());
        }
        gs = next;
    }
    let total: f64 = (0..m).fold(1.0, |acc, i| acc * (n + m - i) as f64 / (i + 1) as f64).round();
    let count: u64 = gs[m].iter().filter(|(k, _)| **k as f64 >= zeta).map(|(_, c)| *c).sum();
    count as f64 / total
}

// ANCHOR: cramervonmises_2samp
/// The two-sample Cramér-von Mises test (SciPy's `cramervonmises_2samp`), from the average
/// ranks: `U = n_x Σ (r_i − i)² + n_y Σ (s_j − j)²` and `T = U/(n_x n_y N) − (4 n_x n_y − 1)/(6N)`,
/// with an exact p-value (lattice-path counting) or the asymptotic one after standardising `T`.
pub fn cramervonmises_2samp(x: &[f64], y: &[f64], method: CvmMethod) -> Result<TestResult, ErrorsJSL> {
    let (nx, ny) = (x.len(), y.len());
    if nx < 2 || ny < 2 || x.iter().chain(y).any(|v| !v.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("cramervonmises_2samp: need at least two finite observations per sample"));
    }
    let exact = match method {
        CvmMethod::Exact => true,
        CvmMethod::Asymptotic => false,
        CvmMethod::Auto => nx.max(ny) <= 20,
    };
    let mut xs = x.to_vec();
    let mut ys = y.to_vec();
    xs.sort_by(f64::total_cmp);
    ys.sort_by(f64::total_cmp);
    let z: Vec<f64> = xs.iter().chain(&ys).copied().collect();
    let r = rank(&z);
    let (nxf, nyf) = (nx as f64, ny as f64);
    let u = nxf * (0..nx).map(|i| (r[i] - (i + 1) as f64).powi(2)).sum::<f64>() + nyf * (0..ny).map(|j| (r[nx + j] - (j + 1) as f64).powi(2)).sum::<f64>();
    let (k, n) = (nxf * nyf, nxf + nyf);
    let t = u / (k * n) - (4.0 * k - 1.0) / (6.0 * n);
    let p = if exact {
        cvm_2samp_exact(u, nx, ny)
    } else {
        let et = (1.0 + 1.0 / n) / 6.0;
        let vt = (n + 1.0) * (4.0 * k * n - 3.0 * (nxf * nxf + nyf * nyf) - 2.0 * k) / (45.0 * n * n * 4.0 * k);
        let tn = 1.0 / 6.0 + (t - et) / (45.0 * vt).sqrt();
        if tn >= 0.003 { (1.0 - cramer_von_mises_cdf_limit(tn)).max(0.0) } else { 1.0 }
    };
    Ok(TestResult::new(t, p, f64::NAN))
}
// ANCHOR_END: cramervonmises_2samp

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limiting_distribution_and_exact_two_sample_p_values() {
        // Anderson and Darling's table: the 95 % point of ω² is 0.46136.
        assert!((cramer_von_mises_cdf_limit(0.46136) - 0.95).abs() < 1e-4);
        // The exact p-value by brute force over all C(7, 3) = 35 assignments.
        let (x, y) = ([0.3, 1.9, 2.5], [0.1, 1.1, 3.2, 4.0]);
        let r = cramervonmises_2samp(&x, &y, CvmMethod::Exact).unwrap();
        let pooled: Vec<f64> = x.iter().chain(&y).copied().collect();
        let mut count = 0;
        let mut total = 0;
        for mask in 0u32..128 {
            if mask.count_ones() != 3 {
                continue;
            }
            let xs: Vec<f64> = (0..7).filter(|i| mask >> i & 1 == 1).map(|i| pooled[i]).collect();
            let ys: Vec<f64> = (0..7).filter(|i| mask >> i & 1 == 0).map(|i| pooled[i]).collect();
            let t = cramervonmises_2samp(&xs, &ys, CvmMethod::Asymptotic).unwrap().statistic;
            total += 1;
            if t >= r.statistic - 1e-12 {
                count += 1;
            }
        }
        assert!((r.p_value - count as f64 / total as f64).abs() < 1e-12, "{} vs {count}/{total}", r.p_value);
        // A sample from the hypothesised distribution is not rejected.
        let u: Vec<f64> = (0..40).map(|i| (i as f64 + 0.5) / 40.0).collect();
        assert!(cramervonmises(&u, |v| v.clamp(0.0, 1.0)).unwrap().p_value > 0.99);
    }
}
