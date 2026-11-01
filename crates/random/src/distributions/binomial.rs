use crate::{distributions::beta::beta_distribution, uniform_generator::UniformGenerator};
use lib_jsl_core::ErrorsJSL;

/// Inversion by sequential search of the pmf, starting from zero (expected cost O(np)).
fn binomial_inversion<U: UniformGenerator>(rng: &mut U, n: u64, p: f64) -> u64 {
    let q = 1.0 - p;
    let ratio = p / q;
    let mut pmf = q.powf(n as f64);
    let mut cdf = pmf;
    let u = rng.next_f64();
    let mut k = 0;
    while u > cdf && k < n {
        pmf *= ratio * (n - k) as f64 / (k + 1) as f64;
        k += 1;
        cdf += pmf;
    }
    k
}

/// A binomial sample: the number of successes in `n` independent trials of probability `p`.
///
/// Small `n·min(p, 1−p)` uses inversion; otherwise Knuth's recursive beta splitting reduces the
/// problem exactly: with `a = 1 + ⌊n/2⌋`, the `a`-th order statistic of `n` uniforms is
/// `X ~ Beta(a, n + 1 − a)`, and the count below `p` follows from one smaller binomial on the
/// side of `X` that contains `p`. Cost is `O(log n)` beta samples.
pub fn binomial_distribution<U: UniformGenerator>(uniform_generator: &mut U, n: u64, p: f64) -> Result<u64, ErrorsJSL> {
    if !(0.0..=1.0).contains(&p) {
        return Err(ErrorsJSL::InvalidInputRange("p must lie in [0, 1]"));
    }
    let (mut n, mut p) = (n, p);
    let mut offset = 0;
    loop {
        if n == 0 || p == 0.0 {
            return Ok(offset);
        }
        if p == 1.0 {
            return Ok(offset + n);
        }
        if (n as f64) * p.min(1.0 - p) < 20.0 {
            // Inversion is cheapest from the side with the smaller mean.
            return Ok(if p <= 0.5 {
                offset + binomial_inversion(uniform_generator, n, p)
            } else {
                offset + n - binomial_inversion(uniform_generator, n, 1.0 - p)
            });
        }
        let a = 1 + n / 2;
        let b = n + 1 - a;
        let x = beta_distribution(uniform_generator, a as f64, b as f64)?;
        if x >= p {
            // Fewer than a successes: the a − 1 uniforms below X are uniform on [0, X).
            n = a - 1;
            p /= x;
        } else {
            // At least a successes: a fixed, the b − 1 above X are uniform on (X, 1].
            offset += a;
            n = b - 1;
            p = (p - x) / (1.0 - x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uniform_generator::UniformRNG;

    #[test]
    fn mean_and_variance_match_np_and_npq_in_both_regimes() {
        let mut rng = UniformRNG::from_seed(2);
        for (n, p) in [(10u64, 0.3), (40, 0.9), (1000, 0.4), (1_000_000, 0.01), (5000, 0.73)] {
            let m = 40_000;
            let s: Vec<f64> = (0..m).map(|_| binomial_distribution(&mut rng, n, p).unwrap() as f64).collect();
            let mean = s.iter().sum::<f64>() / m as f64;
            let var = s.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (m - 1) as f64;
            let (em, ev) = (n as f64 * p, n as f64 * p * (1.0 - p));
            assert!((mean - em).abs() < 5.0 * (ev / m as f64).sqrt(), "n={n} p={p}: mean {mean} vs {em}");
            assert!((var / ev - 1.0).abs() < 0.04, "n={n} p={p}: var {var} vs {ev}");
            assert!(s.iter().all(|&v| v <= n as f64));
        }
    }

    #[test]
    fn small_case_frequencies_match_the_pmf_and_edges() {
        let mut rng = UniformRNG::from_seed(9);
        let (n, p, m) = (4u64, 0.35, 200_000);
        let mut counts = [0usize; 5];
        (0..m).for_each(|_| counts[binomial_distribution(&mut rng, n, p).unwrap() as usize] += 1);
        let pmf = |k: u64| [1.0, 4.0, 6.0, 4.0, 1.0][k as usize] * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32);
        for k in 0..=n {
            assert!((counts[k as usize] as f64 / m as f64 - pmf(k)).abs() < 0.004, "k={k}");
        }
        assert_eq!(binomial_distribution(&mut rng, 7, 0.0).unwrap(), 0);
        assert_eq!(binomial_distribution(&mut rng, 7, 1.0).unwrap(), 7);
        assert!(binomial_distribution(&mut rng, 7, 1.5).is_err());
    }
}
