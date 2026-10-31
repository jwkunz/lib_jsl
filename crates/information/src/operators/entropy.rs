//! Entropy and related information measures (in bits).

/// The Shannon entropy of the byte distribution of `data`, in bits per byte (0 to 8).
/// Empty input has zero entropy.
pub fn entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let n = data.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum()
}

/// The Shannon entropy in bits of a probability distribution. Probabilities should sum to 1;
/// zero entries contribute nothing. Returns `NaN` if any entry is negative or non-finite.
pub fn entropy_from_probabilities(p: &[f64]) -> f64 {
    if p.iter().any(|&x| !x.is_finite() || x < 0.0) {
        return f64::NAN;
    }
    p.iter().filter(|&&x| x > 0.0).map(|&x| -x * x.log2()).sum()
}

/// The cross entropy `-Σ p log2 q` in bits. Infinite if `q` is zero where `p` is positive.
/// Returns `NaN` if the slices differ in length or contain negative values.
pub fn cross_entropy(p: &[f64], q: &[f64]) -> f64 {
    if p.len() != q.len() || p.iter().chain(q).any(|&x| !x.is_finite() || x < 0.0) {
        return f64::NAN;
    }
    p.iter()
        .zip(q)
        .filter(|(pi, _)| **pi > 0.0)
        .map(|(&pi, &qi)| if qi == 0.0 { f64::INFINITY } else { -pi * qi.log2() })
        .sum()
}

/// The Kullback-Leibler divergence `Σ p log2(p / q)` in bits.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    cross_entropy(p, q) - entropy_from_probabilities(p)
}

/// The mutual information in bits of a joint distribution given as a row-major `rows x cols`
/// probability table.
pub fn mutual_information(joint: &[f64], rows: usize, cols: usize) -> f64 {
    if rows * cols != joint.len() || joint.iter().any(|&x| !x.is_finite() || x < 0.0) {
        return f64::NAN;
    }
    let mut pr = vec![0.0; rows];
    let mut pc = vec![0.0; cols];
    for r in 0..rows {
        for c in 0..cols {
            pr[r] += joint[r * cols + c];
            pc[c] += joint[r * cols + c];
        }
    }
    let mut mi = 0.0;
    for r in 0..rows {
        for c in 0..cols {
            let p = joint[r * cols + c];
            if p > 0.0 {
                mi += p * (p / (pr[r] * pc[c])).log2();
            }
        }
    }
    mi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_entropy() {
        assert_eq!(entropy(&[]), 0.0);
        assert_eq!(entropy(&[7u8; 100]), 0.0);
        assert!((entropy(&[0, 1]) - 1.0).abs() < 1e-15);
        let all: Vec<u8> = (0..=255u8).collect();
        assert!((entropy(&all) - 8.0).abs() < 1e-12);
        // 3:1 split -> H(0.25) = 0.811278...
        assert!((entropy(&[0, 0, 0, 1]) - 0.8112781244591328).abs() < 1e-14);
    }

    #[test]
    fn distributions() {
        assert!((entropy_from_probabilities(&[0.5, 0.5]) - 1.0).abs() < 1e-15);
        assert!((entropy_from_probabilities(&[0.25; 4]) - 2.0).abs() < 1e-15);
        assert_eq!(entropy_from_probabilities(&[1.0, 0.0]), 0.0);
        assert!(entropy_from_probabilities(&[-0.1, 1.1]).is_nan());
    }

    #[test]
    fn divergences_and_information() {
        let p = [0.5, 0.5];
        let q = [0.25, 0.75];
        let kl = 0.5 * (0.5f64 / 0.25).log2() + 0.5 * (0.5f64 / 0.75).log2();
        assert!((kl_divergence(&p, &q) - kl).abs() < 1e-14);
        assert!(kl_divergence(&p, &p).abs() < 1e-15);
        assert!(cross_entropy(&[1.0, 0.0], &[0.0, 1.0]).is_infinite());
        // Independent variables carry no mutual information; identical ones carry H.
        assert!(mutual_information(&[0.25, 0.25, 0.25, 0.25], 2, 2).abs() < 1e-15);
        assert!((mutual_information(&[0.5, 0.0, 0.0, 0.5], 2, 2) - 1.0).abs() < 1e-15);
    }
}
