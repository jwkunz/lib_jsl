//! Image feature extraction.

pub mod sift;
pub mod surf;
pub mod hough_transform;
pub mod blob_detection;

use lib_jsl_core::ErrorsJSL;

/// A correspondence between a descriptor in set A and one in set B.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DescriptorMatch {
    pub index_a: usize,
    pub index_b: usize,
    /// Euclidean distance between the two descriptors.
    pub distance: f64,
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// Matches descriptors by nearest neighbour with Lowe's ratio test.
///
/// For each descriptor in `a` the two nearest descriptors in `b` are found; the match is accepted
/// only if the nearest is closer than `ratio` times the second nearest (0.75 is typical, smaller
/// is stricter). This rejects ambiguous descriptors that resemble several candidates. With
/// `mutual` set, a match must also be the nearest neighbour of its partner in the other direction.
pub fn match_descriptors(a: &[Vec<f64>], b: &[Vec<f64>], ratio: f64, mutual: bool) -> Result<Vec<DescriptorMatch>, ErrorsJSL> {
    if !(ratio > 0.0 && ratio <= 1.0) {
        return Err(ErrorsJSL::InvalidInputRange("ratio must be in (0, 1]"));
    }
    let dim = a.first().or(b.first()).map_or(0, |d| d.len());
    if a.iter().chain(b).any(|d| d.len() != dim) {
        return Err(ErrorsJSL::InvalidInputRange("All descriptors must have the same length"));
    }
    if b.is_empty() {
        return Ok(Vec::new());
    }
    let nearest = |query: &[f64], set: &[Vec<f64>]| -> (usize, f64, f64) {
        let (mut best, mut best_d, mut second_d) = (0usize, f64::INFINITY, f64::INFINITY);
        for (i, cand) in set.iter().enumerate() {
            let d = euclidean(query, cand);
            if d < best_d {
                second_d = best_d;
                best_d = d;
                best = i;
            } else if d < second_d {
                second_d = d;
            }
        }
        (best, best_d, second_d)
    };
    // ANCHOR: ratio_test
    let mut out = Vec::new();
    for (i, q) in a.iter().enumerate() {
        let (j, d1, d2) = nearest(q, b);
        // With a single candidate there is no second neighbour to compare against.
        let passes = d2.is_infinite() || d1 < ratio * d2;
        if passes && (!mutual || nearest(&b[j], a).0 == i) {
            out.push(DescriptorMatch { index_a: i, index_b: j, distance: d1 });
        }
    }
    Ok(out)
    // ANCHOR_END: ratio_test
}

#[cfg(test)]
mod matching_tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }

    fn random_set(n: usize, dim: usize, seed: &mut u64) -> Vec<Vec<f64>> {
        (0..n).map(|_| (0..dim).map(|_| lcg(seed)).collect()).collect()
    }

    #[test]
    fn recovers_a_shuffled_noisy_copy() {
        let mut seed = 3;
        let a = random_set(60, 32, &mut seed);
        // b is a permutation of a with a little noise.
        let perm: Vec<usize> = (0..60).map(|i| (i * 17 + 5) % 60).collect();
        let b: Vec<Vec<f64>> = perm.iter().map(|&k| a[k].iter().map(|v| v + 0.01 * lcg(&mut seed)).collect()).collect();
        let m = match_descriptors(&a, &b, 0.8, false).unwrap();
        assert_eq!(m.len(), 60);
        for mm in &m {
            assert_eq!(perm[mm.index_b], mm.index_a);
            assert!(mm.distance < 0.2);
        }
        assert_eq!(match_descriptors(&a, &b, 0.8, true).unwrap().len(), 60);
    }

    #[test]
    fn the_ratio_test_rejects_ambiguous_descriptors() {
        let a = vec![vec![0.0, 0.0]];
        // Two near-identical candidates: the nearest is not clearly better than the second.
        let b = vec![vec![1.0, 0.0], vec![1.0, 0.01]];
        assert!(match_descriptors(&a, &b, 0.8, false).unwrap().is_empty());
        // A distinct nearest neighbour passes.
        let b2 = vec![vec![0.1, 0.0], vec![1.0, 1.0]];
        let m = match_descriptors(&a, &b2, 0.8, false).unwrap();
        assert_eq!((m[0].index_b, (m[0].distance - 0.1).abs() < 1e-12), (0, true));
        // A single candidate has no competitor and is accepted.
        assert_eq!(match_descriptors(&a, &[vec![5.0, 5.0]], 0.5, false).unwrap().len(), 1);
    }

    #[test]
    fn mutual_checking_discards_one_sided_matches() {
        // Both a0 and a1 prefer b0, but b0 prefers a0.
        let a = vec![vec![0.0], vec![0.4]];
        let b = vec![vec![0.1], vec![10.0]];
        assert_eq!(match_descriptors(&a, &b, 1.0, false).unwrap().len(), 2);
        let m = match_descriptors(&a, &b, 1.0, true).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!((m[0].index_a, m[0].index_b), (0, 0));
    }

    #[test]
    fn empty_sets_and_invalid_arguments() {
        let a = vec![vec![1.0, 2.0]];
        assert!(match_descriptors(&a, &[], 0.8, false).unwrap().is_empty());
        assert!(match_descriptors(&[], &a, 0.8, false).unwrap().is_empty());
        assert!(match_descriptors(&a, &a, 0.0, false).is_err() && match_descriptors(&a, &a, 1.5, false).is_err());
        assert!(match_descriptors(&a, &[vec![1.0]], 0.8, false).is_err(), "mismatched dimensions");
    }
}
