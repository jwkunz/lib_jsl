//! Isotonic regression by the pool-adjacent-violators algorithm (SciPy's
//! `optimize.isotonic_regression`).
//!
//! See the book entry [Brute force, isotonic regression and gradient checks](https://jwkunz.github.io/lib_jsl/book/optimization/brute-isotonic.html).

use lib_jsl_core::ErrorsJSL;

/// The result of [`isotonic_regression`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct IsotonicResult {
    /// The fitted monotone values, one per observation.
    pub x: Vec<f64>,
    /// The total weight of each block of equal fitted values.
    pub weights: Vec<f64>,
    /// The start of each block, followed by the number of observations: block `k` covers
    /// `blocks[k]..blocks[k + 1]`.
    pub blocks: Vec<usize>,
}

// ANCHOR: isotonic
/// The weighted least-squares fit of `y` by a non-decreasing (or, with `increasing = false`,
/// non-increasing) sequence, `min Σ wᵢ (xᵢ − yᵢ)²` subject to `x₁ ≤ x₂ ≤ … ≤ xₙ` (SciPy's
/// `isotonic_regression`). The pool-adjacent-violators algorithm merges each new observation
/// with the blocks before it while their weighted means are out of order; every block's value
/// is its weighted mean, in O(n) time. `weights` default to one and must be positive.
pub fn isotonic_regression(y: &[f64], weights: Option<&[f64]>, increasing: bool) -> Result<IsotonicResult, ErrorsJSL> {
    let n = y.len();
    let w: Vec<f64> = match weights {
        Some(w) if w.len() != n => return Err(ErrorsJSL::IncompatibleArraySizes((w.len(), n))),
        Some(w) if w.iter().any(|v| v.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || !v.is_finite()) => {
            return Err(ErrorsJSL::InvalidInputRange("isotonic_regression: weights must be positive and finite"));
        }
        Some(w) => w.to_vec(),
        None => vec![1.0; n],
    };
    let sign = if increasing { 1.0 } else { -1.0 };
    // Blocks as (weighted mean, weight, start).
    let mut blocks: Vec<(f64, f64, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let (mut mean, mut weight, mut start) = (sign * y[i], w[i], i);
        while let Some(&(m, wt, s)) = blocks.last() {
            if m < mean {
                break;
            }
            mean = (m * wt + mean * weight) / (wt + weight);
            weight += wt;
            start = s;
            blocks.pop();
        }
        blocks.push((mean, weight, start));
    }
    let mut x = vec![0.0; n];
    let mut starts: Vec<usize> = blocks.iter().map(|b| b.2).collect();
    starts.push(n);
    for (k, b) in blocks.iter().enumerate() {
        x[starts[k]..starts[k + 1]].fill(sign * b.0);
    }
    Ok(IsotonicResult { x, weights: blocks.iter().map(|b| b.1).collect(), blocks: starts })
}
// ANCHOR_END: isotonic

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pava_matches_definition() {
        let y = [1.0, 3.0, 2.0, 4.0, 3.5, 5.0];
        let r = isotonic_regression(&y, None, true).unwrap();
        assert_eq!(r.x, vec![1.0, 2.5, 2.5, 3.75, 3.75, 5.0]);
        assert_eq!(r.blocks, vec![0, 1, 3, 5, 6]);
        assert_eq!(r.weights, vec![1.0, 2.0, 2.0, 1.0]);
        // Decreasing, weighted: 5 and 6 pool to (5·1 + 6·3)/4.
        let r = isotonic_regression(&[5.0, 6.0, 1.0], Some(&[1.0, 3.0, 1.0]), false).unwrap();
        assert_eq!(r.x, vec![5.75, 5.75, 1.0]);
        // Already monotone: unchanged, one block per observation.
        let r = isotonic_regression(&[1.0, 2.0, 3.0], None, true).unwrap();
        assert_eq!(r.blocks, vec![0, 1, 2, 3]);
        assert!(isotonic_regression(&[1.0], Some(&[0.0]), true).is_err());
    }
}
