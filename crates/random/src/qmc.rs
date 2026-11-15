//! Tools for quasi-Monte Carlo sampling, following `scipy.stats.qmc`: Latin hypercube samples,
//! the discrepancy of a point set (how far it is from uniform), and scaling between the unit cube
//! and a box. The low-discrepancy sequences themselves, plain or scrambled, are
//! [`SobolSequence`](crate::quasi_random::SobolSequence) and
//! [`HaltonSequence`](crate::quasi_random::HaltonSequence).
//!
//! See the book entry [Quasi-random sequences](https://jwkunz.github.io/lib_jsl/book/random/quasi-random.html).

use crate::uniform_generator::UniformGenerator;
use lib_jsl_core::ErrorsJSL;

/// A discrepancy measure (SciPy's `method`). All are `L²` discrepancies with closed forms that
/// cost `O(n² d)`; each is the worst-case integration error, over a unit ball of functions, of
/// the equal-weight rule on the points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Discrepancy {
    /// Centered `L²` discrepancy (`'CD'`, SciPy's default): boxes with a corner at the nearest
    /// vertex of the cube. Invariant under reflections `x_k → 1 − x_k`.
    Centered,
    /// Wrap-around `L²` discrepancy (`'WD'`): boxes that wrap around the torus. Invariant under
    /// shifts modulo 1.
    WrapAround,
    /// Mixture discrepancy (`'MD'`), which combines the two and avoids some of their blind spots.
    Mixture,
    /// The `L²`-star discrepancy (`'L2-star'`): boxes anchored at the origin, Warnock's formula.
    /// Returned as the discrepancy itself; the others are returned squared, as SciPy does.
    L2Star,
}

// ANCHOR: discrepancy
/// The discrepancy of `points` in `[0, 1]^d` (SciPy's `qmc.discrepancy`): smaller is more uniform.
pub fn discrepancy(points: &[Vec<f64>], method: Discrepancy) -> Result<f64, ErrorsJSL> {
    let n = points.len();
    let d = points.first().map_or(0, |p| p.len());
    if n == 0 || d == 0 || points.iter().any(|p| p.len() != d) {
        return Err(ErrorsJSL::InvalidInputRange("discrepancy: need a non-empty set of points of one dimension"));
    }
    if points.iter().flatten().any(|x| !(0.0..=1.0).contains(x)) {
        return Err(ErrorsJSL::InvalidInputRange("discrepancy: the points must lie in the unit cube"));
    }
    let nf = n as f64;
    let df = d as i32;
    // Σ_i Π_k f(x_ik) and Σ_i Σ_j Π_k g(x_ik, x_jk).
    let single = |f: &dyn Fn(f64) -> f64| points.iter().map(|p| p.iter().map(|&x| f(x)).product::<f64>()).sum::<f64>();
    let pairs = |g: &dyn Fn(f64, f64) -> f64| {
        let mut total = 0.0;
        for p in points {
            for q in points {
                total += p.iter().zip(q).map(|(&x, &y)| g(x, y)).product::<f64>();
            }
        }
        total
    };
    Ok(match method {
        Discrepancy::Centered => {
            let s1 = single(&|x| 1.0 + 0.5 * (x - 0.5).abs() - 0.5 * (x - 0.5).powi(2));
            let s2 = pairs(&|x, y| 1.0 + 0.5 * (x - 0.5).abs() + 0.5 * (y - 0.5).abs() - 0.5 * (x - y).abs());
            (13.0f64 / 12.0).powi(df) - 2.0 / nf * s1 + s2 / (nf * nf)
        }
        Discrepancy::WrapAround => {
            let s2 = pairs(&|x, y| {
                let t = (x - y).abs();
                1.5 - t * (1.0 - t)
            });
            -(4.0f64 / 3.0).powi(df) + s2 / (nf * nf)
        }
        Discrepancy::Mixture => {
            let s1 = single(&|x| 5.0 / 3.0 - 0.25 * (x - 0.5).abs() - 0.25 * (x - 0.5).powi(2));
            let s2 = pairs(&|x, y| 15.0 / 8.0 - 0.25 * (x - 0.5).abs() - 0.25 * (y - 0.5).abs() - 0.75 * (x - y).abs() + 0.5 * (x - y).powi(2));
            (19.0f64 / 12.0).powi(df) - 2.0 / nf * s1 + s2 / (nf * nf)
        }
        Discrepancy::L2Star => {
            let s1 = single(&|x| 1.0 - x * x);
            let s2 = pairs(&|x, y| 1.0 - x.max(y));
            (3.0f64.powi(-df) - 2.0f64.powi(1 - df) / nf * s1 + s2 / (nf * nf)).max(0.0).sqrt()
        }
    })
}
// ANCHOR_END: discrepancy

// ANCHOR: latin_hypercube
/// `n` points of a Latin hypercube in `[0, 1)^d` (SciPy's `qmc.LatinHypercube`): in every
/// coordinate, each of the `n` intervals `[i/n, (i+1)/n)` holds exactly one point. Each coordinate
/// takes an independent random permutation of the intervals; the point is placed uniformly in its
/// cell, or at the cell's centre when `centered` (SciPy's `scramble=False`).
pub fn latin_hypercube<G: UniformGenerator>(n: usize, d: usize, centered: bool, rng: &mut G) -> Vec<Vec<f64>> {
    let mut points = vec![vec![0.0; d]; n];
    for k in 0..d {
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        for (p, &cell) in points.iter_mut().zip(&perm) {
            let offset = if centered { 0.5 } else { rng.next_f64() };
            p[k] = (cell as f64 + offset) / n as f64;
        }
    }
    points
}
// ANCHOR_END: latin_hypercube

/// Maps points from the unit cube to the box `[lower, upper]` (SciPy's `qmc.scale`), or back
/// when `reverse`.
pub fn scale(points: &[Vec<f64>], lower: &[f64], upper: &[f64], reverse: bool) -> Result<Vec<Vec<f64>>, ErrorsJSL> {
    let d = lower.len();
    if upper.len() != d || points.iter().any(|p| p.len() != d) {
        return Err(ErrorsJSL::InvalidInputRange("scale: the bounds and points must have one dimension"));
    }
    if lower.iter().zip(upper).any(|(l, u)| l.partial_cmp(u) != Some(std::cmp::Ordering::Less)) {
        return Err(ErrorsJSL::InvalidInputRange("scale: need lower < upper in every coordinate"));
    }
    Ok(points
        .iter()
        .map(|p| p.iter().zip(lower.iter().zip(upper)).map(|(&x, (&l, &u))| if reverse { (x - l) / (u - l) } else { l + x * (u - l) }).collect())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quasi_random::{HaltonSequence, SobolSequence};
    use crate::uniform_generator::UniformRNG;

    #[test]
    fn discrepancy_of_small_sets() {
        // One point at the centre of the unit interval: the centered discrepancy is
        // 13/12 − 2(1) + 1 = 1/12; the L²-star discrepancy is √(1/3 − 3/4 + 1/2) = √(1/12).
        let p = vec![vec![0.5]];
        assert!((discrepancy(&p, Discrepancy::Centered).unwrap() - 1.0 / 12.0).abs() < 1e-15);
        assert!((discrepancy(&p, Discrepancy::L2Star).unwrap() - (1.0f64 / 12.0).sqrt()).abs() < 1e-15);
        assert!(discrepancy(&[vec![1.5]], Discrepancy::Centered).is_err());
        assert!(discrepancy(&[], Discrepancy::Centered).is_err());
    }

    #[test]
    fn scrambled_sequences_beat_random_points() {
        let mut rng = UniformRNG::from_seed(7);
        let n = 256;
        let random: Vec<Vec<f64>> = (0..n).map(|_| (0..4).map(|_| rng.next_f64()).collect()).collect();
        let mut sobol = SobolSequence::scrambled(4, &mut rng).unwrap();
        let sobol_pts: Vec<Vec<f64>> = (0..n).map(|_| sobol.next_point()).collect();
        let mut halton = HaltonSequence::scrambled(4, &mut rng).unwrap();
        let halton_pts: Vec<Vec<f64>> = (0..n).map(|_| halton.next_point()).collect();
        let lhs = latin_hypercube(n, 4, false, &mut rng);
        let cd = |p: &[Vec<f64>]| discrepancy(p, Discrepancy::Centered).unwrap();
        assert!(cd(&sobol_pts) < cd(&random) / 10.0, "{} {}", cd(&sobol_pts), cd(&random));
        assert!(cd(&halton_pts) < cd(&random) / 10.0);
        assert!(cd(&lhs) < cd(&random));
        // Scrambled Sobol keeps the net property: each 1-D projection of 256 points is stratified.
        for k in 0..4 {
            let mut cells: Vec<usize> = sobol_pts.iter().map(|p| (p[k] * 256.0) as usize).collect();
            cells.sort_unstable();
            assert!(cells.iter().enumerate().all(|(i, &c)| c == i));
            let mut cells: Vec<usize> = lhs.iter().map(|p| (p[k] * 256.0) as usize).collect();
            cells.sort_unstable();
            assert!(cells.iter().enumerate().all(|(i, &c)| c == i));
        }
        assert!(sobol_pts.iter().chain(&halton_pts).flatten().all(|&x| (0.0..1.0).contains(&x)));
        let boxed = scale(&lhs, &[-1.0; 4], &[3.0; 4], false).unwrap();
        let back = scale(&boxed, &[-1.0; 4], &[3.0; 4], true).unwrap();
        assert!(back.iter().flatten().zip(lhs.iter().flatten()).all(|(a, b)| (a - b).abs() < 1e-15));
    }
}
