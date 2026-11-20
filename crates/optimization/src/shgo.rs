//! Simplicial homology global optimisation (Endres, Sandrock and Focke, 2018; SciPy's `shgo`
//! with `sampling_method='sobol'`).
//!
//! See the book entry [Global and combinatorial optimisation](https://jwkunz.github.io/lib_jsl/book/optimization/shgo-qap.html).

use crate::constrained::{ConstrainedOptions, ConstrainedProblem, sqp};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_geometry::algorithms::hull_nd::DelaunayNd;
use lib_jsl_random::quasi_random::SobolSequence;

/// Options for [`shgo`], with SciPy's defaults for Sobol sampling.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ShgoOptions {
    /// Sampling points per iteration.
    pub n: usize,
    /// Iterations: each adds `n` points of the sequence and re-triangulates.
    pub iters: usize,
    /// Tolerances of the local SQP solves.
    pub local: ConstrainedOptions,
}

impl Default for ShgoOptions {
    fn default() -> Self {
        ShgoOptions { n: 100, iters: 1, local: ConstrainedOptions { tolerance: 1e-10, max_iterations: 200 } }
    }
}

impl ShgoOptions {
    pub fn with_samples(n: usize, iters: usize) -> Self {
        ShgoOptions { n, iters, ..Default::default() }
    }
}

/// The result of [`shgo`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ShgoResult {
    /// The global minimiser found and its value.
    pub x: Vec<f64>,
    pub fun: f64,
    /// All distinct local minima found, sorted by value (SciPy's `xl`, `funl`).
    pub xl: Vec<Vec<f64>>,
    pub funl: Vec<f64>,
    /// Sampling evaluations, local solves, and the sampled points that were minimisers.
    pub nfev: usize,
    pub nlfev: usize,
    pub minimizers: Vec<Vec<f64>>,
}

/// The problem with the box added as inequalities.
struct Boxed<'a, P: ConstrainedProblem> {
    p: &'a P,
    lo: &'a [f64],
    hi: &'a [f64],
}

impl<P: ConstrainedProblem> ConstrainedProblem for Boxed<'_, P> {
    fn objective(&self, x: &[f64]) -> f64 {
        self.p.objective(x)
    }
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        self.p.gradient(x)
    }
    fn equalities(&self, x: &[f64]) -> Vec<f64> {
        self.p.equalities(x)
    }
    fn equality_jacobian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        self.p.equality_jacobian(x)
    }
    fn inequalities(&self, x: &[f64]) -> Vec<f64> {
        let mut g = self.p.inequalities(x);
        g.extend(x.iter().zip(self.lo).map(|(v, l)| l - v));
        g.extend(x.iter().zip(self.hi).map(|(v, h)| v - h));
        g
    }
    fn inequality_jacobian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let n = x.len();
        let mut j = self.p.inequality_jacobian(x);
        j.extend((0..n).map(|i| (0..n).map(|k| if k == i { -1.0 } else { 0.0 }).collect()));
        j.extend((0..n).map(|i| (0..n).map(|k| if k == i { 1.0 } else { 0.0 }).collect()));
        j
    }
}

// ANCHOR: shgo
/// Finds the global minimum of `problem` in the box `[lower, upper]` (with its inequality and
/// equality constraints) by simplicial homology global optimisation with Sobol sampling.
///
/// The box is sampled with the first `n · iters` points of the (unscrambled) Sobol sequence,
/// samples violating an inequality are dropped, and the rest are Delaunay-triangulated. A
/// sample whose value is lower than at all its neighbours in the triangulation is a
/// *minimiser*: it approximates a local minimum, and the number of them estimates the number of
/// basins (the rank of the complex's homology, in the language of the paper). A local SQP solve
/// is started from each minimiser in order of value, and the distinct results are returned with
/// the best of them.
pub fn shgo<P: ConstrainedProblem>(problem: &P, lower: &[f64], upper: &[f64], options: ShgoOptions) -> Result<ShgoResult, ErrorsJSL> {
    let d = lower.len();
    if d == 0 || upper.len() != d || lower.iter().zip(upper).any(|(l, u)| l.is_nan() || u.is_nan() || l >= u) {
        return Err(ErrorsJSL::InvalidInputRange("need finite bounds with lower < upper"));
    }
    let mut sobol = SobolSequence::new(d)?;
    let total = options.n * options.iters.max(1);
    let mut pts = Vec::new();
    let mut vals = Vec::new();
    let mut nfev = 0;
    for _ in 0..total {
        let u = sobol.next_point();
        let x: Vec<f64> = u.iter().zip(lower.iter().zip(upper)).map(|(t, (l, h))| l + t * (h - l)).collect();
        if problem.inequalities(&x).iter().any(|g| *g > 0.0) {
            continue;
        }
        nfev += 1;
        vals.push(problem.objective(&x));
        pts.push(x);
    }
    if pts.len() < d + 2 {
        return Err(ErrorsJSL::InvalidInputRange("too few feasible sampling points; increase n"));
    }
    // Neighbours in the triangulation (in 1-D, adjacent samples).
    let mut nbrs: Vec<Vec<usize>> = vec![Vec::new(); pts.len()];
    if d == 1 {
        let mut order: Vec<usize> = (0..pts.len()).collect();
        order.sort_by(|&a, &b| pts[a][0].total_cmp(&pts[b][0]));
        for w in order.windows(2) {
            nbrs[w[0]].push(w[1]);
            nbrs[w[1]].push(w[0]);
        }
    } else {
        let tri = DelaunayNd::new(&pts)?;
        for s in &tri.simplices {
            for &a in s {
                for &b in s {
                    if a != b && !nbrs[a].contains(&b) {
                        nbrs[a].push(b);
                    }
                }
            }
        }
    }
    let mut minimizers: Vec<usize> = (0..pts.len()).filter(|&i| !nbrs[i].is_empty() && nbrs[i].iter().all(|&j| vals[i] < vals[j])).collect();
    minimizers.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
    let boxed = Boxed { p: problem, lo: lower, hi: upper };
    let mut xl: Vec<Vec<f64>> = Vec::new();
    let mut funl: Vec<f64> = Vec::new();
    let mut nlfev = 0;
    for &m in &minimizers {
        nlfev += 1;
        let r = sqp(&boxed, pts[m].clone(), options.local)?;
        if r.constraint_violation > 1e-6 {
            continue;
        }
        let scale = lower.iter().zip(upper).map(|(l, h)| h - l).fold(0.0f64, f64::max);
        let dup = xl.iter().any(|x| x.iter().zip(&r.parameters).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max) <= 1e-6 * scale);
        if !dup {
            xl.push(r.parameters);
            funl.push(r.objective_value);
        }
    }
    let mut order: Vec<usize> = (0..xl.len()).collect();
    order.sort_by(|&a, &b| funl[a].total_cmp(&funl[b]));
    let xl: Vec<Vec<f64>> = order.iter().map(|&i| xl[i].clone()).collect();
    let funl: Vec<f64> = order.iter().map(|&i| funl[i]).collect();
    let (x, fun) = match xl.first() {
        Some(x) => (x.clone(), funl[0]),
        None => {
            let b = (0..pts.len()).min_by(|&a, &b| vals[a].total_cmp(&vals[b])).unwrap();
            (pts[b].clone(), vals[b])
        }
    };
    Ok(ShgoResult { x, fun, xl, funl, nfev, nlfev, minimizers: minimizers.iter().map(|&m| pts[m].clone()).collect() })
}
// ANCHOR_END: shgo

#[cfg(test)]
mod tests {
    use super::*;

    struct Himmelblau;
    impl ConstrainedProblem for Himmelblau {
        fn objective(&self, x: &[f64]) -> f64 {
            (x[0] * x[0] + x[1] - 11.0).powi(2) + (x[0] + x[1] * x[1] - 7.0).powi(2)
        }
    }

    #[test]
    fn finds_all_four_minima_of_himmelblau() {
        let r = shgo(&Himmelblau, &[-5.0, -5.0], &[5.0, 5.0], ShgoOptions::with_samples(128, 1)).unwrap();
        assert!(r.fun.abs() < 1e-10, "{}", r.fun);
        // All four global minima, each with f = 0.
        let four = r.funl.iter().filter(|f| f.abs() < 1e-8).count();
        assert_eq!(four, 4, "{:?} {:?}", r.xl, r.funl);
    }
}
