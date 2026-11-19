//! Mixed-integer linear programming by branch and bound (SciPy's `optimize.milp`, which uses
//! HiGHS): a [`LinearProgram`] in which some variables must take integer values.
//!
//! See the book entry [Levenberg-Marquardt for systems, mixed-integer programming and truncated Newton](https://jwkunz.github.io/lib_jsl/book/optimization/milp-tnc.html#branch-and-bound).

use crate::linear_programming::{LinearProgram, LpStatus};
use lib_jsl_core::ErrorsJSL;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Options for [`milp`] (SciPy's `options`).
#[derive(Clone, Copy, Debug)]
pub struct MilpOptions {
    /// Stop when the gap between the best integer solution and the best bound, relative to the
    /// former, is at most this (HiGHS's default 10⁻⁴).
    pub mip_rel_gap: f64,
    /// The largest number of branch-and-bound nodes (LP relaxations) to solve.
    pub node_limit: usize,
    /// How far from an integer a value may be and still count as one (HiGHS's 10⁻⁶).
    pub integrality_tolerance: f64,
}

impl Default for MilpOptions {
    fn default() -> Self {
        MilpOptions { mip_rel_gap: 1e-4, node_limit: 100_000, integrality_tolerance: 1e-6 }
    }
}

/// The outcome of [`milp`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MilpStatus {
    /// An optimal solution within `mip_rel_gap`.
    Optimal,
    /// No integer point satisfies the constraints.
    Infeasible,
    /// The LP relaxation is unbounded.
    Unbounded,
    /// The node limit was reached; `x` is the best integer solution found, if any.
    NodeLimit,
}

/// The result of [`milp`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MilpResult {
    pub status: MilpStatus,
    /// The best integer-feasible point (integer variables rounded exactly); empty if none.
    pub x: Vec<f64>,
    /// Its objective value (`∞` if none).
    pub objective: f64,
    /// The best bound on the optimum, from the open nodes' relaxations.
    pub dual_bound: f64,
    /// `(objective − dual_bound)/|objective|`.
    pub mip_gap: f64,
    /// LP relaxations solved.
    pub nodes: usize,
}

struct Node {
    bound: f64,
    depth: usize,
    bounds: Vec<(f64, f64)>,
}

impl PartialEq for Node {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Node {}
impl PartialOrd for Node {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Node {
    /// Best bound first (a min-heap on the bound), then deeper nodes, which find integer points
    /// sooner.
    fn cmp(&self, o: &Self) -> Ordering {
        o.bound.total_cmp(&self.bound).then(self.depth.cmp(&o.depth))
    }
}

// ANCHOR: milp
/// Minimises `cᵀx` over a [`LinearProgram`]'s constraints with `x[i]` integer where
/// `integrality[i]` (SciPy's `milp`). Branch and bound: each node solves the LP relaxation with
/// tightened bounds; a relaxation no better than the best integer point found (the incumbent)
/// is pruned; an integral one becomes the incumbent; otherwise the most fractional variable
/// `xⱼ = v` splits the node into `xⱼ ≤ ⌊v⌋` and `xⱼ ≥ ⌈v⌉`. Nodes are explored best bound first,
/// and the search stops when the incumbent is within `mip_rel_gap` of the best open bound.
pub fn milp(problem: &LinearProgram, integrality: &[bool], options: MilpOptions) -> Result<MilpResult, ErrorsJSL> {
    let n = problem.n_vars();
    if integrality.len() != n {
        return Err(ErrorsJSL::IncompatibleArraySizes((integrality.len(), n)));
    }
    let tol = options.integrality_tolerance;
    // Integer variables take integer bounds.
    let root_bounds: Vec<(f64, f64)> = problem
        .bounds()
        .iter()
        .zip(integrality)
        .map(|(&(lo, hi), &int)| if int { ((lo - tol).ceil(), (hi + tol).floor()) } else { (lo, hi) })
        .collect();
    let mut result = MilpResult { status: MilpStatus::Infeasible, x: Vec::new(), objective: f64::INFINITY, dual_bound: f64::INFINITY, mip_gap: f64::INFINITY, nodes: 0 };
    if root_bounds.iter().any(|&(lo, hi)| lo > hi) {
        return Ok(result);
    }
    let solve = |bounds: &[(f64, f64)]| {
        let mut lp = problem.clone();
        for (i, &(lo, hi)) in bounds.iter().enumerate() {
            lp = lp.with_bounds(i, lo, hi);
        }
        lp.solve()
    };
    let mut heap = BinaryHeap::new();
    heap.push(Node { bound: f64::NEG_INFINITY, depth: 0, bounds: root_bounds });
    let gap_closed = |incumbent: f64, bound: f64| incumbent - bound <= options.mip_rel_gap * incumbent.abs().max(1e-300) || incumbent - bound <= 1e-9 * (1.0 + incumbent.abs());
    while let Some(node) = heap.pop() {
        if result.objective.is_finite() && gap_closed(result.objective, node.bound) {
            // Every open node's bound is at least this one's.
            heap.push(node);
            break;
        }
        if result.nodes >= options.node_limit {
            heap.push(node);
            result.status = MilpStatus::NodeLimit;
            break;
        }
        let lp = solve(&node.bounds)?;
        result.nodes += 1;
        match lp.status {
            LpStatus::Infeasible => continue,
            LpStatus::Unbounded => {
                result.status = MilpStatus::Unbounded;
                return Ok(result);
            }
            LpStatus::Optimal => {}
        }
        if result.objective.is_finite() && lp.objective >= result.objective - 1e-12 * (1.0 + result.objective.abs()) {
            continue;
        }
        // The most fractional integer variable.
        let branch = (0..n)
            .filter(|&i| integrality[i])
            .map(|i| (i, (lp.x[i] - lp.x[i].round()).abs()))
            .filter(|&(_, f)| f > tol)
            .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)));
        match branch {
            None => {
                let x: Vec<f64> = (0..n).map(|i| if integrality[i] { lp.x[i].round() } else { lp.x[i] }).collect();
                let obj: f64 = problem.objective().iter().zip(&x).map(|(c, v)| c * v).sum();
                result.x = x;
                result.objective = obj;
                result.status = MilpStatus::Optimal;
            }
            Some((j, _)) => {
                let v = lp.x[j];
                let mut down = node.bounds.clone();
                down[j].1 = v.floor();
                let mut up = node.bounds;
                up[j].0 = v.ceil();
                for b in [down, up] {
                    if b[j].0 <= b[j].1 {
                        heap.push(Node { bound: lp.objective, depth: node.depth + 1, bounds: b });
                    }
                }
            }
        }
    }
    let open_bound = heap.iter().map(|nd| nd.bound).fold(f64::INFINITY, f64::min);
    result.dual_bound = open_bound.min(result.objective);
    if result.status == MilpStatus::NodeLimit && result.x.is_empty() {
        result.objective = f64::INFINITY;
    }
    if result.objective.is_finite() {
        result.mip_gap = if result.objective == result.dual_bound { 0.0 } else { (result.objective - result.dual_bound) / result.objective.abs().max(1e-300) };
    }
    Ok(result)
}
// ANCHOR_END: milp

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knapsack_and_scipy_example() {
        // SciPy's documentation example: max x₂ s.t. −x₁ + x₂ ≤ 1, 3x₁ + 2x₂ ≤ 12, 2x₁ + 3x₂ ≤ 12,
        // integers ≥ 0: optimum x = (1, 2) or (2, 2), objective −2.
        let lp = LinearProgram::new(vec![0.0, -1.0]).with_less_equal(vec![-1.0, 1.0], 1.0).with_less_equal(vec![3.0, 2.0], 12.0).with_less_equal(vec![2.0, 3.0], 12.0);
        let r = milp(&lp, &[true, true], MilpOptions::default()).unwrap();
        assert_eq!(r.status, MilpStatus::Optimal);
        assert_eq!(r.objective, -2.0);
        // 0-1 knapsack by brute force.
        let (v, w) = ([10.0, 13.0, 7.0, 8.0, 9.0, 4.0], [5.0, 6.0, 4.0, 4.0, 5.0, 2.0]);
        let mut lp = LinearProgram::new(v.iter().map(|x| -x).collect()).with_less_equal(w.to_vec(), 15.0);
        for i in 0..6 {
            lp = lp.with_bounds(i, 0.0, 1.0);
        }
        let r = milp(&lp, &[true; 6], MilpOptions { mip_rel_gap: 0.0, ..Default::default() }).unwrap();
        let best = (0u32..64)
            .filter(|m| (0..6).filter(|i| m >> i & 1 == 1).map(|i| w[i]).sum::<f64>() <= 15.0)
            .map(|m| (0..6).filter(|i| m >> i & 1 == 1).map(|i| v[i]).sum::<f64>())
            .fold(0.0, f64::max);
        assert_eq!(-r.objective, best);
        // Infeasible: 2x = 1 with x integer.
        let lp = LinearProgram::new(vec![1.0]).with_equal(vec![2.0], 1.0);
        assert_eq!(milp(&lp, &[true], MilpOptions::default()).unwrap().status, MilpStatus::Infeasible);
    }
}
