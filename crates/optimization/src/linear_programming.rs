//! Linear programming by the two-phase simplex method.
//!
//! Solves `minimise cᵀx` subject to
//!
//! - `A_ub x <= b_ub` (inequalities),
//! - `A_eq x  = b_eq` (equalities), and
//! - per-variable bounds `lower <= x <= upper` (default `0 <= x < ∞`; use
//!   `f64::NEG_INFINITY` / `f64::INFINITY` for free sides).
//!
//! The problem is converted to standard form (shifting or splitting variables as the bounds
//! require, adding slack variables), phase one finds a feasible basis with artificial variables,
//! and phase two optimises. Pivoting uses Bland's rule, so the method cannot cycle on degenerate
//! problems; it is reliable rather than fast, and is intended for problems of up to a few hundred
//! variables and constraints. To maximise, negate the objective.

use lib_jsl_core::ErrorsJSL;

/// A linear program (see the module docs). Build with [`LinearProgram::new`] and the `with_*`
/// methods.
#[derive(Debug, Clone)]
pub struct LinearProgram {
    objective: Vec<f64>,
    a_ub: Vec<Vec<f64>>,
    b_ub: Vec<f64>,
    a_eq: Vec<Vec<f64>>,
    b_eq: Vec<f64>,
    bounds: Vec<(f64, f64)>,
}

/// The outcome class of a solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpStatus {
    Optimal,
    Infeasible,
    Unbounded,
}

/// The solution of a linear program. `x` and `objective` are meaningful only when `status` is
/// [`LpStatus::Optimal`].
#[derive(Debug, Clone, PartialEq)]
pub struct LpSolution {
    pub status: LpStatus,
    pub x: Vec<f64>,
    pub objective: f64,
    /// Simplex pivots performed across both phases.
    pub iterations: usize,
}

impl LinearProgram {
    /// A problem with the given objective coefficients and `0 <= x` bounds on every variable.
    pub fn new(objective: Vec<f64>) -> Self {
        let n = objective.len();
        LinearProgram { objective, a_ub: Vec::new(), b_ub: Vec::new(), a_eq: Vec::new(), b_eq: Vec::new(), bounds: vec![(0.0, f64::INFINITY); n] }
    }

    /// Adds the constraint `row · x <= rhs`.
    pub fn with_less_equal(mut self, row: Vec<f64>, rhs: f64) -> Self {
        self.a_ub.push(row);
        self.b_ub.push(rhs);
        self
    }

    /// Adds the constraint `row · x >= rhs`.
    pub fn with_greater_equal(self, row: Vec<f64>, rhs: f64) -> Self {
        self.with_less_equal(row.into_iter().map(|v| -v).collect(), -rhs)
    }

    /// Adds the constraint `row · x = rhs`.
    pub fn with_equal(mut self, row: Vec<f64>, rhs: f64) -> Self {
        self.a_eq.push(row);
        self.b_eq.push(rhs);
        self
    }

    /// Sets the bounds of variable `index`.
    pub fn with_bounds(mut self, index: usize, lower: f64, upper: f64) -> Self {
        self.bounds[index] = (lower, upper);
        self
    }

    /// Makes variable `index` unrestricted in sign.
    pub fn with_free_variable(self, index: usize) -> Self {
        self.with_bounds(index, f64::NEG_INFINITY, f64::INFINITY)
    }

    fn validate(&self) -> Result<(), ErrorsJSL> {
        let n = self.objective.len();
        if n == 0 {
            return Err(ErrorsJSL::Misconfiguration("The objective must have at least one variable."));
        }
        if self.a_ub.iter().chain(&self.a_eq).any(|r| r.len() != n) {
            return Err(ErrorsJSL::IncompatibleArraySizes((n, 0)));
        }
        if self.objective.iter().chain(self.b_ub.iter()).chain(self.b_eq.iter()).chain(self.a_ub.iter().flatten()).chain(self.a_eq.iter().flatten()).any(|v| !v.is_finite()) {
            return Err(ErrorsJSL::InvalidInputRange("Coefficients must be finite."));
        }
        if self.bounds.iter().any(|&(lo, hi)| lo.is_nan() || hi.is_nan() || lo > hi || lo == f64::INFINITY || hi == f64::NEG_INFINITY) {
            return Err(ErrorsJSL::Misconfiguration("Bounds must satisfy lower <= upper."));
        }
        Ok(())
    }

    /// Solves the program. Infeasible and unbounded programs are reported through
    /// [`LpSolution::status`]; an error means invalid input, or (only conceivably, through
    /// rounding) that 50 000 pivots passed without termination, which used to be reported as
    /// optimal.
    pub fn solve(&self) -> Result<LpSolution, ErrorsJSL> {
        self.validate()?;
        let n = self.objective.len();

        // Map original variables onto non-negative standard-form columns:
        // x_i = offset_i + sign_i * y_p  (+ optionally  - y_m  for free variables).
        struct Map {
            offset: f64,
            plus: usize,
            sign: f64,
            minus: Option<usize>,
        }
        let mut cols = 0usize;
        let mut maps = Vec::with_capacity(n);
        let mut upper_rows: Vec<(usize, f64)> = Vec::new(); // (column, upper bound on y)
        for &(lo, hi) in &self.bounds {
            if lo.is_finite() {
                maps.push(Map { offset: lo, plus: cols, sign: 1.0, minus: None });
                if hi.is_finite() {
                    upper_rows.push((cols, hi - lo));
                }
                cols += 1;
            } else if hi.is_finite() {
                maps.push(Map { offset: hi, plus: cols, sign: -1.0, minus: None });
                cols += 1;
            } else {
                maps.push(Map { offset: 0.0, plus: cols, sign: 1.0, minus: Some(cols + 1) });
                cols += 2;
            }
        }
        let structural = cols;

        // Constraint rows over structural columns: (coefficients, rhs, is_equality).
        let expand = |row: &[f64]| -> (Vec<f64>, f64) {
            let mut r = vec![0.0; structural];
            let mut shift = 0.0;
            for (i, m) in maps.iter().enumerate() {
                r[m.plus] += m.sign * row[i];
                if let Some(mi) = m.minus {
                    r[mi] -= row[i];
                }
                shift += row[i] * m.offset;
            }
            (r, shift)
        };
        let mut rows: Vec<(Vec<f64>, f64, bool)> = Vec::new();
        for (a, &b) in self.a_ub.iter().zip(&self.b_ub) {
            let (r, shift) = expand(a);
            rows.push((r, b - shift, false));
        }
        for (a, &b) in self.a_eq.iter().zip(&self.b_eq) {
            let (r, shift) = expand(a);
            rows.push((r, b - shift, true));
        }
        for &(c, ub) in &upper_rows {
            let mut r = vec![0.0; structural];
            r[c] = 1.0;
            rows.push((r, ub, false));
        }

        let m = rows.len();
        let slacks = rows.iter().filter(|r| !r.2).count();
        let total = structural + slacks + m; // structural + slack + artificial (one per row)
        // Tableau rows: [coefficients (total), rhs]. Make every rhs non-negative.
        let mut t = vec![vec![0.0; total + 1]; m];
        let mut slack_col = structural;
        for (i, (r, b, eq)) in rows.iter().enumerate() {
            t[i][..structural].copy_from_slice(r);
            if !eq {
                t[i][slack_col] = 1.0;
                slack_col += 1;
            }
            t[i][total] = *b;
            if *b < 0.0 {
                t[i].iter_mut().for_each(|v| *v = -*v);
            }
            t[i][structural + slacks + i] = 1.0; // artificial
        }
        let mut basis: Vec<usize> = (0..m).map(|i| structural + slacks + i).collect();
        let mut iterations = 0usize;

        // Phase one: minimise the sum of artificials.
        let mut cost1 = vec![0.0; total];
        cost1[structural + slacks..].iter_mut().for_each(|c| *c = 1.0);
        let status = simplex(&mut t, &mut basis, &cost1, total, &mut iterations, None).ok_or(PIVOT_LIMIT)?;
        debug_assert!(status != LpStatus::Unbounded, "phase one is bounded below by zero");
        let infeasibility: f64 = basis.iter().enumerate().filter(|(_, b)| **b >= structural + slacks).map(|(i, _)| t[i][total]).sum();
        let scale = 1.0 + rows.iter().map(|r| r.1.abs()).fold(0.0, f64::max);
        if infeasibility > 1e-9 * scale {
            return Ok(LpSolution { status: LpStatus::Infeasible, x: vec![], objective: f64::NAN, iterations });
        }
        // Drive any artificial still in the basis (at value zero) out, or drop its redundant row.
        let mut i = 0;
        while i < t.len() {
            if basis[i] >= structural + slacks {
                if let Some(j) = (0..structural + slacks).find(|&j| t[i][j].abs() > 1e-9) {
                    pivot(&mut t, &mut basis, i, j, total);
                    i += 1;
                } else {
                    t.remove(i);
                    basis.remove(i);
                }
            } else {
                i += 1;
            }
        }

        // Phase two: the true objective over structural columns; artificial columns are barred.
        let mut cost2 = vec![0.0; total];
        for (i, m_) in maps.iter().enumerate() {
            cost2[m_.plus] += m_.sign * self.objective[i];
            if let Some(mi) = m_.minus {
                cost2[mi] -= self.objective[i];
            }
        }
        let status = simplex(&mut t, &mut basis, &cost2, total, &mut iterations, Some(structural + slacks)).ok_or(PIVOT_LIMIT)?;
        if status == LpStatus::Unbounded {
            return Ok(LpSolution { status, x: vec![], objective: f64::NAN, iterations });
        }
        let mut y = vec![0.0; total];
        for (i, &b) in basis.iter().enumerate() {
            y[b] = t[i][total];
        }
        let x: Vec<f64> = maps.iter().map(|m| m.offset + m.sign * y[m.plus] - m.minus.map_or(0.0, |mi| y[mi])).collect();
        let objective = x.iter().zip(&self.objective).map(|(a, b)| a * b).sum();
        Ok(LpSolution { status: LpStatus::Optimal, x, objective, iterations })
    }
}

const PIVOT_LIMIT: ErrorsJSL = ErrorsJSL::RuntimeError("The simplex method reached its limit of 50 000 pivots.");

fn pivot(t: &mut [Vec<f64>], basis: &mut [usize], row: usize, col: usize, total: usize) {
    let p = t[row][col];
    t[row].iter_mut().for_each(|v| *v /= p);
    let pivot_row = t[row].clone();
    for (i, r) in t.iter_mut().enumerate() {
        if i != row {
            let f = r[col];
            if f != 0.0 {
                for j in 0..=total {
                    r[j] -= f * pivot_row[j];
                }
                r[col] = 0.0;
            }
        }
    }
    basis[row] = col;
}

/// Runs the simplex method on the tableau for the cost vector `cost`, restricting entering
/// columns to `0..limit` (all columns when `None`).
/// Returns `None` if the pivot limit is reached, which Bland's rule rules out in exact
/// arithmetic but rounding could in principle cause.
fn simplex(t: &mut [Vec<f64>], basis: &mut [usize], cost: &[f64], total: usize, iterations: &mut usize, limit: Option<usize>) -> Option<LpStatus> {
    let limit = limit.unwrap_or(total);
    let eps = 1e-9;
    for _ in 0..50_000 {
        // Reduced costs: c_j - c_B · column_j.
        let entering = (0..limit).find(|&j| {
            let reduced = cost[j] - basis.iter().enumerate().map(|(i, &b)| cost[b] * t[i][j]).sum::<f64>();
            reduced < -eps
        });
        let Some(col) = entering else { return Some(LpStatus::Optimal) };
        // Ratio test with Bland's tie-break (lowest basic variable index).
        let mut best: Option<(f64, usize, usize)> = None;
        for (i, row) in t.iter().enumerate() {
            if row[col] > eps {
                let ratio = row[total] / row[col];
                let better = match best {
                    None => true,
                    Some((r, _, b)) => ratio < r - 1e-12 || (ratio <= r + 1e-12 && basis[i] < b),
                };
                if better {
                    best = Some((ratio, i, basis[i]));
                }
            }
        }
        let Some((_, row, _)) = best else { return Some(LpStatus::Unbounded) };
        pivot(t, basis, row, col, total);
        *iterations += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-8)
    }

    #[test]
    fn classic_production_problem() {
        // maximise 3x + 5y  s.t.  x <= 4, 2y <= 12, 3x + 2y <= 18  -> (2, 6), value 36.
        let s = LinearProgram::new(vec![-3.0, -5.0])
            .with_less_equal(vec![1.0, 0.0], 4.0)
            .with_less_equal(vec![0.0, 2.0], 12.0)
            .with_less_equal(vec![3.0, 2.0], 18.0)
            .solve()
            .unwrap();
        assert_eq!(s.status, LpStatus::Optimal);
        assert!(close(&s.x, &[2.0, 6.0]) && (s.objective + 36.0).abs() < 1e-8);
    }

    #[test]
    fn diet_problem_with_greater_equal_constraints() {
        // minimise 2x + 3y  s.t.  x + y >= 4, x + 3y >= 6  -> x = 3, y = 1, cost 9.
        let s = LinearProgram::new(vec![2.0, 3.0]).with_greater_equal(vec![1.0, 1.0], 4.0).with_greater_equal(vec![1.0, 3.0], 6.0).solve().unwrap();
        assert!(close(&s.x, &[3.0, 1.0]) && (s.objective - 9.0).abs() < 1e-8);
    }

    #[test]
    fn equalities_bounds_and_free_variables() {
        // minimise x + 2y + z  s.t.  x + y + z = 10, x <= 2, y in [1, 3], z free-signed but forced by equality.
        let s = LinearProgram::new(vec![1.0, 2.0, 1.0])
            .with_equal(vec![1.0, 1.0, 1.0], 10.0)
            .with_bounds(0, 0.0, 2.0)
            .with_bounds(1, 1.0, 3.0)
            .with_free_variable(2)
            .solve()
            .unwrap();
        assert_eq!(s.status, LpStatus::Optimal);
        // Substituting z = 10 - x - y gives cost 10 + y, so y sits at its lower bound 1: cost 11.
        assert!((s.objective - 11.0).abs() < 1e-8);
        assert!((s.x.iter().sum::<f64>() - 10.0).abs() < 1e-8 && s.x[1] >= 1.0 - 1e-9 && s.x[0] <= 2.0 + 1e-9);
        // A variable bounded only above: maximise x s.t. x <= 5 (declared via bounds (-inf, 5)).
        let s = LinearProgram::new(vec![-1.0]).with_bounds(0, f64::NEG_INFINITY, 5.0).solve().unwrap();
        assert!(close(&s.x, &[5.0]));
        // A negative lower bound: minimise x with x >= -3.
        let s = LinearProgram::new(vec![1.0]).with_bounds(0, -3.0, 10.0).solve().unwrap();
        assert!(close(&s.x, &[-3.0]));
    }

    #[test]
    fn infeasible_and_unbounded() {
        let infeasible = LinearProgram::new(vec![1.0, 1.0]).with_less_equal(vec![1.0, 1.0], 1.0).with_greater_equal(vec![1.0, 1.0], 3.0).solve().unwrap();
        assert_eq!(infeasible.status, LpStatus::Infeasible);
        let unbounded = LinearProgram::new(vec![-1.0, 0.0]).with_less_equal(vec![-1.0, 1.0], 1.0).solve().unwrap();
        assert_eq!(unbounded.status, LpStatus::Unbounded);
        let conflicting_equalities = LinearProgram::new(vec![1.0]).with_equal(vec![1.0], 1.0).with_equal(vec![1.0], 2.0).solve().unwrap();
        assert_eq!(conflicting_equalities.status, LpStatus::Infeasible);
    }

    #[test]
    fn degenerate_and_redundant_problems_terminate() {
        // Beale-style cycling example (cycles under Dantzig's rule without anti-cycling).
        let s = LinearProgram::new(vec![-0.75, 150.0, -0.02, 6.0])
            .with_less_equal(vec![0.25, -60.0, -0.04, 9.0], 0.0)
            .with_less_equal(vec![0.5, -90.0, -0.02, 3.0], 0.0)
            .with_less_equal(vec![0.0, 0.0, 1.0, 0.0], 1.0)
            .solve()
            .unwrap();
        assert_eq!(s.status, LpStatus::Optimal);
        assert!((s.objective + 0.05).abs() < 1e-8, "{s:?}");
        // Redundant equality rows are tolerated.
        let s = LinearProgram::new(vec![1.0, 1.0]).with_equal(vec![1.0, 1.0], 2.0).with_equal(vec![2.0, 2.0], 4.0).solve().unwrap();
        assert_eq!(s.status, LpStatus::Optimal);
        assert!((s.objective - 2.0).abs() < 1e-8);
    }

    #[test]
    fn validation() {
        assert!(LinearProgram::new(vec![]).solve().is_err());
        assert!(LinearProgram::new(vec![1.0, 1.0]).with_less_equal(vec![1.0], 1.0).solve().is_err());
        assert!(LinearProgram::new(vec![f64::NAN]).solve().is_err());
        assert!(LinearProgram::new(vec![1.0]).with_bounds(0, 2.0, 1.0).solve().is_err());
    }
}
