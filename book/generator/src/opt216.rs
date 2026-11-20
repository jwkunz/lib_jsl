//! Tables and figures for release 2.16.0 (optimisation chapter): trust-krylov, COBYLA,
//! trust-constr, least squares with an LSMR trust-region solver, and LSMR, compared with SciPy.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_linalg::sparse::{CooMatrix, LsmrOptions, lsmr};
use lib_jsl_optimization::bounded::{LeastSquaresOptions, Loss, least_squares_lsmr};
use lib_jsl_optimization::cobyla::{CobylaOptions, cobyla};
use lib_jsl_optimization::constrained::{ConstrainedOptions, ConstrainedProblem};
use lib_jsl_optimization::trust_constr::trust_constr;
use lib_jsl_optimization::trust_region::{TrustRegionOptions, minimize_trust_krylov};
use ndarray::Array2;
use std::cell::RefCell;

const CHAPTER: &str = "optimization";

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/opt216_reference.txt"));
    constrained_table(&refs);
    other_table(&refs);
    figure();
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

/// The three test problems in the library's convention (`g ≤ 0`), recording every point at
/// which the objective is evaluated.
struct Problem {
    name: &'static str,
    trail: RefCell<Vec<Vec<f64>>>,
}

impl ConstrainedProblem for Problem {
    fn objective(&self, x: &[f64]) -> f64 {
        self.trail.borrow_mut().push(x.to_vec());
        match self.name {
            "tutorial" => x[0] * x[1],
            "rosendisc" => 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2),
            _ => x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],
        }
    }
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        match self.name {
            "tutorial" => vec![x[1], x[0]],
            "rosendisc" => vec![-400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]), 200.0 * (x[1] - x[0] * x[0])],
            _ => vec![x[3] * (2.0 * x[0] + x[1] + x[2]), x[0] * x[3], x[0] * x[3] + 1.0, x[0] * (x[0] + x[1] + x[2])],
        }
    }
    fn equalities(&self, x: &[f64]) -> Vec<f64> {
        if self.name == "hs071" { vec![x.iter().map(|v| v * v).sum::<f64>() - 40.0] } else { Vec::new() }
    }
    fn inequalities(&self, x: &[f64]) -> Vec<f64> {
        match self.name {
            "tutorial" => vec![x[0] * x[0] + x[1] * x[1] - 1.0, -x[1]],
            "rosendisc" => vec![x[0] * x[0] + x[1] * x[1] - 1.5],
            _ => {
                let mut g = vec![25.0 - x.iter().product::<f64>()];
                g.extend(x.iter().map(|v| 1.0 - v));
                g.extend(x.iter().map(|v| v - 5.0));
                g
            }
        }
    }
}

fn start(name: &str) -> Vec<f64> {
    match name {
        "tutorial" => vec![1.0, 1.0],
        "rosendisc" => vec![-1.0, 0.0],
        _ => vec![1.0, 5.0, 5.0, 1.0],
    }
}

fn constrained_table(refs: &Refs) {
    let mut rows = Vec::new();
    for (name, label) in [("tutorial", "x₀x₁ on the half disc, from (1, 1)"), ("rosendisc", "Rosenbrock in the disc x² + y² ≤ 1.5"), ("hs071", "HS071 (4 variables, 1 equality, 9 inequalities)")] {
        let opt = refs.get(&format!("{name} optimum"));
        if name != "hs071" {
            let p = Problem { name, trail: RefCell::new(Vec::new()) };
            let r = cobyla(&p, start(name), CobylaOptions::with_radii(1.0, 1e-8)).unwrap();
            let s = refs.get(&format!("{name} cobyla"));
            rows.push(vec![
                format!("COBYLA: {label}"),
                format!("{} ({} evaluations)", fmt_diff(max_diff(&r.parameters, &opt[0])), r.iterations),
                format!("{} ({} evaluations)", fmt_diff(max_diff(&s[0], &opt[0])), s[1][1]),
            ]);
            let mut o = CobylaOptions::with_radii(1.0, 1e-8);
            o.max_evaluations = if name == "rosendisc" { 10_000 } else { 0 };
            let p = Problem { name, trail: RefCell::new(Vec::new()) };
            let r = cobyla(&p, start(name), o);
            let s = refs.get(&format!("{name} cobyla10k"));
            if let (Ok(r), true) = (r, name == "rosendisc") { rows.push(vec![
                format!("… with up to 10 000 evaluations"),
                format!("{} ({} evaluations)", fmt_diff(max_diff(&r.parameters, &opt[0])), r.iterations),
                format!("{} ({} evaluations)", fmt_diff(max_diff(&s[0], &opt[0])), s[1][1]),
            ]); }
        }
        let p = Problem { name, trail: RefCell::new(Vec::new()) };
        let r = trust_constr(&p, start(name), ConstrainedOptions { tolerance: 1e-8, max_iterations: 3000 }).unwrap();
        let s = refs.get(&format!("{name} trust-constr"));
        rows.push(vec![
            format!("trust-constr: {label}"),
            format!("{} ({} iterations)", fmt_diff(max_diff(&r.parameters, &opt[0])), r.iterations),
            format!("{} ({} iterations)", fmt_diff(max_diff(&s[0], &opt[0])), s[1][1]),
        ]);
    }
    write(CHAPTER, "opt216_constrained.md", &md_table(&["Method and problem", "Library: distance from the optimum", "SciPy: distance from the optimum"], &rows));
}

fn rosen(x: &[f64]) -> f64 {
    (0..x.len() - 1).map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2)).sum()
}

fn rosen_grad(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut g = vec![0.0; n];
    for i in 0..n - 1 {
        g[i] += -400.0 * x[i] * (x[i + 1] - x[i] * x[i]) - 2.0 * (1.0 - x[i]);
        g[i + 1] += 200.0 * (x[i + 1] - x[i] * x[i]);
    }
    g
}

fn rosen_hessp(x: &[f64], v: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0; n];
    for i in 0..n - 1 {
        out[i] += (1200.0 * x[i] * x[i] - 400.0 * x[i + 1] + 2.0) * v[i] - 400.0 * x[i] * v[i + 1];
        out[i + 1] += 200.0 * v[i + 1] - 400.0 * x[i] * v[i];
    }
    out
}

#[allow(clippy::needless_range_loop)]
fn other_table(refs: &Refs) {
    let mut rows = Vec::new();
    for (name, label) in [("rosen4", "4-D Rosenbrock from (−1.2, 1, −0.5, 0.8)"), ("rosen30", "30-D Rosenbrock from (−1.2, 1, …)")] {
        let w = refs.get(&format!("krylov {name}"));
        let x0: Vec<f64> = if name == "rosen4" { vec![-1.2, 1.0, -0.5, 0.8] } else { (0..30).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect() };
        let r = minimize_trust_krylov(rosen, rosen_grad, rosen_hessp, &x0, TrustRegionOptions { gtol: 1e-8, ..Default::default() }).unwrap();
        rows.push(vec![
            format!("`minimize_trust_krylov`: {label}"),
            format!("x differs by {}; iterations {} vs {}, Hessian-vector products {} vs {}", fmt_diff(max_diff(&r.x, &w[0])), r.iterations, w[1][1], r.nhev, w[1][4]),
        ]);
    }
    // Broyden tridiagonal, sparse Jacobian.
    let n = 100;
    let f = |x: &[f64]| (0..n).map(|i| (3.0 - 2.0 * x[i]) * x[i] - if i > 0 { x[i - 1] } else { 0.0 } - 2.0 * if i + 1 < n { x[i + 1] } else { 0.0 } + 1.0).collect::<Vec<_>>();
    let j = |x: &[f64]| {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 3.0 - 4.0 * x[i]);
            if i > 0 {
                coo.push(i, i - 1, -1.0);
            }
            if i + 1 < n {
                coo.push(i, i + 1, -2.0);
            }
        }
        coo.to_csr().unwrap()
    };
    let r = least_squares_lsmr(f, j, &vec![-1.0; n], &vec![f64::NEG_INFINITY; n], &vec![f64::INFINITY; n], LeastSquaresOptions::default()).unwrap();
    let w = refs.get("broyden");
    rows.push(vec![
        "`least_squares_lsmr`: Broyden tridiagonal system, n = 100, sparse Jacobian".into(),
        format!("x differs by {}; cost {:.1e} vs {:.1e}; evaluations {} vs {}", fmt_diff(max_diff(&r.x, &w[0])), r.cost, w[1][0], r.nfev, w[1][1]),
    ]);
    let t: Vec<f64> = (0..21).map(|i| i as f64 * 0.25).collect();
    let y: Vec<f64> = t.iter().map(|&ti| 2.0 * (-0.7 * ti).exp() + 0.01 * (3.0 * ti).sin()).collect();
    let fun = |p: &[f64]| t.iter().zip(&y).map(|(&ti, &yi)| p[0] * (-p[1] * ti).exp() - yi).collect::<Vec<_>>();
    let jac = |p: &[f64]| Array2::from_shape_fn((t.len(), 2), |(i, k)| if k == 0 { (-p[1] * t[i]).exp() } else { -p[0] * t[i] * (-p[1] * t[i]).exp() });
    for (ub, loss, key, label) in [(5.0, Loss::Linear, "expfit 5.0 linear", "free"), (0.5, Loss::Linear, "expfit 0.5 linear", "rate bounded by 0.5"), (5.0, Loss::SoftL1, "expfit 5.0 soft_l1", "soft-L1 loss")] {
        let r = least_squares_lsmr(fun, jac, &[1.0, 0.1], &[0.0, 0.0], &[10.0, ub], LeastSquaresOptions { loss, ..Default::default() }).unwrap();
        let w = refs.get(key);
        rows.push(vec![format!("`least_squares_lsmr`: exponential fit, {label}"), format!("x differs by {}; evaluations {} vs {}", fmt_diff(max_diff(&r.x, &w[0])), r.nfev, w[1][1])]);
    }
    let av = refs.values("lsmr a");
    let a = Array2::from_shape_vec((60, 20), av.to_vec()).unwrap();
    let b = refs.values("lsmr b");
    for damp in [0.0, 0.5] {
        let mut o = LsmrOptions::with_damp(damp);
        (o.atol, o.btol) = (1e-12, 1e-12);
        let r = lsmr(&a, b, o).unwrap();
        let w = refs.get(&format!("lsmr {damp:?}"));
        rows.push(vec![
            format!("`lsmr`, 60 × 20 sparse, damping {damp}"),
            format!("x differs by {}; istop {} vs {}; iterations {} vs {}", fmt_diff(max_diff(&r.x, &w[0])), r.istop, w[1][0], r.iterations, w[1][1]),
        ]);
    }
    write(CHAPTER, "opt216_other.md", &md_table(&["Function and problem", "Library against SciPy"], &rows));
}

fn figure() {
    let p = Problem { name: "rosendisc", trail: RefCell::new(Vec::new()) };
    let _ = cobyla(&p, start("rosendisc"), CobylaOptions::with_radii(1.0, 1e-8)).unwrap();
    let cob = p.trail.borrow().clone();
    let q = Problem { name: "rosendisc", trail: RefCell::new(Vec::new()) };
    let _ = trust_constr(&q, start("rosendisc"), ConstrainedOptions { tolerance: 1e-8, max_iterations: 3000 }).unwrap();
    let tc = q.trail.borrow().clone();
    let mut plot = Plot::new("Rosenbrock in the disc x² + y² ≤ 1.5: where each method evaluates f", "x", "y");
    plot.legend = Legend::BottomRight;
    plot.x_range = Some((-1.4, 1.4));
    plot.y_range = Some((-1.0, 1.4));
    let r = 1.5f64.sqrt();
    plot.series.push(Series::new("constraint boundary", (0..=200).map(|i| { let t = i as f64 * std::f64::consts::TAU / 200.0; (r * t.cos(), r * t.sin()) }).collect(), Style::Dashed, 7));
    plot.series.push(Series::new("valley y = x²", (0..=100).map(|i| { let x = -1.2 + 2.4 * i as f64 / 100.0; (x, x * x) }).collect(), Style::Line, 6));
    plot.series.push(Series::new(&format!("COBYLA ({} evaluations)", cob.len()), cob.iter().map(|v| (v[0], v[1])).collect(), Style::Markers, 0));
    plot.series.push(Series::new(&format!("trust-constr ({} evaluations)", tc.len()), tc.iter().map(|v| (v[0], v[1])).collect(), Style::LineMarkers, 1));
    write(CHAPTER, "constrained_paths.svg", &plot.render());
}
