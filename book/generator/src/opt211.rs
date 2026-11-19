//! Tables and a figure for release 2.11.0 (optimisation chapter): dogbox least squares,
//! MINPACK's Levenberg-Marquardt (`leastsq`, `root(method="lm")`), mixed-integer linear
//! programming and truncated Newton, compared with SciPy.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_optimization::bounded::{LeastSquaresOptions, Loss, least_squares_dogbox};
use lib_jsl_optimization::linear_programming::LinearProgram;
use lib_jsl_optimization::milp::{MilpOptions, milp};
use lib_jsl_optimization::tnc::{TncOptions, minimize_tnc};
use lib_jsl_root_finding::{LmOptions, leastsq, root_lm};
use ndarray::Array2;

const CHAPTER: &str = "optimization";

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/opt211_reference.txt"));
    least_squares_table(&refs);
    milp_tnc_table(&refs);
    milp_figure();
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn rosen_res(x: &[f64]) -> Vec<f64> {
    vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]]
}

fn least_squares_table(refs: &Refs) {
    let mut rows = Vec::new();
    let rosen_jac = |x: &[f64]| Array2::from_shape_vec((2, 2), vec![-20.0 * x[0], 10.0, -1.0, 0.0]).unwrap();
    let ninf = f64::NEG_INFINITY;
    for (name, x0, lb, ub) in [("rosen1", [2.0, 2.0], [ninf, 1.5], [f64::INFINITY; 2]), ("rosen2", [-1.2, 1.0], [-2.0, -2.0], [0.5, 2.0])] {
        let r = least_squares_dogbox(rosen_res, rosen_jac, &x0, &lb, &ub, LeastSquaresOptions::default()).unwrap();
        let w = refs.get(&format!("dogbox {name}"));
        let label = if name == "rosen1" { "dogbox: Rosenbrock, x₁ ≥ 1.5" } else { "dogbox: Rosenbrock, −2 ≤ x ≤ (0.5, 2)" };
        rows.push(vec![
            label.into(),
            fmt_diff(max_abs(&r.x, &w[0])),
            format!("{} / {}", r.nfev, w[1][1]),
            format!("{} / {}", r.njev, w[1][2]),
            format!("{} / {}", r.status, w[1][3]),
        ]);
    }
    let t: Vec<f64> = (0..30).map(|i| 4.0 * i as f64 / 29.0).collect();
    let y = refs.values("expfit y").to_vec();
    let fe = |p: &[f64]| t.iter().zip(&y).map(|(&v, &w)| p[0] * (-p[1] * v).exp() + p[2] - w).collect::<Vec<_>>();
    let je = |p: &[f64]| Array2::from_shape_fn((30, 3), |(i, k)| match k {
        0 => (-p[1] * t[i]).exp(),
        1 => -p[0] * t[i] * (-p[1] * t[i]).exp(),
        _ => 1.0,
    });
    for (name, loss, ub) in [("expfit", Loss::Linear, [10.0, 1.2, 1.0]), ("expfit soft_l1", Loss::SoftL1, [10.0, 5.0, 1.0])] {
        let opts = LeastSquaresOptions { loss, f_scale: 0.1, ..Default::default() };
        let r = least_squares_dogbox(fe, je, &[1.0, 0.5, 0.1], &[0.0; 3], &ub, opts).unwrap();
        let w = refs.get(&format!("dogbox {name}"));
        let label = if loss == Loss::Linear { "dogbox: exponential fit, rate ≤ 1.2 (active)" } else { "dogbox: exponential fit with outliers, soft L1 loss" };
        rows.push(vec![label.into(), fmt_diff(max_abs(&r.x, &w[0])), format!("{} / {}", r.nfev, w[1][1]), format!("{} / {}", r.njev, w[1][2]), format!("{} / {}", r.status, w[1][3])]);
    }
    let mut rj = |x: &[f64]| vec![vec![-20.0 * x[0], 10.0], vec![-1.0, 0.0]];
    let r = leastsq(rosen_res, Some(&mut rj), &[-1.2, 1.0], LmOptions::default()).unwrap();
    let w = refs.get("leastsq rosen jac");
    rows.push(vec!["`leastsq`: Rosenbrock, analytic Jacobian".into(), fmt_diff(max_abs(&r.x, &w[0])), format!("{} / {}", r.nfev, w[1][0]), format!("{} / {}", r.njev, w[1][1]), format!("{} / {}", r.info, w[1][2])]);
    type Case<'a> = (&'a str, &'a str, Box<dyn Fn(&[f64]) -> Vec<f64> + 'a>, Vec<f64>, LmOptions);
    let cases: Vec<Case> = vec![
        ("leastsq rosen fd", "`leastsq`: Rosenbrock, forward differences", Box::new(rosen_res), vec![-1.2, 1.0], LmOptions::default()),
        ("leastsq expfit", "`leastsq`: exponential fit, 30 residuals", Box::new(fe), vec![1.0, 0.5, 0.1], LmOptions::default()),
        ("leastsq system", "`leastsq`: x² + y² = 4, eˣ + y = 1", Box::new(|x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 4.0, x[0].exp() + x[1] - 1.0]), vec![1.0, -1.0], LmOptions::default()),
        (
            "leastsq powell",
            "`leastsq`: Powell's badly scaled system",
            Box::new(|x: &[f64]| vec![1e4 * x[0] * x[1] - 1.0, (-x[0]).exp() + (-x[1]).exp() - 1.0001]),
            vec![0.0, 1.0],
            LmOptions { xtol: 1e-12, ftol: 1e-12, ..Default::default() },
        ),
    ];
    for (key, label, fun, x0, opts) in cases {
        let r = leastsq(|x| fun(x), None, &x0, opts).unwrap();
        let w = refs.get(key);
        let scale = w[0].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        rows.push(vec![label.into(), fmt_diff(max_abs(&r.x, &w[0]) / scale), format!("{} / {}", r.nfev, w[1][0]), "—".into(), format!("{} / {}", r.info, w[1][1])]);
    }
    let s = root_lm(|x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 4.0, x[0].exp() + x[1] - 1.0], None, &[1.0, -1.0], LmOptions::default()).unwrap();
    rows.push(vec!["`root_lm`: the same system".into(), fmt_diff(max_abs(&s.x, refs.values("root lm"))), format!("{} / —", s.nfev), "—".into(), if s.converged { "converged".into() } else { "not converged".into() }]);
    write(CHAPTER, "opt211_lsq.md", &md_table(&["Problem", "x vs SciPy", "nfev: library / SciPy", "njev", "status / info"], &rows));
}

fn milp_tnc_table(refs: &Refs) {
    let mut rows = Vec::new();
    let exact = MilpOptions { mip_rel_gap: 0.0, ..Default::default() };
    // SciPy's documentation example.
    let lp = LinearProgram::new(vec![0.0, -1.0]).with_less_equal(vec![-1.0, 1.0], 1.0).with_less_equal(vec![3.0, 2.0], 12.0).with_less_equal(vec![2.0, 3.0], 12.0);
    let r = milp(&lp, &[true, true], MilpOptions::default()).unwrap();
    let w = refs.get("milp doc");
    rows.push(vec!["`milp`: SciPy's example, 2 integers".into(), format!("{} / {}", r.objective, w[1][0]), same(&r.x, &w[0]), format!("{}", r.nodes)]);
    let (v, wt) = (refs.values("milp knapsack v"), refs.values("milp knapsack w"));
    let cap = 0.4 * wt.iter().sum::<f64>();
    let mut lp = LinearProgram::new(v.iter().map(|x| -x).collect()).with_less_equal(wt.to_vec(), cap);
    for i in 0..20 {
        lp = lp.with_bounds(i, 0.0, 1.0);
    }
    let r = milp(&lp, &[true; 20], exact).unwrap();
    let w = refs.get("milp knapsack");
    rows.push(vec!["`milp`: 0-1 knapsack, 20 items".into(), format!("{} / {}", r.objective, w[1][0]), same(&r.x, &w[0]), format!("{}", r.nodes)]);
    let lp = LinearProgram::new(vec![3.0, 2.5, 4.0, 1.5])
        .with_greater_equal(vec![1.0, 1.0, 1.0, 0.0], 7.5)
        .with_greater_equal(vec![2.0, 1.0, 3.0, -1.0], 10.0)
        .with_equal(vec![1.0, 2.0, 0.0, 0.0], 6.0)
        .with_bounds(0, 0.0, 10.0)
        .with_bounds(1, 0.0, 10.0)
        .with_bounds(2, 0.0, 10.0)
        .with_bounds(3, 0.0, 4.0);
    let r = milp(&lp, &[true, true, true, false], exact).unwrap();
    let w = refs.get("milp mixed");
    rows.push(vec!["`milp`: 3 integers and 1 continuous, with an equality".into(), format!("{} / {}", r.objective, w[1][0]), same(&r.x, &w[0]), format!("{}", r.nodes)]);
    let a = refs.values("milp random A");
    let b = refs.values("milp random b");
    let c = refs.values("milp random c");
    let mut lp = LinearProgram::new(c.to_vec());
    for i in 0..6 {
        lp = lp.with_less_equal(a[8 * i..8 * i + 8].to_vec(), b[i]);
    }
    for i in 0..8 {
        lp = lp.with_bounds(i, 0.0, 5.0);
    }
    let r = milp(&lp, &[true; 8], exact).unwrap();
    let w = refs.get("milp random");
    rows.push(vec!["`milp`: random, 8 integers in [0, 5], 6 constraints".into(), format!("{} / {}", r.objective, w[1][0]), same(&r.x, &w[0]), format!("{}", r.nodes)]);
    write(CHAPTER, "opt211_milp.md", &md_table(&["Problem", "Objective: library / SciPy", "x", "Nodes"], &rows));

    let mut rows = Vec::new();
    let rosen = |x: &[f64]| {
        let n = x.len();
        let mut f = 0.0;
        let mut g = vec![0.0; n];
        for i in 0..n - 1 {
            let a = x[i + 1] - x[i] * x[i];
            let b = 1.0 - x[i];
            f += 100.0 * a * a + b * b;
            g[i] += -400.0 * x[i] * a - 2.0 * b;
            g[i + 1] += 200.0 * a;
        }
        (f, g)
    };
    for n in [2usize, 6, 20] {
        let r = minimize_tnc(rosen, &vec![-1.2; n], &vec![f64::NEG_INFINITY; n], &vec![f64::INFINITY; n], TncOptions::default()).unwrap();
        let w = refs.get(&format!("tnc rosen {n}"));
        let err = r.x.iter().map(|v| (v - 1.0).abs()).fold(0.0, f64::max);
        let serr = w[0].iter().map(|v| (v - 1.0).abs()).fold(0.0, f64::max);
        rows.push(vec![format!("Rosenbrock, n = {n}"), fmt_diff(err), fmt_diff(serr), format!("{} / {}", r.nfev, w[1][1])]);
    }
    let mut ub = vec![f64::INFINITY; 6];
    ub[0] = 0.5;
    let r = minimize_tnc(rosen, &[-1.2; 6], &[f64::NEG_INFINITY; 6], &ub, TncOptions::default()).unwrap();
    let w = refs.get("tnc rosen bounded");
    let truth = refs.values("lbfgsb rosen bounded");
    rows.push(vec!["Rosenbrock, n = 6, x₁ ≤ 0.5 (against tight L-BFGS-B)".into(), fmt_diff(max_abs(&r.x, truth)), fmt_diff(max_abs(&w[0], truth)), format!("{} / {}", r.nfev, w[1][1])]);
    write(CHAPTER, "opt211_tnc.md", &md_table(&["Problem", "Library: error in x", "SciPy TNC: error in x", "nfev: library / SciPy"], &rows));
}

fn same(a: &[f64], b: &[f64]) -> String {
    if max_abs(a, b) < 1e-9 { "same".into() } else { "different optimum".into() }
}

fn milp_figure() {
    // max x₂ s.t. −x₁ + x₂ ≤ 1, 3x₁ + 2x₂ ≤ 12, 2x₁ + 3x₂ ≤ 12: the LP optimum and integer points.
    let mut plot = Plot::new("Integer points, the LP relaxation and the integer optimum", "x₁", "x₂");
    plot.legend = Legend::TopRight;
    plot.x_range = Some((-0.5, 5.0));
    plot.y_range = Some((-0.5, 4.5));
    let line = |a: f64, b: f64, c: f64| -> Vec<(f64, f64)> { (0..=50).map(|i| -0.5 + 5.5 * i as f64 / 50.0).map(|x| (x, (c - a * x) / b)).filter(|p| p.1 > -1.0 && p.1 < 5.0).collect() };
    plot.series.push(Series::new("−x₁ + x₂ = 1", line(-1.0, 1.0, 1.0), Style::Line, 7));
    plot.series.push(Series::new("3x₁ + 2x₂ = 12", line(3.0, 2.0, 12.0), Style::Dashed, 7));
    plot.series.push(Series::new("2x₁ + 3x₂ = 12", line(2.0, 3.0, 12.0), Style::Line, 6));
    let feasible = |x: f64, y: f64| -x + y <= 1.0 && 3.0 * x + 2.0 * y <= 12.0 && 2.0 * x + 3.0 * y <= 12.0;
    let pts: Vec<(f64, f64)> = (0..5).flat_map(|i| (0..5).map(move |j| (i as f64, j as f64))).filter(|&(x, y)| feasible(x, y)).collect();
    plot.series.push(Series::new("feasible integer points", pts, Style::Markers, 0));
    let lp = LinearProgram::new(vec![0.0, -1.0]).with_less_equal(vec![-1.0, 1.0], 1.0).with_less_equal(vec![3.0, 2.0], 12.0).with_less_equal(vec![2.0, 3.0], 12.0);
    let relax = lp.solve().unwrap();
    plot.series.push(Series::new(&format!("LP optimum ({:.1}, {:.1})", relax.x[0], relax.x[1]), vec![(relax.x[0], relax.x[1])], Style::Markers, 1));
    let r = milp(&lp, &[true, true], MilpOptions::default()).unwrap();
    plot.series.push(Series::new(&format!("integer optimum ({}, {})", r.x[0], r.x[1]), vec![(r.x[0], r.x[1])], Style::Markers, 2));
    write(CHAPTER, "milp_lattice.svg", &plot.render());
}
