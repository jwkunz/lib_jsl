//! Figures and tables for "Bound-constrained optimisation" (the optimisation chapter): the
//! comparison with SciPy, and robust fits to data with outliers.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_optimization::bounded::{LbfgsbOptions, LeastSquaresOptions, Loss, least_squares_bounded, lsq_linear, minimize_lbfgsb, nnls};
use ndarray::{Array1, Array2};

const CHAPTER: &str = "optimization";

pub fn generate() {
    scipy_reference();
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn scipy_reference() {
    let refs = Refs::parse(include_str!("../data/bounded_reference.txt"));
    let mat = |name: &str| {
        let m = refs.get(&format!("data {name}"));
        Array2::from_shape_vec((m[0][0] as usize, m[1][0] as usize), m[2].clone()).unwrap()
    };
    let mut rows: Vec<Vec<String>> = Vec::new();

    // L-BFGS-B on Rosenbrock.
    let rosen = |x: &[f64]| {
        let n = x.len();
        let (mut f, mut g) = (0.0, vec![0.0; n]);
        for i in 0..n - 1 {
            let (a, b) = (x[i + 1] - x[i] * x[i], 1.0 - x[i]);
            f += 100.0 * a * a + b * b;
            g[i] += -400.0 * x[i] * a - 2.0 * b;
            g[i + 1] += 200.0 * a;
        }
        (f, g)
    };
    let x0: Vec<f64> = (0..10).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect();
    let ub = refs.values("data rosen ub").to_vec();
    let r = minimize_lbfgsb(rosen, &x0, &[-2.0; 10], &ub, LbfgsbOptions::default()).unwrap();
    let want = refs.get("lbfgsb rosen");
    rows.push(vec![
        "`minimize_lbfgsb`: Rosenbrock, n = 10, three upper bounds active".into(),
        fmt_diff(max_diff(&r.x, &want[0])),
        fmt_diff((r.fun - want[1][0]).abs()),
        format!("{} / {}", r.iterations, want[1][1]),
        format!("{} / {}", r.function_evaluations, want[1][2]),
    ]);
    let h = mat("H");
    let c = refs.get("data c")[2].clone();
    let quad = |x: &[f64]| {
        let hx = h.dot(&Array1::from(x.to_vec()));
        (0.5 * x.iter().zip(hx.iter()).map(|(a, b)| a * b).sum::<f64>() + x.iter().zip(&c).map(|(a, b)| a * b).sum::<f64>(), hx.iter().zip(&c).map(|(a, b)| a + b).collect::<Vec<_>>())
    };
    let o = LbfgsbOptions { ftol: 1e-15, gtol: 1e-10, ..Default::default() };
    let r = minimize_lbfgsb(quad, &[0.0; 30], &[-0.5; 30], &[0.5; 30], o).unwrap();
    let want = refs.get("lbfgsb quadratic");
    rows.push(vec![
        "`minimize_lbfgsb`: convex quadratic, n = 30, −0.5 ≤ xᵢ ≤ 0.5".into(),
        fmt_diff(max_diff(&r.x, &want[0])),
        fmt_diff((r.fun - want[1][0]).abs()),
        format!("{} / {}", r.iterations, want[1][1]),
        format!("{} / {}", r.function_evaluations, want[1][2]),
    ]);

    // Nonlinear least squares with each loss.
    let t = refs.values("data t").to_vec();
    let y = refs.values("data y").to_vec();
    let fun = |p: &[f64]| t.iter().zip(&y).map(|(&ti, &yi)| p[0] * (-p[1] * ti).exp() + p[2] - yi).collect::<Vec<_>>();
    let jac = |p: &[f64]| Array2::from_shape_fn((t.len(), 3), |(i, k)| match k {
        0 => (-p[1] * t[i]).exp(),
        1 => -p[0] * t[i] * (-p[1] * t[i]).exp(),
        _ => 1.0,
    });
    let mut fits = Vec::new();
    for (loss, key) in [(Loss::Linear, "linear"), (Loss::SoftL1, "soft_l1"), (Loss::Huber, "huber"), (Loss::Cauchy, "cauchy"), (Loss::Arctan, "arctan")] {
        let o = LeastSquaresOptions { loss, f_scale: 0.2, ..Default::default() };
        let r = least_squares_bounded(fun, jac, &[1.0, 0.1, 0.0], &[0.0, 0.0, -1.0], &[10.0, 2.0, 1.0], o).unwrap();
        let want = refs.get(&format!("trf {key}"));
        rows.push(vec![
            format!("`least_squares_bounded`: exponential with outliers, {key} loss"),
            fmt_diff(max_diff(&r.x, &want[0])),
            fmt_diff((r.cost - want[1][0]).abs()),
            "".into(),
            format!("{} / {}", r.nfev, want[1][1]),
        ]);
        fits.push((key, r.x));
    }
    let r = least_squares_bounded(fun, jac, &[1.0, 0.1, 0.0], &[0.0, 0.0, -1.0], &[10.0, 0.4, 1.0], LeastSquaresOptions::default()).unwrap();
    let want = refs.get("trf bounded");
    rows.push(vec![
        "`least_squares_bounded`: the same, decay rate ≤ 0.4 (active)".into(),
        fmt_diff(max_diff(&r.x, &want[0])),
        fmt_diff((r.cost - want[1][0]).abs()),
        "".into(),
        format!("{} / {}", r.nfev, want[1][1]),
    ]);

    // Linear problems.
    let a = mat("A");
    let b = refs.get("data b")[2].clone();
    let r = lsq_linear(&a, &b, &[-0.15; 12], &[0.15; 12], 1e-12).unwrap();
    let want = refs.get("lsq_linear");
    rows.push(vec!["`lsq_linear` (BVLS): 40 × 12, −0.15 ≤ xᵢ ≤ 0.15".into(), fmt_diff(max_diff(&r.x, &want[0])), fmt_diff((r.cost - want[1][0]).abs()), "".into(), "".into()]);
    let r = nnls(&a, &b).unwrap();
    let want = refs.get("nnls");
    rows.push(vec!["`nnls`: 40 × 12".into(), fmt_diff(max_diff(&r.x, &want[0])), fmt_diff((r.residual_norm - want[1][0]).abs()), "".into(), "".into()]);
    write(CHAPTER, "bounded_scipy.md", &md_table(&["problem (SciPy reference)", "largest difference in x", "difference in the objective", "iterations (library / SciPy)", "evaluations (library / SciPy)"], &rows));

    // Robust fits figure.
    let mut plot = Plot::new("Fitting a e^(−kt) + c to data with three outliers", "t", "y");
    plot.legend = Legend::TopRight;
    plot.series.push(Series::new("data", t.iter().zip(&y).map(|(&a, &b)| (a, b)).collect(), Style::Markers, 7));
    let fine: Vec<f64> = (0..=200).map(|i| i as f64 * 0.05).collect();
    plot.series.push(Series::new("true curve 3e^(−0.6t) + 0.5", fine.iter().map(|&ti| (ti, 3.0 * (-0.6 * ti).exp() + 0.5)).collect(), Style::Dashed, 6));
    for (i, (key, p)) in fits.iter().enumerate().filter(|(_, (k, _))| ["linear", "soft_l1", "cauchy"].contains(k)) {
        plot.series.push(Series::new(&format!("{key} loss: k = {:.3}, c = {:.3}", p[1], p[2]), fine.iter().map(|&ti| (ti, p[0] * (-p[1] * ti).exp() + p[2])).collect(), Style::Line, i));
    }
    write(CHAPTER, "bounded_robust.svg", &plot.render());
}
