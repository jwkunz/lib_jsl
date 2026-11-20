//! Tables and figures for release 2.19.0: periodic smoothing splines, smoothing surfaces and
//! sparse Jacobians in `solve_ivp`, compared with SciPy.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::write;
use crate::svg::{Legend, Plot, Series, Style};
use lib_jsl_integration::ivp::{IvpMethod, IvpOptions, SparseJacobian, solve_ivp, solve_ivp_sparse};
use lib_jsl_interpolation::one_d::fitpack::{SplrepOptions, make_splrep_periodic};
use lib_jsl_interpolation::two_d::smooth_bivariate::{bisplrep, bisplrep_lsq};
use lib_jsl_linalg::sparse::CooMatrix;

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/spl219_reference.txt"));
    periodic_table(&refs);
    surface_table(&refs);
    ode_table(&refs);
    figure(&refs);
}

fn surface_table(refs: &Refs) {
    let d = refs.get("surface data");
    let (x, y, z, g) = (&d[0], &d[1], &d[2], &d[3]);
    let truth: Vec<f64> = g.iter().flat_map(|&a| g.iter().map(move |&b| (6.0 * a).sin() * (4.0 * b).cos())).collect();
    let err = |v: &[f64]| v.iter().zip(&truth).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    let mut rows = Vec::new();
    let t = vec![0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0];
    let lsq = bisplrep_lsq(x, y, z, None, &t, &t, 3, 3).unwrap();
    let w = refs.get("surface lsq");
    let c: Vec<f64> = lsq.c.iter().flatten().copied().collect();
    let v: Vec<f64> = lsq.evaluate_grid(g, g).into_iter().flatten().collect();
    rows.push(vec![
        "`bisplrep_lsq`, knots 0, ¼, ½, ¾, 1 in each direction (SciPy's `task=-1`)".into(),
        format!("coefficients {}, values {}", fmt_diff(max_diff(&c, &w[0])), fmt_diff(max_diff(&v, &w[1]))),
        format!("{:.3}", err(&v)),
        format!("{:.3}", err(&w[1])),
    ]);
    let ws = vec![20.0; x.len()];
    let sm = bisplrep(x, y, z, Some(&ws), 3, 3, None).unwrap();
    let w = refs.get("surface smooth");
    let v: Vec<f64> = sm.evaluate_grid(g, g).into_iter().flatten().collect();
    rows.push(vec![
        "`bisplrep`, weights 1/σ, default s = m − √(2m) = 371.7".into(),
        format!("fp {:.2} vs {:.2}; {} × {} knots vs {} × {}; surfaces differ by {}", sm.fp, w[0][0], sm.tx.len(), sm.ty.len(), w[0][1], w[0][2], fmt_diff(max_diff(&v, &w[1]))),
        format!("{:.3}", err(&v)),
        format!("{:.3}", err(&w[1])),
    ]);
    write("interpolation", "spl219_surface.md", &md_table(&["Surface, 400 noisy samples of sin 6x cos 4y (σ = 0.05)", "Library against SciPy", "Library error on a 21 × 21 grid", "SciPy error"], &rows));
}

fn ode_table(refs: &Refs) {
    let n = 200;
    let d = 400.0;
    let f = move |_t: f64, y: &[f64]| (0..n).map(|i| d * ((if i > 0 { y[i - 1] } else { 0.0 }) - 2.0 * y[i] + (if i + 1 < n { y[i + 1] } else { 0.0 })) - y[i].powi(3)).collect::<Vec<_>>();
    let y0: Vec<f64> = (0..n).map(|i| (std::f64::consts::PI * (i + 1) as f64 / (n + 1) as f64).sin()).collect();
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        for j in i.saturating_sub(1)..(i + 2).min(n) {
            coo.push(i, j, 1.0);
        }
    }
    let pattern = coo.to_csr().unwrap();
    let o = IvpOptions { rtol: 1e-8, atol: 1e-10, ..Default::default() };
    let mut rows = Vec::new();
    for (method, name) in [(IvpMethod::Radau, "Radau"), (IvpMethod::Lsoda, "LSODA")] {
        let dense = solve_ivp(f, None::<fn(f64, &[f64]) -> Vec<Vec<f64>>>, (0.0, 0.5), &y0, method, o, &[]).unwrap();
        let sp = solve_ivp_sparse(f, SparseJacobian::Pattern(pattern.clone()), (0.0, 0.5), &y0, method, o, &[]).unwrap();
        let w = refs.get(&format!("ode {name} True"));
        let yl = sp.y.last().unwrap();
        let scale = w[0].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        rows.push(vec![
            format!("{name}, n = 200"),
            fmt_diff(max_diff(yl, &w[0]) / scale),
            format!("{} / {} / {}", sp.t.len() - 1, sp.njev, sp.nlu),
            format!("{} / {} / {}", w[1][3], w[1][1], w[1][2]),
            format!("{} vs {} dense", sp.nfev, dense.nfev),
        ]);
    }
    write("integration", "spl219_ode.md", &md_table(&["Stiff reaction-diffusion chain", "Final state vs SciPy (relative)", "Library steps / Jacobians / LUs", "SciPy steps / Jacobians / LUs", "Library f evaluations, sparse vs dense"], &rows));
}

fn figure(refs: &Refs) {
    let d = refs.get("periodic data");
    let (x, y) = (&d[0], &d[1]);
    let per = 2.0 * std::f64::consts::PI;
    let o = SplrepOptions { k: 3, s: 0.6, ..Default::default() };
    let (spl, _) = make_splrep_periodic(x, y, None, &o).unwrap();
    let mut plot = Plot::new("A periodic smoothing spline (k = 3, s = 0.6) drawn over two periods", "x", "y");
    plot.legend = Legend::TopRight;
    plot.y_range = Some((-2.0, 2.4));
    let pts: Vec<(f64, f64)> = x.iter().zip(y).flat_map(|(&a, &b)| [(a, b), (a + per, b)]).collect();
    plot.series.push(Series::new("data (repeated)", pts, Style::Markers, 7));
    let curve: Vec<(f64, f64)> = (0..=400).map(|i| { let t = 2.0 * per * i as f64 / 400.0; (t, spl.evaluate(t % per)) }).collect();
    plot.series.push(Series::new(&format!("periodic spline ({} knots)", spl.t.len()), curve, Style::Line, 0));
    write("interpolation", "periodic_spline.svg", &plot.render());
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn periodic_table(refs: &Refs) {
    let d = refs.get("periodic data");
    let (x, y, xe) = (&d[0], &d[1], &d[2]);
    let mut rows = Vec::new();
    for k in [3usize, 2] {
        for s in [0.0, 0.3, 0.6, 2.0] {
            let w = refs.get(&format!("periodic {k} {s:?}"));
            let o = SplrepOptions { k, s, ..Default::default() };
            let (spl, info) = make_splrep_periodic(x, y, None, &o).unwrap();
            let v: Vec<f64> = xe.iter().map(|&t| spl.evaluate(t)).collect();
            rows.push(vec![
                format!("k = {k}, s = {s}"),
                format!("{} vs {}{}", spl.t.len(), w[0].len(), if max_diff(&spl.t, &w[0]) < 1e-12 { ", identical" } else { ", differ" }),
                fmt_diff(max_diff(&v, &w[1])),
                format!("{:.4}", info.fp),
            ]);
        }
    }
    write("interpolation", "spl219_periodic.md", &md_table(&["Periodic smoothing spline", "Knots (library vs SciPy)", "Values on 101 points", "fp (library)"], &rows));
}
