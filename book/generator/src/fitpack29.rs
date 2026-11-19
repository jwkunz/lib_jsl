//! Tables and figures for release 2.9.0 (interpolation chapter): smoothing splines with
//! automatically placed knots (`make_splrep`, `make_splprep`, `generate_knots`) and
//! Bernstein-basis piecewise polynomials (`BPoly`), compared with SciPy.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_interpolation::one_d::b_spline::{BSpline, make_lsq_spline};
use lib_jsl_interpolation::one_d::bpoly::BPoly;
use lib_jsl_interpolation::one_d::fitpack::{SplrepOptions, generate_knots, make_splprep, make_splrep};
use lib_jsl_interpolation::one_d::ppoly::PPoly;
use ndarray::Array2;

const CHAPTER: &str = "interpolation";

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/fitpack_reference.txt"));
    splrep_table(&refs);
    bpoly_table(&refs);
    figures(&refs);
}

fn max_rel(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0.0f64, |m, (a, b)| m.max((a - b).abs())) / scale
}

fn options<'a>(name: &str, w: &'a [f64], m: usize) -> SplrepOptions<'a> {
    let d = SplrepOptions::default();
    match name {
        "cubic s=0.5" => SplrepOptions { s: 0.5, ..d },
        "cubic s=2" => SplrepOptions { s: 2.0, ..d },
        "cubic s=8" => SplrepOptions { s: 8.0, ..d },
        "linear s=1" => SplrepOptions { k: 1, s: 1.0, ..d },
        "quadratic s=1" => SplrepOptions { k: 2, s: 1.0, ..d },
        "quartic s=1" => SplrepOptions { k: 4, s: 1.0, ..d },
        "quintic s=1" => SplrepOptions { k: 5, s: 1.0, ..d },
        "weighted s=m" => SplrepOptions { s: m as f64, w: Some(w), ..d },
        "interval [-1, 11]" => SplrepOptions { s: 2.0, xb: Some(-1.0), xe: Some(11.0), ..d },
        "nest=14" => SplrepOptions { s: 0.01, nest: Some(14), ..d },
        "s huge" => SplrepOptions { s: 1e4, ..d },
        _ => unreachable!(),
    }
}

const CASES: [&str; 11] =
    ["cubic s=0.5", "cubic s=2", "cubic s=8", "linear s=1", "quadratic s=1", "quartic s=1", "quintic s=1", "weighted s=m", "interval [-1, 11]", "nest=14", "s huge"];

fn splrep_table(refs: &Refs) {
    let (x, y, w) = (refs.values("x"), refs.values("y"), refs.values("w"));
    let m = x.len();
    let mut rows = Vec::new();
    for name in CASES {
        let opts = options(name, w, m);
        let (spl, info) = make_splrep(x, y, None, &opts).unwrap();
        let r = refs.get(&format!("splrep {name}"));
        let same = spl.t == r[0];
        let dc = if same { fmt_diff(max_rel(&spl.c, &r[1])) } else { "—".into() };
        let fortran = if r[3].is_empty() {
            "(no `nest` in `splrep`)".to_string()
        } else if r[3] == spl.t {
            format!("same knots; {}", fmt_diff(max_rel(&spl.c, &r[4])))
        } else {
            format!("{} knots", r[3].len())
        };
        rows.push(vec![
            format!("`{name}`").replace("s=m", "s=80"),
            format!("{}", spl.t.len()),
            if same { "identical".into() } else { format!("{} knots in SciPy", r[0].len()) },
            dc,
            format!("{:.4} / {:.4}", info.fp, r[2][0]),
            fortran,
        ]);
    }
    let curve = refs.get("curve");
    let pts: Vec<Vec<f64>> = (0..curve[0].len()).map(|i| curve.iter().map(|c| c[i]).collect()).collect();
    let (u, sp, info) = make_splprep(&pts, None, None, &SplrepOptions { s: 0.05, ..Default::default() }).unwrap();
    let r = refs.get("splprep");
    let du = max_rel(&u, &r[0]);
    let same = sp[0].t == r[1];
    let dc = (0..3).map(|j| max_rel(&sp[j].c, &r[2 + j])).fold(0.0, f64::max);
    rows.push(vec![
        "`make_splprep`, 3-D curve, s=0.05".into(),
        format!("{}", sp[0].t.len()),
        if same { "identical".into() } else { "differ".into() },
        format!("{} (u: {})", fmt_diff(dc), fmt_diff(du)),
        format!("{:.4}", info.fp),
        if r[5] == sp[0].t { "same knots".into() } else { format!("{} knots", r[5].len()) },
    ]);
    let seq: Vec<f64> = generate_knots(x, y, &SplrepOptions { s: 0.5, ..Default::default() }).unwrap().iter().map(|t| t.len() as f64).collect();
    let same_seq = seq == refs.values("generate_knots");
    rows.push(vec![
        "`generate_knots`, s=0.5".into(),
        seq.iter().map(|v| format!("{v}")).collect::<Vec<_>>().join(", "),
        if same_seq { "identical".into() } else { "differ".into() },
        "".into(),
        "".into(),
        "".into(),
    ]);
    write(
        CHAPTER,
        "fitpack_splrep.md",
        &md_table(&["Case (80 points)", "Knots", "Knots vs `make_splrep`", "Coefficients", "fp: library / SciPy", "FITPACK's `splrep`"], &rows),
    );
}

fn bpoly_table(refs: &Refs) {
    let xi = [0.0, 1.0, 2.5, 4.0];
    let yi = vec![vec![1.0, 0.0, -2.0], vec![0.5, -1.0], vec![2.0, 0.3, 0.1], vec![-1.0, 1.5]];
    let ev: Vec<f64> = (0..23).map(|i| -0.5 + 5.0 * i as f64 / 22.0).collect();
    let mut rows = Vec::new();
    for (name, orders) in [("none", None), ("3", Some(vec![3usize])), ("1", Some(vec![1usize]))] {
        let bp = BPoly::from_derivatives(&xi, &yi, orders.as_deref()).unwrap();
        let r = refs.get(&format!("bpoly {name}"));
        let c: Vec<f64> = bp.c.iter().copied().collect();
        let vals: Vec<f64> = ev.iter().map(|&v| bp.evaluate(v)).collect();
        let d1: Vec<f64> = ev.iter().map(|&v| bp.evaluate_derivative(v, 1)).collect();
        let anti = bp.antiderivative(2);
        let a2: Vec<f64> = ev.iter().map(|&v| anti.evaluate(v)).collect();
        let label = match name {
            "none" => "all given derivatives (degree 4)".to_string(),
            o => format!("orders = {o}"),
        };
        rows.push(vec![
            format!("`from_derivatives`, {label}"),
            fmt_diff(max_rel(&c, &r[0])),
            fmt_diff(max_rel(&vals, &r[1])),
            fmt_diff(max_rel(&d1, &r[2])),
            fmt_diff(max_rel(&a2, &r[3])),
            fmt_diff((bp.integrate(-0.3, 3.7) - r[4][0]).abs() / r[4][0].abs()),
        ]);
    }
    // From a natural... SciPy's default CubicSpline (not-a-knot) through five points, as PPoly.
    let r = refs.get("power");
    let x = vec![0.0, 1.0, 2.5, 4.0, 5.0];
    let m = x.len() - 1;
    // Rebuild SciPy's PPoly from the reference (the round trip returns it).
    let pp_c = Array2::from_shape_vec((4, m), r[1].clone()).unwrap();
    let pp = PPoly::new(pp_c, x, true).unwrap();
    let bp = BPoly::from_power_basis(&pp);
    let c: Vec<f64> = bp.c.iter().copied().collect();
    let back: Vec<f64> = bp.to_power_basis().c.iter().copied().collect();
    rows.push(vec!["`from_power_basis`, `to_power_basis` (cubic spline)".into(), fmt_diff(max_rel(&c, &r[0])), fmt_diff(max_rel(&back, &r[1])), "".into(), "".into(), "".into()]);
    write(CHAPTER, "fitpack_bpoly.md", &md_table(&["Construction", "Coefficients", "Values", "First derivative", "Second antiderivative", "Integral"], &rows));
}

fn figures(refs: &Refs) {
    let (x, y) = (refs.values("x"), refs.values("y"));
    let mut plot = Plot::new("Smoothing splines with automatically placed knots (marked above)", "x", "y");
    plot.legend = Legend::BottomLeft;
    plot.y_range = Some((-1.8, 2.2));
    plot.series.push(Series::new("data (80 points)", x.iter().zip(y).map(|(&a, &b)| (a, b)).collect(), Style::Markers, 7));
    let grid: Vec<f64> = (0..=500).map(|i| 10.0 * i as f64 / 500.0).collect();
    for (c, s) in [0.5, 2.0, 8.0].into_iter().enumerate() {
        let (spl, _) = make_splrep(x, y, None, &SplrepOptions { s, ..Default::default() }).unwrap();
        let n_int = spl.t.len() - 8;
        plot.series.push(Series::new(&format!("s = {s}: {n_int} interior knots"), grid.iter().map(|&v| (v, spl.evaluate(v))).collect(), Style::Line, c));
        let y0 = 2.05 - 0.13 * c as f64;
        plot.series.push(Series::new("", spl.t[4..spl.t.len() - 4].iter().map(|&k| (k, y0)).collect(), Style::Markers, c));
    }
    write(CHAPTER, "fitpack_fits.svg", &plot.render());

    // The residual of the least-squares spline on each knot vector FITPACK tries.
    let mut plot = Plot::new("Adding knots until the least-squares residual reaches s", "number of knots", "Σ (yᵢ − g(xᵢ))²");
    plot.log_y = true;
    plot.legend = Legend::TopRight;
    for (c, s) in [0.5, 2.0, 8.0].into_iter().enumerate() {
        let seq = generate_knots(x, y, &SplrepOptions { s, ..Default::default() }).unwrap();
        let pts: Vec<(f64, f64)> = seq
            .iter()
            .map(|t| {
                let spl: BSpline = make_lsq_spline(x, y, t, 3, None).unwrap();
                (t.len() as f64, x.iter().zip(y).map(|(&a, &b)| (b - spl.evaluate(a)).powi(2)).sum::<f64>().max(1e-6))
            })
            .collect();
        plot.series.push(Series::new(&format!("s = {s}"), pts, Style::LineMarkers, c));
        plot.series.push(Series::new("", vec![(8.0, s), (64.0, s)], Style::Dashed, c));
    }
    write(CHAPTER, "fitpack_knots.svg", &plot.render());
}
