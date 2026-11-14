//! Figures and tables for "B-splines and smoothing splines" (the interpolation chapter): the
//! SciPy comparison, the basis functions, convergence under each end condition and smoothing
//! splines with GCV.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_interpolation::one_d::b_spline::{BSpline, BoundaryCondition, make_interp_curve, make_interp_spline, make_lsq_spline};
use lib_jsl_interpolation::one_d::smoothing_spline::make_smoothing_spline;
use lib_jsl_random::uniform_generator::UniformGenerator;
use lib_jsl_random::xoshiro256plusplus::Xoshiro256PlusPlus;
use ndarray::Array2;

const CHAPTER: &str = "interpolation";

pub fn generate() {
    scipy_reference();
    basis_figure();
    convergence_table();
    smoothing_figure();
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

fn scipy_reference() {
    let refs = Refs::parse(include_str!("../data/bspline_reference.txt"));
    let x = refs.values("data x").to_vec();
    let y = refs.values("data y").to_vec();
    let xe = refs.values("data xe").to_vec();
    let mut rows: Vec<Vec<String>> = Vec::new();
    let given = |l: Vec<(usize, f64)>, r: Vec<(usize, f64)>| BoundaryCondition::Derivatives { left: l, right: r };
    let cases: Vec<(&str, usize, BoundaryCondition)> = vec![
        ("cubic not-a-knot", 3, BoundaryCondition::NotAKnot),
        ("quadratic not-a-knot", 2, BoundaryCondition::NotAKnot),
        ("quartic not-a-knot", 4, BoundaryCondition::NotAKnot),
        ("quintic not-a-knot", 5, BoundaryCondition::NotAKnot),
        ("cubic natural", 3, BoundaryCondition::Natural),
        ("cubic clamped", 3, BoundaryCondition::Clamped),
        ("cubic given slopes", 3, given(vec![(1, 0.5)], vec![(1, -1.0)])),
        ("quintic given derivatives", 5, given(vec![(1, 0.5), (2, 0.0)], vec![(1, -1.0), (2, 0.25)])),
        ("linear", 1, BoundaryCondition::NotAKnot),
    ];
    for (name, k, bc) in cases {
        let s = make_interp_spline(&x, &y, k, &bc).unwrap();
        let values = max_diff(&s.evaluate_many(&xe), refs.values(&format!("interp {name}")));
        let d = s.derivative(1).unwrap();
        let derivs = max_diff(&d.evaluate_many(&xe), refs.values(&format!("derivative {name}")));
        let integrals = max_diff(&[s.integrate(0.3, 5.1), s.integrate(-0.5, 6.5)], refs.values(&format!("integral {name}")));
        let anti = max_diff(&s.antiderivative(1).evaluate_many(&xe), refs.values(&format!("antiderivative {name}")));
        rows.push(vec![format!("{name} (`make_interp_spline`)"), fmt_diff(values), fmt_diff(derivs), fmt_diff(integrals.max(anti))]);
    }
    let yp = refs.values("data yp").to_vec();
    for k in [3, 4] {
        let s = make_interp_spline(&x, &yp, k, &BoundaryCondition::Periodic).unwrap();
        let v = max_diff(&s.evaluate_many(&xe), refs.values(&format!("periodic {k}")));
        let i = (s.integrate(-0.5, 13.0) - refs.values(&format!("periodic integral {k}"))[0]).abs();
        rows.push(vec![format!("periodic, degree {k}"), fmt_diff(v), String::new(), fmt_diff(i)]);
    }
    let (xd, yd, wd, t) = (refs.values("data xd").to_vec(), refs.values("data yd").to_vec(), refs.values("data wd").to_vec(), refs.values("data t").to_vec());
    let s = make_lsq_spline(&xd, &yd, &t, 3, Some(&wd)).unwrap();
    rows.push(vec!["weighted least squares, 5 interior knots (`make_lsq_spline`)".into(), fmt_diff(max_diff(&s.evaluate_many(&xe), refs.values("lsq"))), String::new(), String::new()]);
    for lam in [0.05, 2.0] {
        let s = make_smoothing_spline(&xd, &yd, Some(&wd), Some(lam)).unwrap();
        rows.push(vec![format!("weighted smoothing spline, λ = {lam} (`make_smoothing_spline`)"), fmt_diff(max_diff(&s.spline.evaluate_many(&xe), refs.values(&format!("smoothing {lam:?}")))), String::new(), String::new()]);
    }
    let s = make_smoothing_spline(&xd, &yd, None, None).unwrap();
    rows.push(vec![format!("smoothing spline, λ by GCV (here {:.4})", s.lam), fmt_diff(max_diff(&s.spline.evaluate_many(&xe), refs.values("smoothing gcv"))), String::new(), String::new()]);
    let c = refs.get("data curve");
    let pts = Array2::from_shape_vec((c[0].len() / 2, 2), c[0].clone()).unwrap();
    let (u, curve) = make_interp_curve(&pts, 3, &BoundaryCondition::NotAKnot).unwrap();
    let uu: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
    let want = refs.get("splprep curve");
    let d = max_diff(&curve[0].evaluate_many(&uu), &want[0]).max(max_diff(&curve[1].evaluate_many(&uu), &want[1]));
    rows.push(vec![format!("parametric cubic curve (`splprep`, s = 0; parameters differ by {})", fmt_diff(max_diff(&u, refs.values("splprep u")))), fmt_diff(d), String::new(), String::new()]);
    write(CHAPTER, "bspline_scipy.md", &md_table(&["computation (SciPy reference)", "values", "first derivative", "integrals, antiderivative"], &rows));
}

fn basis_figure() {
    let t = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.5, 3.0, 4.0, 4.0, 4.0, 4.0];
    let n = t.len() - 4;
    let mut plot = Plot::new("Cubic B-splines on the knots 0, 0, 0, 0, 1, 1.5, 3, 4, 4, 4, 4", "x", "Bⱼ,₃(x)");
    plot.legend = Legend::TopRight;
    let xs: Vec<f64> = (0..=400).map(|i| i as f64 * 4.0 / 400.0).collect();
    for j in 0..n {
        let mut c = vec![0.0; n];
        c[j] = 1.0;
        let s = BSpline::new(t.clone(), c, 3).unwrap();
        plot.series.push(Series::new(&format!("B{}", sub(j)), xs.iter().map(|&x| (x, s.evaluate(x))).collect(), Style::Line, j));
    }
    let ones = BSpline::new(t.clone(), vec![1.0; n], 3).unwrap();
    plot.series.push(Series::new("sum = 1", xs.iter().map(|&x| (x, ones.evaluate(x))).collect(), Style::Dashed, 7));
    plot.y_range = Some((0.0, 1.6));
    write(CHAPTER, "bspline_basis.svg", &plot.render());
}

fn sub(j: usize) -> String {
    j.to_string().chars().map(|c| char::from_u32(0x2080 + c.to_digit(10).unwrap()).unwrap()).collect()
}

fn convergence_table() {
    let f = |x: f64| (2.0 * x).sin() + (x / 3.0).exp();
    let fp = |x: f64| 2.0 * (2.0 * x).cos() + (x / 3.0).exp() / 3.0;
    let fine: Vec<f64> = (0..=3000).map(|i| i as f64 * 3.0 / 3000.0).collect();
    let truth: Vec<f64> = fine.iter().map(|&x| f(x)).collect();
    type EndCondition = Box<dyn Fn(f64, f64) -> BoundaryCondition>;
    let methods: Vec<(&str, usize, EndCondition)> = vec![
        ("linear", 1, Box::new(|_, _| BoundaryCondition::NotAKnot)),
        ("cubic, zero end slopes (clamped)", 3, Box::new(|_, _| BoundaryCondition::Clamped)),
        ("cubic, natural", 3, Box::new(|_, _| BoundaryCondition::Natural)),
        ("cubic, not-a-knot", 3, Box::new(|_, _| BoundaryCondition::NotAKnot)),
        ("cubic, exact end slopes", 3, Box::new(|a, b| BoundaryCondition::Derivatives { left: vec![(1, a)], right: vec![(1, b)] })),
        ("quintic, not-a-knot", 5, Box::new(|_, _| BoundaryCondition::NotAKnot)),
    ];
    let sizes = [8usize, 16, 32, 64, 128];
    let mut rows = Vec::new();
    for (name, k, bc) in &methods {
        let mut row = vec![name.to_string()];
        let mut prev: Option<f64> = None;
        let mut last_order = 0.0;
        for &n in &sizes {
            let x: Vec<f64> = (0..n).map(|i| i as f64 * 3.0 / (n - 1) as f64).collect();
            let y: Vec<f64> = x.iter().map(|&v| f(v)).collect();
            let s = make_interp_spline(&x, &y, *k, &bc(fp(0.0), fp(3.0))).unwrap();
            let err = max_diff(&s.evaluate_many(&fine), &truth);
            row.push(format!("{err:.1e}"));
            if let Some(p) = prev {
                last_order = (p / err).log2();
            }
            prev = Some(err);
        }
        row.push(format!("{last_order:.1}"));
        rows.push(row);
    }
    let header: Vec<String> = std::iter::once("spline".to_string()).chain(sizes.iter().map(|n| format!("n = {n}"))).chain(std::iter::once("order".to_string())).collect();
    let header: Vec<&str> = header.iter().map(|s| s.as_str()).collect();
    write(CHAPTER, "bspline_convergence.md", &md_table(&header, &rows));
}

fn smoothing_figure() {
    let f = |x: f64| (2.0 * x).sin() + 0.5 * x;
    let mut rng = Xoshiro256PlusPlus::from_seed(41);
    let n = 80;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 4.0 / (n - 1) as f64 + 0.01 * (i as f64 * 1.3).sin()).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&v| {
            let (u, w) = (1.0 - rng.next_f64(), rng.next_f64());
            f(v) + 0.3 * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * w).cos()
        })
        .collect();
    let fine: Vec<f64> = (0..=400).map(|i| i as f64 * x[n - 1] / 400.0).collect();
    let mut plot = Plot::new("Smoothing splines of 80 noisy samples", "x", "y");
    plot.legend = Legend::TopLeft;
    plot.series.push(Series::new("data", x.iter().zip(&y).map(|(&a, &b)| (a, b)).collect(), Style::Markers, 7));
    plot.series.push(Series::new("true function", fine.iter().map(|&v| (v, f(v))).collect(), Style::Dashed, 6));
    let gcv = make_smoothing_spline(&x, &y, None, None).unwrap();
    let mut rows = Vec::new();
    for (i, (label, lam)) in [("λ = 10⁻⁵ (nearly interpolating)", 1e-5), ("λ by GCV", gcv.lam), ("λ = 10 (oversmoothed)", 10.0)].into_iter().enumerate() {
        let s = make_smoothing_spline(&x, &y, None, Some(lam)).unwrap();
        plot.series.push(Series::new(label, fine.iter().map(|&v| (v, s.spline.evaluate(v))).collect(), Style::Line, i));
        let rms_truth = (fine.iter().map(|&v| (s.spline.evaluate(v) - f(v)).powi(2)).sum::<f64>() / fine.len() as f64).sqrt();
        let rms_res = (x.iter().zip(&y).map(|(&a, &b)| (s.spline.evaluate(a) - b).powi(2)).sum::<f64>() / n as f64).sqrt();
        rows.push(vec![format!("{lam:.2e}"), format!("{:.1}", s.degrees_of_freedom), format!("{rms_res:.3}"), format!("{rms_truth:.3}")]);
    }
    write(CHAPTER, "smoothing_spline.svg", &plot.render());
    write(CHAPTER, "smoothing_spline.md", &md_table(&["λ", "degrees of freedom tr A", "RMS residual", "RMS error against the true function"], &rows));
}
