//! Tables and figures for "Piecewise polynomials, Hermite data and scattered interpolation"
//! (interpolation chapter): the SciPy comparison, Akima-type interpolants on data with flat
//! stretches, and the accuracy of scattered-data methods as points are added.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_interpolation::interpolation_trait::InterpolationTrait;
use lib_jsl_interpolation::nd::rbf::{RbfInterpolator, RbfKernel};
use lib_jsl_interpolation::one_d::akima::{akima, makima};
use lib_jsl_interpolation::one_d::b_spline::{BoundaryCondition, make_interp_spline};
use lib_jsl_interpolation::one_d::krogh::KroghInterpolator;
use lib_jsl_interpolation::one_d::pchip::pchip;
use lib_jsl_interpolation::one_d::ppoly::PPoly;
use lib_jsl_interpolation::two_d::rect_bivariate_spline::RectBivariateSpline;
use lib_jsl_interpolation::two_d::scattered::{CloughTocher2DInterpolator, GriddataMethod, LinearNDInterpolator, NearestNDInterpolator, TriangulationIndex, estimate_gradients_2d_global, griddata};
use ndarray::{Array1, Array2};

const CHAPTER: &str = "interpolation";

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/piecewise_reference.txt"));
    scipy_table(&refs);
    akima_figure();
    scattered_figure();
}

fn rel_scaled(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    got.iter().zip(want).fold(0.0f64, |m, (g, w)| if g.is_nan() && w.is_nan() { m } else { m.max((g - w).abs()) }) / scale
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
}

fn scipy_table(refs: &Refs) {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row = |what: &str, d: f64| rows.push(vec![what.to_string(), fmt_diff(d)]);
    let xs = linspace(-0.2, 4.2, 45);
    let x: Vec<f64> = (0..9).map(|i| i as f64 * 0.5).collect();
    let y: Vec<f64> = x.iter().map(|v| (1.3 * v).sin()).collect();
    let pp = PPoly::from_spline(&make_interp_spline(&x, &y, 3, &BoundaryCondition::NotAKnot).unwrap());
    let ev = |p: &PPoly, nu: usize| xs.iter().map(|&v| p.evaluate_derivative(v, nu)).collect::<Vec<_>>();
    row("`PPoly.from_spline`: values, first and second derivatives (with extrapolation)", rel_scaled(&ev(&pp, 0), refs.values("ppoly spline values")).max(rel_scaled(&ev(&pp, 1), refs.values("ppoly spline d1"))).max(rel_scaled(&ev(&pp, 2), refs.values("ppoly spline d2"))));
    row("`integrate` on [0.3, 3.7] and [−0.5, 4.5]; second antiderivative", rel_scaled(&[pp.integrate(0.3, 3.7), pp.integrate(-0.5, 4.5)], refs.values("ppoly spline integral")).max(rel_scaled(&ev(&pp.antiderivative(2), 0), refs.values("ppoly spline anti2"))));
    row("`roots` (4, two by extrapolation) and `solve(0.5)`", rel_scaled(&pp.roots(), refs.values("ppoly spline roots")).max(rel_scaled(&pp.solve(0.5, true), refs.values("ppoly spline solve"))));
    let xh = [0.0, 0.4, 1.1, 1.5, 2.6, 3.0, 4.0];
    let yh = [1.0, -0.5, 0.3, 2.0, -1.0, 0.0, 0.8];
    let dh = [0.0, -2.0, 3.0, 1.0, -0.5, 2.0, 0.0];
    let ch = PPoly::cubic_hermite(&xh, &yh, &dh).unwrap();
    row("`CubicHermiteSpline`: values and slopes", rel_scaled(&ev(&ch, 0), refs.values("hermite values")).max(rel_scaled(&ev(&ch, 1), refs.values("hermite d1"))));
    row("`CubicHermiteSpline`: integral, roots and `solve(0.25)`", rel_scaled(&[ch.integrate(0.1, 3.9)], refs.values("hermite integral")).max(rel_scaled(&ch.roots(), refs.values("hermite roots"))).max(rel_scaled(&ch.solve(0.25, true), refs.values("hermite solve"))));
    let xa = Array1::from_vec((0..11).map(|i| i as f64).collect());
    let ya = Array1::from_vec(vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.5, 15.0, 50.0, 60.0, 85.0]);
    let xb = Array1::from_vec((0..8).map(|i| i as f64).collect());
    let yb = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.0]);
    let sample = |p: PPoly, q: &[f64]| q.iter().map(|&v| p.evaluate(v)).collect::<Vec<_>>();
    let xq = linspace(0.0, 10.0, 101);
    let xq2 = linspace(0.0, 7.0, 71);
    row("`Akima1DInterpolator`, Akima's data and data with flat stretches", rel_scaled(&sample(akima(xa.view(), ya.view()).unwrap().to_ppoly(), &xq), refs.values("akima classic")).max(rel_scaled(&sample(akima(xb.view(), yb.view()).unwrap().to_ppoly(), &xq2), refs.values("akima flat"))));
    row("`Akima1DInterpolator(method=\"makima\")`, the same data", rel_scaled(&sample(makima(xa.view(), ya.view()).unwrap().to_ppoly(), &xq), refs.values("makima classic")).max(rel_scaled(&sample(makima(xb.view(), yb.view()).unwrap().to_ppoly(), &xq2), refs.values("makima flat"))));
    let kr = KroghInterpolator::new(&[0.0, 0.0, 1.0, 1.0, 1.0, 2.5, 3.0], &[1.0, 2.0, 0.0, -1.0, 4.0, 0.5, -0.3]).unwrap();
    let xk = linspace(-0.3, 3.3, 37);
    let kd = refs.values("krogh derivatives");
    let got: Vec<f64> = (0..4).flat_map(|d| xk.iter().map(move |&v| (d, v))).map(|(d, v)| kr.derivatives(v, 4)[d]).collect();
    row("`KroghInterpolator` with derivative data: values and derivatives 0–3", rel_scaled(&xk.iter().map(|&v| kr.evaluate(v)).collect::<Vec<_>>(), refs.values("krogh values")).max(rel_scaled(&got, kd)));
    let gx: [f64; 8] = [0.0, 0.3, 0.7, 1.2, 1.6, 2.1, 2.5, 3.0];
    let gy: [f64; 7] = [-1.0, -0.6, -0.1, 0.4, 0.8, 1.3, 2.0];
    let gz = Array2::from_shape_fn((8, 7), |(i, j)| gx[i].sin() * (1.5 * gy[j]).cos() + 0.2 * gx[i] * gy[j]);
    let qx = linspace(0.05, 2.95, 9);
    let qy = linspace(-0.95, 1.95, 8);
    for (kx, ky) in [(3, 3), (1, 1), (5, 3)] {
        let rb = RectBivariateSpline::new(&gx, &gy, &gz, kx, ky).unwrap();
        let grid = |dx: usize, dy: usize| qx.iter().flat_map(|&a| qy.iter().map(move |&b| (a, b))).map(|(a, b)| rb.evaluate(a, b, dx, dy)).collect::<Vec<_>>();
        let mut d = rel_scaled(&grid(0, 0), refs.values(&format!("rbs {kx}{ky} values"))).max(rel_scaled(&[rb.integral(0.2, 2.8, -0.7, 1.5)], refs.values(&format!("rbs {kx}{ky} integral"))));
        if kx > 1 {
            d = d.max(rel_scaled(&grid(1, 0), refs.values(&format!("rbs {kx}{ky} dx")))).max(rel_scaled(&grid(1, 1), refs.values(&format!("rbs {kx}{ky} dxdy"))));
        }
        row(&format!("`RectBivariateSpline`, degrees ({kx}, {ky}): values{}, integral", if kx > 1 { ", ∂x and ∂x∂y" } else { "" }), d);
    }
    let pts: Vec<[f64; 2]> = refs.values("input points").chunks(2).map(|c| [c[0], c[1]]).collect();
    let vals: Vec<f64> = pts.iter().map(|p| (3.0 * p[0]).sin() * (2.0 * p[1]).cos() + p[0]).collect();
    let g = linspace(0.02, 0.98, 13);
    let qq: Vec<[f64; 2]> = g.iter().flat_map(|&a| g.iter().map(move |&b| [a, b])).collect();
    let lin = LinearNDInterpolator::new(&pts, &vals).unwrap();
    row("`LinearNDInterpolator`, 64 scattered points, 169 queries", rel_scaled(&qq.iter().map(|&q| lin.evaluate(q)).collect::<Vec<_>>(), refs.values("linear values")));
    let tri = TriangulationIndex::new(&pts).unwrap();
    let grads: Vec<f64> = estimate_gradients_2d_global(&tri, &vals, 1e-6, 400).concat();
    row("`estimate_gradients_2d_global` (Nielson's method, tolerance 10⁻⁶)", rel_scaled(&grads, refs.values("cubic gradients")));
    let ct = CloughTocher2DInterpolator::new(&pts, &vals).unwrap();
    row("`CloughTocher2DInterpolator`", rel_scaled(&qq.iter().map(|&q| ct.evaluate(q)).collect::<Vec<_>>(), refs.values("cubic values")));
    let nn = NearestNDInterpolator::new(&pts.iter().map(|p| p.to_vec()).collect::<Vec<_>>(), &vals).unwrap();
    row("`NearestNDInterpolator`", rel_scaled(&qq.iter().map(|q| nn.evaluate(q)).collect::<Vec<_>>(), refs.values("nearest values")));
    let outside = griddata(&pts, &vals, &[[1.2, 0.5], [-0.1, 0.3]], GriddataMethod::Linear).unwrap();
    row(&format!("`griddata` outside the hull: {} and {} (SciPy: NaN)", outside[0], outside[1]), if outside.iter().all(|v| v.is_nan()) { 0.0 } else { 1.0 });
    let p3: Vec<Vec<f64>> = refs.values("input points3").chunks(3).map(|c| c.to_vec()).collect();
    let q3: Vec<Vec<f64>> = refs.values("input queries3").chunks(3).map(|c| c.to_vec()).collect();
    let f3: Vec<f64> = p3.iter().map(|v| (-(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])).exp() + v[0] * v[2]).collect();
    type Case = (&'static str, RbfKernel, Option<f64>, Option<i32>, f64);
    let cases: [Case; 9] = [
        ("tps", RbfKernel::ThinPlateSpline, None, None, 0.0),
        ("cubic", RbfKernel::Cubic, None, None, 0.0),
        ("quintic", RbfKernel::Quintic, None, None, 0.0),
        ("linear", RbfKernel::Linear, None, None, 0.0),
        ("gaussian", RbfKernel::Gaussian, Some(2.0), None, 0.0),
        ("multiquadric", RbfKernel::Multiquadric, Some(1.5), None, 0.0),
        ("inverse_quadratic", RbfKernel::InverseQuadratic, Some(1.0), None, 0.0),
        ("tps smooth", RbfKernel::ThinPlateSpline, None, None, 0.1),
        ("tps degree2", RbfKernel::ThinPlateSpline, None, Some(2), 0.0),
    ];
    let mut worst = 0.0f64;
    let mut worst_name = "";
    for (name, kernel, eps, degree, smoothing) in cases {
        let r = RbfInterpolator::new(&p3, &f3, kernel, eps, degree, smoothing).unwrap();
        let d = rel_scaled(&q3.iter().map(|q| r.evaluate(q)).collect::<Vec<_>>(), refs.values(&format!("rbf {name}")));
        if d > worst {
            worst = d;
            worst_name = name;
        }
    }
    row(&format!("`RBFInterpolator` in 3-D: 8 kernels, smoothing, degree 2 (largest: {worst_name})"), worst);
    write(CHAPTER, "piecewise_scipy.md", &md_table(&["computation (SciPy reference)", "largest difference (relative to the largest value)"], &rows));
}

fn akima_figure() {
    let x = Array1::from_vec((0..8).map(|i| i as f64).collect());
    let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.0]);
    let q = linspace(0.0, 7.0, 281);
    let mut plot = Plot::new("Four cubic interpolants through data with flat stretches", "x", "y");
    plot.legend = Legend::TopLeft;
    let spline = make_interp_spline(&x.to_vec(), &y.to_vec(), 3, &BoundaryCondition::NotAKnot).unwrap();
    plot.series.push(Series::new("cubic spline (not-a-knot)", q.iter().map(|&v| (v, spline.evaluate(v))).collect(), Style::Line, 7));
    let mut a = akima(x.view(), y.view()).unwrap();
    plot.series.push(Series::new("Akima", q.iter().map(|&v| (v, a.interpolate_at(v).unwrap())).collect(), Style::Line, 1));
    let mut m = makima(x.view(), y.view()).unwrap();
    plot.series.push(Series::new("makima", q.iter().map(|&v| (v, m.interpolate_at(v).unwrap())).collect(), Style::Line, 0));
    let mut p = pchip(x.view(), y.view()).unwrap();
    plot.series.push(Series::new("PCHIP", q.iter().map(|&v| (v, p.interpolate_at(v).unwrap())).collect(), Style::Dashed, 2));
    plot.series.push(Series::new("data", x.iter().zip(y.iter()).map(|(&a, &b)| (a, b)).collect(), Style::Markers, 6));
    write(CHAPTER, "akima_family.svg", &plot.render());
}

/// RMS error on a 30 × 30 interior grid of scattered-data methods for Franke's function.
fn scattered_figure() {
    let franke = |x: f64, y: f64| {
        0.75 * (-((9.0 * x - 2.0).powi(2) + (9.0 * y - 2.0).powi(2)) / 4.0).exp() + 0.75 * (-((9.0 * x + 1.0).powi(2) / 49.0 + (9.0 * y + 1.0) / 10.0)).exp()
            + 0.5 * (-((9.0 * x - 7.0).powi(2) + (9.0 * y - 3.0).powi(2)) / 4.0).exp()
            - 0.2 * (-((9.0 * x - 4.0).powi(2) + (9.0 * y - 7.0).powi(2))).exp()
    };
    let mut state: u64 = 2024;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let grid: Vec<[f64; 2]> = (0..30).flat_map(|i| (0..30).map(move |j| [0.05 + 0.9 * i as f64 / 29.0, 0.05 + 0.9 * j as f64 / 29.0])).collect();
    let (mut lin, mut ct, mut rbf) = (Vec::new(), Vec::new(), Vec::new());
    for n in [25usize, 50, 100, 200, 400, 800] {
        let mut pts: Vec<[f64; 2]> = (0..n - 4).map(|_| [next(), next()]).collect();
        pts.extend([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
        let vals: Vec<f64> = pts.iter().map(|p| franke(p[0], p[1])).collect();
        let rms = |f: &dyn Fn([f64; 2]) -> f64| (grid.iter().map(|&q| (f(q) - franke(q[0], q[1])).powi(2)).sum::<f64>() / grid.len() as f64).sqrt();
        let l = LinearNDInterpolator::new(&pts, &vals).unwrap();
        lin.push((n as f64, rms(&|q| l.evaluate(q))));
        let c = CloughTocher2DInterpolator::new(&pts, &vals).unwrap();
        ct.push((n as f64, rms(&|q| c.evaluate(q))));
        let r = RbfInterpolator::new(&pts.iter().map(|p| p.to_vec()).collect::<Vec<_>>(), &vals, RbfKernel::ThinPlateSpline, None, None, 0.0).unwrap();
        rbf.push((n as f64, rms(&|q| r.evaluate(&q))));
    }
    let mut plot = Plot::new("Scattered-data interpolation of Franke's function", "number of points", "RMS error on a 30 × 30 grid");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::BottomLeft;
    plot.series.push(Series::new("linear on the triangulation", lin, Style::LineMarkers, 1));
    plot.series.push(Series::new("Clough-Tocher", ct, Style::LineMarkers, 0));
    plot.series.push(Series::new("thin-plate spline RBF", rbf, Style::LineMarkers, 2));
    write(CHAPTER, "scattered_convergence.svg", &plot.render());
}
