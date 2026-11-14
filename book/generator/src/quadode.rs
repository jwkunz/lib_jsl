//! Tables and figures for release 2.1.0 (integration chapter): explicit Runge-Kutta methods in
//! `solve_ivp_explicit`, `quad_vec`, and `quad_weighted`, each compared with SciPy.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_integration::ivp::{Event, ExplicitMethod, ExplicitOptions, solve_ivp_explicit};
use lib_jsl_integration::quad_vec::{QuadVecOptions, VecNorm, VecRule, quad_vec};
use lib_jsl_integration::weighted::{LogFactor, Weight, WeightedOptions, quad_weighted};
use std::f64::consts::PI;

const CHAPTER: &str = "integration";

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/quadode_reference.txt"));
    ivp_table(&refs);
    work_figure(&refs);
    quad_vec_table(&refs);
    weighted_table(&refs);
}

type Rhs = fn(f64, &[f64]) -> Vec<f64>;

fn vdp(_: f64, y: &[f64]) -> Vec<f64> {
    vec![y[1], 2.0 * (1.0 - y[0] * y[0]) * y[1] - y[0]]
}
fn lorenz(_: f64, y: &[f64]) -> Vec<f64> {
    vec![10.0 * (y[1] - y[0]), y[0] * (28.0 - y[2]) - y[1], y[0] * y[1] - 8.0 / 3.0 * y[2]]
}
fn kepler(_: f64, y: &[f64]) -> Vec<f64> {
    let r3 = (y[0] * y[0] + y[1] * y[1]).powf(1.5);
    vec![y[2], y[3], -y[0] / r3, -y[1] / r3]
}

fn superscript(n: i32) -> String {
    n.to_string().chars().map(|c| ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'][c.to_digit(10).unwrap() as usize]).collect()
}

fn rel(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    got.iter().zip(want).fold(0.0f64, |m, (a, b)| m.max((a - b).abs())) / scale
}

#[allow(clippy::type_complexity)]
fn ivp_table(refs: &Refs) {
    let problems: [(&str, Rhs, (f64, f64), Vec<f64>, [f64; 2]); 4] = [
        ("vdp", vdp, (0.0, 20.0), vec![2.0, 0.0], [3.3, 11.7]),
        ("lorenz", lorenz, (0.0, 2.0), vec![1.0, 1.0, 1.0], [0.77, 1.5]),
        ("kepler", kepler, (0.0, 20.0), vec![0.4, 0.0, 0.0, 2.0], [5.5, 13.1]),
        ("vdp back", vdp, (20.0, 0.0), vec![2.0, 0.0], [15.2, 4.4]),
    ];
    let mut rows = Vec::new();
    for (method, mname) in [(ExplicitMethod::Rk45, "RK45"), (ExplicitMethod::Dop853, "DOP853")] {
        for p in [3, 6, 10] {
            let rtol = 10f64.powi(-p);
            let (mut same, mut worst_y, mut worst_d) = (0, 0.0f64, 0.0f64);
            let (mut steps, mut nfev, mut ssteps, mut snfev) = (0usize, 0usize, 0.0, 0.0);
            for (name, f, span, y0, td) in &problems {
                let o = ExplicitOptions { rtol, atol: rtol * 1e-3, ..Default::default() };
                let s = solve_ivp_explicit(f, *span, y0, method, &o, &[]).unwrap();
                let w = refs.get(&format!("ivp {mname} {p} {name}"));
                if s.t.len() as f64 == w[0][0] && s.nfev as f64 == w[0][1] {
                    same += 1;
                }
                steps += s.t.len();
                nfev += s.nfev;
                ssteps += w[0][0];
                snfev += w[0][1];
                worst_y = worst_y.max(rel(s.y.last().unwrap(), &w[1]));
                // SciPy's sol(t) is (components × times).
                let n = y0.len();
                for (k, &t) in td.iter().enumerate() {
                    let want: Vec<f64> = (0..n).map(|i| w[2][i * 2 + k]).collect();
                    worst_d = worst_d.max(rel(&s.dense.evaluate(t), &want));
                }
            }
            rows.push(vec![format!("{mname}, rtol = 10⁻{}", superscript(p)), format!("{steps} / {ssteps}"), format!("{nfev} / {snfev}"), format!("{same} of 4"), fmt_diff(worst_y), fmt_diff(worst_d)]);
        }
    }
    write(CHAPTER, "explicit_scipy.md", &md_table(&["method and tolerance (atol = rtol/1000)", "steps, 4 problems (library / SciPy)", "function evaluations", "problems with identical steps and evaluations", "final state (relative)", "dense output (relative)"], &rows));

    // Events and per-component tolerances.
    let mut rows = Vec::new();
    let ground = Event { function: Box::new(|_, y: &[f64]| y[0]), direction: -1.0, terminal: Some(1) };
    let o = ExplicitOptions { rtol: 1e-10, atol: 1e-12, ..Default::default() };
    let s = solve_ivp_explicit(|_, y: &[f64]| vec![y[1], -9.81 - 0.1 * y[1] * y[1].abs()], (0.0, 10.0), &[10.0, 5.0], ExplicitMethod::Dop853, &o, std::slice::from_ref(&ground)).unwrap();
    let w = refs.get("event ground");
    rows.push(vec!["DOP853, a falling body with quadratic drag: the landing time (terminal event)".into(), format!("{} / {}", s.nfev, w[2][1]), fmt_diff((s.t_events[0][0] - w[0][0]).abs() / w[0][0])]);
    let o = ExplicitOptions { rtol: 1e-6, atol_components: vec![1e-9, 1e-3], ..Default::default() };
    let s = solve_ivp_explicit(vdp, (0.0, 20.0), &[2.0, 0.0], ExplicitMethod::Rk45, &o, &[]).unwrap();
    let w = refs.get("atol vector");
    rows.push(vec!["RK45, Van der Pol with atol = (10⁻⁹, 10⁻³): the final state".into(), format!("{} / {}", s.nfev, w[0][1]), fmt_diff(rel(s.y.last().unwrap(), &w[1]))]);
    write(CHAPTER, "explicit_events.md", &md_table(&["case", "function evaluations (library / SciPy)", "difference (relative)"], &rows));
}

fn work_figure(refs: &Refs) {
    let e: f64 = 0.6;
    let y0 = [1.0 - e, 0.0, 0.0, ((1.0 + e) / (1.0 - e)).sqrt()];
    let mut plot = Plot::new("Work and precision on an eccentric Kepler orbit (one period)", "function evaluations", "error after one period");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::TopRight;
    for (k, (method, mname)) in [(ExplicitMethod::Rk45, "RK45"), (ExplicitMethod::Dop853, "DOP853")].into_iter().enumerate() {
        let mut pts = Vec::new();
        for p in 3..13 {
            let rtol = 10f64.powi(-p);
            let o = ExplicitOptions { rtol, atol: rtol, ..Default::default() };
            let s = solve_ivp_explicit(kepler, (0.0, 2.0 * PI), &y0, method, &o, &[]).unwrap();
            let err = s.y.last().unwrap().iter().zip(&y0).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            pts.push((s.nfev as f64, err.max(1e-16)));
        }
        plot.series.push(Series::new(&format!("{mname} (library)"), pts, Style::LineMarkers, 2 * k));
        let w = refs.values(&format!("kepler work {mname}"));
        let spts: Vec<(f64, f64)> = w.chunks(3).map(|c| (c[1], c[2].max(1e-16))).collect();
        plot.series.push(Series::new(&format!("{mname} (SciPy)"), spts, Style::Markers, 2 * k + 1));
    }
    write(CHAPTER, "explicit_work.svg", &plot.render());
}

#[allow(clippy::type_complexity)]
fn quad_vec_table(refs: &Refs) {
    let f5 = |x: f64| -> Vec<f64> { (1..=5).map(|k| (k as f64 * x).sin() / (1.0 + x * x).sqrt() + (x - 0.3).abs().sqrt()).collect() };
    let g3 = |x: f64| -> Vec<f64> { (1..=3).map(|k| (-(x * x) * k as f64).exp() * (1.0 + x.sin())).collect() };
    let bessel = |x: f64| -> Vec<f64> { (0..8).map(|n| (n as f64 * x.sin() - x * n as f64).cos() / PI).collect() };
    let d = QuadVecOptions::default;
    let cases: Vec<(&str, &str, Box<dyn Fn(f64) -> Vec<f64>>, f64, f64, QuadVecOptions)> = vec![
        ("f5 max", "5 components with a √\\|x − 0.3\\| kink, [0, 3], max norm", Box::new(f5), 0.0, 3.0, QuadVecOptions { norm: VecNorm::Max, ..d() }),
        ("f5 2", "the same, 2-norm (the default)", Box::new(f5), 0.0, 3.0, d()),
        ("f5 points", "the same, with the kink as a break point", Box::new(f5), 0.0, 3.0, QuadVecOptions { points: vec![0.3], ..d() }),
        ("f5 trapezoid", "the same, trapezoid rule, epsrel = 10⁻⁵", Box::new(f5), 0.0, 3.0, QuadVecOptions { rule: Some(VecRule::Trapezoid), epsrel: 1e-5, ..d() }),
        ("g3 semi", "3 Gaussians, [0, ∞)", Box::new(g3), 0.0, f64::INFINITY, d()),
        ("g3 neg semi", "the same, (−∞, 0]", Box::new(g3), f64::NEG_INFINITY, 0.0, d()),
        ("g3 double", "the same, (−∞, ∞)", Box::new(g3), f64::NEG_INFINITY, f64::INFINITY, d()),
        ("bessel", "Bessel integrals J₀ … J₇ at x = n, epsrel = 10⁻¹²", Box::new(bessel), 0.0, PI, QuadVecOptions { epsrel: 1e-12, ..d() }),
    ];
    let mut rows = Vec::new();
    for (key, label, f, a, b, o) in cases {
        let r = quad_vec(f, a, b, &o).unwrap();
        let w = refs.get(&format!("qv {key}"));
        let s = &w[1];
        rows.push(vec![label.to_string(), fmt_diff(rel(&r.value, &w[0])), format!("{} / {}", r.nfev, s[1]), format!("{} / {}", r.intervals, s[2]), fmt_diff((r.error - s[0]).abs() / s[0])]);
    }
    write(CHAPTER, "quad_vec_scipy.md", &md_table(&["integrand", "values (relative difference)", "evaluations (library / SciPy)", "subintervals", "error estimate (relative difference)"], &rows));
}

#[allow(clippy::type_complexity)]
fn weighted_table(refs: &Refs) {
    let o = WeightedOptions::default();
    let al = |alpha: f64, beta: f64, log: LogFactor| Weight::Algebraic { alpha, beta, log };
    let cases: Vec<(&str, &str, Box<dyn Fn(f64) -> f64>, f64, f64, Weight)> = vec![
        ("cos finite", "x e^{−x} cos 25x on [0, 10]", Box::new(|x: f64| (-x).exp() * x), 0.0, 10.0, Weight::Cos(25.0)),
        ("sin finite", "sin 40x / (1 + x²) on [−2, 7]", Box::new(|x: f64| 1.0 / (1.0 + x * x)), -2.0, 7.0, Weight::Sin(40.0)),
        ("cos fourier", "cos 3x / (1 + x²) on [0, ∞)", Box::new(|x: f64| 1.0 / (1.0 + x * x)), 0.0, f64::INFINITY, Weight::Cos(3.0)),
        ("sin fourier", "e^{−x/10} sin 2.5x / (1 + x) on [1, ∞)", Box::new(|x: f64| (-0.1 * x).exp() / (1.0 + x)), 1.0, f64::INFINITY, Weight::Sin(2.5)),
        ("sin fourier slow", "sin x / √x on [1, ∞) (slow decay)", Box::new(|x: f64| 1.0 / x.sqrt()), 1.0, f64::INFINITY, Weight::Sin(1.0)),
        ("alg", "cos x · x^{−1/2} (2 − x)^{0.3} on [0, 2]", Box::new(f64::cos), 0.0, 2.0, al(-0.5, 0.3, LogFactor::None)),
        ("alg-loga", "eˣ ln x on [0, 1]", Box::new(f64::exp), 0.0, 1.0, al(0.0, 0.0, LogFactor::Left)),
        ("alg-logb", "(1 + x²)(x + 1)^{1/2}(1 − x)^{−0.7} ln(1 − x) on [−1, 1]", Box::new(|x: f64| 1.0 + x * x), -1.0, 1.0, al(0.5, -0.7, LogFactor::Right)),
        ("alg-log", "cos x · x^{−0.3}(3 − x)^{−0.3} ln x ln(3 − x) on [0, 3]", Box::new(f64::cos), 0.0, 3.0, al(-0.3, -0.3, LogFactor::Both)),
        ("cauchy", "⨍ eˣ/(x − 1/2) on [−1, 2]", Box::new(f64::exp), -1.0, 2.0, Weight::Cauchy(0.5)),
        ("cauchy asym", "⨍ 1/((1 + x²)(x − 1)) on [0, 10]", Box::new(|x: f64| 1.0 / (1.0 + x * x)), 0.0, 10.0, Weight::Cauchy(1.0)),
    ];
    let mut rows = Vec::new();
    for (key, label, f, a, b, w) in cases {
        let r = quad_weighted(f, a, b, w, &o).unwrap();
        let s = refs.values(&format!("w {key}"));
        rows.push(vec![label.to_string(), format!("{:.12}", r.value), format!("{:.12}", s[0]), fmt_diff((r.value - s[0]).abs() / s[0].abs().max(1e-300)), format!("{} / {}", r.nfev, s[2])]);
    }
    write(CHAPTER, "weighted_scipy.md", &md_table(&["integral", "library", "SciPy (QUADPACK)", "relative difference", "evaluations (library / SciPy)"], &rows));
}
