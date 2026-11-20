//! Tables and figures for release 2.18.0 (optimisation chapter): `shgo` and
//! `quadratic_assignment`, compared with SciPy.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_optimization::constrained::ConstrainedProblem;
use lib_jsl_optimization::qap::{QapMethod, quadratic_assignment};
use lib_jsl_optimization::shgo::{ShgoOptions, shgo};
use ndarray::Array2;
use std::f64::consts::{E, PI};

const CHAPTER: &str = "optimization";

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/glob218_reference.txt"));
    shgo_table(&refs);
    qap_table(&refs);
    figure();
}

struct Test(&'static str);

impl ConstrainedProblem for Test {
    fn objective(&self, x: &[f64]) -> f64 {
        match self.0 {
            "himmelblau" => (x[0] * x[0] + x[1] - 11.0).powi(2) + (x[0] + x[1] * x[1] - 7.0).powi(2),
            "eggholder" => -(x[1] + 47.0) * (x[0] / 2.0 + x[1] + 47.0).abs().sqrt().sin() - x[0] * (x[0] - (x[1] + 47.0)).abs().sqrt().sin(),
            "ackley" => -20.0 * (-0.2 * (0.5 * (x[0] * x[0] + x[1] * x[1])).sqrt()).exp() - (0.5 * ((2.0 * PI * x[0]).cos() + (2.0 * PI * x[1]).cos())).exp() + E + 20.0,
            "rosen4" => (0..3).map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2)).sum(),
            _ => (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2),
        }
    }
    fn inequalities(&self, x: &[f64]) -> Vec<f64> {
        if self.0 == "constrained" { vec![x[0] * x[0] + x[1] * x[1] - 1.5, -x[0] - x[1]] } else { Vec::new() }
    }
}

/// The number of distinct points, merging those within `tol` (max-norm).
fn distinct(xs: &[Vec<f64>], tol: f64) -> usize {
    let mut kept: Vec<&Vec<f64>> = Vec::new();
    for x in xs {
        if !kept.iter().any(|k| k.iter().zip(x).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max) <= tol) {
            kept.push(x);
        }
    }
    kept.len()
}

fn shgo_table(refs: &Refs) {
    let mut rows = Vec::new();
    for (name, label, lo, hi, n, iters) in [
        ("himmelblau", "Himmelblau, [−5, 5]², n = 128", -5.0, 5.0, 128, 1),
        ("eggholder", "Eggholder, [−512, 512]², n = 256, 3 iterations", -512.0, 512.0, 256, 3),
        ("ackley", "Ackley, [−5, 5]², n = 128", -5.0, 5.0, 128, 1),
        ("rosen4", "Rosenbrock 4-D, [−2, 2]⁴, n = 128", -2.0, 2.0, 128, 1),
        ("constrained", "SciPy's constrained example (a disc and a half-plane), n = 128", -3.0, 3.0, 128, 1),
    ] {
        let d = if name == "rosen4" { 4 } else { 2 };
        let r = shgo(&Test(name), &vec![lo; d], &vec![hi; d], ShgoOptions::with_samples(n, iters)).unwrap();
        let w = refs.get(&format!("shgo {name}"));
        let theirs: Vec<Vec<f64>> = w[2].chunks(d).map(|c| c.to_vec()).collect();
        let tol = 1e-6 * (hi - lo);
        let dx = r.x.iter().zip(&w[0]).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        rows.push(vec![
            label.into(),
            format!("{:.6}", r.fun),
            format!("{:.6}", w[1][0]),
            fmt_diff(dx),
            format!("{} / {}", r.xl.len(), distinct(&theirs, tol)),
            format!("{} / {}", r.nlfev, w[1][2]),
        ]);
    }
    write(CHAPTER, "glob218_shgo.md", &md_table(&["Problem", "Library minimum", "SciPy minimum", "Minimiser difference", "Distinct local minima, library / SciPy", "Local solves / SciPy's local evaluations"], &rows));
}

fn qap_table(refs: &Refs) {
    let mut rows = Vec::new();
    for n in [10usize, 20, 40] {
        for maximize in [false, true] {
            let w = refs.get(&format!("qap {n} {}", if maximize { "True" } else { "False" }));
            let a = Array2::from_shape_vec((n, n), w[0].clone()).unwrap();
            let b = Array2::from_shape_vec((n, n), w[1].clone()).unwrap();
            let f = quadratic_assignment(&a, &b, QapMethod::default(), maximize).unwrap();
            let t = quadratic_assignment(&a, &b, QapMethod::TwoOpt { initial: Vec::new() }, maximize).unwrap();
            let fp: Vec<f64> = f.perm.iter().map(|&v| v as f64).collect();
            let tp: Vec<f64> = t.perm.iter().map(|&v| v as f64).collect();
            rows.push(vec![
                format!("n = {n}, {}", if maximize { "maximise" } else { "minimise" }),
                format!("{} ({} it.); permutation {}", f.fun, f.nit, if fp == w[2] && f.nit as f64 == w[3][1] { "identical" } else { "differs" }),
                format!("{}", w[3][0]),
                format!("{}; permutation {}", t.fun, if tp == w[4] && t.nit as f64 == w[5][1] { "identical" } else { "differs" }),
                format!("{}", w[5][0]),
            ]);
        }
    }
    write(CHAPTER, "glob218_qap.md", &md_table(&["Random integer instance", "FAQ: library", "FAQ: SciPy", "2-opt from the identity: library", "2-opt: SciPy"], &rows));
}

fn figure() {
    let r = shgo(&Test("himmelblau"), &[-5.0, -5.0], &[5.0, 5.0], ShgoOptions::with_samples(128, 1)).unwrap();
    let mut sob = lib_jsl_random::quasi_random::SobolSequence::new(2).unwrap();
    let samples: Vec<(f64, f64)> = (0..128).map(|_| { let u = sob.next_point(); (-5.0 + 10.0 * u[0], -5.0 + 10.0 * u[1]) }).collect();
    let mut plot = Plot::new("shgo on Himmelblau's function: samples, minimisers and minima", "x", "y");
    plot.legend = Legend::TopRight;
    plot.x_range = Some((-5.0, 5.0));
    plot.y_range = Some((-5.0, 6.8));
    plot.series.push(Series::new("Sobol samples", samples, Style::Markers, 7));
    plot.series.push(Series::new(&format!("minimisers ({})", r.minimizers.len()), r.minimizers.iter().map(|m| (m[0], m[1])).collect(), Style::Markers, 1));
    plot.series.push(Series::new(&format!("local minima found ({})", r.xl.len()), r.xl.iter().map(|m| (m[0], m[1])).collect(), Style::Markers, 0));
    write(CHAPTER, "shgo_himmelblau.svg", &plot.render());
}
