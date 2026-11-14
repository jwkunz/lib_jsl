//! Tables and figures for "Assignment, trust regions and global optimisation" (optimisation
//! chapter): the SciPy comparison of `linear_sum_assignment`, the trust-region methods and the
//! global optimisers, and the points DIRECT samples.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_optimization::assignment::linear_sum_assignment;
use lib_jsl_optimization::global::{BasinhoppingOptions, DirectOptions, DualAnnealingOptions, basinhopping, direct, dual_annealing};
use lib_jsl_optimization::trust_region::{TrustRegionMethod, TrustRegionOptions, minimize_trust_region};
use ndarray::Array2;
use std::f64::consts::{E, PI};

const CHAPTER: &str = "optimization";

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/optim2_reference.txt"));
    assignment_table(&refs);
    trust_region_table(&refs);
    global_table(&refs);
    direct_figure();
}

fn matrix(refs: &Refs, name: &str) -> Array2<f64> {
    let m = refs.get(&format!("input {name}"));
    Array2::from_shape_vec((m[0][0] as usize, m[1][0] as usize), m[2].clone()).unwrap()
}

fn assignment_table(refs: &Refs) {
    let mut rows = Vec::new();
    for (name, label) in [("square", "60 × 60, uniform costs"), ("wide", "40 × 70"), ("tall", "70 × 40"), ("ties", "30 × 30, integer costs 0–4 (many ties)")] {
        let c = matrix(refs, name);
        for maximize in [false, true] {
            let (r, k) = linear_sum_assignment(&c, maximize).unwrap();
            let w = refs.get(&format!("lsap {name} {}", maximize as u8));
            let same = r.iter().zip(&w[0]).all(|(a, b)| *a as f64 == *b) && k.iter().zip(&w[1]).all(|(a, b)| *a as f64 == *b);
            let total: f64 = r.iter().zip(&k).map(|(&i, &j)| c[(i, j)]).sum();
            rows.push(vec![label.to_string(), if maximize { "maximise" } else { "minimise" }.into(), if same { "identical".into() } else { "differs".into() }, format!("{total:.12}"), fmt_diff((total - w[2][0]).abs() / w[2][0].abs().max(1.0))]);
        }
    }
    write(CHAPTER, "assignment_scipy.md", &md_table(&["cost matrix", "objective", "assignment vs SciPy", "total cost", "difference from SciPy's total"], &rows));
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
fn rosen_hess(x: &[f64]) -> Array2<f64> {
    let n = x.len();
    let mut h = Array2::zeros((n, n));
    for i in 0..n - 1 {
        h[(i, i)] += 1200.0 * x[i] * x[i] - 400.0 * x[i + 1] + 2.0;
        h[(i + 1, i + 1)] += 200.0;
        h[(i, i + 1)] -= 400.0 * x[i];
        h[(i + 1, i)] -= 400.0 * x[i];
    }
    h
}

fn trust_region_table(refs: &Refs) {
    let mut rows = Vec::new();
    let o = TrustRegionOptions { gtol: 1e-8, ..Default::default() };
    let starts: [(&str, Vec<f64>, &str); 3] = [("r4", vec![-1.2, 1.0, -0.5, 0.8], "Rosenbrock, n = 4"), ("r10", [-1.2, 1.0].repeat(5), "Rosenbrock, n = 10"), ("near", vec![1.2, 1.3, 1.1, 1.0], "Rosenbrock, n = 4, near the minimum")];
    let methods = [("trust-ncg", TrustRegionMethod::Ncg), ("trust-exact", TrustRegionMethod::Exact), ("dogleg", TrustRegionMethod::Dogleg)];
    let mut row = |label: &str, mname: &str, r: lib_jsl_optimization::trust_region::TrustRegionResult, w: &Vec<Vec<f64>>| {
        let xd = r.x.iter().zip(&w[0]).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        let s = &w[1];
        rows.push(vec![label.to_string(), format!("`{mname}`"), format!("{} / {}", r.iterations, s[1]), format!("{} / {}", r.nfev, s[2]), format!("{} / {}", r.status, s[4]), fmt_diff(xd)]);
    };
    for (key, x0, label) in &starts {
        for (mname, m) in methods {
            let r = minimize_trust_region(rosen, rosen_grad, rosen_hess, x0, m, o).unwrap();
            row(label, mname, r, refs.get(&format!("tr {mname} {key}")));
        }
    }
    let f = |v: &[f64]| v[0].powi(4) - 2.0 * v[0] * v[0] + v[1] * v[1] + 0.5 * v[0] * v[1];
    let g = |v: &[f64]| vec![4.0 * v[0].powi(3) - 4.0 * v[0] + 0.5 * v[1], 2.0 * v[1] + 0.5 * v[0]];
    let h = |v: &[f64]| Array2::from_shape_vec((2, 2), vec![12.0 * v[0] * v[0] - 4.0, 0.5, 0.5, 2.0]).unwrap();
    for (mname, m) in [("trust-ncg", TrustRegionMethod::Ncg), ("trust-exact", TrustRegionMethod::Exact)] {
        let r = minimize_trust_region(f, g, h, &[0.01, 0.5], m, TrustRegionOptions { gtol: 1e-10, ..Default::default() }).unwrap();
        row("x⁴ − 2x² + y² + xy/2 from near the saddle", mname, r, refs.get(&format!("tr {mname} saddle")));
    }
    write(CHAPTER, "trust_region_scipy.md", &md_table(&["problem", "method", "iterations (library / SciPy)", "function evaluations", "status", "largest difference in x"], &rows));
}

type Problem = (&'static str, fn(&[f64]) -> f64, Vec<(f64, f64)>, f64);

fn problems() -> Vec<Problem> {
    fn camel(x: &[f64]) -> f64 {
        let (a, b) = (x[0], x[1]);
        (4.0 - 2.1 * a * a + a.powi(4) / 3.0) * a * a + a * b + (-4.0 + 4.0 * b * b) * b * b
    }
    fn branin(x: &[f64]) -> f64 {
        let (a, b) = (x[0], x[1]);
        (b - 5.1 / (4.0 * PI * PI) * a * a + 5.0 / PI * a - 6.0).powi(2) + 10.0 * (1.0 - 1.0 / (8.0 * PI)) * a.cos() + 10.0
    }
    fn rastrigin(x: &[f64]) -> f64 {
        10.0 * x.len() as f64 + x.iter().map(|v| v * v - 10.0 * (2.0 * PI * v).cos()).sum::<f64>()
    }
    fn ackley(x: &[f64]) -> f64 {
        let (a, b) = (x[0], x[1]);
        -20.0 * (-0.2 * (0.5 * (a * a + b * b)).sqrt()).exp() - (0.5 * ((2.0 * PI * a).cos() + (2.0 * PI * b).cos())).exp() + E + 20.0
    }
    fn eggholder(x: &[f64]) -> f64 {
        let (a, b) = (x[0], x[1]);
        -(b + 47.0) * (a / 2.0 + b + 47.0).abs().sqrt().sin() - a * (a - (b + 47.0)).abs().sqrt().sin()
    }
    fn styblinski(x: &[f64]) -> f64 {
        0.5 * x.iter().map(|v| v.powi(4) - 16.0 * v * v + 5.0 * v).sum::<f64>()
    }
    vec![
        ("camel", camel, vec![(-3.0, 3.0), (-2.0, 2.0)], -1.031628453489877),
        ("branin", branin, vec![(-5.0, 10.0), (0.0, 15.0)], 0.39788735772973816),
        ("rastrigin", rastrigin, vec![(-5.12, 5.12); 4], 0.0),
        ("ackley", ackley, vec![(-5.0, 5.0); 2], 0.0),
        ("eggholder", eggholder, vec![(-512.0, 512.0); 2], -959.6406627208506),
        ("styblinski", styblinski, vec![(-5.0, 5.0); 3], -39.16616570377142 * 3.0),
    ]
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn global_table(refs: &Refs) {
    let mut rows = Vec::new();
    for (name, f, bounds, fstar) in problems() {
        let ok = |v: f64| v - fstar <= 1e-4 * fstar.abs().max(1.0);
        let lb: Vec<f64> = bounds.iter().map(|b| b.0).collect();
        let ub: Vec<f64> = bounds.iter().map(|b| b.1).collect();
        let d = direct(f, &lb, &ub, DirectOptions::default()).unwrap();
        let sd = refs.values(&format!("direct {name}"));
        let mut da_ok = 0;
        let mut da_n = Vec::new();
        let mut bh_ok = 0;
        let mut bh_n = Vec::new();
        for seed in 0..20u64 {
            let r = dual_annealing(f, &lb, &ub, DualAnnealingOptions { seed, ..Default::default() }).unwrap();
            da_ok += ok(r.fun) as usize;
            da_n.push(r.nfev as f64);
            let mut s: u64 = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            let x0: Vec<f64> = bounds.iter().map(|b| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                b.0 + (b.1 - b.0) * ((s >> 11) as f64 / (1u64 << 53) as f64)
            }).collect();
            let r = basinhopping(f, &x0, BasinhoppingOptions { seed, ..Default::default() });
            bh_ok += ok(r.fun) as usize;
            bh_n.push(r.nfev as f64);
        }
        let sda = refs.values(&format!("dual {name}"));
        let sbh = refs.values(&format!("basin {name}"));
        let label = format!("{name} ({}-D, f* = {fstar:.6})", bounds.len());
        rows.push(vec![
            label,
            format!("{:.6} ({}) / {:.6} ({})", d.fun, d.nfev, sd[0], sd[1]),
            format!("{da_ok} ({:.0}) / {} ({:.0})", median(da_n), sda[0], sda[1]),
            format!("{bh_ok} ({:.0}) / {} ({:.0})", median(bh_n), sbh[0], sbh[1]),
        ]);
    }
    write(CHAPTER, "global_scipy.md", &md_table(&["problem", "DIRECT: best value (evaluations), library / SciPy", "dual annealing: successes of 20 (median evaluations)", "basin hopping, 100 hops: successes of 20 (median evaluations)"], &rows));
}

/// The centres DIRECT evaluates on the six-hump camel function.
fn direct_figure() {
    use std::cell::RefCell;
    let pts = RefCell::new(Vec::new());
    let camel = |x: &[f64]| {
        pts.borrow_mut().push((x[0], x[1]));
        let (a, b) = (x[0], x[1]);
        (4.0 - 2.1 * a * a + a.powi(4) / 3.0) * a * a + a * b + (-4.0 + 4.0 * b * b) * b * b
    };
    let r = direct(camel, &[-3.0, -2.0], &[3.0, 2.0], DirectOptions { max_fun: Some(600), ..Default::default() }).unwrap();
    let mut plot = Plot::new("Points DIRECT samples on the six-hump camel function (600 evaluations)", "x", "y");
    plot.legend = Legend::BottomRight;
    plot.x_range = Some((-3.0, 3.0));
    plot.y_range = Some((-2.0, 2.0));
    plot.series.push(Series::new("evaluated centres", pts.into_inner(), Style::Markers, 0));
    plot.series.push(Series::new("global minima (±0.0898, ∓0.7126)", vec![(0.0898, -0.7126), (-0.0898, 0.7126)], Style::Markers, 1));
    plot.series.push(Series::new(&format!("best found, f = {:.6}", r.fun), vec![(r.x[0], r.x[1])], Style::Markers, 6));
    write(CHAPTER, "direct_samples.svg", &plot.render());
}
