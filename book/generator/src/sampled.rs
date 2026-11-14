//! Tables and figures for "Integrating sampled data" (integration chapter) and "Adaptive
//! derivatives" (derivatives chapter): comparisons with SciPy, convergence of the rules, and the
//! error of derivatives against the step.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_derivatives::adaptive::{DerivativeOptions, derivative};
use lib_jsl_integration::sampled::{Spacing, cumulative_simpson, cumulative_trapezoid, romb, simpson, trapezoid};

pub fn generate() {
    integration_reference();
    convergence_table();
    derivative_reference();
    derivative_figure();
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs())) / scale
}

fn integration_reference() {
    let refs = Refs::parse(include_str!("../data/sampled_reference.txt"));
    let mut rows = Vec::new();
    for name in ["uniform", "nonuniform"] {
        let x = refs.values(&format!("data x {name}")).to_vec();
        let y = refs.values(&format!("data y {name}")).to_vec();
        let sp = Spacing::Points(&x);
        let one = |v: f64, key: &str| fmt_diff(((v - refs.values(key)[0]) / refs.values(key)[0]).abs());
        rows.push(vec![format!("`trapezoid`, 25 {name} samples"), one(trapezoid(&y, sp).unwrap(), &format!("trapezoid {name}"))]);
        rows.push(vec![format!("`simpson`, 25 {name} samples"), one(simpson(&y, sp).unwrap(), &format!("simpson {name}"))]);
        let xs = &x[..x.len() - 1];
        rows.push(vec![format!("`simpson`, 24 {name} samples (even: end correction)"), one(simpson(&y[..y.len() - 1], Spacing::Points(xs)).unwrap(), &format!("simpson even {name}"))]);
        rows.push(vec![format!("`cumulative_trapezoid`, {name}"), fmt_diff(max_rel(&cumulative_trapezoid(&y, sp, Some(0.0)).unwrap(), refs.values(&format!("cumulative_trapezoid {name}"))))]);
        rows.push(vec![format!("`cumulative_simpson`, {name}"), fmt_diff(max_rel(&cumulative_simpson(&y, sp, Some(0.0)).unwrap(), refs.values(&format!("cumulative_simpson {name}"))))]);
    }
    let yr = refs.values("data y romb").to_vec();
    let r = romb(&yr, 2.0 / 32.0).unwrap();
    rows.push(vec!["`romb`, 33 samples".into(), fmt_diff(((r - refs.values("romb")[0]) / refs.values("romb")[0]).abs())]);
    write("integration", "sampled_scipy.md", &md_table(&["computation (SciPy reference)", "largest relative difference"], &rows));
}

/// Errors of the three rules for `∫₀³ e^{−x} sin 2x + 0.1x² dx` as the number of samples grows.
fn convergence_table() {
    let f = |x: f64| (-x).exp() * (2.0 * x).sin() + 0.1 * x * x;
    // ∫ e^{−x} sin 2x = −e^{−x}(sin 2x + 2 cos 2x)/5.
    let anti = |x: f64| -(-x).exp() * ((2.0 * x).sin() + 2.0 * (2.0 * x).cos()) / 5.0 + 0.1 * x * x * x / 3.0;
    let exact = anti(3.0) - anti(0.0);
    let mut rows = Vec::new();
    for k in [2u32, 3, 4, 5, 6, 7] {
        let n = (1usize << k) + 1;
        let h = 3.0 / (n - 1) as f64;
        let y: Vec<f64> = (0..n).map(|i| f(i as f64 * h)).collect();
        let sp = Spacing::Uniform(h);
        rows.push(vec![
            format!("{n}"),
            format!("{:.1e}", (trapezoid(&y, sp).unwrap() - exact).abs()),
            format!("{:.1e}", (simpson(&y, sp).unwrap() - exact).abs()),
            format!("{:.1e}", (romb(&y, h).unwrap() - exact).abs()),
        ]);
    }
    write("integration", "sampled_convergence.md", &md_table(&["samples", "trapezoid", "Simpson", "Romberg"], &rows));
}

fn derivative_reference() {
    let refs = Refs::parse(include_str!("../data/sampled_reference.txt"));
    type Named = (&'static str, &'static str, Box<dyn Fn(f64) -> f64>);
    let funcs: Vec<Named> = vec![
        ("exp", "eˣ", Box::new(f64::exp)),
        ("sin3x", "sin 3x", Box::new(|t: f64| (3.0 * t).sin())),
        ("log1p", "ln(1 + x)", Box::new(f64::ln_1p)),
        ("atan10", "arctan 10x", Box::new(|t: f64| (10.0 * t).atan())),
    ];
    let pts = [0.3, 1.0, 2.7];
    let mut rows = Vec::new();
    for (key, label, f) in funcs {
        let r = refs.get(&format!("derivative {key}"));
        let (scipy_df, scipy_err, scipy_nfev, exact) = (&r[0], &r[1], &r[2], &r[3]);
        let mut ours_err: f64 = 0.0;
        let mut ours_est: f64 = 0.0;
        let mut ours_nfev = 0usize;
        for (i, &x) in pts.iter().enumerate() {
            let d = derivative(&f, x, DerivativeOptions::default()).unwrap();
            ours_err = ours_err.max((d.df - exact[i]).abs() / exact[i].abs());
            ours_est = ours_est.max(d.error / exact[i].abs());
            ours_nfev = ours_nfev.max(d.nfev);
        }
        let scipy_true = (0..3).fold(0.0f64, |m, i| m.max((scipy_df[i] - exact[i]).abs() / exact[i].abs()));
        let scipy_est = (0..3).fold(0.0f64, |m, i| m.max(scipy_err[i] / exact[i].abs()));
        let scipy_n = scipy_nfev.iter().fold(0.0f64, |m, v| m.max(*v));
        rows.push(vec![label.to_string(), format!("{} / {}", fmt_diff(ours_err), fmt_diff(scipy_true)), format!("{} / {}", fmt_diff(ours_est), fmt_diff(scipy_est)), format!("{ours_nfev} / {scipy_n}")]);
    }
    write("derivatives", "adaptive_scipy.md", &md_table(&["f(x), at x = 0.3, 1, 2.7", "true relative error (library / SciPy)", "estimated error (library / SciPy)", "evaluations (library / SciPy)"], &rows));
}

/// The error of a fixed central difference against the step, with Ridders' result.
fn derivative_figure() {
    let x = 1.0f64;
    let exact = x.exp();
    let mut plot = Plot::new("Derivative of eˣ at x = 1", "step h", "absolute error");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::BottomLeft;
    let hs: Vec<f64> = (0..=60).map(|i| 10f64.powf(-12.0 + i as f64 * 0.2)).collect();
    let central: Vec<(f64, f64)> = hs.iter().map(|&h| (h, (((x + h).exp() - (x - h).exp()) / (2.0 * h) - exact).abs().max(1e-17))).collect();
    plot.series.push(Series::new("central difference with step h", central, Style::Line, 0));
    let r = derivative(f64::exp, x, DerivativeOptions::default()).unwrap();
    plot.series.push(Series::new(&format!("Ridders' extrapolation ({} evaluations)", r.nfev), vec![(1e-12, (r.df - exact).abs().max(1e-17)), (1e0, (r.df - exact).abs().max(1e-17))], Style::Dashed, 1));
    write("derivatives", "adaptive_steps.svg", &plot.render());
}
