//! Figures and tables for the integration chapter.

use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_integration::can_integrate::CanIntegrate1D;
use lib_jsl_integration::adaptive_trapezoid_integration_engine::AdaptiveTrapezoidIntegrationEngine;
use lib_jsl_integration::gaussian_quadrature::{GaussLegendreEngine, gauss_legendre};
use lib_jsl_integration::midpoint_integration_engine::MidpointIntegrationEngine;
use lib_jsl_integration::romberg::RombergEngine;
use lib_jsl_integration::simpsons_1_3_integration_engine::Simpson13Engine;
use lib_jsl_integration::tanh_sinh::TanhSinhEngine;
use lib_jsl_integration::trapezoidal_integration_engine::TrapezoidalIntegrationEngine;
use lib_jsl_special_functions::legendre;
use std::cell::{Cell, RefCell};
use std::f64::consts::{FRAC_PI_2, PI};

const CHAPTER: &str = "integration";

/// Runs `integrate` with a counting wrapper around `f`; returns `(value, evaluations)`.
fn counted<F: Fn(f64) -> f64>(f: &F, integrate: impl Fn(&dyn Fn(f64) -> f64) -> f64) -> (f64, usize) {
    let count = Cell::new(0);
    let wrapped = |x: f64| {
        count.set(count.get() + 1);
        f(x)
    };
    let value = integrate(&wrapped);
    (value, count.get())
}

/// Error-versus-evaluations curves of the classical rules for `f` on `[a, b]`.
fn convergence_series(f: &dyn Fn(f64) -> f64, a: f64, b: f64, exact: f64, with_closed_rules: bool) -> Vec<Series> {
    let err = |v: f64| (v - exact).abs().max(1e-17);
    let mut out = Vec::new();
    if with_closed_rules {
        let trap: Vec<(f64, f64)> = (1..=10).map(|k| (1usize << k) + 1).map(|n| { let (v, e) = counted(&f, |g| TrapezoidalIntegrationEngine::new(n, a, b, false).integrate(g).unwrap()); (e as f64, err(v)) }).collect();
        out.push(Series::new("Trapezoid", trap, Style::LineMarkers, 0));
        let simpson: Vec<(f64, f64)> = (1..=10).map(|k| (1usize << k) + 1).map(|n| { let (v, e) = counted(&f, |g| Simpson13Engine::new(n, a, b, false).integrate(g).unwrap()); (e as f64, err(v)) }).collect();
        out.push(Series::new("Simpson", simpson, Style::LineMarkers, 4));
        let romberg: Vec<(f64, f64)> = (1..=10).map(|k| { let (t, e) = counted(&f, |g| RombergEngine::new(a, b).tableau(g, k).unwrap()[k][k]); (e as f64, err(t)) }).collect();
        out.push(Series::new("Romberg", romberg, Style::LineMarkers, 2));
    }
    let gauss: Vec<(f64, f64)> = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024].iter().map(|&n| { let (v, e) = counted(&f, |g| GaussLegendreEngine::new(n, a, b).unwrap().integrate(g).unwrap()); (e as f64, err(v)) }).collect();
    out.push(Series::new("Gauss-Legendre", gauss, Style::LineMarkers, 1));
    let tanh: Vec<(f64, f64)> = (0..=7).map(|levels| {
        let mut engine = TanhSinhEngine::new(a, b);
        engine.tolerance = 0.0;
        engine.max_levels = levels;
        let (v, e) = counted(&f, |g| engine.integrate(g).unwrap());
        (e as f64, err(v))
    }).collect();
    out.push(Series::new("Tanh-sinh", tanh, Style::LineMarkers, 3));
    out
}

fn convergence_figures() {
    // A smooth (analytic) integrand with poles near the interval: the Runge function.
    let runge = |x: f64| 1.0 / (1.0 + 25.0 * x * x);
    let mut plot = Plot::new("Smooth integrand: ∫ 1/(1+25x²) dx on [−1, 1]", "function evaluations", "absolute error");
    plot.log_x = true;
    plot.log_y = true;
    plot.y_range = Some((1e-17, 1.0));
    plot.series = convergence_series(&runge, -1.0, 1.0, 0.4 * 5f64.atan(), true);
    write(CHAPTER, "convergence_smooth.svg", &plot.render());

    // An endpoint singularity in the derivative: √x on [0, 1].
    let root = |x: f64| x.sqrt();
    let mut plot = Plot::new("Endpoint singularity: ∫ √x dx on [0, 1]", "function evaluations", "absolute error");
    plot.log_x = true;
    plot.log_y = true;
    plot.y_range = Some((1e-17, 1.0));
    plot.series = convergence_series(&root, 0.0, 1.0, 2.0 / 3.0, true);
    write(CHAPTER, "convergence_singular.svg", &plot.render());
}

fn gauss_figures() {
    let rule = gauss_legendre(6).unwrap();
    let curve: Vec<(f64, f64)> = (0..=400).map(|i| { let x = -1.0 + 2.0 * i as f64 / 400.0; (x, legendre(6, x)) }).collect();
    let mut plot = Plot::new("The 6 Gauss-Legendre nodes are the zeros of P₆", "x", "P₆(x)");
    plot.series.push(Series::new("P₆(x)", curve, Style::Line, 0));
    plot.series.push(Series::new("nodes xᵢ", rule.nodes.iter().map(|&x| (x, 0.0)).collect(), Style::Markers, 1));
    plot.series.push(Series::new("", vec![(-1.0, 0.0), (1.0, 0.0)], Style::Dashed, 6));
    write(CHAPTER, "gauss_legendre_nodes.svg", &plot.render());

    let mut plot = Plot::new("Gauss-Legendre weights", "node xᵢ", "weight wᵢ");
    for (k, &n) in [5usize, 10, 20].iter().enumerate() {
        let r = gauss_legendre(n).unwrap();
        plot.series.push(Series::new(&format!("n = {n}"), r.nodes.iter().zip(&r.weights).map(|(&x, &w)| (x, w)).collect(), Style::LineMarkers, k));
    }
    let asymptote: Vec<(f64, f64)> = (0..=200).map(|i| { let x = -1.0 + 2.0 * i as f64 / 200.0; (x, PI / 20.0 * (1.0 - x * x).sqrt()) }).collect();
    plot.series.push(Series::new("(π/20)√(1 − x²)", asymptote, Style::Dashed, 6));
    write(CHAPTER, "gauss_legendre_weights.svg", &plot.render());

    let mut table = String::from("| $i$ | node $x_i$ | weight $w_i$ |\n|---:|---:|---:|\n");
    for (i, (x, w)) in gauss_legendre(5).unwrap().nodes.iter().zip(gauss_legendre(5).unwrap().weights).enumerate() {
        table.push_str(&format!("| {} | {:+.16} | {:.16} |\n", i + 1, x, w));
    }
    write(CHAPTER, "gauss_legendre_table.md", &table);

    let rule = gauss_legendre(3).unwrap();
    let mut table = String::from("| degree $k$ | exact $\\int_{-1}^1 x^k\\,dx$ | 3-point Gauss | error |\n|---:|---:|---:|---:|\n");
    for k in 0..=8 {
        let exact = if k % 2 == 1 { 0.0 } else { 2.0 / (k + 1) as f64 };
        let q = rule.apply(|x| x.powi(k));
        table.push_str(&format!("| {k} | {exact:.10} | {q:.10} | {:.1e} |\n", (q - exact).abs()));
    }
    write(CHAPTER, "gauss_legendre_exactness.md", &table);
}

fn romberg_figures() {
    let exact = std::f64::consts::E - 1.0;
    let levels = 7;
    let t = RombergEngine::new(0.0, 1.0).tableau(f64::exp, levels).unwrap();
    let mut plot = Plot::new("Each Richardson column gains two orders: ∫₀¹ eˣ dx", "panel width h", "absolute error");
    plot.legend = Legend::BottomRight;
    plot.log_x = true;
    plot.log_y = true;
    plot.y_range = Some((1e-17, 1.0));
    for j in 0..5 {
        let pts: Vec<(f64, f64)> = t.iter().enumerate().skip(j).map(|(k, row)| (1.0 / (1u64 << k) as f64, (row[j] - exact).abs().max(1e-17))).collect();
        let label = match j {
            0 => "column 0 (trapezoid, h²)".to_string(),
            1 => "column 1 (Simpson, h⁴)".to_string(),
            2 => "column 2 (Boole, h⁶)".to_string(),
            3 => "column 3 (h⁸)".to_string(),
            _ => "column 4 (h¹⁰)".to_string(),
        };
        plot.series.push(Series::new(&label, pts, Style::LineMarkers, j));
    }
    write(CHAPTER, "romberg_columns.svg", &plot.render());

    let mut table = String::from("| $k$ | $h$ | $R(k,0)$ | $R(k,1)$ | $R(k,2)$ | $R(k,3)$ | $R(k,4)$ |\n|---:|---:|---:|---:|---:|---:|---:|\n");
    for (k, row) in t.iter().enumerate().take(5) {
        let cells: Vec<String> = (0..5).map(|j| row.get(j).map_or(String::new(), |v| format!("{v:.12}"))).collect();
        table.push_str(&format!("| {k} | 1/{} | {} |\n", 1u64 << k, cells.join(" | ")));
    }
    table.push_str(&format!("\nExact value: $e - 1 = {exact:.12}$.\n"));
    write(CHAPTER, "romberg_tableau.md", &table);
}

fn tanh_sinh_figures() {
    // The transformed integrand on [0, 1]: g(t) = f(x(t)) · ½ · w(t), w = (π/2) cosh t / cosh²u.
    let transformed = |f: &dyn Fn(f64) -> f64, t: f64| {
        let u = FRAC_PI_2 * t.sinh();
        let x = 1.0 / (1.0 + (-2.0 * u).exp());
        let w = FRAC_PI_2 * t.cosh() / u.cosh().powi(2);
        (f(x) * 0.5 * w).abs()
    };
    let ts: Vec<f64> = (0..=360).map(|i| -4.5 + 9.0 * i as f64 / 360.0).collect();
    let mut plot = Plot::new("After x = ½(1 + tanh(½π sinh t)) every integrand decays double exponentially", "t", "|transformed integrand|");
    plot.log_y = true;
    plot.y_range = Some((1e-30, 10.0));
    plot.legend = Legend::BottomRight;
    let integrands: [(&str, &dyn Fn(f64) -> f64); 3] = [("f(x) = 1", &|_| 1.0), ("f(x) = ln x", &|x: f64| x.ln()), ("f(x) = 1/√x", &|x: f64| 1.0 / x.sqrt())];
    for (k, (label, f)) in integrands.iter().enumerate() {
        plot.series.push(Series::new(label, ts.iter().map(|&t| (t, transformed(*f, t))).collect(), Style::Line, k));
    }
    write(CHAPTER, "tanh_sinh_transformed.svg", &plot.render());

    type Case<'a> = (&'a str, &'a dyn Fn(f64) -> f64, f64);
    let cases: [Case; 4] = [
        ("$\\ln x$", &|x: f64| x.ln(), -1.0),
        ("$1/\\sqrt{x}$", &|x: f64| 1.0 / x.sqrt(), 2.0),
        ("$\\sqrt{x}$", &|x: f64| x.sqrt(), 2.0 / 3.0),
        ("$x^{-0.9}$", &|x: f64| x.powf(-0.9), 10.0),
    ];
    let mut table = String::from("| integrand on $[0,1]$ | exact | tanh-sinh error | evaluations | Gauss-Legendre (100 pts) error |\n|---|---:|---:|---:|---:|\n");
    for (label, f, exact) in cases {
        let (v, e) = counted(&f, |g| TanhSinhEngine::new(0.0, 1.0).integrate(g).unwrap());
        let gl = GaussLegendreEngine::new(100, 0.0, 1.0).unwrap().integrate(f).unwrap();
        table.push_str(&format!("| {label} | {exact} | {:.1e} | {e} | {:.1e} |\n", (v - exact).abs(), (gl - exact).abs()));
    }
    write(CHAPTER, "tanh_sinh_table.md", &table);
}

pub fn generate() {
    newton_cotes_figures();
    convergence_figures();
    gauss_figures();
    romberg_figures();
    tanh_sinh_figures();
}

/// The weights of the closed Newton-Cotes rule with `points` equally spaced nodes on [0, 1], from
/// the moment equations Σ wᵢ tᵢᵏ = 1/(k + 1) (small enough for elimination with pivoting).
fn newton_cotes_weights(points: usize) -> Vec<f64> {
    let n = points;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let mut m: Vec<Vec<f64>> = (0..n).map(|k| { let mut row: Vec<f64> = t.iter().map(|&x| x.powi(k as i32)).collect(); row.push(1.0 / (k + 1) as f64); row }).collect();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs())).unwrap();
        m.swap(c, p);
        let pivot_row = m[c].clone();
        for row in m.iter_mut().skip(c + 1) {
            let f = row[c] / pivot_row[c];
            for (x, p) in row.iter_mut().zip(&pivot_row).skip(c) {
                *x -= f * p;
            }
        }
    }
    let mut w = vec![0.0; n];
    for r in (0..n).rev() {
        w[r] = (m[r][n] - (r + 1..n).map(|k| m[r][k] * w[k]).sum::<f64>()) / m[r][r];
    }
    w
}

fn newton_cotes_figures() {
    let exact = std::f64::consts::E - 1.0;
    // Error against the number of panels for the three composite rules.
    let mut plot = Plot::new("Composite rules on ∫₀¹ eˣ dx: slopes 2, 2 and 4", "number of panels", "absolute error");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::BottomLeft;
    let panels: Vec<usize> = (0..=12).map(|k| 1usize << k).collect();
    let mid: Vec<(f64, f64)> = panels.iter().map(|&n| (n as f64, (MidpointIntegrationEngine::new(n, 0.0, 1.0, false).integrate(f64::exp).unwrap() - exact).abs())).collect();
    let trap: Vec<(f64, f64)> = panels.iter().map(|&n| (n as f64, (TrapezoidalIntegrationEngine::new(n + 1, 0.0, 1.0, false).integrate(f64::exp).unwrap() - exact).abs())).collect();
    let simp: Vec<(f64, f64)> = panels.iter().filter(|&&n| n >= 2).map(|&n| (n as f64, (Simpson13Engine::new(n + 1, 0.0, 1.0, false).integrate(f64::exp).unwrap() - exact).abs().max(1e-17))).collect();
    plot.series.push(Series::new("trapezoid", trap, Style::LineMarkers, 0));
    plot.series.push(Series::new("midpoint", mid, Style::LineMarkers, 1));
    plot.series.push(Series::new("Simpson 1/3", simp, Style::LineMarkers, 4));
    write(CHAPTER, "newton_cotes_convergence.svg", &plot.render());

    // Midpoint and trapezoid errors, side by side.
    let mut table = String::from("| panels $n$ | midpoint $M_n - I$ | trapezoid $T_n - I$ | ratio | $(2M_n + T_n)/3 - I$ |\n|---:|---:|---:|---:|---:|\n");
    for n in [1usize, 2, 4, 8, 16] {
        let m = MidpointIntegrationEngine::new(n, 0.0, 1.0, false).integrate(f64::exp).unwrap();
        let t = TrapezoidalIntegrationEngine::new(n + 1, 0.0, 1.0, false).integrate(f64::exp).unwrap();
        table.push_str(&format!("| {n} | {:+.3e} | {:+.3e} | {:.4} | {:+.3e} |\n", m - exact, t - exact, (m - exact) / (t - exact), (2.0 * m + t) / 3.0 - exact));
    }
    write(CHAPTER, "newton_cotes_midpoint_trapezoid.md", &table);

    // Closed Newton-Cotes weights, degree of exactness and Σ|w|.
    let mut table = String::from("| points | weights on $[0, 1]$ | degree | $\\sum_i \\lvert w_i \\rvert$ |\n|---:|---|---:|---:|\n");
    for points in 2..=11usize {
        let w = newton_cotes_weights(points);
        let panels = points - 1;
        let degree = if panels % 2 == 0 { panels + 1 } else { panels };
        let shown: Vec<String> = w.iter().take(points.div_ceil(2)).map(|v| format!("{v:.5}")).collect();
        let more = if points > 2 { ", …" } else { "" };
        table.push_str(&format!("| {points} | {}{more} | {degree} | {:.3} |\n", shown.join(", "), w.iter().map(|v| v.abs()).sum::<f64>()));
    }
    write(CHAPTER, "newton_cotes_weights.md", &table);

    // The trapezoid rule on a periodic integrand converges exponentially.
    let periodic_exact = 2.0 * PI * lib_jsl_special_functions::bessel_functions_integer::bessel_i_integer(0, 1.0);
    let mut plot = Plot::new("The trapezoid rule on a periodic integrand converges exponentially", "number of panels", "absolute error");
    plot.log_y = true;
    plot.y_range = Some((1e-17, 10.0));
    let periodic: Vec<(f64, f64)> = (1..=24).map(|n| (n as f64, (TrapezoidalIntegrationEngine::new(n + 1, 0.0, 2.0 * PI, false).integrate(|x| x.cos().exp()).unwrap() - periodic_exact).abs().max(1e-17))).collect();
    let nonperiodic: Vec<(f64, f64)> = (1..=24).map(|n| (n as f64, (TrapezoidalIntegrationEngine::new(n + 1, 0.0, 1.0, false).integrate(f64::exp).unwrap() - exact).abs())).collect();
    plot.series.push(Series::new("trapezoid, e^(cos x) on [0, 2π]", periodic, Style::LineMarkers, 0));
    plot.series.push(Series::new("trapezoid, eˣ on [0, 1]", nonperiodic, Style::LineMarkers, 1));
    write(CHAPTER, "trapezoid_periodic.svg", &plot.render());

    // The adaptive trapezoid rule against the uniform rule with the same number of evaluations.
    let c = 1.0 / PI;
    let r = 1e-3f64.sqrt();
    type Case<'a> = (&'a str, &'a dyn Fn(f64) -> f64, f64);
    let peak = |x: f64| 1.0 / (1e-3 + (x - 0.3) * (x - 0.3));
    let kink = move |x: f64| (x - c).abs().sqrt();
    let jump = move |x: f64| if x < c { 0.0 } else { 1.0 };
    let cases: [Case; 3] = [
        ("smooth peak", &peak, ((0.7 / r).atan() + (0.3 / r).atan()) / r),
        ("kink", &kink, 2.0 / 3.0 * (c.powf(1.5) + (1.0 - c).powf(1.5))),
        ("jump", &jump, 1.0 - c),
    ];
    let mut table = String::from("| integrand on $[0, 1]$ | tolerance | evaluations | adaptive error | uniform error, same evaluations |\n|---|---:|---:|---:|---:|\n");
    for (label, f, exact) in cases {
        for tol in [1e-4, 1e-8] {
            let (v, n) = counted(&f, |g| AdaptiveTrapezoidIntegrationEngine::new(tol, 0.0, 1.0).integrate(g).unwrap());
            let u = TrapezoidalIntegrationEngine::new(n, 0.0, 1.0, false).integrate(f).unwrap();
            table.push_str(&format!("| {label} | {tol:.0e} | {n} | {:.1e} | {:.1e} |\n", (v - exact).abs(), (u - exact).abs()));
        }
    }
    write(CHAPTER, "adaptive_trapezoid_table.md", &table);

    let samples = RefCell::new(Vec::new());
    AdaptiveTrapezoidIntegrationEngine::new(1e-4, 0.0, 1.0).integrate(|x| { samples.borrow_mut().push(x); kink(x) }).unwrap();
    let xs = samples.into_inner();
    let mut plot = Plot::new(&format!("Adaptive trapezoid samples cluster at the kink ({} samples, tolerance 10⁻⁴)", xs.len()), "x", "f(x) = √|x − 1/π|");
    plot.legend = Legend::BottomRight;
    let curve: Vec<(f64, f64)> = (0..=800).map(|i| { let x = i as f64 / 800.0; (x, kink(x)) }).collect();
    plot.series.push(Series::new("f(x)", curve, Style::Line, 0));
    plot.series.push(Series::new("samples", xs.iter().map(|&x| (x, kink(x))).collect(), Style::Markers, 1));
    write(CHAPTER, "adaptive_trapezoid_samples.svg", &plot.render());
}
