//! Figures and tables for the integration chapter.

use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_integration::can_integrate::CanIntegrate1D;
use lib_jsl_integration::adaptive_gl7k15_integration_engine::AdaptiveGL7K15IntegrationEngine;
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
    finite_difference_figures();
    bvp_figures();
    stiff_figures();
    runge_kutta_figures();
    monte_carlo_figures();
    cubature_figures();
    contour_figures();
    gauss_kronrod_figures();
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

fn gauss_kronrod_figures() {
    // The G7-K15 pair: Kronrod adds 8 nodes between and beyond the 7 Gauss nodes.
    let g7 = gauss_legendre(7).unwrap();
    let k15: [f64; 8] = [0.0, 0.2077849550078985, 0.4058451513773972, 0.5860872354676911, 0.7415311855993945, 0.8648644233597691, 0.9491079123427585, 0.9914553711208126];
    let wk: [f64; 8] = [0.2094821410847278, 0.2044329400752989, 0.1903505780647854, 0.1690047266392679, 0.1406532597155259, 0.1047900103222502, 0.0630920926299786, 0.0229353220105292];
    let mut kronrod_points: Vec<(f64, f64)> = k15.iter().zip(&wk).flat_map(|(&x, &w)| [(-x, w), (x, w)]).collect();
    kronrod_points.dedup();
    let mut plot = Plot::new("The 15 Kronrod nodes contain the 7 Gauss nodes", "node x", "weight");
    plot.series.push(Series::new("Kronrod (15 points)", kronrod_points, Style::Stems, 0));
    plot.series.push(Series::new("Gauss (7 points)", g7.nodes.iter().zip(&g7.weights).map(|(&x, &w)| (x, w)).collect(), Style::Markers, 1));
    plot.y_range = Some((0.0, 0.45));
    write(CHAPTER, "gauss_kronrod_nodes.svg", &plot.render());

    // Where the global strategy samples an integrand with a peak and a kink.
    let f = |x: f64| 1.0 / (1.0 + 1e4 * (x - 0.3) * (x - 0.3)) + (x - 0.7).abs().sqrt();
    let samples = RefCell::new(Vec::new());
    let (_, _, n) = AdaptiveGL7K15IntegrationEngine::new(1e-10, 0.0, 1.0).integrate_with_error(|x| { samples.borrow_mut().push(x); f(x) }).unwrap();
    let xs = samples.into_inner();
    let mut plot = Plot::new(&format!("Global adaptivity concentrates panels at the peak and the kink ({n} evaluations)"), "x", "f(x)");
    let curve: Vec<(f64, f64)> = (0..=1000).map(|i| { let x = i as f64 / 1000.0; (x, f(x)) }).collect();
    plot.series.push(Series::new("f(x) = 1/(1 + 10⁴(x − 0.3)²) + √|x − 0.7|", curve, Style::Line, 0));
    plot.series.push(Series::new("samples", xs.iter().map(|&x| (x, f(x))).collect(), Style::Markers, 1));
    plot.legend = Legend::TopRight;
    plot.y_range = Some((0.0, 1.8));
    write(CHAPTER, "gauss_kronrod_samples.svg", &plot.render());

    // Evaluations, true and estimated errors at tolerance 1e-10, against the adaptive trapezoid rule.
    let c = 1.0 / PI;
    type Case<'a> = (&'a str, &'a dyn Fn(f64) -> f64, f64);
    let r = 1e-4f64.sqrt();
    let cases: [Case; 5] = [
        ("$e^x$", &f64::exp, std::f64::consts::E - 1.0),
        ("$1/(10^{-4} + (x - 0.3)^2)$", &|x: f64| 1.0 / (1e-4 + (x - 0.3) * (x - 0.3)), ((0.7 / r).atan() + (0.3 / r).atan()) / r),
        ("$\\sqrt{\\lvert x - 1/\\pi \\rvert}$", &move |x: f64| (x - c).abs().sqrt(), 2.0 / 3.0 * (c.powf(1.5) + (1.0 - c).powf(1.5))),
        ("$\\ln x$", &f64::ln, -1.0),
        ("$x^{-0.9}$", &|x: f64| x.powf(-0.9), 10.0),
    ];
    let tol = 1e-10;
    let mut table = String::from("| integrand on $[0, 1]$ | evaluations | true error | estimated error | adaptive trapezoid evaluations |\n|---|---:|---:|---:|---:|\n");
    for (label, f, exact) in cases {
        let engine = AdaptiveGL7K15IntegrationEngine::new(tol, 0.0, 1.0);
        let (v, est, n) = engine.integrate_with_error(f).unwrap();
        let trap = {
            let count = Cell::new(0);
            let ok = AdaptiveTrapezoidIntegrationEngine::new(tol, 0.0, 1.0).integrate(|x| { count.set(count.get() + 1); f(x) });
            match ok { Ok(_) => count.get().to_string(), Err(_) => "fails (f(0) is infinite)".to_string() }
        };
        table.push_str(&format!("| {label} | {n} | {:.1e} | {est:.1e} | {trap} |\n", (v - exact).abs()));
    }
    write(CHAPTER, "gauss_kronrod_table.md", &table);
}

fn contour_figures() {
    use lib_jsl_integration::complex_path_integrals::{C64, Contour, PathSegment, cauchy_derivative, contour_integral};
    // ∮ e^z / z³ dz around the unit circle = 2πi · ½: trapezoid against adaptive Gauss-Kronrod.
    let f = |z: C64| z.exp() / (z * z * z);
    let exact = C64::new(0.0, PI);
    let trapezoid: Vec<(f64, f64)> = (1..=24).map(|n| {
        let sum: C64 = (0..n).map(|k| { let z = C64::from_polar(1.0, 2.0 * PI * k as f64 / n as f64); f(z) * C64::new(0.0, 1.0) * z }).sum();
        (n as f64, ((sum * (2.0 * PI / n as f64)) - exact).norm().max(1e-17))
    }).collect();
    let circle = Contour::circle(C64::new(0.0, 0.0), 1.0);
    let gk: Vec<(f64, f64)> = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14].iter().map(|&tol| {
        let r = contour_integral(f, &circle, tol).unwrap();
        (r.evaluations as f64, (r.value - exact).norm().max(1e-17))
    }).collect();
    let mut plot = Plot::new("∮ eᶻ/z³ dz around the unit circle: the periodic trapezoid rule wins", "function evaluations", "absolute error");
    plot.log_x = true;
    plot.log_y = true;
    plot.y_range = Some((1e-17, 10.0));
    plot.series.push(Series::new("trapezoid rule (equally spaced angles)", trapezoid, Style::LineMarkers, 0));
    plot.series.push(Series::new("adaptive Gauss-Kronrod (varying tolerance)", gk, Style::LineMarkers, 1));
    write(CHAPTER, "contour_trapezoid_vs_gk.svg", &plot.render());

    // Cauchy's formula for f^(n)(0) with f = e^z: the error as a function of the radius.
    let mut plot = Plot::new("Derivatives of eᶻ at 0 by Cauchy's formula: the radius matters", "radius r of the circle", "absolute error");
    plot.log_x = true;
    plot.log_y = true;
    plot.y_range = Some((1e-17, 1e3));
    plot.legend = Legend::TopRight;
    for (k, n) in [1u32, 4, 8, 12].iter().enumerate() {
        let pts: Vec<(f64, f64)> = (0..=40).map(|i| 10f64.powf(-3.0 + 4.5 * i as f64 / 40.0)).map(|r| {
            let d = cauchy_derivative(|z| z.exp(), C64::new(0.0, 0.0), *n, r, 1e-14).unwrap();
            (r, (d - C64::new(1.0, 0.0)).norm().max(1e-17))
        }).collect();
        plot.series.push(Series::new(&format!("n = {n}"), pts, Style::Line, k));
    }
    write(CHAPTER, "cauchy_derivative_radius.svg", &plot.render());

    // A real integral from a closed contour: ∫ dx/(1 + x²) over the real line = π.
    let g = |z: C64| (z * z + 1.0).inv();
    let mut table = String::from("| radius $R$ | segment $[-R, R]$ | arc $\\lvert z \\rvert = R$ | closed contour | $\\pi$ |\n|---:|---:|---:|---:|---:|\n");
    for r in [2.0, 10.0, 100.0, 1000.0] {
        let line = contour_integral(g, &Contour::new(vec![PathSegment::line(C64::new(-r, 0.0), C64::new(r, 0.0))]), 1e-13).unwrap().value;
        let arc = contour_integral(g, &Contour::new(vec![PathSegment::arc(C64::new(0.0, 0.0), r, 0.0, PI)]), 1e-13).unwrap().value;
        let total = line + arc;
        table.push_str(&format!("| {r} | {:.12} | {:+.3e} | {:.12} | {:.12} |\n", line.re, arc.re, total.re, PI));
    }
    write(CHAPTER, "contour_semicircle.md", &table);
}

fn cubature_figures() {
    use lib_jsl_integration::cubature::{CubatureOptions, adaptive_cubature, tensor_gauss_legendre};
    // Points per degree-7 rule against dimension.
    let mut table = String::from("| dimension $n$ | monomials of degree $\\le 7$ | Genz-Malik, degree 7 | tensor Gauss, 4 points per axis |\n|---:|---:|---:|---:|\n");
    for n in [1u64, 2, 3, 4, 5, 6, 8, 10, 15, 20] {
        let monomials: u64 = (1..=7).fold(1u64, |acc, k| acc * (n + k) / k);
        let gm = (1u64 << n) + 2 * n * n + 2 * n + 1;
        table.push_str(&format!("| {n} | {monomials} | {gm} | {} |\n", 4u64.pow(n as u32)));
    }
    write(CHAPTER, "cubature_points.md", &table);

    // Where adaptive cubature samples a peaked 2D integrand.
    let samples = RefCell::new(Vec::new());
    let s = 0.05;
    let peak = |x: &[f64]| (-((x[0] - 0.3).powi(2) + (x[1] + 0.2).powi(2)) / (2.0 * s * s)).exp();
    let r = adaptive_cubature(|x: &[f64]| { samples.borrow_mut().push((x[0], x[1])); peak(x) }, &[-1.0, -1.0], &[1.0, 1.0], CubatureOptions { absolute_tolerance: 1e-8, relative_tolerance: 1e-8, ..Default::default() }).unwrap();
    let pts = samples.into_inner();
    let mut plot = Plot::new(&format!("Adaptive Genz-Malik samples for a Gaussian peak at (0.3, −0.2): {} evaluations", r.evaluations), "x", "y");
    plot.series.push(Series::new("", pts, Style::Markers, 0));
    plot.x_range = Some((-1.0, 1.0));
    plot.y_range = Some((-1.0, 1.0));
    write(CHAPTER, "cubature_samples.svg", &plot.render());

    // Convergence on a smooth integrand in 3 and 6 dimensions.
    let erf1 = 1.0 - lib_jsl_special_functions::error_function::erfc(1.0);
    let mut plot = Plot::new("∫ exp(−|x|²) over [0, 1]ⁿ: adaptive Genz-Malik against tensor Gauss", "function evaluations", "relative error");
    plot.log_x = true;
    plot.log_y = true;
    plot.y_range = Some((1e-16, 1.0));
    plot.legend = Legend::BottomLeft;
    for (k, n) in [3usize, 6].iter().enumerate() {
        let exact = (0.5 * PI.sqrt() * erf1).powi(*n as i32);
        let f = |x: &[f64]| (-x.iter().map(|v| v * v).sum::<f64>()).exp();
        let tensor: Vec<(f64, f64)> = (1..=12).filter(|&p| (p as f64).powi(*n as i32) < 3e6).map(|p| {
            let v = tensor_gauss_legendre(f, &vec![0.0; *n], &vec![1.0; *n], p).unwrap();
            ((p as f64).powi(*n as i32), ((v - exact) / exact).abs().max(1e-17))
        }).collect();
        let adaptive: Vec<(f64, f64)> = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12].iter().map(|&tol| {
            let r = adaptive_cubature(f, &vec![0.0; *n], &vec![1.0; *n], CubatureOptions { absolute_tolerance: 0.0, relative_tolerance: tol, max_evaluations: 3_000_000 }).unwrap();
            (r.evaluations as f64, ((r.value - exact) / exact).abs().max(1e-17))
        }).collect();
        plot.series.push(Series::new(&format!("tensor Gauss, n = {n}"), tensor, Style::LineMarkers, 2 * k));
        plot.series.push(Series::new(&format!("adaptive Genz-Malik, n = {n}"), adaptive, Style::LineMarkers, 2 * k + 1));
    }
    write(CHAPTER, "cubature_convergence.svg", &plot.render());

    // A corner singularity in 2D: here adaptivity pays.
    let singular = |x: &[f64]| 1.0 / (x[0] * x[0] + x[1] * x[1]).sqrt();
    let exact = 2.0 * (1.0 + 2f64.sqrt()).ln();
    let mut plot = Plot::new("A corner singularity: ∫∫ 1/√(x² + y²) over [0, 1]²", "function evaluations", "relative error");
    plot.log_x = true;
    plot.log_y = true;
    plot.y_range = Some((1e-16, 10.0));
    plot.legend = Legend::BottomLeft;
    let tensor: Vec<(f64, f64)> = [2usize, 4, 8, 16, 32, 64, 128, 256, 512, 1024].iter().map(|&p| {
        let v = tensor_gauss_legendre(singular, &[0.0, 0.0], &[1.0, 1.0], p).unwrap();
        ((p * p) as f64, ((v - exact) / exact).abs().max(1e-17))
    }).collect();
    let adaptive: Vec<(f64, f64)> = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10].iter().map(|&tol| {
        let r = adaptive_cubature(singular, &[0.0, 0.0], &[1.0, 1.0], CubatureOptions { absolute_tolerance: 0.0, relative_tolerance: tol, max_evaluations: 3_000_000 }).unwrap();
        (r.evaluations as f64, ((r.value - exact) / exact).abs().max(1e-17))
    }).collect();
    plot.series.push(Series::new("tensor Gauss", tensor, Style::LineMarkers, 0));
    plot.series.push(Series::new("adaptive Genz-Malik", adaptive, Style::LineMarkers, 1));
    write(CHAPTER, "cubature_singular.svg", &plot.render());
}

fn monte_carlo_figures() {
    use lib_jsl_integration::monte_carlo_integration::monte_carlo;
    use lib_jsl_integration::quasi_monte_carlo::{QuasiRandomSequence, quasi_monte_carlo_integration};
    use lib_jsl_random::quasi_random::{HaltonSequence, SobolSequence};
    use lib_jsl_random::uniform_generator::{UniformGenerator, UniformRNG};

    // 256 points in the unit square: random, Sobol, Halton.
    let n = 256;
    let mut rng = UniformRNG::from_seed(5);
    let random: Vec<(f64, f64)> = (0..n).map(|_| (rng.next_f64(), rng.next_f64())).collect();
    let mut sobol = SobolSequence::new(2).unwrap();
    let sobol_pts: Vec<(f64, f64)> = (0..n).map(|_| { let p = sobol.next_point(); (p[0], p[1]) }).collect();
    let mut halton = HaltonSequence::new(2).unwrap();
    let halton_pts: Vec<(f64, f64)> = (0..n).map(|_| { let p = halton.next_point(); (p[0], p[1]) }).collect();
    for (name, pts, colour) in [("random", random, 0), ("sobol", sobol_pts, 1), ("halton", halton_pts, 2)] {
        let title = match name { "random" => "256 pseudo-random points", "sobol" => "256 Sobol points", _ => "256 Halton points" };
        let mut plot = Plot::new(title, "x₁", "x₂");
        plot.x_range = Some((0.0, 1.0));
        plot.y_range = Some((0.0, 1.0));
        plot.series.push(Series::new("", pts, Style::Markers, colour));
        write(CHAPTER, &format!("points_{name}.svg"), &plot.render());
    }

    // RMS error over seeds against N in five dimensions.
    let product = |x: &[f64]| x.iter().map(|&v| 0.5 * PI * (PI * v).sin()).product::<f64>();
    let bounds = vec![(0.0, 1.0); 5];
    let seeds = 20u64;
    let rms = |errors: Vec<f64>| (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();
    let sizes: Vec<usize> = (6..=16).map(|k| 1usize << k).collect();
    let mc: Vec<(f64, f64)> = sizes.iter().map(|&n| (n as f64, rms((0..seeds).map(|s| monte_carlo(product, &bounds, n, s).unwrap().value - 1.0).collect()))).collect();
    let qmc = |sequence| -> Vec<(f64, f64)> { sizes.iter().map(|&n| (n as f64, rms((0..seeds).map(|s| quasi_monte_carlo_integration(product, &bounds, n, 1, sequence, s).unwrap().value - 1.0).collect()))).collect() };
    let mut plot = Plot::new("∫ Π (π/2) sin(πxᵢ) over [0, 1]⁵: RMS error over 20 random seeds", "number of points N", "RMS error");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::BottomLeft;
    let first = mc[0].1;
    plot.series.push(Series::new("Monte Carlo", mc, Style::LineMarkers, 0));
    plot.series.push(Series::new("randomised Sobol", qmc(QuasiRandomSequence::Sobol), Style::LineMarkers, 1));
    plot.series.push(Series::new("randomised Halton", qmc(QuasiRandomSequence::Halton), Style::LineMarkers, 2));
    plot.series.push(Series::new("N^(−1/2)", sizes.iter().map(|&n| (n as f64, first * (64.0 / n as f64).sqrt())).collect(), Style::Dashed, 6));
    plot.series.push(Series::new("N^(−1)", sizes.iter().map(|&n| (n as f64, first * 64.0 / n as f64)).collect(), Style::Dashed, 5));
    write(CHAPTER, "monte_carlo_convergence.svg", &plot.render());

    // The error does not depend on the dimension, only on the variance.
    let mut table = String::from("| dimension $d$ | $\\sigma$ | Monte Carlo error ($N = 10^5$) | standard error | randomised Halton error ($N = 10^5$) | tensor rule with 2 points per axis |\n|---:|---:|---:|---:|---:|---:|\n");
    for d in [1usize, 5, 20, 50] {
        // f = Π (1 + c (x_i − ½)) with c chosen so that Var f = 1/4 in every dimension; ∫ f = 1.
        let c = (12.0 * (1.25f64.powf(1.0 / d as f64) - 1.0)).sqrt();
        let f = move |x: &[f64]| x.iter().map(|&v| 1.0 + c * (v - 0.5)).product::<f64>();
        let b = vec![(0.0, 1.0); d];
        let mc = monte_carlo(f, &b, 100_000, 1).unwrap();
        let q = quasi_monte_carlo_integration(f, &b, 100_000, 1, QuasiRandomSequence::Halton, 1).unwrap();
        let tensor = if d <= 20 { format!("{}", 1u64 << d) } else { format!("{:.1e}", 2f64.powi(d as i32)) };
        table.push_str(&format!("| {d} | 0.5 | {:.1e} | {:.1e} | {:.1e} | {tensor} evaluations |\n", (mc.value - 1.0).abs(), mc.standard_error, (q.value - 1.0).abs()));
    }
    write(CHAPTER, "monte_carlo_dimensions.md", &table);
}

/// The restricted three-body problem (Earth-Moon) with the periodic Arenstorf orbit.
const ARENSTORF_MU: f64 = 0.012277471;
const ARENSTORF_Y0: [f64; 4] = [0.994, 0.0, 0.0, -2.001_585_106_379_082_5];
const ARENSTORF_PERIOD: f64 = 17.065_216_560_157_96;

fn arenstorf(_t: f64, y: &[f64]) -> Vec<f64> {
    let (mu, mu1) = (ARENSTORF_MU, 1.0 - ARENSTORF_MU);
    let d1 = ((y[0] + mu).powi(2) + y[1] * y[1]).powf(1.5);
    let d2 = ((y[0] - mu1).powi(2) + y[1] * y[1]).powf(1.5);
    vec![
        y[2],
        y[3],
        y[0] + 2.0 * y[3] - mu1 * (y[0] + mu) / d1 - mu * (y[0] - mu1) / d2,
        y[1] - 2.0 * y[2] - mu1 * y[1] / d1 - mu * y[1] / d2,
    ]
}

fn runge_kutta_figures() {
    use lib_jsl_integration::dormand_prince::dormand_prince_solve;
    use lib_jsl_integration::runge_kutta_4::rk4_solve;
    use lib_jsl_integration::stiff_ode::OdeOptions;
    let opts = |tol: f64| OdeOptions { relative_tolerance: tol, absolute_tolerance: tol, ..Default::default() };
    // The orbit, with the accepted Dormand-Prince steps.
    let s = dormand_prince_solve(arenstorf, 0.0, &ARENSTORF_Y0, ARENSTORF_PERIOD, opts(1e-7)).unwrap();
    let fine = dormand_prince_solve(arenstorf, 0.0, &ARENSTORF_Y0, ARENSTORF_PERIOD, OdeOptions { initial_step: Some(1e-4), ..opts(1e-12) }).unwrap();
    let mut plot = Plot::new(&format!("The Arenstorf orbit: {} adaptive steps at tolerance 10⁻⁷", s.t.len() - 1), "x", "y");
    plot.series.push(Series::new("orbit", fine.y.iter().map(|y| (y[0], y[1])).collect(), Style::Line, 0));
    plot.series.push(Series::new("accepted steps", s.y.iter().map(|y| (y[0], y[1])).collect(), Style::Markers, 1));
    plot.series.push(Series::new("Earth and Moon", vec![(-ARENSTORF_MU, 0.0), (1.0 - ARENSTORF_MU, 0.0)], Style::Markers, 6));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "arenstorf_orbit.svg", &plot.render());

    // Error after one period against evaluations.
    let error = |y: &[f64]| ((y[0] - ARENSTORF_Y0[0]).powi(2) + (y[1] - ARENSTORF_Y0[1]).powi(2)).sqrt();
    let rk4: Vec<(f64, f64)> = [2000usize, 4000, 8000, 16000, 32000, 64000, 128000].iter().map(|&n| {
        let s = rk4_solve(arenstorf, 0.0, &ARENSTORF_Y0, ARENSTORF_PERIOD, n).unwrap();
        (s.function_evaluations as f64, error(s.y.last().unwrap()).max(1e-16))
    }).collect();
    let dopri: Vec<(f64, f64)> = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12].iter().map(|&tol| {
        let s = dormand_prince_solve(arenstorf, 0.0, &ARENSTORF_Y0, ARENSTORF_PERIOD, opts(tol)).unwrap();
        (s.function_evaluations as f64, error(s.y.last().unwrap()).max(1e-16))
    }).collect();
    let mut plot = Plot::new("Arenstorf orbit: error after one period", "function evaluations", "distance from the starting point");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::BottomLeft;
    plot.series.push(Series::new("RK4, fixed step", rk4, Style::LineMarkers, 0));
    plot.series.push(Series::new("Dormand-Prince 5(4), adaptive", dopri, Style::LineMarkers, 1));
    write(CHAPTER, "arenstorf_convergence.svg", &plot.render());
}

fn stiff_figures() {
    use lib_jsl_integration::dormand_prince::dormand_prince_solve;
    use lib_jsl_integration::stiff_ode::{OdeOptions, OdeSolution, bdf, rosenbrock23};
    type NoJacobian = fn(f64, &[f64]) -> Vec<Vec<f64>>;
    let opts = |rtol: f64, atol: f64| OdeOptions { relative_tolerance: rtol, absolute_tolerance: atol, max_steps: 20_000_000, ..Default::default() };

    let scalar = |t: f64, y: &[f64]| vec![-1000.0 * (y[0] - t.cos())];
    let van_der_pol = |_t: f64, y: &[f64]| vec![y[1], 100.0 * ((1.0 - y[0] * y[0]) * y[1]) - y[0]];
    let robertson = |_t: f64, y: &[f64]| vec![-0.04 * y[0] + 1e4 * y[1] * y[2], 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1], 3e7 * y[1] * y[1]];
    // Reference solutions.
    let scalar_ref = {
        // y = (10⁶ cos t + 10³ sin t − 10⁶ e^{−1000 t}) / (10⁶ + 1).
        let t: f64 = 1.0;
        (1e6 * t.cos() + 1e3 * t.sin() - 1e6 * (-1000.0 * t).exp()) / (1e6 + 1.0)
    };
    let vdp_ref = rosenbrock23(van_der_pol, None::<NoJacobian>, 0.0, &[2.0, 0.0], 200.0, opts(1e-11, 1e-13)).unwrap().last()[0];
    let robertson_ref = 0.715_827_068_7;

    type Problem<'a> = (&'a str, &'a dyn Fn(f64, &[f64]) -> Vec<f64>, Vec<f64>, f64, f64, f64, f64);
    let problems: [Problem; 3] = [
        ("$y' = -1000(y - \\cos t)$ on $[0, 1]$", &scalar, vec![0.0], 1.0, scalar_ref, 1e-6, 1e-9),
        ("Van der Pol, $\\mu = 100$, on $[0, 200]$", &van_der_pol, vec![2.0, 0.0], 200.0, vdp_ref, 1e-6, 1e-9),
        ("Robertson kinetics on $[0, 40]$", &robertson, vec![1.0, 0.0, 0.0], 40.0, robertson_ref, 1e-6, 1e-10),
    ];
    let mut table = String::from("| problem | solver | steps | rejected | evaluations of $f$ | Jacobians | error in $y_1$ |\n|---|---|---:|---:|---:|---:|---:|\n");
    let row = |name: &str, label: &str, s: &OdeSolution, exact: f64| format!("| {name} | {label} | {} | {} | {} | {} | {:.1e} |\n", s.t.len() - 1, s.rejected_steps, s.function_evaluations, s.jacobian_evaluations, (s.last()[0] - exact).abs());
    for (name, f, y0, t1, exact, rtol, atol) in problems {
        let o = opts(rtol, atol);
        table.push_str(&row(name, "Dormand-Prince", &dormand_prince_solve(f, 0.0, &y0, t1, o).unwrap(), exact));
        table.push_str(&row("", "Rosenbrock 2(3)", &rosenbrock23(f, None::<NoJacobian>, 0.0, &y0, t1, o).unwrap(), exact));
        table.push_str(&row("", "BDF, orders 1–5", &bdf(f, None::<NoJacobian>, 0.0, &y0, t1, 5, o).unwrap(), exact));
    }
    write(CHAPTER, "stiff_table.md", &table);

    // Step sizes along the Van der Pol solution.
    let o = opts(1e-6, 1e-9);
    let steps = |s: &OdeSolution| -> Vec<(f64, f64)> { s.t.windows(2).map(|w| (w[1], w[1] - w[0])).collect() };
    let dp = dormand_prince_solve(van_der_pol, 0.0, &[2.0, 0.0], 200.0, o).unwrap();
    let ros = rosenbrock23(van_der_pol, None::<NoJacobian>, 0.0, &[2.0, 0.0], 200.0, o).unwrap();
    let bd = bdf(van_der_pol, None::<NoJacobian>, 0.0, &[2.0, 0.0], 200.0, 5, o).unwrap();
    let mut plot = Plot::new("Van der Pol (μ = 100): step size along the solution", "t", "step size h");
    plot.log_y = true;
    plot.legend = Legend::BottomRight;
    plot.series.push(Series::new(&format!("Dormand-Prince ({} steps)", dp.t.len() - 1), steps(&dp), Style::Line, 0));
    plot.series.push(Series::new(&format!("Rosenbrock 2(3) ({} steps)", ros.t.len() - 1), steps(&ros), Style::Line, 1));
    plot.series.push(Series::new(&format!("BDF ({} steps)", bd.t.len() - 1), steps(&bd), Style::Line, 2));
    write(CHAPTER, "stiff_step_sizes.svg", &plot.render());
    let mut plot = Plot::new("Van der Pol (μ = 100): slow drift and sudden jumps", "t", "y₁(t)");
    plot.series.push(Series::new("", ros.t.iter().zip(&ros.y).map(|(&t, y)| (t, y[0])).collect(), Style::Line, 1));
    write(CHAPTER, "van_der_pol.svg", &plot.render());
}

fn bvp_figures() {
    use lib_jsl_integration::boundary_value_problems::{BvpOptions, solve_bvp_collocation, solve_bvp_shooting};
    // Shooting against collocation on y'' = λ² y, y(0) = y(1) = 1.
    let mut table = String::from("| $\\lambda$ | growth $e^{\\lambda}$ | shooting: error at $x = \\tfrac12$ | collocation: error at $x = \\tfrac12$ | collocation nodes |\n|---:|---:|---:|---:|---:|\n");
    for lambda in [5.0f64, 15.0, 25.0, 35.0, 45.0] {
        let f = move |_: f64, y: &[f64]| vec![y[1], lambda * lambda * y[0]];
        let bc = |ya: &[f64], yb: &[f64]| vec![ya[0] - 1.0, yb[0] - 1.0];
        let exact = 1.0 / (lambda / 2.0).cosh();
        let shooting = match solve_bvp_shooting(f, bc, 0.0, 1.0, vec![1.0, -lambda], BvpOptions { tolerance: 1e-8, ..Default::default() }) {
            Ok(sol) if sol.converged => format!("{:.1e}", (sol.evaluate(0.5)[0] - exact).abs()),
            Ok(sol) => format!("not converged (residual {:.0e})", sol.max_residual),
            Err(_) => "fails".to_string(),
        };
        let x: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
        let y: Vec<Vec<f64>> = x.iter().map(|_| vec![1.0, 0.0]).collect();
        let col = solve_bvp_collocation(f, bc, x, y, BvpOptions { tolerance: 1e-8, ..Default::default() }).unwrap();
        table.push_str(&format!("| {lambda} | {:.1e} | {shooting} | {:.1e} | {} |\n", lambda.exp(), (col.evaluate(0.5)[0] - exact).abs(), col.x.len()));
    }
    write(CHAPTER, "bvp_shooting_vs_collocation.md", &table);

    // Bratu's problem has two solutions; the starting guess decides which one Newton finds.
    let bratu = |_: f64, y: &[f64]| vec![y[1], -y[0].exp()];
    let bc = |ya: &[f64], yb: &[f64]| vec![ya[0], yb[0]];
    let mut plot = Plot::new("Bratu's problem u'' + eᵘ = 0, u(0) = u(1) = 0: two solutions", "x", "u(x)");
    plot.legend = Legend::TopRight;
    for (k, amplitude) in [0.5f64, 16.0].iter().enumerate() {
        let x: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
        let y: Vec<Vec<f64>> = x.iter().map(|&t| vec![amplitude * t * (1.0 - t), amplitude * (1.0 - 2.0 * t)]).collect();
        let sol = solve_bvp_collocation(bratu, bc, x, y, BvpOptions { tolerance: 1e-4, ..Default::default() }).unwrap();
        let curve: Vec<(f64, f64)> = (0..=200).map(|i| { let x = i as f64 / 200.0; (x, sol.evaluate(x)[0]) }).collect();
        let peak = sol.evaluate(0.5)[0];
        plot.series.push(Series::new(&format!("from guess {amplitude}·x(1 − x): u(½) = {peak:.6}"), curve, Style::Line, k));
        plot.series.push(Series::new("", sol.x.iter().zip(&sol.y).map(|(&x, y)| (x, y[0])).collect(), Style::Markers, k));
    }
    write(CHAPTER, "bratu_branches.svg", &plot.render());

    // A boundary layer: ε y'' + y' = 0 with ε = 10⁻³.
    let eps = 1e-3;
    let layer = move |_: f64, y: &[f64]| vec![y[1], -y[1] / eps];
    let bc1 = |ya: &[f64], yb: &[f64]| vec![ya[0], yb[0] - 1.0];
    let x: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
    let y: Vec<Vec<f64>> = x.iter().map(|&t| vec![t, 1.0]).collect();
    let sol = solve_bvp_collocation(layer, bc1, x, y, BvpOptions { tolerance: 1e-6, ..Default::default() }).unwrap();
    let mut plot = Plot::new(&format!("Boundary layer ε y'' + y' = 0, ε = 10⁻³: {} mesh points after refinement", sol.x.len()), "x (logarithmic)", "y(x)");
    plot.log_x = true;
    plot.legend = Legend::BottomRight;
    let curve: Vec<(f64, f64)> = (0..=400).map(|i| { let x = 10f64.powf(-6.0 + 6.0 * i as f64 / 400.0); (x, sol.evaluate(x)[0]) }).collect();
    plot.series.push(Series::new("collocation solution", curve, Style::Line, 0));
    plot.series.push(Series::new("mesh points", sol.x.iter().zip(&sol.y).filter(|(x, _)| **x > 0.0).map(|(&x, y)| (x, y[0])).collect(), Style::Markers, 1));
    write(CHAPTER, "bvp_boundary_layer.svg", &plot.render());
}

fn finite_difference_figures() {
    use lib_jsl_integration::pde::finite_differences::{HeatScheme, heat_equation_1d, poisson_2d, wave_equation_1d};
    // Heat equation u_t = u_xx, u(x, 0) = sin πx, at t = 0.1: exact e^{−π² t} sin πx.
    let t_end = 0.1;
    let mut table = String::from("| grid points | $\\Delta x$ | explicit steps ($r \\le \\tfrac12$) | explicit error | Crank-Nicolson steps ($\\Delta t = \\Delta x$) | Crank-Nicolson error |\n|---:|---:|---:|---:|---:|---:|\n");
    for n in [11usize, 21, 41, 81, 161] {
        let dx = 1.0 / (n - 1) as f64;
        let u0: Vec<f64> = (0..n).map(|i| (PI * i as f64 * dx).sin()).collect();
        let exact: Vec<f64> = (0..n).map(|i| (-PI * PI * t_end).exp() * (PI * i as f64 * dx).sin()).collect();
        let err = |u: &[f64]| u.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let explicit_steps = (t_end / (0.5 * dx * dx)).ceil() as usize;
        let e = heat_equation_1d(&u0, 1.0, 1.0, t_end / explicit_steps as f64, explicit_steps, 0.0, 0.0, HeatScheme::Explicit).unwrap();
        let cn_steps = (t_end / dx).ceil() as usize;
        let c = heat_equation_1d(&u0, 1.0, 1.0, t_end / cn_steps as f64, cn_steps, 0.0, 0.0, HeatScheme::CrankNicolson).unwrap();
        table.push_str(&format!("| {n} | {dx:.4} | {explicit_steps} | {:.2e} | {cn_steps} | {:.2e} |\n", err(&e), err(&c)));
    }
    write(CHAPTER, "heat_convergence.md", &table);

    // A wave pulse after t = 0.4 with Courant numbers 1 and 0.5, against d'Alembert's solution.
    let n = 101;
    let dx = 1.0 / (n - 1) as f64;
    let pulse = |x: f64| (-3000.0 * (x - 0.3).powi(2)).exp();
    // Odd, 2-periodic extension (fixed ends reflect with a sign change).
    let extended = |x: f64| { let y = x.rem_euclid(2.0); if y <= 1.0 { pulse(y) } else { -pulse(2.0 - y) } };
    let t = 0.4;
    let u0: Vec<f64> = (0..n).map(|i| pulse(i as f64 * dx)).collect();
    let v0 = vec![0.0; n];
    let mut plot = Plot::new("A wave pulse at t = 0.4: leapfrog with Courant numbers 1 and 0.5", "x", "u(x, 0.4)");
    plot.legend = Legend::BottomRight;
    let exact: Vec<(f64, f64)> = (0..=1000).map(|i| { let x = i as f64 / 1000.0; (x, 0.5 * (extended(x - t) + extended(x + t))) }).collect();
    plot.series.push(Series::new("exact (d'Alembert)", exact, Style::Line, 6));
    for (k, courant) in [1.0f64, 0.5].iter().enumerate() {
        let dt = courant * dx;
        let steps = (t / dt).round() as usize;
        let u = wave_equation_1d(&u0, &v0, 1.0, 1.0, dt, steps).unwrap();
        plot.series.push(Series::new(&format!("Courant number {courant}"), u.iter().enumerate().map(|(i, &v)| (i as f64 * dx, v)).collect(), Style::LineMarkers, k));
    }
    write(CHAPTER, "wave_courant.svg", &plot.render());

    // Poisson on the unit square: −∇²u = 2π² sin πx sin πy, u = 0 on the boundary.
    let mut table = String::from("| grid | $h$ | maximum error | ratio |\n|---:|---:|---:|---:|\n");
    let mut previous: Option<f64> = None;
    for n in [9usize, 17, 33, 65, 129] {
        let u = poisson_2d(|x, y| 2.0 * PI * PI * (PI * x).sin() * (PI * y).sin(), |_, _| 0.0, 1.0, 1.0, n, n).unwrap();
        let h = 1.0 / (n - 1) as f64;
        let err = (0..n).flat_map(|j| (0..n).map(move |i| (i, j))).map(|(i, j)| (u[j][i] - (PI * i as f64 * h).sin() * (PI * j as f64 * h).sin()).abs()).fold(0.0, f64::max);
        let ratio = previous.map_or(String::new(), |p| format!("{:.2}", p / err));
        table.push_str(&format!("| {n} × {n} | 1/{} | {err:.2e} | {ratio} |\n", n - 1));
        previous = Some(err);
    }
    write(CHAPTER, "poisson_convergence.md", &table);
}
