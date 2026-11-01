//! Figures and tables for the interpolation chapter.

use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_core::R1D;
use lib_jsl_interpolation::interpolation_trait::InterpolationTrait;
use lib_jsl_interpolation::one_d::chebyshev_approximations::ChebyshevApproximation;
use lib_jsl_interpolation::one_d::lagrange_interpolant::LagrangeInterpolant;
use std::f64::consts::PI;

const CHAPTER: &str = "interpolation";

fn runge(x: f64) -> f64 {
    1.0 / (1.0 + 25.0 * x * x)
}

fn equispaced(n: usize) -> Vec<f64> {
    (0..n).map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64).collect()
}

fn chebyshev_nodes(n: usize) -> Vec<f64> {
    (0..n).map(|j| (PI * j as f64 / (n - 1) as f64).cos()).collect()
}

/// The maximum error of the polynomial interpolant of `runge` through `nodes`, on a fine grid.
fn interpolation_error(nodes: &[f64]) -> f64 {
    let x = R1D::from_vec(nodes.to_vec());
    let y = R1D::from_vec(nodes.iter().map(|&v| runge(v)).collect());
    let mut p = LagrangeInterpolant::new(x.view(), y.view()).unwrap();
    (0..=2000).map(|i| { let t = -1.0 + i as f64 / 1000.0; (p.interpolate_at(t).unwrap() - runge(t)).abs() }).fold(0.0, f64::max)
}

fn polynomial_figures() {
    let n = 13;
    let mut plot = Plot::new("Interpolating 1/(1 + 25x²) at 13 points", "x", "y");
    plot.y_range = Some((-1.0, 2.0));
    plot.legend = Legend::TopRight;
    plot.series.push(Series::new("f(x)", (0..=400).map(|i| { let x = -1.0 + i as f64 / 200.0; (x, runge(x)) }).collect(), Style::Line, 6));
    for (k, (label, nodes)) in [("equally spaced points", equispaced(n)), ("Chebyshev points", chebyshev_nodes(n))].into_iter().enumerate() {
        let x = R1D::from_vec(nodes.clone());
        let y = R1D::from_vec(nodes.iter().map(|&v| runge(v)).collect());
        let mut p = LagrangeInterpolant::new(x.view(), y.view()).unwrap();
        let curve: Vec<(f64, f64)> = (0..=400).map(|i| { let t = -1.0 + i as f64 / 200.0; (t, p.interpolate_at(t).unwrap()) }).collect();
        plot.series.push(Series::new(&format!("interpolant, {label}"), curve, Style::Line, k));
        plot.series.push(Series::new("", nodes.iter().map(|&v| (v, runge(v))).collect(), Style::Markers, k));
    }
    write(CHAPTER, "runge.svg", &plot.render());

    let mut plot = Plot::new("Maximum error for 1/(1 + 25x²) as the degree grows", "number of points n", "maximum error on [−1, 1]");
    plot.log_y = true;
    plot.y_range = Some((1e-16, 1e6));
    plot.legend = Legend::BottomLeft;
    let sizes: Vec<usize> = (3..=80).step_by(3).collect();
    plot.series.push(Series::new("equally spaced points", sizes.iter().filter(|&&n| n <= 60).map(|&n| (n as f64, interpolation_error(&equispaced(n)))).collect(), Style::LineMarkers, 0));
    plot.series.push(Series::new("Chebyshev points", sizes.iter().map(|&n| (n as f64, interpolation_error(&chebyshev_nodes(n)).max(1e-17))).collect(), Style::LineMarkers, 1));
    let series: Vec<(f64, f64)> = sizes.iter().map(|&n| {
        let c = ChebyshevApproximation::new(runge, -1.0, 1.0, n).unwrap();
        (n as f64, (0..=2000).map(|i| { let t = -1.0 + i as f64 / 1000.0; (c.evaluate(t) - runge(t)).abs() }).fold(0.0, f64::max).max(1e-17))
    }).collect();
    plot.series.push(Series::new("Chebyshev series (ChebyshevApproximation)", series, Style::LineMarkers, 2));
    let rho = 1.0 / 5.0 + (1.0f64 + 1.0 / 25.0).sqrt();
    plot.series.push(Series::new("ρ⁻ⁿ, ρ = 1/5 + √(1 + 1/25)", sizes.iter().map(|&n| (n as f64, rho.powi(-(n as i32)))).collect(), Style::Dashed, 6));
    write(CHAPTER, "runge_convergence.svg", &plot.render());
}

pub fn generate() {
    trigonometric_figures();
    savitzky_golay_figures();
    rational_figures();
    spline_figures();
    polynomial_figures();
}

fn spline_figures() {
    use lib_jsl_interpolation::one_d::akima::akima;
    use lib_jsl_interpolation::one_d::cubic_basis::cubic_basis;
    use lib_jsl_interpolation::one_d::linear_interpolator::LinearInterpolator;
    use lib_jsl_interpolation::one_d::natural_cubic_spline_interpolator::NaturalCubicSplineInterpolator;
    use lib_jsl_interpolation::one_d::pchip::pchip;
    // Akima's (1970) test data: flat, then a steep rise.
    let xs = R1D::from_vec(vec![0.0, 2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0, 12.0, 14.0, 15.0]);
    let ys = R1D::from_vec(vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.5, 15.0, 50.0, 60.0, 85.0]);
    let grid: Vec<f64> = (0..=600).map(|i| 15.0 * i as f64 / 600.0).collect();
    let mut plot = Plot::new("Akima's data: a global spline overshoots, local methods do not", "x", "y");
    plot.legend = Legend::TopLeft;
    plot.y_range = Some((0.0, 90.0));
    let mut spline = NaturalCubicSplineInterpolator::new(xs.view(), ys.view()).unwrap();
    plot.series.push(Series::new("natural cubic spline", grid.iter().map(|&x| (x, spline.interpolate_at(x).unwrap())).collect(), Style::Line, 0));
    let mut p = pchip(xs.view(), ys.view()).unwrap();
    plot.series.push(Series::new("PCHIP", grid.iter().map(|&x| (x, p.interpolate_at(x).unwrap())).collect(), Style::Line, 1));
    let mut a = akima(xs.view(), ys.view()).unwrap();
    plot.series.push(Series::new("Akima", grid.iter().map(|&x| (x, a.interpolate_at(x).unwrap())).collect(), Style::Line, 2));
    plot.series.push(Series::new("data", xs.iter().zip(ys.iter()).map(|(&x, &y)| (x, y)).collect(), Style::Markers, 6));
    write(CHAPTER, "akima_data.svg", &plot.render());

    // Convergence on e^x over [0, 1], where f'' does not vanish at the ends.
    let mut table = String::from("| points | linear | natural spline | natural spline, middle half only | PCHIP | Akima | Catmull-Rom |\n|---:|---:|---:|---:|---:|---:|---:|\n");
    for n in [6usize, 11, 21, 41, 81] {
        let x = R1D::from_vec((0..n).map(|i| i as f64 / (n - 1) as f64).collect());
        let y = R1D::from_vec(x.iter().map(|v| v.exp()).collect());
        let fine: Vec<f64> = (0..=4000).map(|i| i as f64 / 4000.0).collect();
        let err = |f: &mut dyn FnMut(f64) -> f64, lo: f64, hi: f64| fine.iter().filter(|&&t| t >= lo && t <= hi).map(|&t| (f(t) - t.exp()).abs()).fold(0.0, f64::max);
        let mut lin = LinearInterpolator::new(x.view(), y.view()).unwrap();
        let mut spl = NaturalCubicSplineInterpolator::new(x.view(), y.view()).unwrap();
        let mut pc = pchip(x.view(), y.view()).unwrap();
        let mut ak = akima(x.view(), y.view()).unwrap();
        let mut cr = cubic_basis(x.view(), y.view(), 0.0).unwrap();
        let e_lin = err(&mut |t| lin.interpolate_at(t).unwrap(), 0.0, 1.0);
        let e_spl = err(&mut |t| spl.interpolate_at(t).unwrap(), 0.0, 1.0);
        let e_mid = err(&mut |t| spl.interpolate_at(t).unwrap(), 0.25, 0.75);
        let e_pc = err(&mut |t| pc.interpolate_at(t).unwrap(), 0.0, 1.0);
        let e_ak = err(&mut |t| ak.interpolate_at(t).unwrap(), 0.0, 1.0);
        let e_cr = err(&mut |t| cr.interpolate_at(t).unwrap(), 0.0, 1.0);
        table.push_str(&format!("| {n} | {e_lin:.1e} | {e_spl:.1e} | {e_mid:.1e} | {e_pc:.1e} | {e_ak:.1e} | {e_cr:.1e} |\n"));
    }
    write(CHAPTER, "spline_convergence.md", &table);
}

fn rational_figures() {
    use lib_jsl_interpolation::one_d::aaa::AaaApproximation;
    use lib_jsl_interpolation::one_d::bulirsch_stoer::BulirschStoer;
    use lib_jsl_interpolation::one_d::floater_hormann::FloaterHormann;
    use lib_jsl_interpolation::one_d::pade_approximants::PadeApproximant;
    // The Runge function at equally spaced points: polynomial against rational interpolants.
    let fine: Vec<f64> = (0..=2000).map(|i| -1.0 + i as f64 / 1000.0).collect();
    let max_err = |f: &mut dyn FnMut(f64) -> f64| fine.iter().map(|&t| (f(t) - runge(t)).abs()).fold(0.0, f64::max).max(1e-17);
    let sizes: Vec<usize> = (5..=101).step_by(8).collect();
    let mut poly = Vec::new();
    let mut fh = Vec::new();
    let mut bs = Vec::new();
    let mut aaa = Vec::new();
    for &n in &sizes {
        let nodes = equispaced(n);
        let (x, y) = (R1D::from_vec(nodes.clone()), R1D::from_vec(nodes.iter().map(|&v| runge(v)).collect()));
        if n <= 45 {
            let mut p = LagrangeInterpolant::new(x.view(), y.view()).unwrap();
            poly.push((n as f64, max_err(&mut |t| p.interpolate_at(t).unwrap())));
        }
        let mut f = FloaterHormann::new(x.view(), y.view(), 3).unwrap();
        fh.push((n as f64, max_err(&mut |t| f.interpolate_at(t).unwrap())));
        let mut b = BulirschStoer::new(x.view(), y.view()).unwrap();
        bs.push((n as f64, max_err(&mut |t| b.interpolate_at(t).unwrap_or(f64::NAN)).min(1e6)));
        let a = AaaApproximation::new(x.view(), y.view(), 1e-13, 100).unwrap();
        aaa.push((n as f64, max_err(&mut |t| a.evaluate(t))));
    }
    let mut plot = Plot::new("1/(1 + 25x²) from equally spaced samples: rational interpolants converge", "number of points n", "maximum error on [−1, 1]");
    plot.log_y = true;
    plot.y_range = Some((1e-16, 1e6));
    plot.legend = Legend::TopRight;
    plot.series.push(Series::new("polynomial (diverges)", poly, Style::LineMarkers, 0));
    plot.series.push(Series::new("Floater-Hormann, d = 3", fh, Style::LineMarkers, 1));
    plot.series.push(Series::new("Bulirsch-Stoer", bs, Style::LineMarkers, 2));
    plot.series.push(Series::new("AAA", aaa, Style::LineMarkers, 3));
    write(CHAPTER, "rational_runge.svg", &plot.render());

    // Padé against Taylor for ln(1 + x), whose series converges only for |x| < 1.
    let taylor: Vec<f64> = (0..=20).map(|k| if k == 0 { 0.0 } else { (if k % 2 == 1 { 1.0 } else { -1.0 }) / k as f64 }).collect();
    let pade = PadeApproximant::new(&taylor, 10, 10).unwrap();
    let mut table = String::from("| $x$ | $\\ln(1 + x)$ | Taylor polynomial, degree 20: error | Padé $[10/10]$: error |\n|---:|---:|---:|---:|\n");
    for x in [0.5f64, 0.9, 1.0, 2.0, 5.0, 20.0] {
        let t: f64 = taylor.iter().enumerate().map(|(k, c)| c * x.powi(k as i32)).sum();
        let exact = x.ln_1p();
        table.push_str(&format!("| {x} | {exact:.10} | {:.1e} | {:.1e} |\n", (t - exact).abs(), (pade.evaluate(x) - exact).abs()));
    }
    write(CHAPTER, "pade_log.md", &table);
}

fn trigonometric_figures() {
    use lib_jsl_interpolation::one_d::fourier_1d::FourierInterpolator1D;
    use lib_jsl_interpolation::one_d::pronys_method::PronyModel;
    use lib_jsl_random::uniform_generator::{UniformGenerator, UniformRNG};
    // Fourier interpolation of a periodic function and of a non-periodic one, 16 samples each.
    let n = 16;
    let grid: Vec<f64> = (0..=600).map(|i| i as f64 / 600.0).collect();
    let mut plot = Plot::new("Trigonometric interpolation from 16 samples on [0, 1)", "x", "y");
    plot.legend = Legend::BottomLeft;
    let cases: [(&str, &dyn Fn(f64) -> f64, usize); 2] = [
        ("periodic: exp(sin 2πx)", &|x: f64| (2.0 * PI * x).sin().exp(), 0),
        ("not periodic: x", &|x: f64| x, 1),
    ];
    for (label, f, k) in cases {
        let samples = R1D::from_vec((0..n).map(|i| f(i as f64 / n as f64)).collect());
        let interp = FourierInterpolator1D::new(samples.view(), 0.0, 1.0).unwrap();
        plot.series.push(Series::new(&format!("{label}, interpolant"), grid.iter().map(|&x| (x, interp.evaluate(x))).collect(), Style::Line, k));
        plot.series.push(Series::new("", grid.iter().map(|&x| (x, f(x))).collect(), Style::Dashed, k));
        plot.series.push(Series::new("", (0..n).map(|i| { let x = i as f64 / n as f64; (x, f(x)) }).collect(), Style::Markers, k));
    }
    write(CHAPTER, "fourier_interpolation.svg", &plot.render());

    // Prony's method: two damped oscillations plus noise.
    let truth = |t: f64| 1.0 * (-0.3 * t).exp() * (2.0 * PI * 1.2 * t).cos() + 0.5 * (-0.1 * t).exp() * (2.0 * PI * 0.45 * t + 0.7).cos();
    let dt = 0.1;
    let mut rng = UniformRNG::from_seed(4);
    let data: Vec<f64> = (0..80).map(|k| truth(k as f64 * dt) + 0.02 * (rng.next_f64() - 0.5)).collect();
    let exact_order = PronyModel::new(R1D::from_vec(data.clone()).view(), 0.0, dt, 4).unwrap();
    let over_order = PronyModel::new(R1D::from_vec(data.clone()).view(), 0.0, dt, 8).unwrap();
    let mut plot = Plot::new("Prony's method on 8 s of noisy data, extrapolated to 16 s", "t", "y");
    plot.legend = Legend::TopRight;
    plot.series.push(Series::new("data", data.iter().enumerate().map(|(k, &y)| (k as f64 * dt, y)).collect(), Style::Markers, 6));
    plot.series.push(Series::new("true signal", (0..=1600).map(|i| { let t = i as f64 * 0.01; (t, truth(t)) }).collect(), Style::Dashed, 6));
    plot.series.push(Series::new("4 exponentials (the true number)", (0..=1600).map(|i| { let t = i as f64 * 0.01; (t, exact_order.evaluate(t)) }).collect(), Style::Line, 0));
    plot.series.push(Series::new("8 exponentials", (0..=1600).map(|i| { let t = i as f64 * 0.01; (t, over_order.evaluate(t)) }).collect(), Style::Line, 1));
    write(CHAPTER, "prony.svg", &plot.render());
    let mut table = String::from("| model | oscillatory components found: damping, frequency (Hz) |\n|---|---|\n");
    table.push_str("| true signal | 0.100, 0.450 Hz; 0.300, 1.200 Hz |\n");
    for (label, model) in [("4 exponentials", &exact_order), ("8 exponentials", &over_order)] {
        let mut comps: Vec<(f64, f64)> = model.exponents().into_iter().filter(|s| s.im > 1e-9).map(|s| (-s.re, s.im / (2.0 * PI))).collect();
        comps.sort_by(|a, b| a.0.total_cmp(&b.0));
        let text: Vec<String> = comps.iter().map(|(d, f)| format!("{d:.3}, {f:.3} Hz")).collect();
        table.push_str(&format!("| {label} | {} |\n", text.join("; ")));
    }
    write(CHAPTER, "prony_table.md", &table);
}

fn savitzky_golay_figures() {
    use lib_jsl_interpolation::one_d::savitzky_golay::savitzky_golay;
    use lib_jsl_random::uniform_generator::{UniformGenerator, UniformRNG};
    // A narrow peak on a slope, with noise.
    let n = 400;
    let dx = 1.0 / n as f64;
    let signal = |x: f64| 0.5 * x + (-((x - 0.5) / 0.02).powi(2)).exp();
    let mut rng = UniformRNG::from_seed(9);
    let noisy: Vec<f64> = (0..n).map(|i| signal(i as f64 * dx) + 0.1 * (2.0 * rng.next_f64() - 1.0)).collect();
    let y = R1D::from_vec(noisy.clone());
    let average = savitzky_golay(y.view(), 21, 0, 0, dx).unwrap();
    let sg = savitzky_golay(y.view(), 21, 4, 0, dx).unwrap();
    let mut plot = Plot::new("Smoothing a noisy peak with a 21-point window", "x", "y");
    plot.x_range = Some((0.3, 0.7));
    plot.legend = Legend::TopRight;
    let pts = |v: &[f64]| -> Vec<(f64, f64)> { v.iter().enumerate().map(|(i, &y)| (i as f64 * dx, y)).filter(|p| p.0 >= 0.3 && p.0 <= 0.7).collect() };
    plot.series.push(Series::new("noisy data", pts(&noisy), Style::Markers, 6));
    plot.series.push(Series::new("moving average (order 0)", pts(average.as_slice().unwrap()), Style::Line, 0));
    plot.series.push(Series::new("Savitzky-Golay, order 4", pts(sg.as_slice().unwrap()), Style::Line, 1));
    plot.series.push(Series::new("true signal", (0..=400).map(|i| { let x = 0.3 + 0.4 * i as f64 / 400.0; (x, signal(x)) }).collect(), Style::Dashed, 2));
    write(CHAPTER, "savitzky_golay_peak.svg", &plot.render());

    // Derivatives of noisy data: central differences against Savitzky-Golay.
    let dsignal = |x: f64| 0.5 - 2.0 * (x - 0.5) / (0.02f64 * 0.02) * (-((x - 0.5) / 0.02).powi(2)).exp();
    let rms = |v: &[f64]| (v.iter().enumerate().skip(30).take(n - 60).map(|(i, d)| (d - dsignal(i as f64 * dx)).powi(2)).sum::<f64>() / (n - 60) as f64).sqrt();
    let central: Vec<f64> = (0..n).map(|i| if i == 0 || i == n - 1 { 0.0 } else { (noisy[i + 1] - noisy[i - 1]) / (2.0 * dx) }).collect();
    let mut table = String::from("| derivative estimate | RMS error |\n|---|---:|\n");
    table.push_str(&format!("| central differences | {:.1} |\n", rms(&central)));
    for (window, order) in [(11, 2), (21, 2), (21, 4), (41, 4)] {
        let d = savitzky_golay(y.view(), window, order, 1, dx).unwrap();
        table.push_str(&format!("| Savitzky-Golay, window {window}, order {order} | {:.1} |\n", rms(d.as_slice().unwrap())));
    }
    write(CHAPTER, "savitzky_golay_derivative.md", &table);
}
