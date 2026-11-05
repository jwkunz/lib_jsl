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
    scipy_reference();
    grid_figures();
    scattered_figures();
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
    type Case<'a> = (&'a str, &'a dyn Fn(f64) -> f64, usize);
    let cases: [Case; 2] = [
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

fn grid_figures() {
    use lib_jsl_core::R2D;
    use lib_jsl_interpolation::two_d::Interpolation2D;
    use lib_jsl_interpolation::two_d::{all_phase::AllPhase2D, bicubic::Bicubic, bilinear::Bilinear, cubic_basis::CubicBasis2D, fourier_2d::FourierInterpolator2D, lanczos::Lanczos2D, nearest::Nearest2D};
    // Effective kernels: interpolate a unit impulse at x = 0 (constant in y).
    let xs = R1D::from_vec((-8..=8).map(|i| i as f64).collect());
    let ys = R1D::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    let z = R2D::from_shape_fn((17, 4), |(i, _)| if i == 8 { 1.0 } else { 0.0 });
    let grid: Vec<f64> = (0..=800).map(|i| -4.0 + 8.0 * i as f64 / 800.0).collect();
    let mut plot = Plot::new("Each method's response to a single unit sample (its interpolation kernel)", "distance in grid steps", "weight");
    plot.legend = Legend::TopLeft;
    plot.y_range = Some((-0.2, 1.1));
    let methods: Vec<(&str, Box<dyn Interpolation2D>)> = vec![
        ("nearest", Box::new(Nearest2D::new(xs.view(), ys.view(), z.view()).unwrap())),
        ("bilinear", Box::new(Bilinear::new(xs.view(), ys.view(), z.view()).unwrap())),
        ("bicubic (Hermite, central-difference slopes)", Box::new(Bicubic::new(xs.view(), ys.view(), z.view()).unwrap())),
        ("Lanczos, a = 3", Box::new(Lanczos2D::new(xs.view(), ys.view(), z.view(), 3).unwrap())),
        ("all-phase, n = 4", Box::new(AllPhase2D::new(xs.view(), ys.view(), z.view(), 4).unwrap())),
    ];
    for (k, (label, m)) in methods.iter().enumerate() {
        plot.series.push(Series::new(label, grid.iter().map(|&x| (x, m.interpolate_at(x, 1.5).unwrap())).collect(), Style::Line, k));
    }
    write(CHAPTER, "grid_kernels.svg", &plot.render());

    // Accuracy on a 16 × 16 grid over [0, 1): periodic and non-periodic smooth functions.
    let n = 16;
    let axis = R1D::from_vec((0..n).map(|i| i as f64 / n as f64).collect());
    let periodic = |x: f64, y: f64| (2.0 * PI * x).sin() * (2.0 * PI * y).cos();
    let plain = |x: f64, y: f64| (x + 0.5 * y).exp();
    // Interior points stay at least four samples from the edges, where the kernel methods replicate
    // edge values; "near edges" points cover the whole span between the first and last samples.
    let region = |lo: f64, hi: f64| -> Vec<(f64, f64)> { (0..40).flat_map(|i| (0..40).map(move |j| (lo + (hi - lo) * i as f64 / 39.0, lo + (hi - lo) * j as f64 / 39.0))).collect() };
    let (interior, whole) = (region(0.25, 0.68), region(0.0, 15.0 / 16.0));
    let mut table = String::from("| method | periodic $\\sin 2\\pi x \\cos 2\\pi y$, interior | non-periodic $e^{x + y/2}$, interior | non-periodic, whole grid |\n|---|---:|---:|---:|\n");
    let rms_on = |m: &dyn Interpolation2D, f: &dyn Fn(f64, f64) -> f64, pts: &[(f64, f64)]| (pts.iter().map(|&(x, y)| (m.interpolate_at(x, y).unwrap() - f(x, y)).powi(2)).sum::<f64>() / pts.len() as f64).sqrt();
    type Build<'a> = (&'a str, Box<dyn Fn(&R2D) -> Box<dyn Interpolation2D> + 'a>);
    let builders: Vec<Build> = vec![
        ("nearest", Box::new(|z: &R2D| Box::new(Nearest2D::new(axis.view(), axis.view(), z.view()).unwrap()) as Box<dyn Interpolation2D>)),
        ("bilinear", Box::new(|z: &R2D| Box::new(Bilinear::new(axis.view(), axis.view(), z.view()).unwrap()) as Box<dyn Interpolation2D>)),
        ("bicubic", Box::new(|z: &R2D| Box::new(Bicubic::new(axis.view(), axis.view(), z.view()).unwrap()) as Box<dyn Interpolation2D>)),
        ("cubic basis (Catmull-Rom)", Box::new(|z: &R2D| Box::new(CubicBasis2D::new(axis.view(), axis.view(), z.view(), 0.0).unwrap()) as Box<dyn Interpolation2D>)),
        ("Lanczos, a = 3", Box::new(|z: &R2D| Box::new(Lanczos2D::new(axis.view(), axis.view(), z.view(), 3).unwrap()) as Box<dyn Interpolation2D>)),
        ("all-phase, n = 4", Box::new(|z: &R2D| Box::new(AllPhase2D::new(axis.view(), axis.view(), z.view(), 4).unwrap()) as Box<dyn Interpolation2D>)),
        ("Fourier", Box::new(|z: &R2D| Box::new(FourierInterpolator2D::new(z.view(), 0.0, 0.0, 1.0, 1.0).unwrap()) as Box<dyn Interpolation2D>)),
    ];
    for (label, build) in &builders {
        let zp = R2D::from_shape_fn((n, n), |(i, j)| periodic(axis[i], axis[j]));
        let zn = R2D::from_shape_fn((n, n), |(i, j)| plain(axis[i], axis[j]));
        let (mp, mn) = (build(&zp), build(&zn));
        table.push_str(&format!("| {label} | {:.1e} | {:.1e} | {:.1e} |\n", rms_on(mp.as_ref(), &periodic, &interior), rms_on(mn.as_ref(), &plain, &interior), rms_on(mn.as_ref(), &plain, &whole)));
    }
    write(CHAPTER, "grid_accuracy.md", &table);

    // 3D convergence: trilinear, tricubic Lagrange and B-spline.
    use lib_jsl_interpolation::three_d::Interpolation3D;
    use lib_jsl_interpolation::three_d::{b_spline::BSpline3D, cubic_lagrangian::CubicLagrangian3D, trilinear::Trilinear};
    let f3 = |x: f64, y: f64, z: f64| (x + 2.0 * y).sin() * (1.0 + z * z).ln();
    let mut table = String::from("| grid | trilinear | tricubic Lagrange | tricubic B-spline |\n|---:|---:|---:|---:|\n");
    for m in [5usize, 9, 17, 33] {
        let ax = R1D::from_vec((0..m).map(|i| i as f64 / (m - 1) as f64).collect());
        let v = ndarray::Array3::from_shape_fn((m, m, m), |(i, j, k)| f3(ax[i], ax[j], ax[k]));
        let methods: Vec<Box<dyn Interpolation3D>> = vec![
            Box::new(Trilinear::new(ax.view(), ax.view(), ax.view(), &v).unwrap()),
            Box::new(CubicLagrangian3D::new(ax.view(), ax.view(), ax.view(), &v).unwrap()),
            Box::new(BSpline3D::new(ax.view(), ax.view(), ax.view(), &v).unwrap()),
        ];
        let pts: Vec<[f64; 3]> = (0..10).flat_map(|i| (0..10).flat_map(move |j| (0..10).map(move |k| [0.25 + 0.5 * i as f64 / 9.0, 0.25 + 0.5 * j as f64 / 9.0, 0.25 + 0.5 * k as f64 / 9.0]))).collect();
        let errs: Vec<String> = methods.iter().map(|mth| format!("{:.1e}", pts.iter().map(|p| (mth.interpolate_at(p[0], p[1], p[2]).unwrap() - f3(p[0], p[1], p[2])).abs()).fold(0.0, f64::max))).collect();
        table.push_str(&format!("| {m}³ | {} |\n", errs.join(" | ")));
    }
    write(CHAPTER, "grid_3d.md", &table);
}

fn scattered_figures() {
    use lib_jsl_interpolation::three_d::Interpolation3D;
    use lib_jsl_interpolation::three_d::shepards::Shepards;
    use lib_jsl_interpolation::two_d::Interpolation2D;
    use lib_jsl_interpolation::two_d::radial_basis_functions::{RadialBasisInterpolator, RadialKernel};
    use lib_jsl_random::quasi_random::HaltonSequence;
    // Franke's test function on the unit square, sampled at 100 Halton points.
    let franke = |x: f64, y: f64| {
        0.75 * (-((9.0 * x - 2.0).powi(2) + (9.0 * y - 2.0).powi(2)) / 4.0).exp()
            + 0.75 * (-(9.0 * x + 1.0).powi(2) / 49.0 - (9.0 * y + 1.0) / 10.0).exp()
            + 0.5 * (-((9.0 * x - 7.0).powi(2) + (9.0 * y - 3.0).powi(2)) / 4.0).exp()
            - 0.2 * (-(9.0 * x - 4.0).powi(2) - (9.0 * y - 7.0).powi(2)).exp()
    };
    let mut h = HaltonSequence::new(2).unwrap();
    let pts: Vec<(f64, f64)> = (0..100).map(|_| { let p = h.next_point(); (p[0], p[1]) }).collect();
    let (xs, ys) = (R1D::from_vec(pts.iter().map(|p| p.0).collect()), R1D::from_vec(pts.iter().map(|p| p.1).collect()));
    let fs = R1D::from_vec(pts.iter().map(|&(x, y)| franke(x, y)).collect());
    let test: Vec<(f64, f64)> = (0..50).flat_map(|i| (0..50).map(move |j| (i as f64 / 49.0, j as f64 / 49.0))).collect();
    let mut table = String::from("| method | RMS error | maximum error |\n|---|---:|---:|\n");
    let mut record = |label: &str, eval: &dyn Fn(f64, f64) -> f64| {
        let errs: Vec<f64> = test.iter().map(|&(x, y)| (eval(x, y) - franke(x, y)).abs()).collect();
        let rms = (errs.iter().map(|e| e * e).sum::<f64>() / errs.len() as f64).sqrt();
        table.push_str(&format!("| {label} | {rms:.1e} | {:.1e} |\n", errs.iter().cloned().fold(0.0, f64::max)));
    };
    let kernels = [
        ("RBF, Gaussian (ε = 5)", RadialKernel::Gaussian(5.0)),
        ("RBF, multiquadric (ε = 3)", RadialKernel::Multiquadric(3.0)),
        ("RBF, inverse multiquadric (ε = 3)", RadialKernel::InverseMultiquadric(3.0)),
        ("RBF, cubic", RadialKernel::Cubic),
        ("RBF, thin-plate spline", RadialKernel::ThinPlateSpline),
    ];
    for (label, kernel) in kernels {
        let r = RadialBasisInterpolator::new(xs.view(), ys.view(), fs.view(), kernel).unwrap();
        record(label, &|x, y| r.interpolate_at(x, y).unwrap());
    }
    let zeros = R1D::zeros(100);
    for power in [2.0, 4.0] {
        let s = Shepards::new(xs.view(), ys.view(), zeros.view(), fs.view(), power).unwrap();
        record(&format!("Shepard, power {power}"), &|x, y| s.interpolate_at(x, y, 0.0).unwrap());
    }
    write(CHAPTER, "scattered_table.md", &table);

    // A slice at y = 0.5.
    let tps = RadialBasisInterpolator::new(xs.view(), ys.view(), fs.view(), RadialKernel::ThinPlateSpline).unwrap();
    let shep = Shepards::new(xs.view(), ys.view(), zeros.view(), fs.view(), 2.0).unwrap();
    let grid: Vec<f64> = (0..=400).map(|i| i as f64 / 400.0).collect();
    let mut plot = Plot::new("Franke's function from 100 scattered samples: the slice y = 0.5", "x", "f(x, 0.5)");
    plot.legend = Legend::TopRight;
    plot.series.push(Series::new("exact", grid.iter().map(|&x| (x, franke(x, 0.5))).collect(), Style::Dashed, 6));
    plot.series.push(Series::new("thin-plate spline RBF", grid.iter().map(|&x| (x, tps.interpolate_at(x, 0.5).unwrap())).collect(), Style::Line, 0));
    plot.series.push(Series::new("Shepard, power 2", grid.iter().map(|&x| (x, shep.interpolate_at(x, 0.5, 0.0).unwrap())).collect(), Style::Line, 1));
    write(CHAPTER, "scattered_slice.svg", &plot.render());
}

// ------------------------------------------------------------------ SciPy references

fn scipy_reference() {
    use crate::refs::{Refs, fmt_diff, max_abs_diff, max_rel_diff};
    use crate::spectral::md_table;
    use lib_jsl_interpolation::nd::InterpolationND;
    use lib_jsl_interpolation::nd::grid_data::GridData;
    use lib_jsl_interpolation::one_d::aaa::AaaApproximation;
    use lib_jsl_interpolation::one_d::akima::akima;
    use lib_jsl_interpolation::one_d::floater_hormann::FloaterHormann;
    use lib_jsl_interpolation::one_d::fourier_1d::FourierInterpolator1D;
    use lib_jsl_interpolation::one_d::linear_interpolator::LinearInterpolator;
    use lib_jsl_interpolation::one_d::natural_cubic_spline_interpolator::NaturalCubicSplineInterpolator;
    use lib_jsl_interpolation::one_d::pade_approximants::PadeApproximant;
    use lib_jsl_interpolation::one_d::pchip::pchip;
    use lib_jsl_interpolation::one_d::polynomial_interpolator::PolynomialInterpolator;
    use lib_jsl_interpolation::one_d::savitzky_golay::{savitzky_golay, savitzky_golay_coefficients};
    use lib_jsl_interpolation::three_d::Interpolation3D;
    use lib_jsl_interpolation::three_d::b_spline::BSpline3D;
    use lib_jsl_interpolation::three_d::trilinear::Trilinear;
    use lib_jsl_interpolation::two_d::Interpolation2D;
    use lib_jsl_interpolation::two_d::bilinear::Bilinear;
    use lib_jsl_interpolation::two_d::nearest::Nearest2D;
    use lib_jsl_interpolation::two_d::radial_basis_functions::{RadialBasisInterpolator, RadialKernel};
    use ndarray::{Array2, Array3, ArrayD, IxDyn};

    let refs = Refs::parse(include_str!("../data/interpolation_reference.txt"));
    let mut rows: Vec<Vec<String>> = Vec::new();
    let row = |rows: &mut Vec<Vec<String>>, label: &str, got: &[f64], key: &str| {
        let d = max_abs_diff(got, refs.values(key));
        rows.push(vec![label.to_string(), fmt_diff(d)]);
    };

    // One-dimensional interpolants on uneven nodes.
    let x: Vec<f64> = (0..12).map(|i| i as f64 + 0.3 * (i as f64).sin()).collect();
    let smooth: Vec<f64> = x.iter().map(|&t| t.sin() + 0.05 * t * t).collect();
    let steps = vec![0.0, 0.0, 0.5, 1.0, 1.0, 1.2, 3.0, 3.1, 3.1, 4.0, 6.0, 6.2];
    let xq: Vec<f64> = (0..61).map(|k| x[0] + (x[11] - x[0]) * k as f64 / 60.0).collect();
    let xa = R1D::from_vec(x.clone());
    for (name, y) in [("smooth", smooth), ("steps", steps)] {
        let ya = R1D::from_vec(y);
        let mut spline = NaturalCubicSplineInterpolator::new(xa.view(), ya.view()).unwrap();
        let mut p = pchip(xa.view(), ya.view()).unwrap();
        let mut a = akima(xa.view(), ya.view()).unwrap();
        let mut l = LinearInterpolator::new(xa.view(), ya.view()).unwrap();
        let data = if name == "smooth" { "a smooth curve" } else { "data with flat stretches" };
        row(&mut rows, &format!("natural cubic spline, {data} (`CubicSpline`, natural)"), &xq.iter().map(|&t| spline.interpolate_at(t).unwrap()).collect::<Vec<_>>(), &format!("natural spline {name}"));
        row(&mut rows, &format!("PCHIP, {data} (`PchipInterpolator`)"), &xq.iter().map(|&t| p.interpolate_at(t).unwrap()).collect::<Vec<_>>(), &format!("pchip {name}"));
        row(&mut rows, &format!("Akima, {data} (`Akima1DInterpolator`)"), &xq.iter().map(|&t| a.interpolate_at(t).unwrap()).collect::<Vec<_>>(), &format!("akima {name}"));
        row(&mut rows, &format!("linear, {data} (`numpy.interp`)"), &xq.iter().map(|&t| l.interpolate_at(t).unwrap()).collect::<Vec<_>>(), &format!("linear {name}"));
    }

    // Polynomial and rational interpolation of Runge's function.
    let tq: Vec<f64> = (0..101).map(|k| -1.0 + k as f64 / 50.0).collect();
    let cheb: Vec<f64> = (0..9).map(|j| (PI * j as f64 / 8.0).cos()).collect();
    let (cx, cy) = (R1D::from_vec(cheb.clone()), R1D::from_vec(cheb.iter().map(|&t| runge(t)).collect()));
    let mut poly = PolynomialInterpolator::new(cx.view(), cy.view()).unwrap();
    row(&mut rows, "polynomial through 9 Chebyshev points (`BarycentricInterpolator`)", &tq.iter().map(|&t| poly.interpolate_at(t).unwrap()).collect::<Vec<_>>(), "polynomial chebyshev 9");
    let eq = equispaced(21);
    let (ex, ey) = (R1D::from_vec(eq.clone()), R1D::from_vec(eq.iter().map(|&t| runge(t)).collect()));
    let mut fh = FloaterHormann::new(ex.view(), ey.view(), 3).unwrap();
    row(&mut rows, "Floater-Hormann, 21 equispaced points, d = 3 (`FloaterHormannInterpolator`)", &tq.iter().map(|&t| fh.interpolate_at(t).unwrap()).collect::<Vec<_>>(), "floater-hormann 21 d3");
    let z: Vec<f64> = (0..1000).map(|k| -1.0 + 2.0 * k as f64 / 999.0).collect();
    let tan = |t: f64| (PI * t / 2.2).tan();
    let aaa = AaaApproximation::new(R1D::from_vec(z.clone()).view(), R1D::from_vec(z.iter().map(|&t| tan(t)).collect()).view(), 1e-13, 100).unwrap();
    let got: Vec<f64> = tq.iter().map(|&t| aaa.evaluate(t)).collect();
    rows.push(vec!["AAA rational approximation of tan(pi x / 2.2) (`AAA`)".into(), format!("{} (relative); each within {} of the function", fmt_diff(max_rel_diff(&got, refs.values("aaa tan"))), fmt_diff(max_rel_diff(&got, &tq.iter().map(|&t| tan(t)).collect::<Vec<_>>())))]);
    let taylor: Vec<f64> = (0..9).map(|k| 1.0 / (1..=k).map(|j| j as f64).product::<f64>()).collect();
    let pade = PadeApproximant::new(&taylor, 4, 4).unwrap();
    let want = refs.get("pade exp 4 4");
    rows.push(vec!["[4/4] Pade approximant of exp, coefficients (`pade`)".into(), format!("numerator {}, denominator {}", fmt_diff(max_abs_diff(pade.numerator(), &want[0])), fmt_diff(max_abs_diff(pade.denominator(), &want[1])))]);
    let c = ChebyshevApproximation::new(f64::exp, -1.0, 1.0, 12).unwrap();
    let mut coeffs = c.coefficients().to_vec();
    coeffs[0] /= 2.0; // the library's series starts with c0/2
    row(&mut rows, "Chebyshev coefficients of exp, 12 terms (`numpy.polynomial.chebyshev.chebinterpolate`)", &coeffs, "chebyshev exp 12");

    // Savitzky-Golay smoothing and differentiation.
    for (deriv, delta) in [(0, 1.0), (1, 0.1)] {
        let mut c = savitzky_golay_coefficients(11, 3, deriv, delta).unwrap();
        let want = refs.values(&format!("savgol coeffs 11 3 {deriv}"));
        if max_abs_diff(&c, want) > 1e-9 {
            c.reverse(); // SciPy lists convolution coefficients in reverse order
        }
        row(&mut rows, &format!("Savitzky-Golay coefficients, 11 points, cubic, derivative {deriv} (`savgol_coeffs`)"), &c, &format!("savgol coeffs 11 3 {deriv}"));
        let noisy = R1D::from_vec((0..100).map(|k| (k as f64 / 8.0).sin() + 0.1 * (k as f64 * 2.3).sin()).collect());
        let filtered = savitzky_golay(noisy.view(), 11, 3, deriv, delta).unwrap();
        row(&mut rows, &format!("Savitzky-Golay filter, derivative {deriv}, with end fitting (`savgol_filter`, interp)"), filtered.as_slice().unwrap(), &format!("savgol filter 11 3 {deriv}"));
    }

    // Fourier interpolation of periodic samples onto a four times finer grid.
    for n in [15usize, 16] {
        let y = R1D::from_vec((0..n).map(|j| { let t = 2.0 * PI * j as f64 / n as f64; t.sin().exp() + 0.3 * (3.0 * t).cos() }).collect());
        let f = FourierInterpolator1D::new(y.view(), 0.0, 2.0 * PI).unwrap();
        let got: Vec<f64> = (0..4 * n).map(|j| f.evaluate(2.0 * PI * j as f64 / (4 * n) as f64)).collect();
        row(&mut rows, &format!("trigonometric interpolation, {n} samples (`signal.resample`)"), &got, &format!("fourier {n}"));
    }

    // Gridded data in two and three dimensions.
    let (gx, gy): (Vec<f64>, Vec<f64>) = (vec![0.0, 0.5, 1.3, 2.0, 3.1], vec![0.0, 1.0, 1.5, 2.5]);
    let g2 = Array2::from_shape_fn((5, 4), |(i, j)| gx[i].sin() * gy[j].cos() + gx[i] * gy[j] / 4.0);
    let q2: Vec<[f64; 2]> = (0..40).map(|j| [0.05 + 3.0 * ((7 * j) % 41) as f64 / 41.0, 0.05 + 2.4 * ((11 * j) % 37) as f64 / 37.0]).collect();
    let (gxa, gya) = (R1D::from_vec(gx.clone()), R1D::from_vec(gy.clone()));
    let bil = Bilinear::new(gxa.view(), gya.view(), g2.view()).unwrap();
    row(&mut rows, "bilinear on an uneven grid (`RegularGridInterpolator`, linear)", &q2.iter().map(|p| bil.interpolate_at(p[0], p[1]).unwrap()).collect::<Vec<_>>(), "grid2 linear");
    let near = Nearest2D::new(gxa.view(), gya.view(), g2.view()).unwrap();
    row(&mut rows, "nearest neighbour on an uneven grid (`RegularGridInterpolator`, nearest)", &q2.iter().map(|p| near.interpolate_at(p[0], p[1]).unwrap()).collect::<Vec<_>>(), "grid2 nearest");
    let ax: Vec<f64> = (0..6).map(|k| 0.4 * k as f64).collect();
    let g3 = Array3::from_shape_fn((6, 6, 6), |(i, j, k)| ax[i].sin() + (1.3 * ax[j]).cos() * ax[k] * ax[k] / 4.0);
    let q3: Vec<[f64; 3]> = (0..40).map(|j| [2.0 * ((5 * j) % 29) as f64 / 29.0, 2.0 * ((13 * j) % 31) as f64 / 31.0, 2.0 * ((17 * j) % 23) as f64 / 23.0]).collect();
    let axa = R1D::from_vec(ax.clone());
    let tri = Trilinear::new(axa.view(), axa.view(), axa.view(), &g3).unwrap();
    row(&mut rows, "trilinear (`RegularGridInterpolator`, linear)", &q3.iter().map(|p| tri.interpolate_at(p[0], p[1], p[2]).unwrap()).collect::<Vec<_>>(), "grid3 linear");
    let nd = GridData::new(&[&ax, &ax, &ax], &ArrayD::from_shape_vec(IxDyn(&[6, 6, 6]), g3.iter().copied().collect()).unwrap()).unwrap();
    row(&mut rows, "n-dimensional multilinear, in 3D (`RegularGridInterpolator`, linear)", &q3.iter().map(|p| nd.interpolate_at(p).unwrap()).collect::<Vec<_>>(), "grid3 linear");
    let bs = BSpline3D::new(axa.view(), axa.view(), axa.view(), &g3).unwrap();
    row(&mut rows, "tricubic natural B-spline (`CubicSpline`, natural, along each axis)", &q3.iter().map(|p| bs.interpolate_at(p[0], p[1], p[2]).unwrap()).collect::<Vec<_>>(), "grid3 natural spline");

    // Scattered data and radial basis functions.
    let sp: Vec<[f64; 2]> = (0..30).map(|j| [((37 * j) % 53) as f64 / 53.0, ((61 * j) % 47) as f64 / 47.0]).collect();
    let sf: Vec<f64> = sp.iter().map(|p| (3.0 * p[0]).sin() * (2.0 * p[1]).cos() + p[0] * p[0]).collect();
    let sq: Vec<[f64; 2]> = (0..40).map(|j| [((29 * j) % 43) as f64 / 43.0, ((19 * j) % 41) as f64 / 41.0]).collect();
    let (sx, sy, sv) = (R1D::from_vec(sp.iter().map(|p| p[0]).collect()), R1D::from_vec(sp.iter().map(|p| p[1]).collect()), R1D::from_vec(sf));
    for (name, kernel) in [("thin_plate_spline", RadialKernel::ThinPlateSpline), ("cubic", RadialKernel::Cubic), ("gaussian", RadialKernel::Gaussian(1.5)), ("multiquadric", RadialKernel::Multiquadric(1.5)), ("inverse_multiquadric", RadialKernel::InverseMultiquadric(1.5))] {
        let rbf = RadialBasisInterpolator::new(sx.view(), sy.view(), sv.view(), kernel).unwrap();
        row(&mut rows, &format!("radial basis functions, {} kernel, 30 scattered points (`RBFInterpolator`)", name.replace('_', " ")), &sq.iter().map(|p| rbf.interpolate_at(p[0], p[1]).unwrap()).collect::<Vec<_>>(), &format!("rbf {name}"));
    }
    write(CHAPTER, "scipy_reference.md", &md_table(&["interpolant (SciPy reference)", "largest difference"], &rows));
}
