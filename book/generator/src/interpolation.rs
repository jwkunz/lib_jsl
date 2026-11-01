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
    polynomial_figures();
}
