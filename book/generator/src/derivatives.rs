//! Figures and tables for the numerical differentiation chapter.

use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_derivatives::derivatives_1d_scalar::{DerivativeStencilOrder, derivative_2_point, derivative_stencil};

const CHAPTER: &str = "derivatives";

fn step_size_figure() {
    // Errors in derivatives of e^x at x = 1 (all equal to e) as the step varies.
    let e = std::f64::consts::E;
    let steps: Vec<f64> = (0..=60).map(|i| 10f64.powf(-14.0 + 13.0 * i as f64 / 60.0)).collect();
    let mut plot = Plot::new("Derivatives of eˣ at x = 1: truncation against rounding", "step h", "absolute error");
    plot.log_x = true;
    plot.log_y = true;
    plot.y_range = Some((1e-14, 1e4));
    plot.legend = Legend::TopRight;
    let central: Vec<(f64, f64)> = steps.iter().map(|&h| (h, (derivative_2_point(&f64::exp, 1.0, Some(h)).unwrap() - e).abs().max(1e-17))).collect();
    plot.series.push(Series::new("f′, central difference (h²)", central, Style::Line, 0));
    let five: Vec<(f64, f64)> = steps.iter().map(|&h| (h, (derivative_stencil(&f64::exp, 1.0, Some(h), DerivativeStencilOrder::FirstOrder).unwrap()[0] - e).abs().max(1e-17))).collect();
    plot.series.push(Series::new("f′, five-point (h⁴)", five, Style::Line, 1));
    let second: Vec<(f64, f64)> = steps.iter().map(|&h| (h, (derivative_stencil(&f64::exp, 1.0, Some(h), DerivativeStencilOrder::SecondOrder).unwrap()[0] - e).abs().max(1e-17))).collect();
    plot.series.push(Series::new("f″, five-point (h⁴)", second, Style::Line, 2));
    let fourth: Vec<(f64, f64)> = steps.iter().map(|&h| (h, (derivative_stencil(&f64::exp, 1.0, Some(h), DerivativeStencilOrder::FourthOrder).unwrap()[0] - e).abs().max(1e-17))).collect();
    plot.series.push(Series::new("f⁗, five-point (h²)", fourth, Style::Line, 3));
    write(CHAPTER, "step_size.svg", &plot.render());

    // Default steps and the accuracy they achieve.
    let mut table = String::from("| formula | truncation error | rounding error | default relative step | error for $e^x$ at $x = 1$ |\n|---|---:|---:|---:|---:|\n");
    let rows: [(&str, &str, &str, i32, i32, f64); 5] = [
        ("central difference, $f'$", "$h^2$", "$\\varepsilon/h$", 2, 1, derivative_2_point(&f64::exp, 1.0, None).unwrap()),
        ("five-point, $f'$", "$h^4$", "$\\varepsilon/h$", 4, 1, derivative_stencil(&f64::exp, 1.0, None, DerivativeStencilOrder::FirstOrder).unwrap()[0]),
        ("five-point, $f''$", "$h^4$", "$\\varepsilon/h^2$", 4, 2, derivative_stencil(&f64::exp, 1.0, None, DerivativeStencilOrder::SecondOrder).unwrap()[0]),
        ("five-point, $f'''$", "$h^2$", "$\\varepsilon/h^3$", 2, 3, derivative_stencil(&f64::exp, 1.0, None, DerivativeStencilOrder::ThirdOrder).unwrap()[0]),
        ("five-point, $f''''$", "$h^2$", "$\\varepsilon/h^4$", 2, 4, derivative_stencil(&f64::exp, 1.0, None, DerivativeStencilOrder::FourthOrder).unwrap()[0]),
    ];
    for (name, trunc, round, p, m, value) in rows {
        let h = f64::EPSILON.powf(1.0 / (p + m) as f64);
        table.push_str(&format!("| {name} | {trunc} | {round} | $\\varepsilon^{{1/{}}} \\approx {:.1e}$ | {:.1e} |\n", p + m, h, (value - e).abs()));
    }
    write(CHAPTER, "default_steps.md", &table);
}

pub fn generate() {
    step_size_figure();
}
