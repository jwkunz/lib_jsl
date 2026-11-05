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
    scipy_reference();
    step_size_figure();
}

// ------------------------------------------------------------------ SciPy references

/// A test function, the point, and its exact first and second derivatives there.
type TestFunction = (&'static str, fn(f64) -> f64, f64, f64, f64);

fn scipy_reference() {
    use crate::refs::{Refs, fmt_diff, max_rel_diff};
    use crate::spectral::md_table;
    use lib_jsl_derivatives::derivatives_1d_scalar::{DerivativeStencilOrder, derivative_2_point, derivative_stencil};
    use lib_jsl_derivatives::derivatives_vectors::{gradient_2_point, hessian_2_point, jacobian_2_point};

    let refs = Refs::parse(include_str!("../data/derivatives_reference.txt"));
    let functions: [TestFunction; 5] = [
        ("exp(x) at 1", f64::exp, 1.0, 1f64.exp(), 1f64.exp()),
        ("sin(x) at 1", f64::sin, 1.0, 1f64.cos(), -1f64.sin()),
        ("ln(x) at 0.1", f64::ln, 0.1, 10.0, -100.0),
        ("1/(1 + 25 x^2) at 0.3", |x| 1.0 / (1.0 + 25.0 * x * x), 0.3, -15.0 / (3.25f64 * 3.25), 50.0 * (75.0 * 0.09 - 1.0) / 3.25f64.powi(3)),
        ("x^(5/2) at 2", |x| x.powf(2.5), 2.0, 2.5 * 2f64.powf(1.5), 3.75 * 2f64.sqrt()),
    ];
    let rel = |v: f64, exact: f64| if v.is_nan() { "NaN".to_string() } else { fmt_diff(((v - exact) / exact).abs()) };
    let mut rows = Vec::new();
    for (name, f, x, d1, d2) in functions {
        let scipy1 = refs.values(&format!("d1 {name}"));
        let scipy2 = refs.values(&format!("d2 {name}"))[0];
        rows.push(vec![
            name.to_string(),
            rel(derivative_2_point(&f, x, None).unwrap(), d1),
            rel(derivative_stencil(&f, x, None, DerivativeStencilOrder::FirstOrder).unwrap()[0], d1),
            format!("{} ({} with a first step of 0.01)", rel(scipy1[0], d1), rel(scipy1[1], d1)),
            rel(derivative_stencil(&f, x, None, DerivativeStencilOrder::SecondOrder).unwrap()[0], d2),
            rel(scipy2, d2),
        ]);
    }
    write(CHAPTER, "scipy_scalar.md", &md_table(&["function", "f': central difference", "f': five-point stencil", "f': SciPy `derivative`", "f'': five-point stencil", "f'': SciPy, nested"], &rows));

    let rosenbrock = |v: &[f64]| (1.0 - v[0]).powi(2) + 100.0 * (v[1] - v[0] * v[0]).powi(2);
    let x = [-1.2, 1.0];
    let (g_exact, h_exact) = (vec![-215.6, -88.0], vec![1330.0, 480.0, 480.0, 200.0]);
    let polar = |v: &[f64]| vec![v[0] * v[1].cos(), v[0] * v[1].sin()];
    let j_exact = vec![0.5f64.cos(), -2.0 * 0.5f64.sin(), 0.5f64.sin(), 2.0 * 0.5f64.cos()];
    let flat = |m: Vec<Vec<f64>>| m.into_iter().flatten().collect::<Vec<f64>>();
    let scipy_flat = |key: &str| refs.get(key).iter().flatten().copied().collect::<Vec<f64>>();
    let rows = vec![
        vec!["gradient of Rosenbrock's function at (-1.2, 1) (`differentiate.jacobian`)".into(), fmt_diff(max_rel_diff(&gradient_2_point(&rosenbrock, &x, None).unwrap(), &g_exact)), fmt_diff(max_rel_diff(refs.values("gradient rosenbrock"), &g_exact))],
        vec!["its Hessian (`differentiate.hessian`)".into(), fmt_diff(max_rel_diff(&flat(hessian_2_point(&rosenbrock, &x, None).unwrap()), &h_exact)), fmt_diff(max_rel_diff(&scipy_flat("hessian rosenbrock"), &h_exact))],
        vec!["Jacobian of polar coordinates at (2, 0.5) (`differentiate.jacobian`)".into(), fmt_diff(max_rel_diff(&flat(jacobian_2_point(&polar, &[2.0, 0.5], None).unwrap()), &j_exact)), fmt_diff(max_rel_diff(&scipy_flat("jacobian polar"), &j_exact))],
    ];
    write(CHAPTER, "scipy_vectors.md", &md_table(&["derivative (SciPy reference)", "library: relative error", "SciPy: relative error"], &rows));
}
