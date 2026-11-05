//! Figures and tables for the root finding chapter.

use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_root_finding::{
    SystemOptions, bisection, brent_method, broyden_system, false_position, newton_raphson, newton_system, polynomial_roots,
    polynomial_roots_complex, ridders_method, secant_method,
};
use std::cell::RefCell;

const CHAPTER: &str = "root_finding";

type NoJacobian = fn(&[f64]) -> Vec<Vec<f64>>;

/// Runs a solver on a recording copy of `f`; returns the result and every point evaluated.
fn recorded<F: Fn(f64) -> f64>(f: F, solve: impl FnOnce(&dyn Fn(f64) -> f64) -> Result<f64, String>) -> (Result<f64, String>, Vec<f64>) {
    let log = RefCell::new(Vec::new());
    let g = |x: f64| {
        log.borrow_mut().push(x);
        f(x)
    };
    let r = solve(&g);
    (r, log.into_inner())
}

type Solver = fn(&dyn Fn(f64) -> f64, &dyn Fn(f64) -> f64, f64, f64) -> Result<f64, String>;

/// The methods compared, each given `f`, `f'` and a bracket `[a, b]`; the open methods start
/// from the bracket's ends (secant) or its midpoint (Newton).
fn methods() -> Vec<(&'static str, Solver)> {
    vec![
        ("bisection", |f, _, a, b| bisection(f, a, b, 1e-13)),
        ("false position (Illinois)", |f, _, a, b| false_position(f, a, b, 1e-13)),
        ("Ridders", |f, _, a, b| ridders_method(f, a, b, 1e-13)),
        ("Brent", |f, _, a, b| brent_method(f, a, b, 1e-13)),
        ("secant", |f, _, a, b| secant_method(f, a, b, 1e-13)),
        ("Newton", |f, df, a, b| newton_raphson(f, df, 0.5 * (a + b), 1e-13)),
    ]
}

fn bracketing_figures() {
    // Wallis's and Newton's example x³ − 2x − 5 = 0, on [2, 3].
    let f = |x: f64| x * x * x - 2.0 * x - 5.0;
    let df = |x: f64| 3.0 * x * x - 2.0;
    let root = brent_method(f, 2.0, 3.0, 0.0).unwrap();
    let mut plot = Plot::new("Solving x³ − 2x − 5 = 0 on [2, 3]", "evaluations of f", "error of the best point so far");
    plot.log_y = true;
    plot.y_range = Some((1e-16, 1.0));
    plot.x_range = Some((0.0, 45.0));
    plot.legend = Legend::TopRight;
    for (k, (name, solve)) in methods().into_iter().enumerate() {
        let (_, xs) = recorded(f, |g| solve(g, &df, 2.0, 3.0));
        // The closest point evaluated so far: what the method knows about the root after i calls.
        let mut best = f64::INFINITY;
        let pts = xs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                best = best.min((x - root).abs().max(1e-16));
                ((i + 1) as f64, best)
            })
            .collect();
        plot.series.push(Series::new(name, pts, Style::LineMarkers, k));
    }
    write(CHAPTER, "convergence.svg", &plot.render());

    // Evaluations needed on harder problems.
    type Problem = (&'static str, fn(f64) -> f64, fn(f64) -> f64, f64, f64, f64);
    let problems: [Problem; 5] = [
        ("$x^3 - 2x - 5$ on $[2, 3]$", |x| x * x * x - 2.0 * x - 5.0, |x| 3.0 * x * x - 2.0, 2.0, 3.0, 2.0945514815423265),
        ("$\\cos x - x$ on $[0, 1]$", |x| x.cos() - x, |x| -x.sin() - 1.0, 0.0, 1.0, 0.7390851332151607),
        ("$x^{10} - 1$ on $[0, 1.3]$", |x| x.powi(10) - 1.0, |x| 10.0 * x.powi(9), 0.0, 1.3, 1.0),
        ("$(x - 1/3)^3$ on $[0, 1]$", |x| (x - 1.0 / 3.0).powi(3), |x| 3.0 * (x - 1.0 / 3.0).powi(2), 0.0, 1.0, 1.0 / 3.0),
        ("$\\tanh(20(x - 0.3))$ on $[-1, 1]$", |x| (20.0 * (x - 0.3)).tanh(), |x| 20.0 / (20.0 * (x - 0.3)).cosh().powi(2), -1.0, 1.0, 0.3),
    ];
    let mut table = String::from("| problem |");
    for (name, _) in methods() {
        table.push_str(&format!(" {name} |"));
    }
    table.push_str("\n|---|");
    table.push_str(&"---:|".repeat(methods().len()));
    table.push('\n');
    for (label, f, df, a, b, root) in problems {
        table.push_str(&format!("| {label} |"));
        for (_, solve) in methods() {
            // Newton's evaluations of f′ count too.
            let calls = RefCell::new(0usize);
            let dfc = |x: f64| {
                *calls.borrow_mut() += 1;
                df(x)
            };
            let (r, xs) = recorded(f, |g| solve(g, &dfc, a, b));
            let n = xs.len() + calls.into_inner();
            let cell = match r {
                Ok(x) if (x - root).abs() < 1e-6 => format!("{n}"),
                Ok(x) => format!("wrong root ({x:.3})"),
                Err(_) => "fails".to_string(),
            };
            table.push_str(&format!(" {cell} |"));
        }
        table.push('\n');
    }
    write(CHAPTER, "evaluations.md", &table);
}

fn newton_figures() {
    let s2 = 2f64.sqrt();
    let mut plot = Plot::new("Orders of convergence", "iteration k", "|xₖ − root|");
    plot.log_y = true;
    plot.y_range = Some((1e-16, 1.0));
    plot.legend = Legend::TopRight;
    // Errors of the iterates, up to the first that is exact to rounding.
    let err = |xs: Vec<f64>, root: f64| -> Vec<(f64, f64)> {
        let mut pts: Vec<(f64, f64)> = xs.iter().enumerate().map(|(i, x)| (i as f64, (x - root).abs().max(1e-16))).collect();
        if let Some(k) = pts.iter().position(|p| p.1 <= 1e-16) {
            pts.truncate(k + 1);
        }
        pts
    };
    let (_, xs) = recorded(|x| x * x - 2.0, |g| newton_raphson(g, |x| 2.0 * x, 1.0, 1e-15));
    plot.series.push(Series::new("Newton, x² − 2 (quadratic)", err(xs, s2), Style::LineMarkers, 0));
    let (_, xs) = recorded(|x| x * x - 2.0, |g| secant_method(g, 1.0, 2.0, 1e-15));
    plot.series.push(Series::new("secant, x² − 2 (order 1.618)", err(xs, s2), Style::LineMarkers, 1));
    let (_, xs) = recorded(|x| (x - 1.0).powi(2) * (x + 2.0), |g| newton_raphson(g, |x| 3.0 * (x - 1.0) * (x + 1.0), 2.0, 1e-15));
    plot.series.push(Series::new("Newton, double root (linear, ratio ½)", err(xs, 1.0), Style::LineMarkers, 2));
    plot.x_range = Some((0.0, 30.0));
    write(CHAPTER, "orders.svg", &plot.render());

    // Newton on arctan: converges from |x₀| < 1.3917, diverges beyond.
    let mut plot = Plot::new("Newton's method on arctan x = 0", "iteration k", "xₖ");
    plot.legend = Legend::TopLeft;
    plot.y_range = Some((-6.0, 6.0));
    for (k, x0) in [1.35, 1.39, 1.395, 1.42].into_iter().enumerate() {
        let (_, xs) = recorded(f64::atan, |g| newton_raphson(g, |x| 1.0 / (1.0 + x * x), x0, 1e-15));
        let pts = xs.iter().enumerate().take(12).map(|(i, &x)| (i as f64, x.clamp(-7.0, 7.0))).collect();
        plot.series.push(Series::new(&format!("x₀ = {x0}"), pts, Style::LineMarkers, k));
    }
    write(CHAPTER, "newton_atan.svg", &plot.render());
}

fn poly_from_roots(r: &[f64]) -> Vec<f64> {
    let mut c = vec![1.0];
    for &x in r {
        let mut n = vec![0.0; c.len() + 1];
        for (i, &a) in c.iter().enumerate() {
            n[i] += a;
            n[i + 1] -= a * x;
        }
        c = n;
    }
    c
}

fn polynomial_figures() {
    let roots: Vec<f64> = (1..=20).map(|k| k as f64).collect();
    let w = poly_from_roots(&roots);
    let mut perturbed = w.clone();
    perturbed[1] -= 2f64.powi(-23);
    let mut plot = Plot::new("Wilkinson's polynomial ∏(x − k), k = 1…20", "Re z", "Im z");
    plot.legend = Legend::TopLeft;
    plot.y_range = Some((-3.5, 3.5));
    plot.x_range = Some((0.0, 22.0));
    let to_pts = |c: &[f64]| polynomial_roots_complex(c).unwrap().iter().map(|z| (z.re, z.im)).collect::<Vec<_>>();
    plot.series.push(Series::new("roots as computed", to_pts(&w), Style::Markers, 0));
    plot.series.push(Series::new("x¹⁹ coefficient changed by 2⁻²³", to_pts(&perturbed), Style::Markers, 1));
    write(CHAPTER, "wilkinson.svg", &plot.render());

    let mut table = String::from("| polynomial | real roots found | largest error |\n|---|---:|---:|\n");
    for n in [5usize, 10, 15, 20] {
        let r: Vec<f64> = (1..=n).map(|k| k as f64).collect();
        let found = polynomial_roots(&poly_from_roots(&r));
        let e = found.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        table.push_str(&format!("| $\\prod_{{k=1}}^{{{n}}}(x - k)$ | {} | {e:.1e} |\n", found.len()));
    }
    let found = polynomial_roots(&perturbed);
    table.push_str(&format!("| the same, $n = 20$, with $a_{{19}}$ changed by $2^{{-23}}$ | {} | n/a |\n", found.len()));
    let triple = polynomial_roots(&poly_from_roots(&[1.0, 1.0, 1.0]));
    table.push_str(&format!("| $(x - 1)^3$ | {} | {:.1e} |\n", triple.len(), (triple[0] - 1.0).abs()));
    write(CHAPTER, "polynomial_accuracy.md", &table);
}

fn rosenbrock(x: &[f64]) -> Vec<f64> {
    vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]]
}

fn power_sums(x: &[f64]) -> Vec<f64> {
    vec![x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - 14.0, x[0] + x[1] + x[2] - 6.0, x[0] * x[1] * x[2] - 6.0]
}

/// The discretised Bratu problem u″ + eᵘ = 0 on (0, 1), u(0) = u(1) = 0.
fn bratu(u: &[f64]) -> Vec<f64> {
    let n = u.len();
    let h = 1.0 / (n + 1) as f64;
    (0..n)
        .map(|i| {
            let left = if i == 0 { 0.0 } else { u[i - 1] };
            let right = if i + 1 == n { 0.0 } else { u[i + 1] };
            left - 2.0 * u[i] + right + h * h * u[i].exp()
        })
        .collect()
}

fn systems_figures() {
    let o = SystemOptions::default();
    let mut table = String::from("| system | Newton, exact Jacobian | Newton, difference Jacobian | Broyden |\n|---|---:|---:|---:|\n");
    let cell = |r: lib_jsl_root_finding::SystemRootResult| {
        if r.converged { format!("{} it., {} ev.", r.iterations, r.function_evaluations) } else { "fails".to_string() }
    };
    let jac_r = |x: &[f64]| vec![vec![-20.0 * x[0], 10.0], vec![-1.0, 0.0]];
    table.push_str(&format!(
        "| Rosenbrock, $n = 2$, from $(-1.2, 1)$ | {} | {} | {} |\n",
        cell(newton_system(rosenbrock, Some(jac_r), vec![-1.2, 1.0], o).unwrap()),
        cell(newton_system(rosenbrock, None::<NoJacobian>, vec![-1.2, 1.0], o).unwrap()),
        cell(broyden_system(rosenbrock, vec![-1.2, 1.0], o).unwrap())
    ));
    let jac_p = |x: &[f64]| vec![vec![2.0 * x[0], 2.0 * x[1], 2.0 * x[2]], vec![1.0, 1.0, 1.0], vec![x[1] * x[2], x[0] * x[2], x[0] * x[1]]];
    let x0 = vec![0.7, 2.4, 2.6];
    table.push_str(&format!(
        "| power sums, $n = 3$ | {} | {} | {} |\n",
        cell(newton_system(power_sums, Some(jac_p), x0.clone(), o).unwrap()),
        cell(newton_system(power_sums, None::<NoJacobian>, x0.clone(), o).unwrap()),
        cell(broyden_system(power_sums, x0, o).unwrap())
    ));
    let n = 40;
    let h2 = 1.0 / ((n + 1) as f64).powi(2);
    let jac_b = move |u: &[f64]| {
        let mut j = vec![vec![0.0; n]; n];
        for i in 0..n {
            j[i][i] = -2.0 + h2 * u[i].exp();
            if i > 0 {
                j[i][i - 1] = 1.0;
            }
            if i + 1 < n {
                j[i][i + 1] = 1.0;
            }
        }
        j
    };
    table.push_str(&format!(
        "| Bratu, $n = 40$, from $u = 0$ | {} | {} | {} |\n",
        cell(newton_system(bratu, Some(jac_b), vec![0.0; n], o).unwrap()),
        cell(newton_system(bratu, None::<NoJacobian>, vec![0.0; n], o).unwrap()),
        cell(broyden_system(bratu, vec![0.0; n], o).unwrap())
    ));
    write(CHAPTER, "systems.md", &table);

    // The two solutions of the Bratu problem, from two starting guesses.
    let grid: Vec<f64> = (1..=n).map(|i| i as f64 / (n + 1) as f64).collect();
    let lower = newton_system(bratu, Some(jac_b), vec![0.0; n], o).unwrap();
    let upper_start: Vec<f64> = grid.iter().map(|x| 16.0 * x * (1.0 - x)).collect();
    let upper = newton_system(bratu, Some(jac_b), upper_start.clone(), o).unwrap();
    assert!(lower.converged && upper.converged);
    let with_ends = |u: &[f64]| {
        let mut p = vec![(0.0, 0.0)];
        p.extend(grid.iter().copied().zip(u.iter().copied()));
        p.push((1.0, 0.0));
        p
    };
    let mut plot = Plot::new("The two solutions of u″ + eᵘ = 0, u(0) = u(1) = 0", "x", "u");
    plot.legend = Legend::TopRight;
    plot.y_range = Some((0.0, 5.0));
    plot.series.push(Series::new("from u = 0", with_ends(&lower.x), Style::LineMarkers, 0));
    plot.series.push(Series::new("from u = 16x(1 − x)", with_ends(&upper.x), Style::LineMarkers, 1));
    plot.series.push(Series::new("second starting guess", with_ends(&upper_start), Style::Dashed, 1));
    write(CHAPTER, "bratu.svg", &plot.render());
}

pub fn generate() {
    scipy_reference();
    bracketing_figures();
    newton_figures();
    polynomial_figures();
    systems_figures();
}

// ------------------------------------------------------------------ SciPy references

/// A scalar test equation: its function, derivative, bracket and Newton start.
type Equation = (&'static str, fn(f64) -> f64, fn(f64) -> f64, (f64, f64), f64);
/// A test polynomial (highest power first) and, when known, its exact roots.
type TestPolynomial = (&'static str, Vec<f64>, Option<Vec<f64>>);
/// A Jacobian function for the system solvers' optional argument.
type SystemJacobian = fn(&[f64]) -> Vec<Vec<f64>>;
/// A test system and its starting point.
type TestSystem<'a> = (&'static str, &'a dyn Fn(&[f64]) -> Vec<f64>, Vec<f64>);

fn scipy_reference() {
    use std::f64::consts::PI;
    use crate::refs::{Refs, fmt_diff, max_abs_diff};
    use crate::spectral::md_table;
    use lib_jsl_root_finding::bisection::bisection;
    use lib_jsl_root_finding::false_position_method::false_position;
    use lib_jsl_root_finding::newton_raphson::newton_raphson;
    use lib_jsl_root_finding::nonlinear_systems::{SystemOptions, broyden_system, newton_system};
    use lib_jsl_root_finding::polynomial_roots::polynomial_roots_complex;
    use lib_jsl_root_finding::ridders_method::ridders_method;
    use lib_jsl_root_finding::secant_method::secant_method;
    use lib_jsl_root_finding::van_wijngaarden_dekker_brent::brent_method;

    let refs = Refs::parse(include_str!("../data/root_finding_reference.txt"));
    let equations: [Equation; 5] = [
        ("cos(x) = x", |x| x.cos() - x, |x| -x.sin() - 1.0, (0.0, 1.0), 0.5),
        ("x^3 - 2x - 5 = 0", |x| x * x * x - 2.0 * x - 5.0, |x| 3.0 * x * x - 2.0, (2.0, 3.0), 2.0),
        ("Kepler: E - 0.9 sin E = 0.3", |x| x - 0.9 * x.sin() - 0.3, |x| 1.0 - 0.9 * x.cos(), (0.0, PI), 1.0),
        ("x e^-x = 0.1", |x| x * (-x).exp() - 0.1, |x| (1.0 - x) * (-x).exp(), (0.0, 0.5), 0.0),
        ("ln x + x^2 = 3", |x| x.ln() + x * x - 3.0, |x| 1.0 / x + 2.0 * x, (1.0, 2.0), 1.5),
    ];
    let mut rows = Vec::new();
    for (name, f, df, (a, b), x0) in equations {
        let want = refs.values(&format!("scalar {name}"));
        let root = want[0];
        let rel = |v: f64| fmt_diff(((v - root) / root).abs());
        let tol = 1e-15;
        rows.push(vec![
            name.to_string(),
            format!("{root:.16}"),
            rel(brent_method(f, a, b, tol).unwrap()),
            rel(bisection(f, a, b, tol).unwrap()),
            rel(ridders_method(f, a, b, tol).unwrap()),
            rel(false_position(f, a, b, tol).unwrap()),
            rel(newton_raphson(f, df, x0, tol).unwrap()),
            rel(secant_method(f, x0, x0 + 0.1, tol).unwrap()),
            fmt_diff(want.iter().map(|&v| ((v - root) / root).abs()).fold(0.0, f64::max)),
        ]);
    }
    write(CHAPTER, "scipy_scalar.md", &md_table(&["equation", "root (`brentq`)", "Brent", "bisection", "Ridders", "false position", "Newton", "secant", "spread of SciPy's six methods"], &rows));

    let polynomials: [TestPolynomial; 4] = [
        ("Wilkinson-like, roots 1 to 10", { let mut c = vec![1.0]; for r in 1..=10 { let mut next = vec![0.0; c.len() + 1]; for (i, &v) in c.iter().enumerate() { next[i] += v; next[i + 1] -= v * r as f64; } c = next; } c }, Some((1..=10).map(|r| r as f64).collect())),
        ("x^5 - x - 1", vec![1.0, 0.0, 0.0, 0.0, -1.0, -1.0], None),
        ("(x - 1)^2 (x + 2) (x^2 + 1)", vec![1.0, 0.0, -2.0, 2.0, -3.0, 2.0], None),
        ("degree 12, coefficients 1 to 13", (1..=13).map(|k| k as f64).collect(), None),
    ];
    let mut rows = Vec::new();
    for (name, coeffs, exact) in polynomials {
        let want = refs.get(&format!("poly {name}"));
        let got = polynomial_roots_complex(&coeffs).unwrap();
        let re: Vec<f64> = got.iter().map(|z| z.re).collect();
        // Pair each root with the nearest unused NumPy root: conjugate pairs share a real part
        // only to rounding, so sorting alone can swap their members.
        let mut unused: Vec<usize> = (0..want[0].len()).collect();
        let mut d = 0.0f64;
        for z in &got {
            let (k, dist) = unused.iter().enumerate().map(|(k, &j)| (k, (z.re - want[0][j]).hypot(z.im - want[1][j]))).min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
            d = d.max(dist);
            unused.remove(k);
        }
        let against_exact = exact
            .map(|e| {
                let polished = lib_jsl_root_finding::polynomial_roots::polynomial_roots(&coeffs);
                format!("; from the exact roots, library {} and NumPy {}, and the library's Newton-polished real roots {}", fmt_diff(max_abs_diff(&re, &e)), fmt_diff(max_abs_diff(&want[0], &e)), fmt_diff(max_abs_diff(&polished, &e)))
            })
            .unwrap_or_default();
        rows.push(vec![format!("{name} (`numpy.roots`)"), format!("{}{against_exact}", fmt_diff(d))]);
    }
    let options = SystemOptions { function_tolerance: 1e-14, step_tolerance: 1e-16, max_iterations: 200 };
    let no_jacobian: Option<SystemJacobian> = None;
    let circle = |v: &[f64]| vec![v[0] * v[0] + v[1] * v[1] - 4.0, v[0].exp() + v[1] - 1.0];
    let trig = |v: &[f64]| vec![3.0 * v[0] - (v[1] * v[2]).cos() - 0.5, v[0] * v[0] - 81.0 * (v[1] + 0.1).powi(2) + v[2].sin() + 1.06, (-v[0] * v[1]).exp() + 20.0 * v[2] + (10.0 * PI - 3.0) / 3.0];
    let tridiagonal = |v: &[f64]| { let x: Vec<f64> = std::iter::once(0.0).chain(v.iter().copied()).chain(std::iter::once(0.0)).collect(); (1..6).map(|i| (3.0 - 2.0 * x[i]) * x[i] - x[i - 1] - 2.0 * x[i + 1] + 1.0).collect::<Vec<f64>>() };
    let systems: [TestSystem; 3] = [("circle and exponential", &circle, vec![1.0, -1.0]), ("three trigonometric equations", &trig, vec![0.1, 0.1, -0.1]), ("Broyden tridiagonal, 5 variables", &tridiagonal, vec![-1.0; 5])];
    for (name, f, x0) in systems {
        let want = refs.values(&format!("system {name}"));
        let n = newton_system(f, no_jacobian, x0.clone(), options).unwrap();
        let b = broyden_system(f, x0, options).unwrap();
        rows.push(vec![format!("system: {name} (`optimize.fsolve`)"), format!("Newton {}, Broyden {}", fmt_diff(max_abs_diff(&n.x, want)), fmt_diff(max_abs_diff(&b.x, want)))]);
    }
    write(CHAPTER, "scipy_polynomials_systems.md", &md_table(&["problem (SciPy reference)", "largest difference in the roots"], &rows));
}
