//! Figures and tables for the optimisation chapter.

use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_optimization::conjugate_gradient::{ConjugateGradientBeta, NonlinearConjugateGradient};
use lib_jsl_optimization::golden_section_search::GoldenSectionSearch;
use lib_jsl_optimization::gradient_descent::GradientDescent;
use lib_jsl_optimization::line_search::strong_wolfe;
use lib_jsl_optimization::nelder_mead_method::NelderMeadMethod;
use lib_jsl_optimization::newton_raphson_method::NewtonRaphsonMethod;
use lib_jsl_optimization::optimization_traits::{
    GradientBasedMinimizationEngine, GradientFreeMinimizationEngine, GradientFunction, HessianBasedMinimizationEngine, HessianFunction,
    MinimizationControls, ObjectiveFunction, OptimizationResult,
};
use lib_jsl_optimization::powells_method::PowellsMethod;
use lib_jsl_optimization::quasi_newton::{Bfgs, Lbfgs};
use std::cell::Cell;

pub(crate) const CHAPTER: &str = "optimization";

/// A problem given by closures, counting evaluations of the objective and of the gradient.
pub(crate) struct Problem<'a> {
    pub f: &'a dyn Fn(&[f64]) -> f64,
    pub g: &'a dyn Fn(&[f64]) -> Vec<f64>,
    pub h: &'a dyn Fn(&[f64]) -> Vec<Vec<f64>>,
    pub f_calls: Cell<usize>,
    pub g_calls: Cell<usize>,
}

impl<'a> Problem<'a> {
    pub fn new(f: &'a dyn Fn(&[f64]) -> f64, g: &'a dyn Fn(&[f64]) -> Vec<f64>, h: &'a dyn Fn(&[f64]) -> Vec<Vec<f64>>) -> Self {
        Problem { f, g, h, f_calls: Cell::new(0), g_calls: Cell::new(0) }
    }
    pub fn reset(&self) {
        self.f_calls.set(0);
        self.g_calls.set(0);
    }
}

impl ObjectiveFunction for Problem<'_> {
    fn evaluate(&self, p: &[f64]) -> f64 {
        self.f_calls.set(self.f_calls.get() + 1);
        (self.f)(p)
    }
}
impl GradientFunction for Problem<'_> {
    fn evaluate(&self, p: &[f64]) -> Vec<f64> {
        self.g_calls.set(self.g_calls.get() + 1);
        (self.g)(p)
    }
}
impl HessianFunction for Problem<'_> {
    fn evaluate(&self, p: &[f64]) -> Vec<Vec<f64>> {
        (self.h)(p)
    }
}

pub(crate) fn rosenbrock(p: &[f64]) -> f64 {
    p.windows(2).map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2)).sum()
}

pub(crate) fn rosenbrock_gradient(p: &[f64]) -> Vec<f64> {
    let mut g = vec![0.0; p.len()];
    for i in 0..p.len() - 1 {
        let t = p[i + 1] - p[i] * p[i];
        g[i] += -400.0 * p[i] * t - 2.0 * (1.0 - p[i]);
        g[i + 1] += 200.0 * t;
    }
    g
}

pub(crate) fn rosenbrock_hessian(p: &[f64]) -> Vec<Vec<f64>> {
    let n = p.len();
    let mut h = vec![vec![0.0; n]; n];
    for i in 0..n - 1 {
        h[i][i] += 1200.0 * p[i] * p[i] - 400.0 * p[i + 1] + 2.0;
        h[i][i + 1] -= 400.0 * p[i];
        h[i + 1][i] -= 400.0 * p[i];
        h[i + 1][i + 1] += 200.0;
    }
    h
}

/// The classic starting point (−1.2, 1, −1.2, 1, …).
pub(crate) fn rosenbrock_start(n: usize) -> Vec<f64> {
    (0..n).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect()
}

/// The iterates of a deterministic method, recovered by re-running it with `max_iterations = k`.
fn path(k_max: usize, run: impl Fn(usize) -> OptimizationResult) -> Vec<OptimizationResult> {
    (0..=k_max).map(run).collect()
}

/// Level sets `f = c` of the quadratic ½(x² + κy²), as closed curves.
fn ellipses(kappa: f64, levels: &[f64]) -> Vec<Series> {
    levels
        .iter()
        .map(|&c| {
            let pts = (0..=200)
                .map(|i| {
                    let t = i as f64 / 200.0 * std::f64::consts::TAU;
                    ((2.0 * c).sqrt() * t.cos(), (2.0 * c / kappa).sqrt() * t.sin())
                })
                .collect();
            Series::new("", pts, Style::Line, 7)
        })
        .collect()
}

/// The valley floor y = x² of the Rosenbrock function and a few of its level sets, which are
/// y = x² ± √((c − (1 − x)²)/100).
fn rosenbrock_levels() -> Vec<Series> {
    let mut out = vec![];
    for c in [0.5, 4.0, 25.0] {
        let mut upper = vec![];
        let mut lower = vec![];
        for i in 0..=400 {
            let x = -2.0 + 4.0 * i as f64 / 400.0;
            let r = c - (1.0 - x) * (1.0 - x);
            if r >= 0.0 {
                let d = (r / 100.0).sqrt();
                upper.push((x, x * x + d));
                lower.push((x, x * x - d));
            }
        }
        lower.reverse();
        upper.extend(lower);
        upper.push(upper[0]);
        out.push(Series::new("", upper, Style::Line, 7));
    }
    out
}

fn line_search_figures() {
    // φ(α) = f(x₀ + α d) along steepest descent from (−1.2, 1) on Rosenbrock.
    let p = Problem::new(&rosenbrock, &rosenbrock_gradient, &rosenbrock_hessian);
    let x0 = [-1.2, 1.0];
    let f0 = rosenbrock(&x0);
    let g0 = rosenbrock_gradient(&x0);
    let norm = g0.iter().map(|v| v * v).sum::<f64>().sqrt();
    let d: Vec<f64> = g0.iter().map(|v| -v / norm).collect();
    let slope = g0.iter().zip(&d).map(|(a, b)| a * b).sum::<f64>();
    let phi = |a: f64| rosenbrock(&[x0[0] + a * d[0], x0[1] + a * d[1]]);
    let alphas: Vec<f64> = (0..=400).map(|i| 0.5 * i as f64 / 400.0).collect();
    let mut plot = Plot::new("A line search along steepest descent from (−1.2, 1)", "step α", "φ(α) = f(x₀ + αd)");
    plot.legend = Legend::TopRight;
    plot.y_range = Some((0.0, 30.0));
    plot.series.push(Series::new("φ(α)", alphas.iter().map(|&a| (a, phi(a))).collect(), Style::Line, 0));
    plot.series.push(Series::new("sufficient decrease, c₁ = 0.1", alphas.iter().map(|&a| (a, f0 + 0.1 * a * slope)).collect(), Style::Dashed, 1));
    plot.series.push(Series::new("tangent at α = 0", alphas.iter().map(|&a| (a, f0 + a * slope)).collect(), Style::Dashed, 7));
    // Strong Wolfe points accepted (c₂ = 0.9 and 0.1) from an initial step of 0.02, doubled
    // until the conditions hold or a bracket is found.
    for (k, c2) in [(2, 0.9), (3, 0.1)] {
        let pt = strong_wolfe(&p, &p, &x0, f0, &g0, &d, 0.02, 1e-4, c2).unwrap();
        plot.series.push(Series::new(&format!("accepted, c₂ = {c2}"), vec![(pt.alpha, pt.value)], Style::Markers, k));
    }
    write(CHAPTER, "line_search.svg", &plot.render());

    // Golden section: interval width against evaluations.
    let mut table = String::from("| function on $[0, 1]$ | tolerance | evaluations | error in $x$ |\n|---|---:|---:|---:|\n");
    type Case = (&'static str, fn(f64) -> f64, f64);
    let cases: [Case; 3] = [
        ("$(x - 0.3)^2$", |x| (x - 0.3).powi(2), 0.3),
        ("$|x - 0.3|$", |x| (x - 0.3).abs(), 0.3),
        ("$-x e^{-5x}$", |x| -x * (-5.0 * x).exp(), 0.2),
    ];
    for (name, f, xmin) in cases {
        for tol in [1e-4, 1e-8, 1e-12] {
            let count = Cell::new(0usize);
            struct One<'a>(&'a dyn Fn(f64) -> f64, &'a Cell<usize>);
            impl ObjectiveFunction for One<'_> {
                fn evaluate(&self, p: &[f64]) -> f64 {
                    self.1.set(self.1.get() + 1);
                    (self.0)(p[0])
                }
            }
            let mut gs = GoldenSectionSearch::new();
            gs.set_bounds(Some(vec![(0.0, 1.0)]));
            gs.set_tolerance(Some(tol));
            let r = gs.gradient_free_minimize(&One(&f, &count), vec![0.5]).unwrap();
            table.push_str(&format!("| {name} | $10^{{{}}}$ | {} | {:.1e} |\n", tol.log10().round(), count.get(), (r.parameters[0] - xmin).abs()));
        }
    }
    write(CHAPTER, "golden_section.md", &table);
}

fn gradient_figures() {
    // Paths on the ill-conditioned quadratic ½(x² + 10y²) from (9, 2.5).
    let kappa = 10.0;
    let f = move |p: &[f64]| 0.5 * (p[0] * p[0] + kappa * p[1] * p[1]);
    let g = move |p: &[f64]| vec![p[0], kappa * p[1]];
    let h = move |_: &[f64]| vec![vec![1.0, 0.0], vec![0.0, kappa]];
    let prob = Problem::new(&f, &g, &h);
    let x0 = vec![9.0, 2.5];
    let mut plot = Plot::new("Minimising ½(x² + 10y²) from (9, 2.5)", "x", "y");
    plot.legend = Legend::TopLeft;
    plot.x_range = Some((-10.0, 10.0));
    plot.y_range = Some((-4.0, 4.0));
    plot.series.extend(ellipses(kappa, &[0.5, 5.0, 20.0, 50.0, 71.75]));
    let to_xy = |rs: Vec<OptimizationResult>| rs.into_iter().map(|r| (r.parameters[0], r.parameters[1])).collect::<Vec<_>>();
    let gd = path(40, |k| {
        let mut m = GradientDescent::new(0.18);
        m.set_max_iterations(Some(k));
        m.set_tolerance(Some(0.0));
        m.gradient_based_minimize(&prob, &prob, x0.clone()).unwrap()
    });
    plot.series.push(Series::new("gradient descent, step 0.18", to_xy(gd), Style::LineMarkers, 0));
    let cg = path(4, |k| {
        let mut m = NonlinearConjugateGradient::new(ConjugateGradientBeta::FletcherReeves);
        m.set_max_iterations(Some(k));
        m.gradient_based_minimize(&prob, &prob, x0.clone()).unwrap()
    });
    plot.series.push(Series::new("conjugate gradient (exact on a quadratic in 2 steps)", to_xy(cg), Style::LineMarkers, 1));
    write(CHAPTER, "quadratic_paths.svg", &plot.render());

    // Gradient descent on the quadratic with several step sizes: f against iterations.
    let mut plot = Plot::new("Gradient descent on ½(x² + 10y²): the step size", "iteration", "f(xₖ)");
    plot.log_y = true;
    plot.legend = Legend::TopRight;
    plot.y_range = Some((1e-16, 1e4));
    for (k, step) in [0.02, 0.1, 0.18, 0.205].into_iter().enumerate() {
        let mut m = GradientDescent::new(step);
        let pts: Vec<(f64, f64)> = (0..=100)
            .map(|i| {
                m.set_max_iterations(Some(i));
                m.set_tolerance(Some(0.0));
                (i as f64, m.gradient_based_minimize(&prob, &prob, x0.clone()).unwrap().objective_value.max(1e-16))
            })
            .collect();
        let label = if step > 0.2 { format!("step {step} (> 2/L = 0.2: diverges)") } else { format!("step {step}") };
        plot.series.push(Series::new(&label, pts, Style::Line, k));
    }
    write(CHAPTER, "step_sizes.svg", &plot.render());
}

/// Iterations and evaluations of the gradient-based methods on standard problems.
fn comparison_table() {
    let mut table = String::from("| problem | gradient descent | CG (PR+) | CG (FR) | BFGS | L-BFGS (m = 10) | Newton |\n|---|---:|---:|---:|---:|---:|---:|\n");
    let quad_n = 100;
    let quad_c = |i: usize| 1e3f64.powf(i as f64 / (quad_n - 1) as f64);
    let qf = move |p: &[f64]| p.iter().enumerate().map(|(i, v)| 0.5 * quad_c(i) * v * v).sum::<f64>();
    let qg = move |p: &[f64]| p.iter().enumerate().map(|(i, v)| quad_c(i) * v).collect::<Vec<_>>();
    let qh = move |p: &[f64]| (0..p.len()).map(|i| (0..p.len()).map(|j| if i == j { quad_c(i) } else { 0.0 }).collect()).collect::<Vec<Vec<f64>>>();
    type Case<'a> = (&'a str, Problem<'a>, Vec<f64>, f64);
    let cases: Vec<Case> = vec![
        ("Rosenbrock, $n = 2$", Problem::new(&rosenbrock, &rosenbrock_gradient, &rosenbrock_hessian), rosenbrock_start(2), 1.0 / 500.0),
        ("Rosenbrock, $n = 10$", Problem::new(&rosenbrock, &rosenbrock_gradient, &rosenbrock_hessian), rosenbrock_start(10), 1.0 / 500.0),
        ("quadratic, $n = 100$, $\\kappa = 10^3$", Problem::new(&qf, &qg, &qh), vec![1.0; quad_n], 1.0 / 1e3),
    ];
    for (name, prob, x0, lr) in &cases {
        let mut cells = vec![];
        let mut run = |r: OptimizationResult, prob: &Problem| {
            let ok = r.converged && r.objective_value < 1e-10;
            cells.push(if ok { format!("{} / {}", r.iterations, prob.f_calls.get()) } else { format!("fails, $f = {:.0e}$", r.objective_value) });
            prob.reset();
        };
        prob.reset();
        let mut gd = GradientDescent::new(*lr);
        gd.set_max_iterations(Some(20000));
        gd.set_tolerance(Some(1e-14));
        run(gd.gradient_based_minimize(prob, prob, x0.clone()).unwrap(), prob);
        run(NonlinearConjugateGradient::new(ConjugateGradientBeta::PolakRibierePlus).gradient_based_minimize(prob, prob, x0.clone()).unwrap(), prob);
        run(NonlinearConjugateGradient::new(ConjugateGradientBeta::FletcherReeves).gradient_based_minimize(prob, prob, x0.clone()).unwrap(), prob);
        run(Bfgs::new().gradient_based_minimize(prob, prob, x0.clone()).unwrap(), prob);
        run(Lbfgs::default().gradient_based_minimize(prob, prob, x0.clone()).unwrap(), prob);
        let mut newton = NewtonRaphsonMethod::new();
        newton.set_tolerance(Some(1e-14));
        run(newton.hessian_based_minimize(prob, prob, prob, x0.clone()).unwrap(), prob);
        table.push_str(&format!("| {name} | {} |\n", cells.join(" | ")));
    }
    write(CHAPTER, "gradient_methods.md", &table);

    // Convergence histories on Rosenbrock n = 2.
    let prob = Problem::new(&rosenbrock, &rosenbrock_gradient, &rosenbrock_hessian);
    let x0 = rosenbrock_start(2);
    let mut plot = Plot::new("Minimising the Rosenbrock function from (−1.2, 1)", "iteration", "f(xₖ) − f*");
    plot.log_y = true;
    plot.legend = Legend::TopRight;
    plot.y_range = Some((1e-20, 1e2));
    plot.x_range = Some((0.0, 60.0));
    type Runner<'a> = Box<dyn Fn(usize) -> OptimizationResult + 'a>;
    let runners: Vec<(&str, Runner)> = vec![
        ("gradient descent, step 0.002", Box::new(|k| {
            let mut m = GradientDescent::new(0.002);
            m.set_max_iterations(Some(k));
            m.set_tolerance(Some(0.0));
            m.gradient_based_minimize(&prob, &prob, x0.clone()).unwrap()
        })),
        ("conjugate gradient (PR+)", Box::new(|k| {
            let mut m = NonlinearConjugateGradient::default();
            m.set_max_iterations(Some(k));
            m.gradient_based_minimize(&prob, &prob, x0.clone()).unwrap()
        })),
        ("BFGS", Box::new(|k| {
            let mut m = Bfgs::new();
            m.set_max_iterations(Some(k));
            m.gradient_based_minimize(&prob, &prob, x0.clone()).unwrap()
        })),
        ("Newton", Box::new(|k| {
            let mut m = NewtonRaphsonMethod::new();
            m.set_max_iterations(Some(k));
            m.set_tolerance(Some(0.0));
            m.hessian_based_minimize(&prob, &prob, &prob, x0.clone()).unwrap()
        })),
    ];
    for (c, (name, run)) in runners.iter().enumerate() {
        let mut pts = vec![];
        for k in 0..=60 {
            let r = run(k);
            pts.push((k as f64, r.objective_value.max(1e-20)));
            if r.objective_value < 1e-20 || (k > 0 && r.iterations < k) {
                break;
            }
        }
        plot.series.push(Series::new(name, pts, Style::LineMarkers, c));
    }
    write(CHAPTER, "rosenbrock_convergence.svg", &plot.render());
}

fn direct_search_figures() {
    let count = Cell::new(0usize);
    struct Counted<'a>(&'a Cell<usize>, &'a dyn Fn(&[f64]) -> f64);
    impl ObjectiveFunction for Counted<'_> {
        fn evaluate(&self, p: &[f64]) -> f64 {
            self.0.set(self.0.get() + 1);
            (self.1)(p)
        }
    }
    let obj = Counted(&count, &rosenbrock);
    let x0 = rosenbrock_start(2);
    let mut plot = Plot::new("Direct search on the Rosenbrock function", "x", "y");
    plot.legend = Legend::BottomRight;
    plot.x_range = Some((-2.0, 2.0));
    plot.y_range = Some((-1.0, 3.0));
    plot.series.extend(rosenbrock_levels());
    let pw = path(20, |k| {
        let mut m = PowellsMethod::new();
        m.set_max_iterations(Some(k.max(1)));
        m.gradient_free_minimize(&obj, x0.clone()).unwrap()
    });
    let mut pts = vec![(x0[0], x0[1])];
    pts.extend(pw.iter().map(|r| (r.parameters[0], r.parameters[1])));
    plot.series.push(Series::new("Powell (after each iteration)", pts, Style::Markers, 1));
    let nm = path(120, |k| {
        let mut m = NelderMeadMethod::new();
        m.set_max_iterations(Some(k.max(1)));
        m.gradient_free_minimize(&obj, x0.clone()).unwrap()
    });
    plot.series.push(Series::new("Nelder-Mead (best vertex)", nm.iter().map(|r| (r.parameters[0], r.parameters[1])).collect(), Style::Line, 0));
    plot.series.push(Series::new("minimum (1, 1)", vec![(1.0, 1.0)], Style::Markers, 6));
    write(CHAPTER, "direct_search.svg", &plot.render());

    // Evaluations to converge.
    let mut table = String::from("| problem | Nelder-Mead | Powell | BFGS (exact gradient) |\n|---|---:|---:|---:|\n");
    for n in [2usize, 5, 10] {
        let x0 = rosenbrock_start(n);
        let mut cells = vec![];
        let mut nm = NelderMeadMethod::new();
        nm.set_max_iterations(Some(100_000));
        count.set(0);
        let r = nm.gradient_free_minimize(&obj, x0.clone()).unwrap();
        cells.push(format!("{} ev., $f = {:.0e}$", count.get(), r.objective_value));
        count.set(0);
        let r = PowellsMethod::new().gradient_free_minimize(&obj, x0.clone()).unwrap();
        cells.push(format!("{} ev., $f = {:.0e}$", count.get(), r.objective_value));
        let prob = Problem::new(&rosenbrock, &rosenbrock_gradient, &rosenbrock_hessian);
        let r = Bfgs::new().gradient_based_minimize(&prob, &prob, x0).unwrap();
        cells.push(format!("{} ev. of $f$ and $\\nabla f$, $f = {:.0e}$", prob.f_calls.get(), r.objective_value));
        table.push_str(&format!("| Rosenbrock, $n = {n}$ | {} |\n", cells.join(" | ")));
    }
    write(CHAPTER, "direct_search.md", &table);
}

pub fn generate() {
    line_search_figures();
    gradient_figures();
    comparison_table();
    direct_search_figures();
}
