//! Figures and tables for the second half of the optimisation chapter: least squares,
//! constrained optimisation, linear and dynamic programming, and stochastic global methods.

use crate::optimization::CHAPTER;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_optimization::constrained::{ConstrainedOptions, ConstrainedProblem, augmented_lagrangian, sqp};
use lib_jsl_optimization::differential_evolution::DifferentialEvolution;
use lib_jsl_optimization::dynamic_programming::{
    FiniteMdp, coin_change_min_coins, edit_distance, knapsack_01, longest_common_subsequence, longest_increasing_subsequence, matrix_chain_order,
};
use lib_jsl_optimization::least_squares::{LevenbergMarquardt, curve_fit};
use lib_jsl_optimization::linear_programming::{LinearProgram, LpStatus};
use lib_jsl_optimization::nelder_mead_method::NelderMeadMethod;
use lib_jsl_optimization::optimization_traits::{GradientFreeMinimizationEngine, MinimizationControls, ObjectiveFunction};
use lib_jsl_optimization::particle_swarm::ParticleSwarm;
use lib_jsl_optimization::simulated_annealing::SimulatedAnnealing;
use lib_jsl_random::distributions::gaussian_distribution_box_muller;
use lib_jsl_random::uniform_generator::{UniformGenerator, UniformRNG};
use std::cell::{Cell, RefCell};

fn least_squares_figures() {
    // A Gaussian peak on a baseline: y = a exp(−(x − μ)²/(2s²)) + b.
    let model = |x: f64, p: &[f64]| p[0] * (-(x - p[1]).powi(2) / (2.0 * p[2] * p[2])).exp() + p[3];
    let truth = [2.0, 4.0, 0.8, 0.3];
    let sigma = 0.05;
    let mut rng = UniformRNG::from_seed(11);
    let x: Vec<f64> = (0..60).map(|i| 8.0 * i as f64 / 59.0).collect();
    let y: Vec<f64> = x.iter().map(|&xi| model(xi, &truth) + gaussian_distribution_box_muller(&mut rng, 0.0, sigma).0).collect();
    let start = vec![1.0, 3.0, 1.5, 0.0];
    let fit = curve_fit(model, &x, &y, Some(&vec![sigma; x.len()]), start.clone(), &LevenbergMarquardt::new()).unwrap();
    let fine: Vec<f64> = (0..=400).map(|i| 8.0 * i as f64 / 400.0).collect();
    let mut plot = Plot::new("Fitting a Gaussian peak by Levenberg-Marquardt", "x", "y");
    plot.legend = Legend::TopRight;
    plot.series.push(Series::new("data (noise σ = 0.05)", x.iter().copied().zip(y.iter().copied()).collect(), Style::Markers, 0));
    plot.series.push(Series::new("starting guess", fine.iter().map(|&t| (t, model(t, &start))).collect(), Style::Dashed, 7));
    plot.series.push(Series::new("least-squares fit", fine.iter().map(|&t| (t, model(t, &fit.parameters))).collect(), Style::Line, 1));
    write(CHAPTER, "curve_fit.svg", &plot.render());
    let names = ["amplitude $a$", "centre $\\mu$", "width $s$", "baseline $b$"];
    let mut table = String::from("| parameter | true value | estimate | standard error | error / standard error |\n|---|---:|---:|---:|---:|\n");
    for i in 0..4 {
        let (e, se) = (fit.parameters[i], fit.standard_errors[i]);
        table.push_str(&format!("| {} | {} | {e:.4} | {se:.4} | {:+.2} |\n", names[i], truth[i], (e - truth[i]) / se));
    }
    table.push_str(&format!("\nReduced $\\chi^2 = {:.3}$ (expected about 1 for a correct model and noise level), {} iterations.\n", fit.reduced_chi_squared, fit.iterations));
    write(CHAPTER, "curve_fit.md", &table);
}

/// min (x − 2)² + (y − 1)² subject to x² − y ≤ 0 and x + y − 2 ≤ 0; solution (1, 1).
struct Disc {
    calls: Cell<usize>,
}
impl ConstrainedProblem for Disc {
    fn objective(&self, x: &[f64]) -> f64 {
        self.calls.set(self.calls.get() + 1);
        (x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2)
    }
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        vec![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]
    }
    fn inequalities(&self, x: &[f64]) -> Vec<f64> {
        vec![x[0] * x[0] - x[1], x[0] + x[1] - 2.0]
    }
    fn inequality_jacobian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        vec![vec![2.0 * x[0], -1.0], vec![1.0, 1.0]]
    }
}

/// Hock and Schittkowski's problem 71: min x₁x₄(x₁ + x₂ + x₃) + x₃ subject to x₁x₂x₃x₄ ≥ 25,
/// Σxᵢ² = 40 and 1 ≤ xᵢ ≤ 5; f* = 17.0140173. Derivatives by central differences.
struct Hs071 {
    calls: Cell<usize>,
}
impl ConstrainedProblem for Hs071 {
    fn objective(&self, x: &[f64]) -> f64 {
        self.calls.set(self.calls.get() + 1);
        x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
    }
    fn equalities(&self, x: &[f64]) -> Vec<f64> {
        vec![x.iter().map(|v| v * v).sum::<f64>() - 40.0]
    }
    fn inequalities(&self, x: &[f64]) -> Vec<f64> {
        let mut g = vec![25.0 - x.iter().product::<f64>()];
        g.extend(x.iter().map(|v| 1.0 - v));
        g.extend(x.iter().map(|v| v - 5.0));
        g
    }
}

fn constrained_figures() {
    let o = ConstrainedOptions::default();
    let x0 = vec![-1.5, 0.5];
    let disc = Disc { calls: Cell::new(0) };
    let mut plot = Plot::new("min (x − 2)² + (y − 1)² subject to y ≥ x², x + y ≤ 2", "x", "y");
    plot.legend = Legend::TopLeft;
    plot.x_range = Some((-2.0, 2.5));
    plot.y_range = Some((-0.5, 3.0));
    for r in [0.5, 1.0, 1.5, 2.5, 3.5] {
        let pts = (0..=200).map(|i| {
            let t = i as f64 / 200.0 * std::f64::consts::TAU;
            (2.0 + r * t.cos(), 1.0 + r * t.sin())
        });
        plot.series.push(Series::new("", pts.collect(), Style::Line, 7));
    }
    plot.series.push(Series::new("y = x²", (0..=100).map(|i| -2.0 + 4.0 * i as f64 / 100.0).map(|x| (x, x * x)).collect(), Style::Line, 6));
    plot.series.push(Series::new("x + y = 2", vec![(-2.0, 4.0), (2.5, -0.5)], Style::Dashed, 6));
    let sqp_path: Vec<(f64, f64)> = (1..=12)
        .map(|k| {
            let r = sqp(&disc, x0.clone(), ConstrainedOptions { max_iterations: k, ..o }).unwrap();
            (r.parameters[0], r.parameters[1])
        })
        .collect();
    let mut pts = vec![(x0[0], x0[1])];
    pts.extend(sqp_path);
    plot.series.push(Series::new("SQP iterates", pts, Style::LineMarkers, 0));
    let al_path: Vec<(f64, f64)> = (1..=8)
        .map(|k| {
            let r = augmented_lagrangian(&disc, x0.clone(), ConstrainedOptions { max_iterations: k, ..o }).unwrap();
            (r.parameters[0], r.parameters[1])
        })
        .collect();
    let mut pts = vec![(x0[0], x0[1])];
    pts.extend(al_path);
    plot.series.push(Series::new("augmented Lagrangian (outer iterates)", pts, Style::LineMarkers, 1));
    write(CHAPTER, "constrained.svg", &plot.render());

    let mut table = String::from("| problem | method | iterations | evaluations of $f$ | $f$ | constraint violation | multipliers (inequalities) |\n|---|---|---:|---:|---:|---:|---|\n");
    let fmt = |v: &[f64]| v.iter().map(|m| format!("{m:.3}")).collect::<Vec<_>>().join(", ");
    for (name, solver) in [("SQP", sqp::<Disc> as fn(&Disc, Vec<f64>, ConstrainedOptions) -> _), ("augmented Lagrangian", augmented_lagrangian::<Disc>)] {
        disc.calls.set(0);
        let r = solver(&disc, x0.clone(), o).unwrap();
        table.push_str(&format!("| the problem above | {name} | {} | {} | {:.8} | {:.0e} | {} |\n", r.iterations, disc.calls.get(), r.objective_value, r.constraint_violation, fmt(&r.inequality_multipliers)));
    }
    let hs = Hs071 { calls: Cell::new(0) };
    let start = vec![1.0, 5.0, 5.0, 1.0];
    for (name, solver) in [("SQP", sqp::<Hs071> as fn(&Hs071, Vec<f64>, ConstrainedOptions) -> _), ("augmented Lagrangian", augmented_lagrangian::<Hs071>)] {
        hs.calls.set(0);
        let r = solver(&hs, start.clone(), o).unwrap();
        let active: Vec<String> = r.inequality_multipliers.iter().enumerate().filter(|(_, m)| **m > 1e-6).map(|(i, m)| format!("$g_{{{}}}$: {m:.3}", i + 1)).collect();
        table.push_str(&format!("| Hock-Schittkowski 71 | {name} | {} | {} | {:.8} | {:.0e} | {} |\n", r.iterations, hs.calls.get(), r.objective_value, r.constraint_violation, active.join(", ")));
    }
    write(CHAPTER, "constrained.md", &table);
}

fn linear_programming_figures() {
    // The Wyndor Glass problem of Hillier and Lieberman: max 3x + 5y.
    let wyndor = LinearProgram::new(vec![-3.0, -5.0]).with_less_equal(vec![1.0, 0.0], 4.0).with_less_equal(vec![0.0, 2.0], 12.0).with_less_equal(vec![3.0, 2.0], 18.0);
    let sol = wyndor.solve().unwrap();
    let mut plot = Plot::new("A linear program: maximise 3x + 5y", "x", "y");
    plot.legend = Legend::TopRight;
    plot.x_range = Some((0.0, 7.0));
    plot.y_range = Some((0.0, 9.0));
    plot.series.push(Series::new("feasible region", vec![(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (2.0, 6.0), (0.0, 6.0), (0.0, 0.0)], Style::LineMarkers, 0));
    for (k, c) in [12.0, 24.0, 36.0].into_iter().enumerate() {
        let label = if k == 0 { "3x + 5y = 12, 24, 36" } else { "" };
        plot.series.push(Series::new(label, vec![(0.0, c / 5.0), (c / 3.0, 0.0)], Style::Dashed, 7));
    }
    plot.series.push(Series::new(&format!("optimum ({}, {}), value {}", sol.x[0], sol.x[1], -sol.objective), vec![(sol.x[0], sol.x[1])], Style::Markers, 1));
    write(CHAPTER, "linear_program.svg", &plot.render());

    let mut table = String::from("| program | status | solution | objective | pivots |\n|---|---|---|---:|---:|\n");
    let mut row = |name: &str, lp: LinearProgram, sign: f64| {
        let s = lp.solve().unwrap();
        let (x, obj) = if s.status == LpStatus::Optimal {
            (format!("({})", s.x.iter().map(|v| format!("{:.4}", v + 0.0).trim_end_matches('0').trim_end_matches('.').to_string()).collect::<Vec<_>>().join(", ")), format!("{:.4}", sign * s.objective))
        } else {
            ("n/a".to_string(), "n/a".to_string())
        };
        table.push_str(&format!("| {name} | {:?} | {x} | {obj} | {} |\n", s.status, s.iterations));
    };
    row("Wyndor: max $3x + 5y$ (above)", wyndor.clone(), -1.0);
    // Transportation: two plants (supply 20, 30) to three markets (demand 10, 25, 15).
    let cost = [8.0, 6.0, 10.0, 9.0, 12.0, 13.0];
    let mut t = LinearProgram::new(cost.to_vec());
    for i in 0..2 {
        t = t.with_equal((0..6).map(|k| f64::from(k / 3 == i)).collect(), [20.0, 30.0][i]);
    }
    for j in 0..3 {
        t = t.with_equal((0..6).map(|k| f64::from(k % 3 == j)).collect(), [10.0, 25.0, 15.0][j]);
    }
    row("transportation, 2 × 3 (min cost)", t, 1.0);
    // Beale's example, on which the textbook pivoting rule cycles forever.
    // (Chvátal's statement; optimum −1/20 at (1/25, 0, 1, 0).)
    let beale = LinearProgram::new(vec![-0.75, 150.0, -0.02, 6.0])
        .with_less_equal(vec![0.25, -60.0, -0.04, 9.0], 0.0)
        .with_less_equal(vec![0.5, -90.0, -0.02, 3.0], 0.0)
        .with_less_equal(vec![0.0, 0.0, 1.0, 0.0], 1.0);
    row("Beale's cycling example (min)", beale, 1.0);
    row("infeasible: $x + y \\le 1$, $x + y \\ge 2$", LinearProgram::new(vec![1.0, 1.0]).with_less_equal(vec![1.0, 1.0], 1.0).with_greater_equal(vec![1.0, 1.0], 2.0), 1.0);
    row("unbounded: max $x + y$, $x - y \\le 1$", LinearProgram::new(vec![-1.0, -1.0]).with_less_equal(vec![1.0, -1.0], 1.0), -1.0);
    row("free variable: min $|x - 3|$ as min $t$, $-t \\le x - 3 \\le t$", LinearProgram::new(vec![0.0, 1.0]).with_free_variable(0).with_less_equal(vec![1.0, -1.0], 3.0).with_greater_equal(vec![1.0, 1.0], 3.0), 1.0);
    write(CHAPTER, "linear_programs.md", &table);
}

fn dynamic_programming_figures() {
    let mut table = String::from("| problem | input | result |\n|---|---|---|\n");
    let (v, items) = knapsack_01(&[12, 2, 1, 1, 4], &[4.0, 2.0, 1.0, 2.0, 10.0], 15).unwrap();
    table.push_str(&format!("| 0/1 knapsack | weights 12, 2, 1, 1, 4; values 4, 2, 1, 2, 10; capacity 15 | value {v}, items {items:?} |\n"));
    let us = coin_change_min_coins(&[1, 5, 10, 25], 63).unwrap();
    table.push_str(&format!("| fewest coins | coins 1, 5, 10, 25; amount 63 | {} coins: {us:?} of each |\n", us.iter().sum::<usize>()));
    let odd = coin_change_min_coins(&[1, 3, 4], 6).unwrap();
    table.push_str(&format!("| fewest coins | coins 1, 3, 4; amount 6 (greedy takes 4 + 1 + 1) | {} coins: {odd:?} of each |\n", odd.iter().sum::<usize>()));
    let (d, _) = edit_distance(&"kitten".chars().collect::<Vec<_>>(), &"sitting".chars().collect::<Vec<_>>());
    table.push_str(&format!("| edit distance | kitten → sitting | {d} |\n"));
    let a: Vec<char> = "ABCBDAB".chars().collect();
    let lcs: String = longest_common_subsequence(&a, &"BDCABA".chars().collect::<Vec<_>>()).iter().map(|&(i, _)| a[i]).collect();
    table.push_str(&format!("| longest common subsequence | ABCBDAB, BDCABA | {lcs} |\n"));
    let seq = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9];
    let lis: Vec<i32> = longest_increasing_subsequence(&seq).iter().map(|&i| seq[i]).collect();
    table.push_str(&format!("| longest increasing subsequence | 3 1 4 1 5 9 2 6 5 3 5 8 9 7 9 | {lis:?} |\n"));
    let (c, p) = matrix_chain_order(&[30, 35, 15, 5, 10, 20, 25]).unwrap();
    table.push_str(&format!("| matrix-chain order | dimensions 30, 35, 15, 5, 10, 20, 25 | {c} multiplications: {p} |\n"));
    write(CHAPTER, "dynamic_programming.md", &table.replace("[", "(").replace("]", ")"));

    // Value iteration on a random 20-state, 3-action MDP. Each value-iteration step from V = 0 is
    // one stage of backward induction on the MDP whose transitions are scaled by γ, with the
    // remaining probability 1 − γ going to an absorbing state of value zero.
    let (n, actions) = (20usize, 3usize);
    let mut rng = UniformRNG::from_seed(4);
    let mut transitions = vec![];
    let mut rewards = vec![];
    for _ in 0..n {
        let mut ts = vec![];
        let mut rs = vec![];
        for _ in 0..actions {
            let w: Vec<f64> = (0..n).map(|_| rng.next_f64().powi(4)).collect();
            let total: f64 = w.iter().sum();
            ts.push(w.iter().enumerate().map(|(t, wi)| (t, wi / total)).collect::<Vec<_>>());
            rs.push(rng.next_f64());
        }
        transitions.push(ts);
        rewards.push(rs);
    }
    let mut plot = Plot::new("Value iteration on a random MDP (20 states, 3 actions)", "iteration k", "max |Vₖ − V*|");
    plot.log_y = true;
    plot.legend = Legend::TopRight;
    plot.y_range = Some((1e-12, 1e3));
    for (k, gamma) in [0.5, 0.9, 0.99].into_iter().enumerate() {
        let mdp = FiniteMdp { transitions: transitions.clone(), rewards: rewards.clone() };
        let exact = mdp.value_iteration(gamma, 1e-13, 100_000).unwrap().values;
        let mut scaled = FiniteMdp { transitions: vec![], rewards: rewards.clone() };
        for ts in &transitions {
            scaled.transitions.push(ts.iter().map(|o| {
                let mut v: Vec<(usize, f64)> = o.iter().map(|&(t, p)| (t, gamma * p)).collect();
                v.push((n, 1.0 - gamma));
                v
            }).collect());
        }
        scaled.transitions.push(vec![vec![(n, 1.0)]]);
        scaled.rewards.push(vec![0.0]);
        let stages = scaled.backward_induction(400, &vec![0.0; n + 1]).unwrap();
        // stages[0] is the value with 400 steps to go; stages[400 − k] has k steps to go.
        let pts = (1..=400)
            .map(|k| {
                let v = &stages[400 - k].values;
                (k as f64, (0..n).map(|s| (v[s] - exact[s]).abs()).fold(0.0, f64::max).max(1e-13))
            })
            .collect();
        plot.series.push(Series::new(&format!("γ = {gamma}"), pts, Style::Line, k));
    }
    write(CHAPTER, "value_iteration.svg", &plot.render());
}

/// A traced objective: counts evaluations and records the best value so far after each.
struct Traced<'a> {
    f: &'a dyn Fn(&[f64]) -> f64,
    trace: RefCell<Vec<f64>>,
}
impl ObjectiveFunction for Traced<'_> {
    fn evaluate(&self, p: &[f64]) -> f64 {
        let v = (self.f)(p);
        let mut t = self.trace.borrow_mut();
        let best = t.last().map_or(v, |b: &f64| b.min(v));
        t.push(best);
        v
    }
}

fn rastrigin(x: &[f64]) -> f64 {
    10.0 * x.len() as f64 + x.iter().map(|v| v * v - 10.0 * (2.0 * std::f64::consts::PI * v).cos()).sum::<f64>()
}

fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let s1 = x.iter().map(|v| v * v).sum::<f64>() / n;
    let s2 = x.iter().map(|v| (2.0 * std::f64::consts::PI * v).cos()).sum::<f64>() / n;
    -20.0 * (-0.2 * s1.sqrt()).exp() - s2.exp() + 20.0 + std::f64::consts::E
}

type Run<'a> = Box<dyn Fn(&Traced, u64, Vec<f64>) -> f64 + 'a>;

fn stochastic_figures() {
    let methods = |n: usize, half: f64| -> Vec<(&'static str, Run)> {
        let bounds = vec![(-half, half); n];
        let b1 = bounds.clone();
        let b2 = bounds.clone();
        let b3 = bounds.clone();
        vec![
            ("Nelder-Mead, random start", Box::new(move |f: &Traced, _, x0| {
                let mut m = NelderMeadMethod::new();
                m.set_max_iterations(Some(20_000));
                m.gradient_free_minimize(f, x0).unwrap().objective_value
            })),
            ("simulated annealing", Box::new(move |f: &Traced, seed, x0| {
                let mut m = SimulatedAnnealing::new();
                m.set_bounds(Some(b1.clone()));
                m.set_seed(seed);
                m.set_maximum_temperature(Some(3.0));
                m.set_minimum_temperature(Some(1e-4));
                m.set_cooling_coefficient(Some(0.95));
                m.set_iterations_per_temperature(Some(100 * n));
                m.gradient_free_minimize(f, x0).unwrap().objective_value
            })),
            ("differential evolution", Box::new(move |f: &Traced, seed, x0| {
                let mut m = DifferentialEvolution::new(b2.clone());
                m.seed = seed;
                m.gradient_free_minimize(f, x0).unwrap().objective_value
            })),
            ("particle swarm", Box::new(move |f: &Traced, seed, x0| {
                let mut m = ParticleSwarm::new(b3.clone());
                m.seed = seed;
                m.gradient_free_minimize(f, x0).unwrap().objective_value
            })),
        ]
    };
    let seeds = 20u64;
    let mut table = String::from("| function | method | runs reaching $f < 10^{-4}$ | median best $f$ | median evaluations |\n|---|---|---:|---:|---:|\n");
    let mut plot = Plot::new("Rastrigin function, n = 5: best value found (seed 1)", "evaluations of f", "best f so far");
    plot.log_y = true;
    plot.log_x = true;
    plot.legend = Legend::BottomLeft;
    plot.y_range = Some((1e-10, 1e3));
    type TestFunction = (&'static str, fn(&[f64]) -> f64, f64);
    let funcs: [TestFunction; 2] = [("Rastrigin", rastrigin, 5.12), ("Ackley", ackley, 32.768)];
    for (fname, func, half) in funcs {
        for n in [2usize, 5] {
            for (k, (mname, run)) in methods(n, half).into_iter().enumerate() {
                let mut best = vec![];
                let mut evals = vec![];
                for seed in 0..seeds {
                    let mut rng = UniformRNG::from_seed(1000 + seed);
                    let x0: Vec<f64> = (0..n).map(|_| (2.0 * rng.next_f64() - 1.0) * half).collect();
                    let traced = Traced { f: &func, trace: RefCell::new(vec![]) };
                    let v = run(&traced, seed, x0);
                    let trace = traced.trace.into_inner();
                    best.push(v.min(*trace.last().unwrap()));
                    evals.push(trace.len() as f64);
                    if fname == "Rastrigin" && n == 5 && seed == 1 {
                        let step = (trace.len() / 300).max(1);
                        let pts = trace.iter().enumerate().filter(|(i, _)| i % step == 0 || *i + 1 == trace.len()).map(|(i, v)| ((i + 1) as f64, v.max(1e-10))).collect();
                        plot.series.push(Series::new(mname, pts, Style::Line, k));
                    }
                }
                let hits = best.iter().filter(|&&b| b < 1e-4).count();
                let median = |v: &mut Vec<f64>| {
                    v.sort_by(f64::total_cmp);
                    0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2])
                };
                table.push_str(&format!("| {fname}, $n = {n}$ | {mname} | {hits} / {seeds} | {:.1e} | {:.0} |\n", median(&mut best), median(&mut evals)));
            }
        }
    }
    write(CHAPTER, "global.md", &table);
    write(CHAPTER, "global.svg", &plot.render());
}

pub fn generate() {
    least_squares_figures();
    constrained_figures();
    linear_programming_figures();
    dynamic_programming_figures();
    stochastic_figures();
}
