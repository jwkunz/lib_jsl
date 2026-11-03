//! Figures and tables for the machine learning chapter on regression and classification.

use crate::linear_algebra::time;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_core::{R1D, R2D};
use lib_jsl_machine_learning::common::{Classifier, MlRng, Regressor, Transformer};
use lib_jsl_machine_learning::datasets::{make_blobs, make_circles, make_classification, make_moons};
use lib_jsl_machine_learning::kernels::Kernel;
use lib_jsl_machine_learning::metrics::{accuracy, confusion_matrix, mean_squared_error, precision_recall_f1};
use lib_jsl_machine_learning::preprocessing::{StandardScaler, select_labels, select_rows, train_test_split};
use lib_jsl_machine_learning::supervised::classification::{DecisionTree, GaussianNaiveBayes, KNearestNeighbors, LogisticRegression, MultinomialNaiveBayes, RandomForest, SupportVectorMachine};
use lib_jsl_machine_learning::supervised::regression::{ElasticNet, GeneralLeastSquares, LinearRegression, Ridge, SupportVectorRegression};
use std::collections::HashMap;
use std::f64::consts::PI;

const CHAPTER: &str = "machine_learning";

pub fn generate() {
    polynomial_figures();
    regularisation_paths();
    weighted_least_squares();
    optimality_table();
    logistic_figures();
    naive_bayes_figures();
    knn_figures();
    tree_figures();
    svm_figures();
    comparison_tables();
}

// ------------------------------------------------------------------ plotting helpers

/// Polylines of the zero level set of `f` on a `cells × cells` grid over the box (marching squares,
/// with saddles resolved by the centre value).
fn contour(f: &dyn Fn(&R2D) -> Vec<f64>, (x0, x1): (f64, f64), (y0, y1): (f64, f64), cells: usize) -> Vec<Vec<(f64, f64)>> {
    let m = cells + 1;
    let xs: Vec<f64> = (0..m).map(|i| x0 + (x1 - x0) * i as f64 / cells as f64).collect();
    let ys: Vec<f64> = (0..m).map(|j| y0 + (y1 - y0) * j as f64 / cells as f64).collect();
    let pts = R2D::from_shape_fn((m * m, 2), |(k, c)| if c == 0 { xs[k % m] } else { ys[k / m] });
    let v = f(&pts);
    let at = |i: usize, j: usize| v[j * m + i];
    let centres = R2D::from_shape_fn((cells * cells, 2), |(k, c)| if c == 0 { 0.5 * (xs[k % cells] + xs[k % cells + 1]) } else { 0.5 * (ys[k / cells] + ys[k / cells + 1]) });
    let vc = f(&centres);
    // Edge keys: (0, i, j) is the horizontal edge from (i, j) to (i+1, j); (1, i, j) the vertical one to (i, j+1).
    let point = |key: (u8, usize, usize)| -> (f64, f64) {
        let (kind, i, j) = key;
        let (a, b, pa, pb) = if kind == 0 { (at(i, j), at(i + 1, j), (xs[i], ys[j]), (xs[i + 1], ys[j])) } else { (at(i, j), at(i, j + 1), (xs[i], ys[j]), (xs[i], ys[j + 1])) };
        let t = if a == b { 0.5 } else { (a / (a - b)).clamp(0.0, 1.0) };
        (pa.0 + t * (pb.0 - pa.0), pa.1 + t * (pb.1 - pa.1))
    };
    type Edge = (u8, usize, usize);
    let mut segments: Vec<(Edge, Edge)> = Vec::new();
    for j in 0..cells {
        for i in 0..cells {
            let corners = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
            let edges = [(0, i, j), (1, i + 1, j), (0, i, j + 1), (1, i, j)]; // bottom, right, top, left
            let pos: Vec<bool> = corners.iter().map(|&c| c >= 0.0).collect();
            let crossing: Vec<usize> = (0..4).filter(|&e| pos[e] != pos[(e + 1) % 4]).collect();
            match crossing.len() {
                2 => segments.push((edges[crossing[0]], edges[crossing[1]])),
                4 => {
                    if (vc[j * cells + i] >= 0.0) == pos[0] {
                        segments.push((edges[0], edges[1]));
                        segments.push((edges[2], edges[3]));
                    } else {
                        segments.push((edges[3], edges[0]));
                        segments.push((edges[1], edges[2]));
                    }
                }
                _ => {}
            }
        }
    }
    // Chain segments into polylines.
    let mut by_key: HashMap<(u8, usize, usize), Vec<usize>> = HashMap::new();
    for (s, (a, b)) in segments.iter().enumerate() {
        by_key.entry(*a).or_default().push(s);
        by_key.entry(*b).or_default().push(s);
    }
    let mut used = vec![false; segments.len()];
    let mut lines = Vec::new();
    for start in 0..segments.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let mut chain = vec![segments[start].0, segments[start].1];
        for forward in [true, false] {
            loop {
                let end = if forward { *chain.last().unwrap() } else { chain[0] };
                let next = by_key[&end].iter().copied().find(|&s| !used[s]);
                let Some(s) = next else { break };
                used[s] = true;
                let other = if segments[s].0 == end { segments[s].1 } else { segments[s].0 };
                if forward { chain.push(other) } else { chain.insert(0, other) }
            }
        }
        lines.push(chain.into_iter().map(point).collect());
    }
    lines
}

fn add_contour(plot: &mut Plot, lines: Vec<Vec<(f64, f64)>>, label: &str, style: Style, color: usize) {
    for (k, line) in lines.into_iter().enumerate() {
        plot.series.push(Series::new(if k == 0 { label } else { "" }, line, style, color));
    }
}

fn add_classes(plot: &mut Plot, x: &R2D, labels: &[usize], names: [&str; 2]) {
    for (c, name) in names.iter().enumerate() {
        let pts = (0..x.nrows()).filter(|&i| labels[i] == c).map(|i| (x[[i, 0]], x[[i, 1]])).collect();
        plot.series.push(Series::new(name, pts, Style::Markers, if c == 0 { 0 } else { 1 }));
    }
}

/// `P(class 1) − 1/2` of a probabilistic classifier, for contouring.
fn proba_minus_half(proba: R2D) -> Vec<f64> {
    proba.column(1).iter().map(|p| p - 0.5).collect()
}

fn boundary_plot(title: &str, x: &R2D, labels: &[usize], score: &dyn Fn(&R2D) -> Vec<f64>, xr: (f64, f64), yr: (f64, f64), file: &str) {
    let mut plot = Plot::new(title, "x₁", "x₂");
    plot.x_range = Some(xr);
    plot.y_range = Some(yr);
    add_classes(&mut plot, x, labels, ["class 0", "class 1"]);
    add_contour(&mut plot, contour(score, xr, yr, 120), "decision boundary", Style::Line, 6);
    plot.legend = Legend::BottomRight;
    write(CHAPTER, file, &plot.render());
}

fn moons(n: usize, seed: u64) -> (R2D, Vec<usize>) {
    make_moons(n, 0.25, seed)
}

const MOON_X: (f64, f64) = (-1.8, 2.8);
const MOON_Y: (f64, f64) = (-1.3, 1.8);

// ------------------------------------------------------------------ linear models

/// `y = sin(πx) + noise` on `[-1, 1]`.
fn sine_data(n: usize, rng: &mut MlRng) -> (Vec<f64>, R1D) {
    let x: Vec<f64> = (0..n).map(|_| rng.range(-1.0, 1.0)).collect();
    let y = R1D::from_iter(x.iter().map(|&v| (PI * v).sin() + rng.gaussian(0.0, 0.25)));
    (x, y)
}

fn powers(x: &[f64], degree: usize) -> R2D {
    R2D::from_shape_fn((x.len(), degree), |(i, j)| x[i].powi(j as i32 + 1))
}

fn polynomial_figures() {
    let mut rng = MlRng::new(3);
    let (x, y) = sine_data(20, &mut rng);
    let grid: Vec<f64> = (0..=400).map(|i| -1.0 + 2.0 * i as f64 / 400.0).collect();
    let mut plot = Plot::new("Polynomial fits to 20 noisy samples of sin(πx)", "x", "y");
    plot.y_range = Some((-2.0, 2.0));
    plot.legend = Legend::BottomRight;
    plot.series.push(Series::new("sin(πx)", grid.iter().map(|&v| (v, (PI * v).sin())).collect(), Style::Dashed, 7));
    let fits: [(&str, usize, Option<f64>, usize); 3] = [("degree 3, least squares", 3, None, 2), ("degree 15, least squares", 15, None, 1), ("degree 15, ridge α = 10⁻³", 15, Some(1e-3), 0)];
    for (name, degree, alpha, color) in fits {
        let px = powers(&x, degree);
        let pred = match alpha {
            None => { let mut m = LinearRegression::new(); m.fit(px.view(), y.view()).unwrap(); m.predict(powers(&grid, degree).view()).unwrap() }
            Some(a) => { let mut m = Ridge::new(a); m.fit(px.view(), y.view()).unwrap(); m.predict(powers(&grid, degree).view()).unwrap() }
        };
        plot.series.push(Series::new(name, grid.iter().zip(pred.iter()).map(|(&g, &p)| (g, p.clamp(-5.0, 5.0))).collect(), Style::Line, color));
    }
    plot.series.push(Series::new("data", x.iter().copied().zip(y.iter().copied()).collect(), Style::Markers, 6));
    write(CHAPTER, "polynomial_fits.svg", &plot.render());

    // Training and test error of degree-15 ridge against the penalty, averaged over data sets.
    let alphas: Vec<f64> = (0..=32).map(|k| 10f64.powf(-9.0 + 0.3125 * k as f64)).collect();
    let reps = 100;
    let (mut train, mut test) = (vec![0.0; alphas.len()], vec![0.0; alphas.len()]);
    let mut rng = MlRng::new(4);
    for _ in 0..reps {
        let (xtr, ytr) = sine_data(20, &mut rng);
        let (xte, yte) = sine_data(500, &mut rng);
        let (ptr, pte) = (powers(&xtr, 15), powers(&xte, 15));
        for (k, &a) in alphas.iter().enumerate() {
            let mut m = Ridge::new(a);
            m.fit(ptr.view(), ytr.view()).unwrap();
            train[k] += mean_squared_error(ytr.view(), m.predict(ptr.view()).unwrap().view()).unwrap() / reps as f64;
            test[k] += mean_squared_error(yte.view(), m.predict(pte.view()).unwrap().view()).unwrap() / reps as f64;
        }
    }
    let mut plot = Plot::new("Degree-15 ridge regression on 20 points: mean squared error", "penalty α", "mean squared error");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::TopLeft;
    plot.series.push(Series::new("test", alphas.iter().copied().zip(test).collect(), Style::LineMarkers, 1));
    plot.series.push(Series::new("training", alphas.iter().copied().zip(train).collect(), Style::LineMarkers, 0));
    plot.series.push(Series::new("noise variance 0.0625", vec![(alphas[0], 0.0625), (alphas[alphas.len() - 1], 0.0625)], Style::Dashed, 7));
    write(CHAPTER, "ridge_error.svg", &plot.render());
}

/// Eight standardised features, of which the first three matter; features 0 and 3 are correlated.
fn sparse_problem(n: usize, rng: &mut MlRng) -> (R2D, R1D) {
    let mut x = R2D::from_shape_fn((n, 8), |_| rng.gaussian(0.0, 1.0));
    for i in 0..n {
        x[[i, 3]] = 0.8 * x[[i, 0]] + 0.6 * x[[i, 3]];
    }
    let y = R1D::from_iter((0..n).map(|i| 3.0 * x[[i, 0]] - 2.0 * x[[i, 1]] + 1.5 * x[[i, 2]] + rng.gaussian(0.0, 1.0)));
    (x, y)
}

fn regularisation_paths() {
    let mut rng = MlRng::new(12);
    let (x, y) = sparse_problem(60, &mut rng);
    for (file, title, lo, hi, lasso) in [("ridge_path.svg", "Ridge coefficient paths", -2.0, 4.0, false), ("lasso_path.svg", "Lasso coefficient paths", -3.0, 1.0, true)] {
        let alphas: Vec<f64> = (0..=60).map(|k| 10f64.powf(lo + (hi - lo) * k as f64 / 60.0)).collect();
        let mut paths = vec![Vec::new(); 8];
        for &a in &alphas {
            let w = if lasso {
                let mut m = ElasticNet::lasso(a);
                m.fit(x.view(), y.view()).unwrap();
                m.weights().clone()
            } else {
                let mut m = Ridge::new(a);
                m.fit(x.view(), y.view()).unwrap();
                m.weights().clone()
            };
            for (p, wj) in paths.iter_mut().zip(w.iter()) {
                p.push((a, *wj));
            }
        }
        let mut plot = Plot::new(title, "penalty α", "coefficient");
        plot.log_x = true;
        plot.legend = Legend::TopRight;
        let names = ["w₀ (true 3)", "w₁ (true −2)", "w₂ (true 1.5)", "w₃ (true 0, correlated with x₀)"];
        for (j, p) in paths.into_iter().enumerate() {
            let (label, color) = if j < 4 { (names[j], j) } else if j == 4 { ("w₄ … w₇ (true 0)", 7) } else { ("", 7) };
            plot.series.push(Series::new(label, p, Style::Line, color));
        }
        write(CHAPTER, file, &plot.render());
    }
}

fn weighted_least_squares() {
    // Noise standard deviation 0.2 + 0.5x on [0, 10].
    let n = 50;
    let xs: Vec<f64> = (0..n).map(|i| 10.0 * i as f64 / (n - 1) as f64).collect();
    let sd: Vec<f64> = xs.iter().map(|v| 0.2 + 0.5 * v).collect();
    let x = R2D::from_shape_fn((n, 1), |(i, _)| xs[i]);
    let weights = R1D::from_iter(sd.iter().map(|s| 1.0 / (s * s)));
    let mut rng = MlRng::new(21);
    let draw = |rng: &mut MlRng| R1D::from_iter((0..n).map(|i| 1.0 + 2.0 * xs[i] + rng.gaussian(0.0, sd[i])));
    let y = draw(&mut rng);
    let mut ols = LinearRegression::new();
    ols.fit(x.view(), y.view()).unwrap();
    let mut wls = GeneralLeastSquares::with_sample_weights(weights.clone());
    wls.fit(x.view(), y.view()).unwrap();
    let mut plot = Plot::new("Noise growing with x: ordinary and weighted least squares", "x", "y");
    plot.legend = Legend::TopLeft;
    plot.series.push(Series::new("data", xs.iter().copied().zip(y.iter().copied()).collect(), Style::Markers, 6));
    plot.series.push(Series::new("true line y = 1 + 2x", vec![(0.0, 1.0), (10.0, 21.0)], Style::Dashed, 7));
    plot.series.push(Series::new(&format!("least squares, slope {:.2}", ols.weights()[0]), vec![(0.0, ols.intercept()), (10.0, ols.intercept() + 10.0 * ols.weights()[0])], Style::Line, 1));
    plot.series.push(Series::new(&format!("weighted, slope {:.2}", wls.weights()[0]), vec![(0.0, wls.intercept()), (10.0, wls.intercept() + 10.0 * wls.weights()[0])], Style::Line, 0));
    write(CHAPTER, "wls.svg", &plot.render());

    let reps = 2000;
    let mut est = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for _ in 0..reps {
        let y = draw(&mut rng);
        let mut ols = LinearRegression::new();
        ols.fit(x.view(), y.view()).unwrap();
        let mut wls = GeneralLeastSquares::with_sample_weights(weights.clone());
        wls.fit(x.view(), y.view()).unwrap();
        est[0].push(ols.weights()[0]);
        est[1].push(wls.weights()[0]);
        est[2].push(ols.intercept());
        est[3].push(wls.intercept());
    }
    let sd_of = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (m, (v.iter().map(|a| (a - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt())
    };
    let mut table = String::from("| estimator | mean slope (true 2) | sd of slope | mean intercept (true 1) | sd of intercept |\n|---|---:|---:|---:|---:|\n");
    for (name, s, b) in [("ordinary least squares", 0, 2), ("weighted least squares", 1, 3)] {
        let ((ms, ss), (mb, sb)) = (sd_of(&est[s]), sd_of(&est[b]));
        table.push_str(&format!("| {name} | {ms:.4} | {ss:.4} | {mb:.4} | {sb:.4} |\n"));
    }
    table.push_str(&format!("\nOver {reps} simulated data sets of {n} points.\n"));
    write(CHAPTER, "wls.md", &table);
}

/// Largest violation of each fitted model's optimality conditions.
fn optimality_table() {
    let mut rng = MlRng::new(31);
    let (x, y) = sparse_problem(100, &mut rng);
    let (n, p) = x.dim();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let xbar: Vec<f64> = (0..p).map(|j| mean(&x.column(j).to_vec())).collect();
    let resid = |w: &R1D, b: f64| -> Vec<f64> { (0..n).map(|i| y[i] - b - (0..p).map(|j| x[[i, j]] * w[j]).sum::<f64>()).collect() };
    let xt_r = |r: &[f64], j: usize| (0..n).map(|i| x[[i, j]] * r[i]).sum::<f64>();
    let scale = (0..n).map(|i| y[i] * y[i]).sum::<f64>().sqrt() * (x.iter().map(|v| v * v).sum::<f64>()).sqrt();
    let mut table = String::from("| model | optimality condition | largest violation |\n|---|---|---:|\n");

    let mut ols = LinearRegression::new();
    ols.fit(x.view(), y.view()).unwrap();
    let r = resid(ols.weights(), ols.intercept());
    let v = (0..p).map(|j| xt_r(&r, j).abs()).fold(r.iter().sum::<f64>().abs(), f64::max) / scale;
    table.push_str(&format!("| least squares | $X^{{\\mathsf T}}r = 0$ (relative to $\\lVert X\\rVert\\lVert y\\rVert$) | {v:.1e} |\n"));

    let alpha = 5.0;
    let mut ridge = Ridge::new(alpha);
    ridge.fit(x.view(), y.view()).unwrap();
    let r = resid(ridge.weights(), ridge.intercept());
    let v = (0..p).map(|j| (xt_r(&r, j) - alpha * ridge.weights()[j]).abs()).fold(r.iter().sum::<f64>().abs(), f64::max) / scale;
    table.push_str(&format!("| ridge, $\\alpha = 5$ | $X^{{\\mathsf T}}r = \\alpha w$ (relative) | {v:.1e} |\n"));

    for (name, a, rho) in [("lasso, $\\alpha = 0.1$", 0.1, 1.0), ("elastic net, $\\alpha = 0.1$, $\\rho = 0.5$", 0.1, 0.5)] {
        let mut m = ElasticNet::new(a, rho);
        m.fit(x.view(), y.view()).unwrap();
        let r = resid(m.weights(), m.intercept());
        let (l1, l2) = (a * rho, a * (1.0 - rho));
        let v = (0..p).map(|j| {
            let g = xt_r(&r, j) / n as f64 - l2 * m.weights()[j];
            let w = m.weights()[j];
            if w != 0.0 { (g - l1 * w.signum()).abs() } else { (g.abs() - l1).max(0.0) }
        }).fold(0.0, f64::max);
        table.push_str(&format!("| {name} | subgradient: $\\tfrac1n x_j^{{\\mathsf T}}r - \\alpha(1-\\rho)w_j \\in \\alpha\\rho\\,\\partial\\lvert w_j\\rvert$ | {v:.1e} ({} of 8 weights non-zero) |\n", m.nonzero()));
    }

    // Logistic regression: the gradient of the penalised mean log-loss.
    let labels: Vec<usize> = (0..n).map(|i| (y[i] > 0.0) as usize).collect();
    let lambda = 1e-2;
    let mut lr = LogisticRegression::new(lambda);
    lr.fit(x.view(), &labels).unwrap();
    let proba = lr.predict_proba(x.view()).unwrap();
    let w = lr.coefficients();
    let mut v: f64 = 0.0;
    for c in 0..2 {
        let err: Vec<f64> = (0..n).map(|i| proba[[i, c]] - f64::from(labels[i] == c)).collect();
        v = v.max(mean(&err).abs());
        for j in 0..p {
            v = v.max(((0..n).map(|i| err[i] * x[[i, j]]).sum::<f64>() / n as f64 + lambda * w[[c, j]]).abs());
        }
    }
    table.push_str(&format!("| logistic regression, $\\lambda = 0.01$ | gradient of the penalised log-loss | {v:.1e} ({} L-BFGS iterations) |\n", lr.iterations()));
    let _ = xbar;
    write(CHAPTER, "optimality.md", &table);
}

// ------------------------------------------------------------------ logistic regression

fn logistic_figures() {
    let (x, labels) = make_blobs(100, &[vec![-1.0, 0.0], vec![1.0, 0.8]], 1.0, 5);
    let mut lr = LogisticRegression::new(1e-3);
    lr.fit(x.view(), &labels).unwrap();
    let (xr, yr) = ((-4.5, 4.5), (-3.5, 4.0));
    let mut plot = Plot::new("Logistic regression: contours of P(class 1)", "x₁", "x₂");
    plot.x_range = Some(xr);
    plot.y_range = Some(yr);
    plot.legend = Legend::BottomRight;
    add_classes(&mut plot, &x, &labels, ["class 0", "class 1"]);
    for (level, style, label) in [(0.5, Style::Line, "P = 0.5"), (0.1, Style::Dashed, "P = 0.1 and 0.9"), (0.9, Style::Dashed, "")] {
        let f = |g: &R2D| lr.predict_proba(g.view()).unwrap().column(1).iter().map(|p| p - level).collect::<Vec<f64>>();
        add_contour(&mut plot, contour(&f, xr, yr, 80), label, style, 6);
    }
    write(CHAPTER, "logistic_contours.svg", &plot.render());

    // Separable, high-dimensional data: the penalty keeps the weights finite.
    let mut rng = MlRng::new(9);
    let make = |n: usize, rng: &mut MlRng| {
        let x = R2D::from_shape_fn((n, 30), |_| rng.gaussian(0.0, 1.0));
        let labels: Vec<usize> = (0..n).map(|i| (x[[i, 0]] + x[[i, 1]] - x[[i, 2]] + rng.gaussian(0.0, 1.0) > 0.0) as usize).collect();
        (x, labels)
    };
    let (xtr, ltr) = make(40, &mut rng);
    let (xte, lte) = make(2000, &mut rng);
    let log_loss = |m: &LogisticRegression, x: &R2D, l: &[usize]| {
        let p = m.predict_proba(x.view()).unwrap();
        -(0..l.len()).map(|i| p[[i, l[i]]].max(1e-300).ln()).sum::<f64>() / l.len() as f64
    };
    let mut table = String::from("| $\\lambda$ | L-BFGS iterations | $\\lVert W\\rVert$ | training log-loss | test log-loss | test accuracy |\n|---:|---:|---:|---:|---:|---:|\n");
    for lambda in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0] {
        let mut m = LogisticRegression::new(lambda);
        m.fit(xtr.view(), &ltr).unwrap();
        let norm = m.coefficients().iter().map(|v| v * v).sum::<f64>().sqrt();
        let acc = accuracy(&lte, &m.predict(xte.view()).unwrap()).unwrap();
        table.push_str(&format!("| {lambda} | {} | {norm:.1} | {:.4} | {:.3} | {acc:.3} |\n", m.iterations(), log_loss(&m, &xtr, &ltr), log_loss(&m, &xte, &lte)));
    }
    table.push_str("\n40 training points in 30 dimensions (3 informative); test on 2000 points. The limit is 500 iterations.\n");
    write(CHAPTER, "logistic_penalty.md", &table);
}

// ------------------------------------------------------------------ naive Bayes

/// Two classes sharing a correlated covariance `[[1, 0.8], [0.8, 1]]`.
fn correlated_classes(per_class: usize, rng: &mut MlRng) -> (R2D, Vec<usize>) {
    let means = [(0.0, 0.0), (1.0, -1.0)];
    let mut x = R2D::zeros((2 * per_class, 2));
    let mut labels = Vec::new();
    for (c, &(mx, my)) in means.iter().enumerate() {
        for i in 0..per_class {
            let (a, b) = (rng.gaussian(0.0, 1.0), rng.gaussian(0.0, 1.0));
            x[[c * per_class + i, 0]] = mx + a;
            x[[c * per_class + i, 1]] = my + 0.8 * a + 0.6 * b;
            labels.push(c);
        }
    }
    (x, labels)
}

fn naive_bayes_figures() {
    let mut rng = MlRng::new(14);
    let (x, labels) = correlated_classes(150, &mut rng);
    let mut nb = GaussianNaiveBayes::new();
    nb.fit(x.view(), &labels).unwrap();
    let mut lr = LogisticRegression::new(0.0);
    lr.fit(x.view(), &labels).unwrap();
    let (xr, yr) = ((-3.5, 4.5), (-4.5, 3.5));
    let mut plot = Plot::new("Correlated features: naive Bayes against the optimal boundary", "x₁", "x₂");
    plot.x_range = Some(xr);
    plot.y_range = Some(yr);
    plot.legend = Legend::TopRight;
    add_classes(&mut plot, &x, &labels, ["class 0", "class 1"]);
    // Bayes-optimal: shared covariance Σ, so the boundary is (μ₁ − μ₀)ᵀΣ⁻¹(x − (μ₀ + μ₁)/2) = 0.
    let bayes = |g: &R2D| -> Vec<f64> {
        let (s11, s12, s22) = (1.0, 0.8, 1.0);
        let det: f64 = s11 * s22 - s12 * s12;
        let (dx, dy) = (1.0, -1.0);
        let (a, b) = ((s22 * dx - s12 * dy) / det, (-s12 * dx + s11 * dy) / det);
        g.rows().into_iter().map(|r| a * (r[0] - 0.5) + b * (r[1] + 0.5)).collect()
    };
    add_contour(&mut plot, contour(&bayes, xr, yr, 60), "optimal (shared covariance)", Style::Dashed, 6);
    add_contour(&mut plot, contour(&|g: &R2D| proba_minus_half(nb.predict_proba(g.view()).unwrap()), xr, yr, 100), "Gaussian naive Bayes", Style::Line, 2);
    add_contour(&mut plot, contour(&|g: &R2D| proba_minus_half(lr.predict_proba(g.view()).unwrap()), xr, yr, 100), "logistic regression", Style::Line, 3);
    write(CHAPTER, "naive_bayes_boundary.svg", &plot.render());

    let (xte, lte) = correlated_classes(5000, &mut rng);
    let bayes_pred: Vec<usize> = bayes(&xte).iter().map(|&v| (v > 0.0) as usize).collect();
    let mut table = String::from("| classifier | test accuracy |\n|---|---:|\n");
    table.push_str(&format!("| optimal (true distributions) | {:.3} |\n", accuracy(&lte, &bayes_pred).unwrap()));
    table.push_str(&format!("| Gaussian naive Bayes | {:.3} |\n", accuracy(&lte, &nb.predict(xte.view()).unwrap()).unwrap()));
    table.push_str(&format!("| logistic regression | {:.3} |\n", accuracy(&lte, &lr.predict(xte.view()).unwrap()).unwrap()));
    write(CHAPTER, "naive_bayes_accuracy.md", &table);

    // Learning curves on synthetic documents: two topics over a 60-word vocabulary.
    let vocabulary = 60;
    let mut rng = MlRng::new(15);
    let distribution = |rng: &mut MlRng| -> Vec<f64> {
        let raw: Vec<f64> = (0..vocabulary).map(|_| rng.uniform().powi(3)).collect();
        let s: f64 = raw.iter().sum();
        raw.iter().map(|v| v / s).collect()
    };
    // Each topic draws 80% of its words from a shared vocabulary distribution.
    let shared = distribution(&mut rng);
    let topic: Vec<Vec<f64>> = (0..2).map(|_| distribution(&mut rng).iter().zip(&shared).map(|(t, s)| 0.2 * t + 0.8 * s).collect()).collect();
    let document = |c: usize, rng: &mut MlRng| -> Vec<f64> {
        let mut counts = vec![0.0; vocabulary];
        for _ in 0..12 {
            let u = rng.uniform();
            let mut acc = 0.0;
            let mut w = vocabulary - 1;
            for (k, p) in topic[c].iter().enumerate() {
                acc += p;
                if u < acc {
                    w = k;
                    break;
                }
            }
            counts[w] += 1.0;
        }
        counts
    };
    let corpus = |n: usize, rng: &mut MlRng| {
        let labels: Vec<usize> = (0..n).map(|i| i % 2).collect();
        let rows: Vec<Vec<f64>> = labels.iter().map(|&c| document(c, rng)).collect();
        (R2D::from_shape_fn((n, vocabulary), |(i, j)| rows[i][j]), labels)
    };
    let (xte, lte) = corpus(4000, &mut rng);
    let sizes = [4, 6, 10, 16, 25, 40, 64, 100, 160, 250, 400, 640, 1000];
    let reps = 30;
    let (mut nb_err, mut lr_err) = (Vec::new(), Vec::new());
    for &n in &sizes {
        let (mut a, mut b) = (0.0, 0.0);
        for _ in 0..reps {
            let (xtr, ltr) = corpus(n, &mut rng);
            let mut nb = MultinomialNaiveBayes::new(1.0);
            nb.fit(xtr.view(), &ltr).unwrap();
            a += (1.0 - accuracy(&lte, &nb.predict(xte.view()).unwrap()).unwrap()) / reps as f64;
            let mut lr = LogisticRegression::new(1.0 / n as f64);
            lr.fit(xtr.view(), &ltr).unwrap();
            b += (1.0 - accuracy(&lte, &lr.predict(xte.view()).unwrap()).unwrap()) / reps as f64;
        }
        nb_err.push((n as f64, a));
        lr_err.push((n as f64, b));
    }
    let mut plot = Plot::new("Classifying 12-word documents from two topics", "training documents", "test error");
    plot.log_x = true;
    plot.legend = Legend::TopRight;
    plot.series.push(Series::new("multinomial naive Bayes", nb_err, Style::LineMarkers, 2));
    plot.series.push(Series::new("logistic regression (λ = 1/n)", lr_err, Style::LineMarkers, 3));
    write(CHAPTER, "naive_bayes_learning.svg", &plot.render());
}

// ------------------------------------------------------------------ k-nearest neighbours

fn knn_figures() {
    let (x, labels) = moons(150, 1);
    for k in [1, 15, 101] {
        let mut m = KNearestNeighbors::new(k);
        m.fit(x.view(), &labels).unwrap();
        let score = |g: &R2D| proba_minus_half(m.predict_proba(g.view()).unwrap());
        boundary_plot(&format!("k-nearest neighbours, k = {k}"), &x, &labels, &score, MOON_X, MOON_Y, &format!("knn_{k}.svg"));
    }
    let ks: Vec<usize> = vec![1, 2, 3, 5, 7, 10, 15, 21, 31, 45, 65, 101, 151, 221];
    let reps = 20;
    let (mut train, mut test) = (vec![0.0; ks.len()], vec![0.0; ks.len()]);
    for r in 0..reps {
        let (xtr, ltr) = moons(150, 100 + r);
        let (xte, lte) = moons(500, 200 + r);
        for (t, &k) in ks.iter().enumerate() {
            let mut m = KNearestNeighbors::new(k);
            m.fit(xtr.view(), &ltr).unwrap();
            train[t] += (1.0 - accuracy(&ltr, &m.predict(xtr.view()).unwrap()).unwrap()) / reps as f64;
            test[t] += (1.0 - accuracy(&lte, &m.predict(xte.view()).unwrap()).unwrap()) / reps as f64;
        }
    }
    let mut plot = Plot::new("k-nearest neighbours on the moons: error against k", "k", "error rate");
    plot.log_x = true;
    plot.legend = Legend::TopLeft;
    plot.series.push(Series::new("test", ks.iter().map(|&k| k as f64).zip(test).collect(), Style::LineMarkers, 1));
    plot.series.push(Series::new("training", ks.iter().map(|&k| k as f64).zip(train).collect(), Style::LineMarkers, 0));
    write(CHAPTER, "knn_error.svg", &plot.render());
}

// ------------------------------------------------------------------ trees and forests

fn tree_figures() {
    let (x, labels) = moons(150, 1);
    let mut shallow = DecisionTree::new(2);
    shallow.fit(x.view(), &labels).unwrap();
    boundary_plot("Decision tree of depth 2", &x, &labels, &|g: &R2D| proba_minus_half(shallow.predict_proba(g.view()).unwrap()), MOON_X, MOON_Y, "tree_depth2.svg");
    let mut deep = DecisionTree::new(50);
    deep.fit(x.view(), &labels).unwrap();
    boundary_plot(&format!("Fully grown decision tree (depth {})", deep.depth()), &x, &labels, &|g: &R2D| proba_minus_half(deep.predict_proba(g.view()).unwrap()), MOON_X, MOON_Y, "tree_deep.svg");
    let mut forest = RandomForest::new(200, 50, 7);
    forest.fit(x.view(), &labels).unwrap();
    boundary_plot("Random forest of 200 trees", &x, &labels, &|g: &R2D| proba_minus_half(forest.predict_proba(g.view()).unwrap()), MOON_X, MOON_Y, "forest.svg");

    // Accuracy against the number of trees, with out-of-bag estimates, on 4 informative + 6 noise features.
    let with_noise = |per_class: usize, seed: u64| {
        let (x, l) = make_classification(per_class, seed);
        let mut rng = MlRng::new(seed + 1000);
        (R2D::from_shape_fn((x.nrows(), 10), |(i, j)| if j < 4 { x[[i, j]] } else { rng.gaussian(0.0, 1.0) }), l)
    };
    let (xtr, ltr) = with_noise(100, 3);
    let (xte, lte) = with_noise(2000, 4);
    let mut tree = DecisionTree::new(50);
    tree.fit(xtr.view(), &ltr).unwrap();
    let tree_acc = accuracy(&lte, &tree.predict(xte.view()).unwrap()).unwrap();
    let counts = [1usize, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300];
    let (mut test, mut oob) = (Vec::new(), Vec::new());
    for &t in &counts {
        let mut f = RandomForest::new(t, 50, 11);
        f.fit(xtr.view(), &ltr).unwrap();
        test.push((t as f64, accuracy(&lte, &f.predict(xte.view()).unwrap()).unwrap()));
        if let Some(a) = f.oob_accuracy() {
            oob.push((t as f64, a));
        }
    }
    let mut plot = Plot::new("Random forest: accuracy against the number of trees", "trees", "accuracy");
    plot.log_x = true;
    plot.legend = Legend::BottomRight;
    plot.series.push(Series::new("test accuracy", test, Style::LineMarkers, 0));
    plot.series.push(Series::new("out-of-bag estimate", oob, Style::LineMarkers, 1));
    plot.series.push(Series::new(&format!("single deep tree ({tree_acc:.3})"), vec![(1.0, tree_acc), (300.0, tree_acc)], Style::Dashed, 7));
    write(CHAPTER, "forest_trees.svg", &plot.render());

    let mut f = RandomForest::new(300, 50, 11);
    f.fit(xtr.view(), &ltr).unwrap();
    let mut table = String::from("| feature | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n| share of splits |");
    for v in f.feature_importances() {
        table.push_str(&format!(" {v:.3} |"));
    }
    table.push_str("\n\nFeatures 0–3 carry the class information; 4–9 are pure noise.\n");
    write(CHAPTER, "forest_importances.md", &table);
}

// ------------------------------------------------------------------ support vector machines

fn svm_figures() {
    // A linear machine on two nearly separable blobs: boundary, margins and support vectors.
    let (x, labels) = make_blobs(40, &[vec![-1.5, -1.0], vec![1.5, 1.0]], 0.9, 2);
    let mut svm = SupportVectorMachine::new(Kernel::Linear, 1.0);
    svm.fit(x.view(), &labels).unwrap();
    let (xr, yr) = ((-4.5, 4.5), (-4.0, 4.0));
    let mut plot = Plot::new(&format!("Linear support vector machine, C = 1 ({} support vectors)", svm.support_vector_count()), "x₁", "x₂");
    plot.x_range = Some(xr);
    plot.y_range = Some(yr);
    plot.legend = Legend::BottomRight;
    add_classes(&mut plot, &x, &labels, ["class 0", "class 1"]);
    let f = svm.decision_function(x.view()).unwrap();
    let sv: Vec<(f64, f64)> = (0..x.nrows()).filter(|&i| (if labels[i] == 1 { f[i] } else { -f[i] }) <= 1.0 + 1e-3).map(|i| (x[[i, 0]], x[[i, 1]])).collect();
    plot.series.push(Series::new("support vectors (y f(x) ≤ 1)", sv, Style::Markers, 4));
    add_contour(&mut plot, contour(&|g: &R2D| svm.decision_function(g.view()).unwrap(), xr, yr, 60), "f(x) = 0", Style::Line, 6);
    add_contour(&mut plot, contour(&|g: &R2D| svm.decision_function(g.view()).unwrap().iter().map(|v| v - 1.0).collect(), xr, yr, 60), "f(x) = ±1 (margins)", Style::Dashed, 6);
    add_contour(&mut plot, contour(&|g: &R2D| svm.decision_function(g.view()).unwrap().iter().map(|v| v + 1.0).collect(), xr, yr, 60), "", Style::Dashed, 6);
    write(CHAPTER, "svm_margin.svg", &plot.render());

    let (x, labels) = moons(150, 1);
    for gamma in [0.3, 3.0, 100.0] {
        let mut m = SupportVectorMachine::new(Kernel::Rbf { gamma }, 1.0);
        m.fit(x.view(), &labels).unwrap();
        boundary_plot(&format!("RBF kernel, γ = {gamma}, C = 1 ({} support vectors)", m.support_vector_count()), &x, &labels, &|g: &R2D| m.decision_function(g.view()).unwrap(), MOON_X, MOON_Y, &format!("svm_rbf_{gamma}.svg"));
    }

    // Test accuracy over a grid of C and γ.
    let (xtr, ltr) = moons(100, 5);
    let (xte, lte) = moons(1000, 6);
    let (cs, gammas) = ([0.1, 1.0, 10.0, 100.0], [0.1, 1.0, 10.0, 100.0]);
    let mut table = String::from("| | γ = 0.1 | γ = 1 | γ = 10 | γ = 100 |\n|---|---:|---:|---:|---:|\n");
    for &c in &cs {
        table.push_str(&format!("| C = {c} |"));
        for &gamma in &gammas {
            let mut m = SupportVectorMachine::new(Kernel::Rbf { gamma }, c);
            m.fit(xtr.view(), &ltr).unwrap();
            table.push_str(&format!(" {:.3} ({}) |", accuracy(&lte, &m.predict(xte.view()).unwrap()).unwrap(), m.support_vector_count()));
        }
        table.push('\n');
    }
    table.push_str("\nTest accuracy on the moons (200 training points, 2000 test points), with the number of support vectors in brackets.\n");
    write(CHAPTER, "svm_grid.md", &table);

    // Support vector regression with its ε-tube.
    let mut rng = MlRng::new(6);
    let (xs, y) = sine_data(60, &mut rng);
    let x = R2D::from_shape_fn((xs.len(), 1), |(i, _)| xs[i]);
    let eps = 0.2;
    let mut svr = SupportVectorRegression::new(Kernel::Rbf { gamma: 5.0 }, 10.0, eps).with_limits(20_000, 1e-9);
    svr.fit(x.view(), y.view()).unwrap();
    let grid: Vec<f64> = (0..=300).map(|i| -1.0 + 2.0 * i as f64 / 300.0).collect();
    let g = R2D::from_shape_fn((grid.len(), 1), |(i, _)| grid[i]);
    let fit = svr.predict(g.view()).unwrap();
    let at_data = svr.predict(x.view()).unwrap();
    let mut plot = Plot::new(&format!("ε-insensitive support vector regression, ε = {eps} ({} support vectors)", svr.support_vector_count()), "x", "y");
    plot.legend = Legend::BottomRight;
    plot.series.push(Series::new("fit", grid.iter().copied().zip(fit.iter().copied()).collect(), Style::Line, 0));
    plot.series.push(Series::new("ε-tube", grid.iter().copied().zip(fit.iter().map(|v| v + eps)).collect(), Style::Dashed, 0));
    plot.series.push(Series::new("", grid.iter().copied().zip(fit.iter().map(|v| v - eps)).collect(), Style::Dashed, 0));
    let (inside, outside): (Vec<usize>, Vec<usize>) = (0..xs.len()).partition(|&i| (y[i] - at_data[i]).abs() < eps - 1e-6);
    plot.series.push(Series::new("inside the tube", inside.iter().map(|&i| (xs[i], y[i])).collect(), Style::Markers, 7));
    plot.series.push(Series::new("on or outside (support vectors)", outside.iter().map(|&i| (xs[i], y[i])).collect(), Style::Markers, 1));
    write(CHAPTER, "svr.svg", &plot.render());
}

// ------------------------------------------------------------------ comparisons

fn comparison_tables() {
    let datasets: Vec<(&str, R2D, Vec<usize>)> = {
        let (a, la) = make_classification(200, 5);
        let (b, lb) = make_moons(300, 0.25, 5);
        let (c, lc) = make_circles(300, 0.08, 0.5, 5);
        vec![("three Gaussian classes, 4D", a, la), ("moons", b, lb), ("circles", c, lc)]
    };
    type Factory = Box<dyn Fn() -> Box<dyn Classifier>>;
    let models: Vec<(&str, Factory)> = vec![
        ("logistic regression", Box::new(|| Box::new(LogisticRegression::new(1e-3)) as Box<dyn Classifier>)),
        ("Gaussian naive Bayes", Box::new(|| Box::new(GaussianNaiveBayes::new()))),
        ("k-nearest neighbours, k = 15", Box::new(|| Box::new(KNearestNeighbors::new(15)))),
        ("decision tree, depth 6", Box::new(|| Box::new(DecisionTree::new(6)))),
        ("random forest, 100 trees", Box::new(|| Box::new(RandomForest::new(100, 50, 3)))),
        ("SVM, RBF kernel γ = 1, C = 1", Box::new(|| Box::new(SupportVectorMachine::new(Kernel::Rbf { gamma: 1.0 }, 1.0)))),
    ];
    let mut table = String::from("| classifier |");
    for (name, _, _) in &datasets {
        table.push_str(&format!(" {name} |"));
    }
    table.push_str(" fit time (moons) |\n|---|---:|---:|---:|---:|\n");
    let split: Vec<_> = datasets
        .iter()
        .map(|(_, x, l)| {
            let (train, test) = train_test_split(x.nrows(), 0.5, 11).unwrap();
            let mut scaler = StandardScaler::new();
            let xtr = scaler.fit_transform(select_rows(x.view(), &train).view()).unwrap();
            let xte = scaler.transform(select_rows(x.view(), &test).view()).unwrap();
            (xtr, select_labels(l, &train), xte, select_labels(l, &test))
        })
        .collect();
    for (name, make) in &models {
        table.push_str(&format!("| {name} |"));
        for (xtr, ltr, xte, lte) in &split {
            let mut m = make();
            m.fit(xtr.view(), ltr).unwrap();
            table.push_str(&format!(" {:.3} |", accuracy(lte, &m.predict(xte.view()).unwrap()).unwrap()));
        }
        let (xtr, ltr, _, _) = &split[1];
        let t = time(|| {
            let mut m = make();
            m.fit(xtr.view(), ltr).unwrap();
        });
        table.push_str(&if t < 1e-3 { format!(" {:.0} µs |\n", t * 1e6) } else { format!(" {:.1} ms |\n", t * 1e3) });
    }
    table.push_str("\nTest accuracy with half of each data set for training (standardised features); the fit time is for 300 points in 2D.\n");
    write(CHAPTER, "comparison.md", &table);

    // Confusion matrix and per-class scores for the forest on the three-class problem.
    let (xtr, ltr, xte, lte) = &split[0];
    let mut f = RandomForest::new(100, 50, 3);
    f.fit(xtr.view(), ltr).unwrap();
    let pred = f.predict(xte.view()).unwrap();
    let m = confusion_matrix(lte, &pred, 3).unwrap();
    let prf = precision_recall_f1(lte, &pred, 3).unwrap();
    let mut table = String::from("| true class | predicted 0 | predicted 1 | predicted 2 | precision | recall | F1 |\n|---|---:|---:|---:|---:|---:|---:|\n");
    for c in 0..3 {
        table.push_str(&format!("| {c} | {} | {} | {} | {:.3} | {:.3} | {:.3} |\n", m[c][0], m[c][1], m[c][2], prf[c].0, prf[c].1, prf[c].2));
    }
    write(CHAPTER, "confusion.md", &table);
}
