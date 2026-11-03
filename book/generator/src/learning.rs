//! Figures and tables for the chapter on neural networks, ensembles, model selection and hidden
//! Markov models.

use crate::linear_algebra::time;
use crate::machine_learning::{MOON_X, MOON_Y, boundary_figure, moons, proba_minus_half};
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_core::{R1D, R2D};
use lib_jsl_machine_learning::common::{Classifier, MlRng, Regressor};
use lib_jsl_machine_learning::datasets::{make_circles, make_classification, make_regression, make_sine};
use lib_jsl_machine_learning::ensembles::{BoostingParameters, GradientBoostingClassifier, GradientBoostingRegressor};
use lib_jsl_machine_learning::kernels::Kernel;
use lib_jsl_machine_learning::metrics::{accuracy, mean_squared_error};
use lib_jsl_machine_learning::model_selection::{cross_validate_classifier, cross_validate_regressor, grid_search, k_fold, mean_score, stratified_k_fold};
use lib_jsl_machine_learning::neural_network::{Activation, MlpClassifier, MlpParameters, MlpRegressor};
use lib_jsl_machine_learning::supervised::classification::{DecisionTree, HiddenMarkovModel, KNearestNeighbors, RandomForest, SupportVectorMachine};
use lib_jsl_machine_learning::supervised::regression::Ridge;
use ndarray::arr2;

const CHAPTER: &str = "learning";

pub fn generate() {
    mlp_figures();
    boosting_figures();
    model_selection_figures();
    hmm_figures();
    comparison_table();
}

fn md_table(header: &[&str], rows: &[Vec<String>]) -> String {
    let mut s = format!("| {} |\n|{}\n", header.join(" | "), header.iter().enumerate().map(|(i, _)| if i == 0 { "---|" } else { "---:|" }).collect::<String>());
    for r in rows {
        s += &format!("| {} |\n", r.join(" | "));
    }
    s
}

/// Moons for training (seed `a`) and a large independent test set.
fn moons_pair(n: usize, a: u64) -> (R2D, Vec<usize>, R2D, Vec<usize>) {
    let (x, l) = moons(n, a);
    let (t, tl) = moons(1000, a + 1000);
    (x, l, t, tl)
}

// ------------------------------------------------------------------ multilayer perceptrons

fn mlp_figures() {
    let (x, l) = moons(150, 1);
    for (hidden, file) in [(vec![2], "mlp_hidden_2.svg"), (vec![8], "mlp_hidden_8.svg"), (vec![64, 64], "mlp_hidden_64_64.svg")] {
        let mut m = MlpClassifier::new(MlpParameters { hidden: hidden.clone(), epochs: 500, seed: 1, ..Default::default() });
        m.fit(x.view(), &l).unwrap();
        let score = |g: &R2D| proba_minus_half(m.predict_proba(g.view()).unwrap());
        let name = hidden.iter().map(|h| h.to_string()).collect::<Vec<_>>().join(" × ");
        let plot = boundary_figure(&format!("MLP, hidden layers {name}"), &x, &l, &score, MOON_X, MOON_Y);
        write(CHAPTER, file, &plot.render());
    }
    // Training loss by epoch for several learning rates.
    let mut plot = Plot::new("Training cross-entropy by epoch (Adam, hidden 32 × 32)", "epoch", "mean training loss");
    plot.log_y = true;
    for (k, rate) in [0.0003, 0.003, 0.03, 0.3].into_iter().enumerate() {
        let mut m = MlpClassifier::new(MlpParameters { learning_rate: rate, epochs: 300, seed: 1, ..Default::default() });
        m.fit(x.view(), &l).unwrap();
        plot.series.push(Series::new(&format!("learning rate {rate}"), m.training_loss.iter().enumerate().map(|(i, &v)| (i as f64 + 1.0, v)).collect(), Style::Line, k));
    }
    plot.legend = Legend::TopRight;
    write(CHAPTER, "mlp_learning_rate.svg", &plot.render());
    // Regression on a sine: activation shapes the fit.
    let (sx, sy) = make_sine(60, 0.1, 3);
    let grid = R2D::from_shape_fn((300, 1), |(i, _)| -1.0 + 8.3 * i as f64 / 299.0);
    let mut plot = Plot::new("MLP regression of a noisy sine (hidden 16 × 16)", "x", "y");
    plot.series.push(Series::new("data", (0..sx.nrows()).map(|i| (sx[[i, 0]], sy[i])).collect(), Style::Markers, 7));
    plot.series.push(Series::new("sin x", (0..300).map(|i| (grid[[i, 0]], grid[[i, 0]].sin())).collect(), Style::Dashed, 6));
    for (k, (act, name)) in [(Activation::Relu, "ReLU"), (Activation::Tanh, "tanh")].into_iter().enumerate() {
        let mut m = MlpRegressor::new(MlpParameters { hidden: vec![16, 16], activation: act, epochs: 1500, batch_size: 16, seed: 2, ..Default::default() });
        m.fit(sx.view(), sy.view()).unwrap();
        let p = m.predict(grid.view()).unwrap();
        plot.series.push(Series::new(name, (0..300).map(|i| (grid[[i, 0]], p[i])).collect(), Style::Line, k));
    }
    plot.y_range = Some((-1.6, 1.6));
    plot.legend = Legend::BottomLeft;
    write(CHAPTER, "mlp_sine.svg", &plot.render());
    // Activations and data sets.
    let (cx, cl) = make_circles(200, 0.08, 0.5, 4);
    let (ctx, ctl) = make_circles(500, 0.08, 0.5, 5);
    let (mx, ml, mtx, mtl) = moons_pair(200, 6);
    let mut rows = Vec::new();
    for (act, name) in [(Activation::Relu, "ReLU"), (Activation::Tanh, "tanh"), (Activation::Sigmoid, "sigmoid")] {
        let mut r = vec![name.to_string()];
        for (x, l, tx, tl) in [(&mx, &ml, &mtx, &mtl), (&cx, &cl, &ctx, &ctl)] {
            let mut m = MlpClassifier::new(MlpParameters { activation: act, epochs: 200, seed: 3, ..Default::default() });
            let t = time(|| {
                let mut m2 = m.clone();
                m2.fit(x.view(), l).unwrap();
            });
            m.fit(x.view(), l).unwrap();
            r.push(format!("{:.3}", accuracy(tl, &m.predict(tx.view()).unwrap()).unwrap()));
            r.push(format!("{:.3}", m.training_loss.last().unwrap()));
            if r.len() == 3 {
                r.push(format!("{:.0} ms", t * 1e3));
            }
        }
        rows.push(r);
    }
    write(CHAPTER, "mlp_activations.md", &md_table(&["activation (32 × 32, 200 epochs)", "moons: test accuracy", "final training loss", "training time", "circles: test accuracy", "final training loss"], &rows));
    // Weight decay on a small noisy problem.
    let (wx, wl, wtx, wtl) = moons_pair(60, 9);
    let mut rows = Vec::new();
    for decay in [0.0, 1e-3, 1e-2, 1e-1, 1.0] {
        let mut m = MlpClassifier::new(MlpParameters { hidden: vec![64, 64], weight_decay: decay, epochs: 1500, seed: 4, ..Default::default() });
        m.fit(wx.view(), &wl).unwrap();
        let train = accuracy(&wl, &m.predict(wx.view()).unwrap()).unwrap();
        let test = accuracy(&wtl, &m.predict(wtx.view()).unwrap()).unwrap();
        rows.push(vec![format!("{decay}"), format!("{train:.3}"), format!("{test:.3}")]);
    }
    write(CHAPTER, "mlp_weight_decay.md", &md_table(&["weight decay λ (64 × 64, 120 training points)", "training accuracy", "test accuracy"], &rows));
}

// ------------------------------------------------------------------ gradient boosting

fn boosting_figures() {
    let (sx, sy) = make_sine(200, 0.15, 3);
    let grid = R2D::from_shape_fn((400, 1), |(i, _)| 6.3 * i as f64 / 399.0);
    let mut plot = Plot::new("Gradient boosting of a noisy sine (depth-2 trees, ν = 0.1)", "x", "y");
    plot.series.push(Series::new("data", (0..sx.nrows()).map(|i| (sx[[i, 0]], sy[i])).collect(), Style::Markers, 7));
    for (k, stages) in [1usize, 10, 100].into_iter().enumerate() {
        let mut gb = GradientBoostingRegressor::new(BoostingParameters { stages, max_depth: 2, learning_rate: 0.1, ..Default::default() });
        gb.fit(sx.view(), sy.view()).unwrap();
        let p = gb.predict(grid.view()).unwrap();
        plot.series.push(Series::new(&format!("{stages} stage{}", if stages == 1 { "" } else { "s" }), (0..400).map(|i| (grid[[i, 0]], p[i])).collect(), Style::Line, k));
    }
    plot.legend = Legend::BottomLeft;
    write(CHAPTER, "gb_stages.svg", &plot.render());
    // Shrinkage: test error against stages.
    let (rx, ry, _, _) = make_regression(600, 8, 1.0, 5);
    // A nonlinear target: square some features and add an interaction.
    let ry = R1D::from_iter((0..600).map(|i| ry[i] + 3.0 * rx[[i, 0]] * rx[[i, 1]] + 2.0 * rx[[i, 2]].powi(2)));
    let (trx, try_) = (rx.slice(ndarray::s![..300, ..]).to_owned(), ry.slice(ndarray::s![..300]).to_owned());
    let (tex, tey) = (rx.slice(ndarray::s![300.., ..]).to_owned(), ry.slice(ndarray::s![300..]).to_owned());
    let stage_grid: Vec<usize> = vec![1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000];
    let mut plot = Plot::new("Test error against boosting stages (depth-3 trees)", "stages", "test mean squared error");
    plot.log_x = true;
    plot.log_y = true;
    let mut rows = Vec::new();
    for (k, rate) in [1.0, 0.3, 0.1, 0.03].into_iter().enumerate() {
        let mut pts = Vec::new();
        let mut best = (f64::INFINITY, 0);
        for &m in &stage_grid {
            let mut gb = GradientBoostingRegressor::new(BoostingParameters { stages: m, learning_rate: rate, ..Default::default() });
            gb.fit(trx.view(), try_.view()).unwrap();
            let e = mean_squared_error(tey.view(), gb.predict(tex.view()).unwrap().view()).unwrap();
            if e < best.0 {
                best = (e, m);
            }
            pts.push((m as f64, e));
        }
        rows.push(vec![format!("{rate}"), format!("{:.2}", best.0), format!("{}", best.1), format!("{:.2}", pts.last().unwrap().1)]);
        plot.series.push(Series::new(&format!("ν = {rate}"), pts, Style::LineMarkers, k));
    }
    plot.legend = Legend::BottomLeft;
    write(CHAPTER, "gb_shrinkage.svg", &plot.render());
    write(CHAPTER, "gb_shrinkage.md", &md_table(&["learning rate ν", "best test MSE", "at stages", "test MSE at 1000 stages"], &rows));
    // Classification boundary.
    let (x, l) = moons(150, 1);
    let mut gb = GradientBoostingClassifier::new(BoostingParameters { stages: 100, max_depth: 2, ..Default::default() });
    gb.fit(x.view(), &l).unwrap();
    let score = |g: &R2D| proba_minus_half(gb.predict_proba(g.view()).unwrap());
    write(CHAPTER, "gb_moons.svg", &boundary_figure("Gradient boosting, 100 stages of depth-2 trees", &x, &l, &score, MOON_X, MOON_Y).render());
    // Subsampling.
    let mut rows = Vec::new();
    for sub in [1.0, 0.8, 0.5, 0.2] {
        let mut errs = Vec::new();
        for seed in 0..5u64 {
            let mut gb = GradientBoostingRegressor::new(BoostingParameters { stages: 300, learning_rate: 0.1, subsample: sub, seed, ..Default::default() });
            gb.fit(trx.view(), try_.view()).unwrap();
            errs.push(mean_squared_error(tey.view(), gb.predict(tex.view()).unwrap().view()).unwrap());
        }
        let mean = errs.iter().sum::<f64>() / errs.len() as f64;
        rows.push(vec![format!("{sub}"), format!("{mean:.2}")]);
    }
    write(CHAPTER, "gb_subsample.md", &md_table(&["subsample (300 stages, ν = 0.1)", "test MSE (mean of 5 seeds)"], &rows));
}

// ------------------------------------------------------------------ model selection

fn model_selection_figures() {
    // Ridge on polynomial features: CV curve with one-standard-error band, against the true test error.
    let mut rng = MlRng::new(7);
    let n = 40;
    let degree = 12;
    let xs: Vec<f64> = (0..n).map(|_| rng.range(-1.0, 1.0)).collect();
    let f = |x: f64| (3.0 * x).sin();
    let design = |xs: &[f64]| R2D::from_shape_fn((xs.len(), degree), |(i, j)| xs[i].powi(j as i32 + 1));
    let x = design(&xs);
    let y = R1D::from_iter(xs.iter().map(|&v| f(v) + rng.gaussian(0.0, 0.3)));
    let test_x: Vec<f64> = (0..2000).map(|_| rng.range(-1.0, 1.0)).collect();
    let tx = design(&test_x);
    let ty = R1D::from_iter(test_x.iter().map(|&v| f(v) + rng.gaussian(0.0, 0.3)));
    let alphas: Vec<f64> = (0..25).map(|i| 10f64.powf(-7.0 + 0.375 * i as f64)).collect();
    let splits = k_fold(n, 10, Some(1)).unwrap();
    let (mut cv, mut upper, mut lower, mut truth) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut best = (f64::INFINITY, 0.0, 0.0);
    let mut per_alpha = Vec::new();
    for &a in &alphas {
        let scores = cross_validate_regressor(|| Ridge::new(a), x.view(), y.view(), &splits, mean_squared_error).unwrap();
        let m = mean_score(&scores);
        let se = (scores.iter().map(|s| (s - m).powi(2)).sum::<f64>() / (scores.len() as f64 - 1.0)).sqrt() / (scores.len() as f64).sqrt();
        cv.push((a, m));
        upper.push((a, m + se));
        lower.push((a, (m - se).max(1e-3)));
        let mut r = Ridge::new(a);
        r.fit(x.view(), y.view()).unwrap();
        truth.push((a, mean_squared_error(ty.view(), r.predict(tx.view()).unwrap().view()).unwrap()));
        if m < best.0 {
            best = (m, a, se);
        }
        per_alpha.push((a, m));
    }
    // The one-standard-error rule: the largest alpha whose CV error is within one SE of the best.
    let one_se = per_alpha.iter().filter(|(_, m)| *m <= best.0 + best.2).map(|(a, _)| *a).fold(0.0, f64::max);
    let mut plot = Plot::new("10-fold cross-validation of ridge regression (degree-12 polynomial, 40 points)", "ridge penalty α", "mean squared error");
    plot.log_x = true;
    plot.log_y = true;
    plot.series.push(Series::new("cross-validation estimate", cv, Style::LineMarkers, 0));
    plot.series.push(Series::new("± one standard error", upper, Style::Dashed, 0));
    plot.series.push(Series::new("", lower, Style::Dashed, 0));
    plot.series.push(Series::new("true test error", truth.clone(), Style::Line, 1));
    let ymax = truth.iter().map(|p| p.1).fold(0.0, f64::max);
    plot.series.push(Series::new("CV minimum", vec![(best.1, 0.05), (best.1, ymax)], Style::Line, 6));
    plot.series.push(Series::new("one-SE rule", vec![(one_se, 0.05), (one_se, ymax)], Style::Dashed, 6));
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "cv_ridge.svg", &plot.render());
    let true_at = |a: f64| truth.iter().find(|p| p.0 == a).unwrap().1;
    let best_true = truth.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let rows = vec![
        vec!["minimum CV error".to_string(), format!("{:.2e}", best.1), format!("{:.4}", best.0), format!("{:.4}", true_at(best.1))],
        vec!["one-standard-error rule".to_string(), format!("{one_se:.2e}"), format!("{:.4}", per_alpha.iter().find(|p| p.0 == one_se).unwrap().1), format!("{:.4}", true_at(one_se))],
        vec!["best possible (oracle)".to_string(), String::new(), String::new(), format!("{best_true:.4}")],
    ];
    write(CHAPTER, "cv_choice.md", &md_table(&["choice of α", "α", "CV estimate", "true test error"], &rows));
    // Bias and variance of the CV estimate itself.
    let reps = 200;
    let mut rows = Vec::new();
    let n = 60;
    let ks = [2usize, 5, 10, 60];
    let mut est: Vec<Vec<f64>> = vec![Vec::new(); ks.len()];
    let mut true_err = Vec::new();
    let (big, bigl) = moons(2500, 999);
    for r in 0..reps {
        let (x, l) = moons(n / 2, 5000 + r as u64);
        let mut m = KNearestNeighbors::new(7);
        m.fit(x.view(), &l).unwrap();
        true_err.push(1.0 - accuracy(&bigl, &m.predict(big.view()).unwrap()).unwrap());
        for (j, &k) in ks.iter().enumerate() {
            let splits = k_fold(n, k, Some(r as u64)).unwrap();
            let s = cross_validate_classifier(|| KNearestNeighbors::new(7), x.view(), &l, &splits, accuracy).unwrap();
            // Pool the folds: total errors over all test points.
            let err = splits.iter().zip(&s).map(|((_, t), a)| (1.0 - a) * t.len() as f64).sum::<f64>() / n as f64;
            est[j].push(err);
        }
    }
    let stats = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (m, (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt())
    };
    let (tm, ts) = stats(&true_err);
    for (j, &k) in ks.iter().enumerate() {
        let (m, s) = stats(&est[j]);
        let rmse = (est[j].iter().zip(&true_err).map(|(e, t)| (e - t).powi(2)).sum::<f64>() / reps as f64).sqrt();
        rows.push(vec![if k == n { format!("{k} (leave-one-out)") } else { format!("{k}") }, format!("{m:.4}"), format!("{s:.4}"), format!("{rmse:.4}")]);
    }
    rows.push(vec!["true error of the model fitted to all 60 points".to_string(), format!("{tm:.4}"), format!("{ts:.4}"), String::new()]);
    write(CHAPTER, "cv_folds.md", &md_table(&["folds (7-NN, 60 points, 200 data sets)", "mean estimate", "standard deviation", "RMS difference from the true error"], &rows));
    // Stratification on imbalanced labels.
    let labels: Vec<usize> = (0..100).map(|i| usize::from(i % 10 == 0)).collect();
    let mut rows = Vec::new();
    for (name, splits) in [("k-fold, shuffled", k_fold(100, 10, Some(3)).unwrap()), ("stratified k-fold", stratified_k_fold(&labels, 10, 3).unwrap())] {
        let counts: Vec<usize> = splits.iter().map(|(_, t)| t.iter().filter(|&&i| labels[i] == 1).count()).collect();
        rows.push(vec![name.to_string(), format!("{counts:?}"), format!("{}", counts.iter().filter(|&&c| c == 0).count())]);
    }
    write(CHAPTER, "cv_stratified.md", &md_table(&["10 folds of 100 points, 10 in the minority class", "minority points in each test fold", "folds with none"], &rows));
    // Selection bias: tuning on pure noise.
    let mut rows = Vec::new();
    let mut best_cv = Vec::new();
    let mut fresh = Vec::new();
    for r in 0..50u64 {
        let mut rng = MlRng::new(100 + r);
        let x = R2D::from_shape_fn((60, 10), |_| rng.gaussian(0.0, 1.0));
        let l: Vec<usize> = (0..60).map(|_| rng.below(2)).collect();
        let tx = R2D::from_shape_fn((1000, 10), |_| rng.gaussian(0.0, 1.0));
        let tl: Vec<usize> = (0..1000).map(|_| rng.below(2)).collect();
        let folds = stratified_k_fold(&l, 5, r).unwrap();
        let ks: Vec<usize> = (1..=40).collect();
        let search = grid_search(&ks, |&k| Ok(mean_score(&cross_validate_classifier(|| KNearestNeighbors::new(k), x.view(), &l, &folds, accuracy)?)), true).unwrap();
        best_cv.push(search.best_score);
        let mut m = KNearestNeighbors::new(search.best);
        m.fit(x.view(), &l).unwrap();
        fresh.push(accuracy(&tl, &m.predict(tx.view()).unwrap()).unwrap());
    }
    let (bm, bs) = stats(&best_cv);
    let (fm, fs) = stats(&fresh);
    rows.push(vec!["best cross-validated accuracy over k = 1, …, 40".to_string(), format!("{bm:.3}"), format!("{bs:.3}")]);
    rows.push(vec!["accuracy of the chosen k on new data".to_string(), format!("{fm:.3}"), format!("{fs:.3}")]);
    write(CHAPTER, "cv_selection_bias.md", &md_table(&["random labels, 60 points, 50 repetitions", "mean", "standard deviation"], &rows));
}

// ------------------------------------------------------------------ hidden Markov models

/// The occasionally dishonest casino: a fair die and a loaded one (six with probability 1/2).
fn casino() -> HiddenMarkovModel {
    let fair = [1.0 / 6.0; 6];
    let loaded = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5];
    HiddenMarkovModel::new(vec![0.5, 0.5], arr2(&[[0.95, 0.05], [0.1, 0.9]]), arr2(&[fair, loaded])).unwrap()
}

fn hmm_figures() {
    let model = casino();
    let (states, rolls) = model.sample(300, 3);
    let (path, _) = model.viterbi(&rolls).unwrap();
    let post = model.posteriors(&rolls).unwrap();
    let mut plot = Plot::new("Occasionally dishonest casino: 300 rolls", "roll", "loaded");
    plot.series.push(Series::new("true die (1 = loaded)", states.iter().enumerate().map(|(t, &s)| (t as f64, s as f64)).collect(), Style::Line, 7));
    plot.series.push(Series::new("P(loaded | all rolls)", (0..300).map(|t| (t as f64, post[[t, 1]])).collect(), Style::Line, 0));
    plot.series.push(Series::new("Viterbi path", path.iter().enumerate().map(|(t, &s)| (t as f64, 1.08 * s as f64 - 0.04)).collect(), Style::Dashed, 1));
    plot.y_range = Some((-0.1, 1.25));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "hmm_casino.svg", &plot.render());
    // Decoding accuracy over many sequences.
    let (mut vit, mut postd, mut base) = (0usize, 0usize, 0usize);
    let mut total = 0usize;
    for seed in 0..100u64 {
        let (s, o) = model.sample(300, 1000 + seed);
        let (p, _) = model.viterbi(&o).unwrap();
        let q = model.posteriors(&o).unwrap();
        vit += s.iter().zip(&p).filter(|(a, b)| a == b).count();
        postd += s.iter().enumerate().filter(|(t, a)| **a == usize::from(q[[*t, 1]] > 0.5)).count();
        base += s.iter().filter(|&&a| a == 0).count();
        total += s.len();
    }
    let rows = vec![
        vec!["always guess fair".to_string(), format!("{:.3}", base as f64 / total as f64)],
        vec!["Viterbi path (most likely sequence)".to_string(), format!("{:.3}", vit as f64 / total as f64)],
        vec!["posterior decoding (most likely state at each roll)".to_string(), format!("{:.3}", postd as f64 / total as f64)],
    ];
    write(CHAPTER, "hmm_decoding.md", &md_table(&["decoder (100 sequences of 300 rolls)", "fraction of rolls labelled correctly"], &rows));
    // Baum-Welch from random starts.
    let sequences: Vec<Vec<usize>> = (0..40).map(|k| model.sample(500, 50 + k).1).collect();
    let true_ll: f64 = sequences.iter().map(|o| model.log_likelihood(o).unwrap()).sum();
    let mut plot = Plot::new("Baum-Welch from five random starts (40 sequences of 500 rolls)", "iteration", "total log-likelihood");
    let mut best: Option<(f64, HiddenMarkovModel, usize)> = None;
    for seed in 0..5u64 {
        let start = HiddenMarkovModel::random(2, 6, seed).unwrap();
        let fit = start.baum_welch(&sequences, 2000, 1e-9).unwrap();
        let ll: f64 = sequences.iter().map(|o| fit.model.log_likelihood(o).unwrap()).sum();
        plot.series.push(Series::new(&format!("start {}", seed + 1), fit.log_likelihood.iter().enumerate().map(|(i, &v)| (i as f64 + 1.0, v)).collect(), Style::Line, seed as usize));
        if best.as_ref().is_none_or(|b| ll > b.0) {
            best = Some((ll, fit.model, if fit.converged { fit.log_likelihood.len() } else { 0 }));
        }
    }
    plot.series.push(Series::new("true model", vec![(1.0, true_ll), (2000.0, true_ll)], Style::Dashed, 6));
    plot.log_x = true;
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "hmm_baum_welch.svg", &plot.render());
    let (ll, fit, iterations) = best.unwrap();
    // Match the learned states to the truth: the loaded state emits more sixes.
    let loaded = usize::from(fit.emission()[[1, 5]] > fit.emission()[[0, 5]]);
    let fair = 1 - loaded;
    let row = |name: &str, t: f64, f: f64| vec![name.to_string(), format!("{t:.3}"), format!("{f:.3}")];
    let rows = vec![
        row("P(fair → loaded)", 0.05, fit.transition()[[fair, loaded]]),
        row("P(loaded → fair)", 0.1, fit.transition()[[loaded, fair]]),
        row("P(six | fair)", 1.0 / 6.0, fit.emission()[[fair, 5]]),
        row("P(six | loaded)", 0.5, fit.emission()[[loaded, 5]]),
        row("P(one | loaded)", 0.1, fit.emission()[[loaded, 0]]),
        vec!["total log-likelihood".to_string(), format!("{true_ll:.1}"), if iterations > 0 { format!("{ll:.1} (converged after {iterations} iterations)") } else { format!("{ll:.1} (not converged in 2000 iterations)") }],
    ];
    write(CHAPTER, "hmm_parameters.md", &md_table(&["parameter", "true", "Baum-Welch (best of five starts)"], &rows));
    // Cost of Viterbi against the number of states.
    let mut rows = Vec::new();
    for states in [2usize, 8, 32, 128] {
        let m = HiddenMarkovModel::random(states, 6, 1).unwrap();
        let (_, o) = m.sample(1000, 2);
        let t = time(|| {
            m.viterbi(&o).unwrap();
        });
        rows.push(vec![format!("{states}"), format!("{:.2}", t * 1e3), format!("{:.1}", t * 1e9 / (1000.0 * (states * states) as f64))]);
    }
    write(CHAPTER, "hmm_viterbi_time.md", &md_table(&["states N (1000 steps)", "Viterbi time (ms)", "ns per step per N²"], &rows));
}

// ------------------------------------------------------------------ overview comparison

fn comparison_table() {
    let (mx, ml, mtx, mtl) = moons_pair(300, 21);
    let (cx, cl) = make_circles(150, 0.1, 0.5, 22);
    let (ctx, ctl) = make_circles(500, 0.1, 0.5, 23);
    let (gx, gl) = make_classification(100, 24);
    let (gtx, gtl) = make_classification(300, 25);
    type Fit = Box<dyn Fn(&R2D, &[usize], &R2D) -> Vec<usize>>;
    let methods: Vec<(&str, Fit)> = vec![
        ("decision tree (depth 6)", Box::new(|x: &R2D, l: &[usize], t: &R2D| {
            let mut m = DecisionTree::new(6);
            m.fit(x.view(), l).unwrap();
            m.predict(t.view()).unwrap()
        })),
        ("random forest (200 trees)", Box::new(|x: &R2D, l: &[usize], t: &R2D| {
            let mut m = RandomForest::new(200, 20, 1);
            m.fit(x.view(), l).unwrap();
            m.predict(t.view()).unwrap()
        })),
        ("gradient boosting (200 × depth 3)", Box::new(|x: &R2D, l: &[usize], t: &R2D| {
            let mut m = GradientBoostingClassifier::new(BoostingParameters::default());
            m.fit(x.view(), l).unwrap();
            m.predict(t.view()).unwrap()
        })),
        ("MLP (32 × 32)", Box::new(|x: &R2D, l: &[usize], t: &R2D| {
            let mut m = MlpClassifier::new(MlpParameters { seed: 1, ..Default::default() });
            m.fit(x.view(), l).unwrap();
            m.predict(t.view()).unwrap()
        })),
        ("SVM (RBF)", Box::new(|x: &R2D, l: &[usize], t: &R2D| {
            let mut m = SupportVectorMachine::new(Kernel::rbf_scaled(x.view()), 1.0);
            m.fit(x.view(), l).unwrap();
            m.predict(t.view()).unwrap()
        })),
    ];
    let mut rows = Vec::new();
    for (name, f) in &methods {
        let mut r = vec![name.to_string()];
        let mut total = 0.0;
        for (x, l, t, tl) in [(&mx, &ml, &mtx, &mtl), (&cx, &cl, &ctx, &ctl), (&gx, &gl, &gtx, &gtl)] {
            let t0 = std::time::Instant::now();
            let p = f(x, l, t);
            total += t0.elapsed().as_secs_f64();
            r.push(format!("{:.3}", accuracy(tl, &p).unwrap()));
        }
        r.push(format!("{:.0} ms", total * 1e3));
        rows.push(r);
    }
    write(CHAPTER, "comparison.md", &md_table(&["classifier (test accuracy)", "moons", "circles", "three Gaussian classes (4D)", "total time"], &rows));
}
