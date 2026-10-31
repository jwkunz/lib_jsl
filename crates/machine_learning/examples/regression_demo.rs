//! Regression demonstration: fitting a noisy sine wave and a sparse linear problem.
//!
//! Run with `cargo run --release -p lib_jsl_machine_learning --example regression_demo`.

mod common;

use common::table;
use lib_jsl_core::{R1D, R2D};
use lib_jsl_machine_learning::common::{MlRng, Regressor};
use lib_jsl_machine_learning::datasets::make_sine;
use lib_jsl_machine_learning::kernels::Kernel;
use lib_jsl_machine_learning::metrics::{mean_squared_error, r2_score};
use lib_jsl_machine_learning::preprocessing::{polynomial_features, select, select_rows, train_test_split};
use lib_jsl_machine_learning::supervised::regression::{ElasticNet, GeneralLeastSquares, LinearRegression, Ridge, SupportVectorRegression};
use lib_jsl_utils::terminal_plot::{PlotOptions, PlotSeries, render_terminal_plot};

fn sine_wave() {
    let (x, y) = make_sine(120, 0.15, 4);
    let (train, test) = train_test_split(x.nrows(), 0.3, 1).unwrap();
    let (xtr, ytr, xte, yte) = (select_rows(x.view(), &train), select(y.view(), &train), select_rows(x.view(), &test), select(y.view(), &test));
    let poly = |x: &R2D| polynomial_features(x.view(), 7).unwrap();
    let mut rows = Vec::new();
    let mut score = |name: &str, model: &mut dyn Regressor, tx: &R2D, te: &R2D| {
        model.fit(tx.view(), ytr.view()).unwrap();
        let p = model.predict(te.view()).unwrap();
        rows.push((name.to_string(), format!("test R² {:+.3}   test MSE {:.4}", r2_score(yte.view(), p.view()).unwrap(), mean_squared_error(yte.view(), p.view()).unwrap())));
    };
    score("straight line (OLS)", &mut LinearRegression::new(), &xtr, &xte);
    score("degree-7 polynomial (OLS)", &mut LinearRegression::new(), &poly(&xtr), &poly(&xte));
    score("degree-7 polynomial (ridge α=0.01)", &mut Ridge::new(0.01), &poly(&xtr), &poly(&xte));
    score("degree-7 polynomial (lasso α=0.001)", &mut ElasticNet::lasso(0.001), &poly(&xtr), &poly(&xte));
    score("support vector regression (RBF)", &mut SupportVectorRegression::new(Kernel::Rbf { gamma: 1.0 }, 10.0, 0.05).with_limits(1500, 1e-6), &xtr, &xte);
    table("== noisy sine wave: models compared on held-out data ==", &rows);

    // Plot the SVR curve against the data.
    let mut svr = SupportVectorRegression::new(Kernel::Rbf { gamma: 1.0 }, 10.0, 0.05).with_limits(1500, 1e-6);
    svr.fit(xtr.view(), ytr.view()).unwrap();
    let grid = R2D::from_shape_fn((80, 1), |(i, _)| i as f64 * std::f64::consts::TAU / 79.0);
    let fit = svr.predict(grid.view()).unwrap();
    let (gx, fx): (Vec<f64>, Vec<f64>) = (grid.column(0).to_vec(), fit.to_vec());
    let (dx, dy): (Vec<f64>, Vec<f64>) = (xtr.column(0).to_vec(), ytr.to_vec());
    let series = [PlotSeries { label: "training data", x: &dx, y: &dy }, PlotSeries { label: "SVR fit", x: &gx, y: &fx }];
    println!("{}\n", render_terminal_plot(&series, &PlotOptions { width: 60, height: 14, title: Some("SVR with an RBF kernel".into()), color: true, ..Default::default() }));
}

fn sparse_features() {
    // 15 candidate features but only 3 matter, and just 40 training samples: which method copes best?
    let mut rng = MlRng::new(2);
    let make = |n: usize, rng: &mut MlRng| {
        let x = R2D::from_shape_fn((n, 15), |_| rng.gaussian(0.0, 1.0));
        let y = R1D::from_iter((0..n).map(|i| 3.0 * x[[i, 0]] - 2.0 * x[[i, 1]] + 1.5 * x[[i, 2]] + rng.gaussian(0.0, 0.5)));
        (x, y)
    };
    let (xtr, ytr) = make(40, &mut rng);
    let (xte, yte) = make(500, &mut rng);
    let mut rows = Vec::new();
    let mut eval = |name: &str, m: &mut dyn Regressor, extra: String| {
        m.fit(xtr.view(), ytr.view()).unwrap();
        let mse = mean_squared_error(yte.view(), m.predict(xte.view()).unwrap().view()).unwrap();
        rows.push((name.to_string(), format!("test MSE {mse:.3}{extra}")));
    };
    eval("OLS", &mut LinearRegression::new(), String::new());
    eval("ridge α=5", &mut Ridge::new(5.0), String::new());
    let mut lasso = ElasticNet::lasso(0.1);
    lasso.fit(xtr.view(), ytr.view()).unwrap();
    eval("lasso α=0.1", &mut lasso.clone(), format!("   non-zero weights: {} of 15", lasso.nonzero()));
    table("== 3 relevant features among 15, only 40 training samples ==", &rows);
    println!("The lasso wins by zeroing the irrelevant weights; ridge shrinks every weight, including the true ones.\n");

    // Heteroscedastic noise: weighted least squares trusts the clean samples more.
    let n = 60;
    let x = R2D::from_shape_fn((n, 1), |(i, _)| i as f64 / 6.0);
    let noisy = |i: usize| i.is_multiple_of(5);
    let y = R1D::from_iter((0..n).map(|i| 2.0 * x[[i, 0]] + 1.0 + rng.gaussian(0.0, if noisy(i) { 6.0 } else { 0.1 })));
    let mut ols = LinearRegression::new();
    ols.fit(x.view(), y.view()).unwrap();
    let mut wls = GeneralLeastSquares::with_sample_weights(R1D::from_iter((0..n).map(|i| if noisy(i) { 1.0 / 36.0 } else { 100.0 })));
    wls.fit(x.view(), y.view()).unwrap();
    table(
        "== line y = 2x + 1 where every 5th sample is very noisy ==",
        &[
            ("OLS".to_string(), format!("slope {:.3}, intercept {:.3}", ols.weights()[0], ols.intercept())),
            ("weighted least squares".to_string(), format!("slope {:.3}, intercept {:.3}", wls.weights()[0], wls.intercept())),
        ],
    );
}

fn main() {
    sine_wave();
    sparse_features();
}
