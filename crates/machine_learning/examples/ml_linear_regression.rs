//! Linear regression with and without penalties on a problem with irrelevant, correlated features.

use lib_jsl_core::{R1D, R2D};
use lib_jsl_machine_learning::common::{MlRng, Regressor};
use lib_jsl_machine_learning::metrics::mean_squared_error;
use lib_jsl_machine_learning::supervised::regression::{ElasticNet, LinearRegression, Ridge};

fn data(n: usize, rng: &mut MlRng) -> (R2D, R1D) {
    // Twenty features; only the first three matter, and feature 3 nearly copies feature 0.
    let mut x = R2D::from_shape_fn((n, 20), |_| rng.gaussian(0.0, 1.0));
    for i in 0..n {
        x[[i, 3]] = x[[i, 0]] + 0.1 * x[[i, 3]];
    }
    let y = R1D::from_iter((0..n).map(|i| 3.0 * x[[i, 0]] - 2.0 * x[[i, 1]] + 1.5 * x[[i, 2]] + rng.gaussian(0.0, 1.0)));
    (x, y)
}

fn main() {
    let mut rng = MlRng::new(1);
    let (x, y) = data(30, &mut rng);
    let (x_test, y_test) = data(1000, &mut rng);
    let models: Vec<(&str, Box<dyn Regressor>)> = vec![
        ("least squares", Box::new(LinearRegression::new())),
        ("ridge, α = 10", Box::new(Ridge::new(10.0))),
        ("lasso, α = 0.2", Box::new(ElasticNet::lasso(0.2))),
        ("elastic net, α = 0.2, ρ = 0.5", Box::new(ElasticNet::new(0.2, 0.5))),
    ];
    for (name, mut model) in models {
        model.fit(x.view(), y.view()).unwrap();
        let mse = mean_squared_error(y_test.view(), model.predict(x_test.view()).unwrap().view()).unwrap();
        println!("{name:<30} test MSE {mse:.3}");
    }
    let mut lasso = ElasticNet::lasso(0.2);
    lasso.fit(x.view(), y.view()).unwrap();
    let kept: Vec<String> = lasso.weights().iter().enumerate().filter(|(_, w)| **w != 0.0).map(|(j, w)| format!("w{j} = {w:+.2}")).collect();
    println!("lasso keeps {} of 20 weights: {}", lasso.nonzero(), kept.join(", "));
}
