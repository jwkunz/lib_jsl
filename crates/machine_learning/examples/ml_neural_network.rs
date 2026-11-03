//! Multilayer perceptrons: classifying two moons and fitting a sine.

use lib_jsl_core::R2D;
use lib_jsl_machine_learning::common::{Classifier, Regressor};
use lib_jsl_machine_learning::datasets::{make_moons, make_sine};
use lib_jsl_machine_learning::metrics::{accuracy, mean_squared_error};
use lib_jsl_machine_learning::neural_network::{Activation, MlpClassifier, MlpParameters, MlpRegressor};

fn main() {
    let (x, labels) = make_moons(200, 0.2, 1);
    let (test, test_labels) = make_moons(500, 0.2, 2);
    let mut clf = MlpClassifier::new(MlpParameters { hidden: vec![32, 32], activation: Activation::Relu, epochs: 200, seed: 1, ..Default::default() });
    clf.fit(x.view(), &labels).unwrap();
    let loss = &clf.training_loss;
    println!("cross-entropy: epoch 1 {:.3}, epoch 50 {:.3}, epoch 200 {:.3}", loss[0], loss[49], loss[199]);
    println!("test accuracy {:.3}", accuracy(&test_labels, &clf.predict(test.view()).unwrap()).unwrap());
    let p = clf.predict_proba(ndarray::arr2(&[[0.5, 0.25], [-1.0, 0.5]]).view()).unwrap();
    println!("P(class 1) at (0.5, 0.25): {:.3}; at (-1, 0.5): {:.3}", p[[0, 1]], p[[1, 1]]);

    let (sx, sy) = make_sine(100, 0.1, 3);
    let mut reg = MlpRegressor::new(MlpParameters { hidden: vec![16, 16], epochs: 1000, batch_size: 16, seed: 2, ..Default::default() });
    reg.fit(sx.view(), sy.view()).unwrap();
    let grid = R2D::from_shape_fn((50, 1), |(i, _)| 0.3 + 5.6 * i as f64 / 49.0);
    let truth = grid.column(0).mapv(f64::sin);
    println!("sine: MSE against sin x {:.5}", mean_squared_error(truth.view(), reg.predict(grid.view()).unwrap().view()).unwrap());
}
