//! Gradient boosting for regression and classification.

use lib_jsl_machine_learning::common::{Classifier, Regressor};
use lib_jsl_machine_learning::datasets::{make_classification, make_regression};
use lib_jsl_machine_learning::ensembles::{BoostingParameters, GradientBoostingClassifier, GradientBoostingRegressor};
use lib_jsl_machine_learning::metrics::{accuracy, r2_score};
use ndarray::s;

fn main() {
    // Eight features, of which the target depends nonlinearly on the first three.
    let (x, linear, _, _) = make_regression(600, 8, 1.0, 5);
    let y = ndarray::Array1::from_iter((0..600).map(|i| linear[i] + 3.0 * x[[i, 0]] * x[[i, 1]] + 2.0 * x[[i, 2]].powi(2)));
    let (train, test) = ((x.slice(s![..400, ..]), y.slice(s![..400])), (x.slice(s![400.., ..]), y.slice(s![400..])));
    let mut gb = GradientBoostingRegressor::new(BoostingParameters { stages: 500, learning_rate: 0.05, max_depth: 3, subsample: 0.8, seed: 1, ..Default::default() });
    gb.fit(train.0, train.1).unwrap();
    println!("regression: test R² {:.3}", r2_score(test.1, gb.predict(test.0).unwrap().view()).unwrap());
    let importances: Vec<String> = gb.feature_importances().iter().map(|v| format!("{v:.3}")).collect();
    println!("feature importances [{}]", importances.join(", "));

    let (cx, cl) = make_classification(150, 2);
    let (tx, tl) = make_classification(300, 3);
    let mut clf = GradientBoostingClassifier::new(BoostingParameters { stages: 150, max_depth: 2, ..Default::default() });
    clf.fit(cx.view(), &cl).unwrap();
    println!("classification: test accuracy {:.3}, training deviance {:.3} → {:.3}", accuracy(&tl, &clf.predict(tx.view()).unwrap()).unwrap(), clf.training_loss[0], clf.training_loss.last().unwrap());
}
