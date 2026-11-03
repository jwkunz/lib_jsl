//! Predict the probability of passing an exam from hours of study and hours of sleep.

use lib_jsl_core::R2D;
use lib_jsl_machine_learning::common::{Classifier, MlRng};
use lib_jsl_machine_learning::supervised::classification::LogisticRegression;

fn main() {
    let mut rng = MlRng::new(3);
    let n = 200;
    let x = R2D::from_shape_fn((n, 2), |(_, j)| if j == 0 { rng.range(0.0, 10.0) } else { rng.range(4.0, 9.0) });
    // True model: log-odds = -6 + 0.8 study + 0.5 sleep.
    let passed: Vec<usize> = (0..n).map(|i| {
        let z = -6.0 + 0.8 * x[[i, 0]] + 0.5 * x[[i, 1]];
        (rng.uniform() < 1.0 / (1.0 + (-z).exp())) as usize
    }).collect();
    let mut model = LogisticRegression::new(1e-4).with_limits(2000, 1e-9);
    model.fit(x.view(), &passed).unwrap();
    // With two classes, the log-odds of class 1 are the difference of the two rows of weights.
    let (w, b) = (model.coefficients(), model.intercepts());
    println!("log-odds = {:+.2} {:+.2} study {:+.2} sleep  ({} iterations)", b[1] - b[0], w[[1, 0]] - w[[0, 0]], w[[1, 1]] - w[[0, 1]], model.iterations());
    let students = R2D::from_shape_vec((3, 2), vec![2.0, 8.0, 5.0, 7.0, 8.0, 5.0]).unwrap();
    let p = model.predict_proba(students.view()).unwrap();
    for i in 0..3 {
        println!("study {:.0} h, sleep {:.0} h: P(pass) = {:.2}", students[[i, 0]], students[[i, 1]], p[[i, 1]]);
    }
}
