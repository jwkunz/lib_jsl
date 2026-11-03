//! The workflow shared by every model: split, scale, fit, predict, score.

use lib_jsl_machine_learning::common::{Classifier, Transformer};
use lib_jsl_machine_learning::datasets::make_classification;
use lib_jsl_machine_learning::metrics::{accuracy, confusion_matrix, precision_recall_f1};
use lib_jsl_machine_learning::preprocessing::{StandardScaler, select_labels, select_rows, train_test_split};
use lib_jsl_machine_learning::supervised::classification::LogisticRegression;

fn main() {
    let (x, labels) = make_classification(150, 1);
    let (train, test) = train_test_split(x.nrows(), 0.3, 7).unwrap();
    // Fit the scaler on the training rows only, then apply it to both sets.
    let mut scaler = StandardScaler::new();
    let x_train = scaler.fit_transform(select_rows(x.view(), &train).view()).unwrap();
    let x_test = scaler.transform(select_rows(x.view(), &test).view()).unwrap();
    let (y_train, y_test) = (select_labels(&labels, &train), select_labels(&labels, &test));

    let mut model = LogisticRegression::new(1e-3);
    model.fit(x_train.view(), &y_train).unwrap();
    let predicted = model.predict(x_test.view()).unwrap();
    println!("test accuracy {:.3} on {} points", accuracy(&y_test, &predicted).unwrap(), y_test.len());
    for (c, row) in confusion_matrix(&y_test, &predicted, 3).unwrap().iter().enumerate() {
        let (p, r, f1) = precision_recall_f1(&y_test, &predicted, 3).unwrap()[c];
        println!("class {c}: {row:?}  precision {p:.2} recall {r:.2} F1 {f1:.2}");
    }
}
