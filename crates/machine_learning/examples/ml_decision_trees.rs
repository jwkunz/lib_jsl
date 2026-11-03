//! A single tree against a random forest, with out-of-bag accuracy.

use lib_jsl_machine_learning::common::Classifier;
use lib_jsl_machine_learning::datasets::make_classification;
use lib_jsl_machine_learning::metrics::accuracy;
use lib_jsl_machine_learning::supervised::classification::{DecisionTree, RandomForest};

fn main() {
    let (x, labels) = make_classification(100, 1);
    let (x_test, test_labels) = make_classification(1000, 2);
    for depth in [1, 2, 4, 8, 50] {
        let mut tree = DecisionTree::new(depth);
        tree.fit(x.view(), &labels).unwrap();
        println!(
            "tree, depth limit {depth:>2} (grown to {}): training {:.3}, test {:.3}",
            tree.depth(),
            accuracy(&labels, &tree.predict(x.view()).unwrap()).unwrap(),
            accuracy(&test_labels, &tree.predict(x_test.view()).unwrap()).unwrap()
        );
    }
    let mut forest = RandomForest::new(200, 50, 1);
    forest.fit(x.view(), &labels).unwrap();
    println!(
        "forest of 200 trees: test {:.3}, out-of-bag estimate {:.3}",
        accuracy(&test_labels, &forest.predict(x_test.view()).unwrap()).unwrap(),
        forest.oob_accuracy().unwrap()
    );
    println!("share of splits per feature: {:.3?}", forest.feature_importances());
}
