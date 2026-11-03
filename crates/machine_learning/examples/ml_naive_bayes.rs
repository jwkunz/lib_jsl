//! Spam filtering with multinomial naive Bayes on word counts.

use lib_jsl_core::R2D;
use lib_jsl_machine_learning::common::Classifier;
use lib_jsl_machine_learning::supervised::classification::MultinomialNaiveBayes;

fn main() {
    let vocabulary = ["free", "winner", "money", "meeting", "report", "lunch", "click", "project"];
    // Word counts of ten training e-mails; label 1 is spam.
    let counts = [
        [2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        [1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [3.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0],
        [0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 3.0],
    ];
    let labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
    let x = R2D::from_shape_fn((10, 8), |(i, j)| counts[i][j]);
    let mut model = MultinomialNaiveBayes::new(1.0);
    model.fit(x.view(), &labels).unwrap();
    let messages = [("free lunch meeting", [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]), ("click to claim money, winner", [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), ("project report", [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])];
    let rows = R2D::from_shape_fn((3, 8), |(i, j)| messages[i].1[j]);
    for ((text, _), label) in messages.iter().zip(model.predict(rows.view()).unwrap()) {
        println!("{text:<30} -> {}", if label == 1 { "spam" } else { "not spam" });
    }
    println!("vocabulary: {}", vocabulary.join(", "));
}
