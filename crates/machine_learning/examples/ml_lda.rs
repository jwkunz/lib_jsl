//! Linear discriminant analysis as a projection and as a classifier.

use lib_jsl_machine_learning::common::Classifier;
use lib_jsl_machine_learning::datasets::make_classification;
use lib_jsl_machine_learning::metrics::accuracy;
use lib_jsl_machine_learning::unsupervised::reduction::Lda;

fn main() {
    let (x, labels) = make_classification(150, 2);
    let (test, test_labels) = make_classification(300, 3);
    let mut lda = Lda::new(2);
    lda.fit(x.view(), &labels).unwrap();
    println!("Fisher ratios {:.3?}", lda.eigenvalues());
    println!("directions (columns):\n{:.3}", lda.directions());
    let z = lda.transform(test.view()).unwrap();
    println!("projected test data: {:?}", z.dim());
    println!("test accuracy {:.3}", accuracy(&test_labels, &lda.predict(test.view()).unwrap()).unwrap());
}
