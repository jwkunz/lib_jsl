//! k-nearest neighbours: choosing k and weighting by distance.

use lib_jsl_machine_learning::common::Classifier;
use lib_jsl_machine_learning::datasets::make_moons;
use lib_jsl_machine_learning::metrics::accuracy;
use lib_jsl_machine_learning::supervised::classification::KNearestNeighbors;

fn main() {
    let (x, labels) = make_moons(100, 0.3, 1);
    let (x_test, test_labels) = make_moons(1000, 0.3, 2);
    for k in [1, 5, 15, 45, 135] {
        let mut plain = KNearestNeighbors::new(k);
        plain.fit(x.view(), &labels).unwrap();
        let mut weighted = KNearestNeighbors::new(k).distance_weighted(true);
        weighted.fit(x.view(), &labels).unwrap();
        println!(
            "k = {k:>3}: test accuracy {:.3}, distance-weighted {:.3}",
            accuracy(&test_labels, &plain.predict(x_test.view()).unwrap()).unwrap(),
            accuracy(&test_labels, &weighted.predict(x_test.view()).unwrap()).unwrap()
        );
    }
    let mut model = KNearestNeighbors::new(5);
    model.fit(x.view(), &labels).unwrap();
    let near = model.neighbours(&[0.5, 0.25]);
    println!("neighbours of (0.5, 0.25): {}", near.iter().map(|(i, d)| format!("#{i} (class {}, {d:.2})", labels[*i])).collect::<Vec<_>>().join(", "));
}
