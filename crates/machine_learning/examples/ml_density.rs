//! Density-based clustering: DBSCAN with a fixed radius against HDBSCAN on clusters of different density.

use lib_jsl_machine_learning::common::Clusterer;
use lib_jsl_machine_learning::datasets::make_blobs;
use ndarray::{Axis, concatenate};
use lib_jsl_machine_learning::unsupervised::clustering::{Dbscan, Hdbscan};

fn summary(labels: &[Option<usize>]) -> String {
    let k = labels.iter().flatten().max().map_or(0, |m| m + 1);
    let sizes: Vec<usize> = (0..k).map(|c| labels.iter().filter(|l| **l == Some(c)).count()).collect();
    format!("{k} clusters of sizes {sizes:?}, {} noise points", labels.iter().filter(|l| l.is_none()).count())
}

fn main() {
    // Two tight blobs and one diffuse blob.
    let (dense, _) = make_blobs(80, &[vec![0.0, 0.0], vec![2.0, 0.0]], 0.25, 1);
    let (sparse, _) = make_blobs(80, &[vec![7.0, 4.0]], 1.6, 2);
    let x = concatenate(Axis(0), &[dense.view(), sparse.view()]).unwrap();
    for eps in [0.3, 1.0] {
        let mut db = Dbscan::new(eps, 5);
        let labels = db.fit_predict(x.view()).unwrap();
        println!("DBSCAN eps = {eps}: {} ({} core points)", summary(&labels), db.core_points().len());
    }
    let mut h = Hdbscan::new(15);
    let labels = h.fit_predict(x.view()).unwrap();
    println!("HDBSCAN:          {}", summary(&labels));
    println!("cluster stabilities: {:?}", h.cluster_stabilities().iter().map(|s| format!("{s:.0}")).collect::<Vec<_>>());
}
