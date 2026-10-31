//! Clustering demonstration: centroid, density, connectivity and probabilistic methods on data
//! with two different geometries.
//!
//! Run with `cargo run --release -p lib_jsl_machine_learning --example clustering_demo`.

mod common;

use common::{as_labels, scatter, table};
use lib_jsl_core::R2D;
use lib_jsl_machine_learning::common::Clusterer;
use lib_jsl_machine_learning::datasets::{make_blobs, make_moons};
use lib_jsl_machine_learning::metrics::adjusted_rand_index;
use lib_jsl_machine_learning::unsupervised::clustering::{Affinity, CovarianceType, Dbscan, GaussianMixture, Hdbscan, HierarchicalClustering, KMeans, KMedoids, Linkage, SpectralClustering};

fn run(name: &str, x: &R2D, truth: &[usize], show: &[&str]) {
    let truth = as_labels(truth);
    let k = truth.iter().flatten().max().map_or(0, |m| m + 1);
    let mut results: Vec<(String, Vec<Option<usize>>)> = Vec::new();
    let mut add = |label: &str, labels: Vec<Option<usize>>| results.push((label.to_string(), labels));
    add("k-means", KMeans::new(k, 1).fit_predict(x.view()).unwrap());
    add("k-medoids", KMedoids::new(k).fit_predict(x.view()).unwrap());
    add("hierarchical (Ward)", HierarchicalClustering::new(k, Linkage::Ward).fit_predict(x.view()).unwrap());
    add("hierarchical (single)", HierarchicalClustering::new(k, Linkage::Single).fit_predict(x.view()).unwrap());
    add("Gaussian mixture", GaussianMixture::new(k, CovarianceType::Full, 1).fit_predict(x.view()).unwrap());
    add("DBSCAN", Dbscan::new(0.25, 5).fit_predict(x.view()).unwrap());
    add("HDBSCAN", Hdbscan::new(12).fit_predict(x.view()).unwrap());
    add("spectral (k-NN graph)", SpectralClustering::new(k, Affinity::NearestNeighbors { k: 10 }, 1).fit_predict(x.view()).unwrap());
    let rows: Vec<(String, String)> = results.iter().map(|(n, l)| (n.clone(), format!("adjusted Rand index {:.3}", adjusted_rand_index(l, &truth).unwrap()))).collect();
    table(&format!("== {name} =="), &rows);
    for (label, assignment) in results.iter().filter(|(n, _)| show.contains(&n.as_str())) {
        scatter(&format!("{name}: {label}"), x, assignment);
    }
}

fn main() {
    let (blobs, blob_truth) = make_blobs(60, &[vec![0.0, 0.0], vec![6.0, 0.5], vec![3.0, 6.0]], 0.8, 7);
    run("three Gaussian blobs", &blobs, &blob_truth, &["k-means"]);
    let (moons, moon_truth) = make_moons(120, 0.07, 3);
    run("two interleaved moons", &moons, &moon_truth, &["k-means", "DBSCAN", "spectral (k-NN graph)"]);
    println!("DBSCAN scores poorly on the blobs only because its fixed eps (0.25) is tuned for the moons.");
    println!("Centroid methods win on convex blobs; density and connectivity methods are needed for the moons.");
}
