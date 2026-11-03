//! Spectral clustering of concentric rings, and the eigengap as a guide to the number of clusters.

use lib_jsl_machine_learning::common::Clusterer;
use lib_jsl_machine_learning::datasets::{make_blobs, make_circles};
use lib_jsl_machine_learning::metrics::adjusted_rand_index;
use lib_jsl_machine_learning::unsupervised::clustering::{Affinity, KMeans, SpectralClustering};

fn main() {
    let (x, truth) = make_circles(150, 0.04, 0.45, 1);
    let truth: Vec<Option<usize>> = truth.into_iter().map(Some).collect();
    let km = KMeans::new(2, 1).fit_predict(x.view()).unwrap();
    let sc = SpectralClustering::new(2, Affinity::NearestNeighbors { k: 8 }, 1).fit_predict(x.view()).unwrap();
    println!("rings: k-means ARI {:.3}, spectral ARI {:.3}", adjusted_rand_index(&km, &truth).unwrap(), adjusted_rand_index(&sc, &truth).unwrap());

    let (b, _) = make_blobs(40, &[vec![0.0, 0.0], vec![8.0, 0.0], vec![4.0, 7.0]], 0.8, 3);
    let mut probe = SpectralClustering::new(6, Affinity::NearestNeighbors { k: 8 }, 1);
    probe.fit_predict(b.view()).unwrap();
    println!("smallest Laplacian eigenvalues for three blobs: {:.4?}", probe.eigenvalues());
}
