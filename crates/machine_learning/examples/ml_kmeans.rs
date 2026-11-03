//! k-means and k-medoids: choosing k by the silhouette, and clustering from a distance matrix.

use lib_jsl_core::R2D;
use lib_jsl_machine_learning::common::Clusterer;
use lib_jsl_machine_learning::datasets::make_blobs;
use lib_jsl_machine_learning::metrics::silhouette_score;
use lib_jsl_machine_learning::unsupervised::clustering::{KMeans, KMedoids};

fn main() {
    let centres = vec![vec![0.0, 0.0], vec![5.0, 0.5], vec![1.0, 5.0], vec![6.0, 5.5]];
    let (x, _) = make_blobs(60, &centres, 1.0, 3);
    for k in 2..=6 {
        let mut km = KMeans::new(k, 1).with_restarts(10);
        let labels = km.fit_predict(x.view()).unwrap();
        println!("k = {k}: inertia {:7.1}, silhouette {:.3}", km.inertia(), silhouette_score(x.view(), &labels).unwrap());
    }
    let mut km = KMeans::new(4, 1);
    km.fit_predict(x.view()).unwrap();
    println!("centroids:\n{:.2}", km.centroids());

    // k-medoids needs only dissimilarities: here, the number of differing letters between words.
    let words = ["cat", "car", "cab", "bat", "dog", "dot", "dig", "fig"];
    let d = R2D::from_shape_fn((8, 8), |(i, j)| words[i].chars().zip(words[j].chars()).filter(|(a, b)| a != b).count() as f64);
    let mut pam = KMedoids::new(2);
    let labels = pam.fit_distances(d.view()).unwrap();
    let medoids: Vec<&str> = pam.medoids().iter().map(|&m| words[m]).collect();
    println!("medoids {medoids:?}, labels {labels:?}, total distance {}", pam.cost());
}
