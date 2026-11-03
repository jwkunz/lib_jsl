//! Nonlinear embeddings: t-SNE and UMAP on clusters in 20 dimensions.

use lib_jsl_machine_learning::datasets::make_blobs;
use lib_jsl_machine_learning::unsupervised::reduction::{Tsne, Umap};

fn main() {
    let centres: Vec<Vec<f64>> = (0..5).map(|c| (0..20).map(|d| if d % 5 == c { 5.0 } else { 0.0 }).collect()).collect();
    let (x, labels) = make_blobs(40, &centres, 1.0, 4);
    let mut tsne = Tsne::new(2, 20.0).with_iterations(500);
    let yt = tsne.fit_transform(x.view()).unwrap();
    println!("t-SNE: KL divergence {:.3}", tsne.kl_divergence());
    let yu = Umap::new(2, 15, 0.1, 1).fit_transform(x.view()).unwrap();
    // The embedded centre of each cluster.
    for (name, y) in [("t-SNE", &yt), ("UMAP", &yu)] {
        let centres: Vec<String> = (0..5)
            .map(|c| {
                let idx: Vec<usize> = (0..y.nrows()).filter(|&i| labels[i] == c).collect();
                let m = |k: usize| idx.iter().map(|&i| y[[i, k]]).sum::<f64>() / idx.len() as f64;
                format!("({:.1}, {:.1})", m(0), m(1))
            })
            .collect();
        println!("{name} cluster centres: {}", centres.join(" "));
    }
}
