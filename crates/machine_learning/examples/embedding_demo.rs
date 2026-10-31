//! Dimensionality-reduction demonstration: PCA, kernel PCA, t-SNE and UMAP on data that PCA alone
//! cannot untangle.
//!
//! Run with `cargo run --release -p lib_jsl_machine_learning --example embedding_demo`.

mod common;

use common::{as_labels, scatter, table};
use lib_jsl_core::R2D;
use lib_jsl_machine_learning::common::Transformer;
use lib_jsl_machine_learning::datasets::{make_blobs, make_circles, make_swiss_roll};
use lib_jsl_machine_learning::kernels::Kernel;
use lib_jsl_machine_learning::unsupervised::reduction::{KernelPca, Pca, Tsne, Umap};

fn main() {
    // Four clusters living in 12 dimensions, each shifted along its own pair of axes.
    let centres: Vec<Vec<f64>> = (0..4).map(|c| (0..12).map(|d| if d / 3 == c { 8.0 } else { 0.0 }).collect()).collect();
    let (high, labels) = make_blobs(40, &centres, 1.2, 9);
    let lab = as_labels(&labels);
    let mut pca = Pca::new(2);
    let flat = pca.fit_transform(high.view()).unwrap();
    table("== 160 points in 12 dimensions, 4 clusters: PCA variance explained ==", &[("first two components".to_string(), format!("{:.1}% of the total variance", 100.0 * pca.explained_variance_ratio().iter().sum::<f64>()))]);
    scatter("PCA projection", &flat, &lab);
    let mut tsne = Tsne::new(2, 20.0).with_iterations(400);
    let t = tsne.fit_transform(high.view()).unwrap();
    scatter(&format!("t-SNE (final KL divergence {:.3})", tsne.kl_divergence()), &t, &lab);
    let u = Umap::new(2, 12, 0.1, 1).fit_transform(high.view()).unwrap();
    scatter("UMAP", &u, &lab);

    // Concentric circles: nonlinear structure that linear PCA cannot separate.
    let (circles, cl) = make_circles(80, 0.03, 0.4, 2);
    let mut kpca = KernelPca::new(Kernel::Rbf { gamma: 2.0 }, 3);
    let k = kpca.fit_transform(circles.view()).unwrap();
    // Plot the radial component (3rd) against the first angular one.
    let pair = R2D::from_shape_fn((k.nrows(), 2), |(i, j)| if j == 0 { k[[i, 2]] } else { k[[i, 0]] });
    scatter("kernel PCA of concentric circles (x = 3rd component separates the rings)", &pair, &as_labels(&cl));

    // A 2D sheet rolled up in 3D: UMAP unrolls it.
    let (roll, angle) = make_swiss_roll(400, 0.05, 3);
    let unrolled = Umap::new(2, 15, 0.3, 1).fit_transform(roll.view()).unwrap();
    let bands: Vec<Option<usize>> = angle.iter().map(|&a| Some(((a - 4.7) / 9.4 * 4.0).clamp(0.0, 3.99) as usize)).collect();
    scatter("UMAP of a Swiss roll (colour = position along the roll)", &unrolled, &bands);
}
