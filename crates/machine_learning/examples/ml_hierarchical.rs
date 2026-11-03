//! Agglomerative clustering, dendrogram cuts, and phylogenetic trees by UPGMA and neighbour joining.

use lib_jsl_core::R2D;
use lib_jsl_machine_learning::common::pairwise_distances;
use lib_jsl_machine_learning::supervised::classification::{neighbor_joining, upgma};
use lib_jsl_machine_learning::unsupervised::clustering::{Linkage, agglomerate};
use ndarray::arr2;

fn main() {
    // Points on a line at 0, 1, 5, 6 and 20.
    let x = arr2(&[[0.0], [1.0], [5.0], [6.0], [20.0]]);
    for linkage in [Linkage::Single, Linkage::Complete, Linkage::Average, Linkage::Ward] {
        let tree = agglomerate(pairwise_distances(x.view()).view(), linkage).unwrap();
        let heights: Vec<String> = tree.merges.iter().map(|m| format!("{:.2}", m.distance)).collect();
        println!("{:<8} merge heights [{}], three clusters {:?}", format!("{linkage:?}"), heights.join(", "), tree.cut_clusters(3).unwrap());
    }

    // An additive (but not ultrametric) tree: A and B evolve faster than C and D.
    //   A:4, B:5 join at u; C:1, D:2 join at v; u and v joined by a branch of length 2.
    let d: R2D = arr2(&[[0.0, 9.0, 7.0, 8.0], [9.0, 0.0, 8.0, 9.0], [7.0, 8.0, 0.0, 3.0], [8.0, 9.0, 3.0, 0.0]]);
    let names = ["A", "B", "C", "D"];
    let nj = neighbor_joining(d.view(), &names).unwrap();
    let up = upgma(d.view(), &names).unwrap();
    println!("neighbour joining: {}", nj.to_newick());
    println!("UPGMA:             {}", up.to_newick());
    let err = |t: &R2D| t.iter().zip(d.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("largest path-length error: NJ {:.1e}, UPGMA {:.2}", err(&nj.patristic_distances()), err(&up.patristic_distances()));
}
