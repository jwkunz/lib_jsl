//! Graph construction, traversal, components, topological order and strongly connected components.

use lib_jsl_graph::{Graph, condensation, strongly_connected_components, topological_sort};

fn main() {
    // An undirected graph with two components.
    let g = Graph::from_unweighted_edges(7, &[(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (5, 6)], false).unwrap();
    println!("breadth-first from 0: {:?}", g.breadth_first(0));
    println!("depth-first from 0:   {:?}", g.depth_first(0));
    println!("components: {:?}", g.connected_components());
    // Course prerequisites: an edge u → v means u comes before v.
    let courses = Graph::from_unweighted_edges(6, &[(0, 2), (1, 2), (2, 3), (1, 4), (4, 5), (3, 5)], true).unwrap();
    println!("a study order: {:?}", topological_sort(&courses).unwrap());
    // A directed graph with cycles 0 → 1 → 2 → 0 and 3 ⇄ 4, and a sink 5.
    let d = Graph::from_unweighted_edges(6, &[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3), (4, 5)], true).unwrap();
    println!("strongly connected components: {:?}", strongly_connected_components(&d));
    let (_, dag) = condensation(&d);
    println!("condensation: {} components, edges {:?}", dag.vertex_count(), dag.edges().iter().map(|e| (e.from, e.to)).collect::<Vec<_>>());
}
