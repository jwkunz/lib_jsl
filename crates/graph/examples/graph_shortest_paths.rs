//! Single-source and all-pairs shortest paths.

use lib_jsl_graph::{Graph, a_star, bellman_ford, dijkstra, floyd_warshall};

fn main() {
    // Road distances between five towns.
    let roads = [(0, 1, 7.0), (0, 2, 9.0), (0, 5, 14.0), (1, 2, 10.0), (1, 3, 15.0), (2, 3, 11.0), (2, 5, 2.0), (3, 4, 6.0), (4, 5, 9.0)];
    let g = Graph::from_edges(6, &roads, false).unwrap();
    let d = dijkstra(&g, 0).unwrap();
    println!("Dijkstra from 0: {:?}", d.distance);
    println!("path to 4: {:?}", d.path_to(4).unwrap());
    // A* with a zero heuristic is Dijkstra stopped at the target.
    println!("A* to 4: {:?}", a_star(&g, 0, 4, |_| 0.0).unwrap());
    // Negative arcs (but no negative cycle): Bellman-Ford.
    let signed = Graph::from_edges(4, &[(0, 1, 4.0), (0, 2, 5.0), (1, 3, 3.0), (2, 1, -3.0), (2, 3, 4.0)], true).unwrap();
    let b = bellman_ford(&signed, 0).unwrap();
    println!("Bellman-Ford: {:?}, path to 3 {:?}", b.distance, b.path_to(3).unwrap());
    let cycle = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, -2.0), (2, 1, 1.0)], true).unwrap();
    println!("with a negative cycle: {:?}", bellman_ford(&cycle, 0).map(|_| ()));
    let all = floyd_warshall(&g).unwrap();
    println!("Floyd-Warshall: 3 → 0 costs {}, via {:?}", all.distance[3][0], all.path(3, 0).unwrap());
}
