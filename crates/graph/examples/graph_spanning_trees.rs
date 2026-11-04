//! Minimum spanning trees and the union-find structure.

use lib_jsl_graph::{Graph, UnionFind, kruskal, prim};

fn main() {
    let edges = [(0, 1, 4.0), (0, 7, 8.0), (1, 2, 8.0), (1, 7, 11.0), (2, 3, 7.0), (2, 8, 2.0), (2, 5, 4.0), (3, 4, 9.0), (3, 5, 14.0), (4, 5, 10.0), (5, 6, 2.0), (6, 7, 1.0), (6, 8, 6.0), (7, 8, 7.0)];
    let g = Graph::from_edges(9, &edges, false).unwrap();
    let k = kruskal(&g).unwrap();
    let p = prim(&g).unwrap();
    println!("Kruskal: weight {}, edges {:?}", k.weight, k.edges.iter().map(|e| (e.from, e.to)).collect::<Vec<_>>());
    println!("Prim:    weight {}, {} edges", p.weight, p.edges.len());
    let mut sets = UnionFind::new(6);
    sets.union(0, 1);
    sets.union(2, 3);
    sets.union(1, 3);
    println!("union-find: 0 and 2 connected {}, 0 and 5 connected {}, {} sets", sets.connected(0, 2), sets.connected(0, 5), sets.set_count());
}
