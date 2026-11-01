//! Graph algorithms on weighted, directed or undirected graphs with vertices `0..n`.

pub mod all_pairs;
pub mod graph;
pub mod shortest_paths;

pub use all_pairs::{AllPairsShortestPaths, floyd_warshall};
pub use graph::{Edge, Graph, UnionFind};
pub use shortest_paths::{ShortestPaths, a_star, bellman_ford, dijkstra};
