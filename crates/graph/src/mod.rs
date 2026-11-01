//! Graph algorithms on weighted, directed or undirected graphs with vertices `0..n`.

pub mod graph;
pub mod shortest_paths;

pub use graph::{Edge, Graph, UnionFind};
pub use shortest_paths::{ShortestPaths, a_star, bellman_ford, dijkstra};
