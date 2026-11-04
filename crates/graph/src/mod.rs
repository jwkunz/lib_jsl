//! Graph algorithms on weighted, directed or undirected graphs with vertices `0..n`.
//!
//! See the book entry [Graph algorithms](https://jwkunz.github.io/lib_jsl/book/graph/index.html).

pub mod all_pairs;
pub mod graph;
pub mod max_flow;
pub mod ordering;
pub mod shortest_paths;
pub mod spanning_trees;

pub use all_pairs::{AllPairsShortestPaths, floyd_warshall};
pub use graph::{Edge, Graph, UnionFind};
pub use max_flow::{FlowNetwork, MaxFlow};
pub use ordering::{condensation, strongly_connected_components, topological_sort};
pub use shortest_paths::{ShortestPaths, a_star, bellman_ford, dijkstra};
pub use spanning_trees::{SpanningForest, kruskal, prim};
