//! Topological sorting and strongly connected components of directed graphs.

use crate::graph::Graph;
use lib_jsl_core::ErrorsJSL;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// A topological order (every edge `u → v` has `u` before `v`) by Kahn's algorithm, taking the
/// smallest ready vertex at each step, so the result is the lexicographically smallest
/// topological order. `O((V + E) log V)`. Returns an error if the graph has a cycle.
pub fn topological_sort(g: &Graph) -> Result<Vec<usize>, ErrorsJSL> {
    if !g.is_directed() {
        return Err(ErrorsJSL::InvalidInputRange("Topological sorting needs a directed graph."));
    }
    let n = g.vertex_count();
    let mut indegree = vec![0usize; n];
    for e in g.edges() {
        indegree[e.to] += 1;
    }
    // A min-heap of the vertices with no remaining incoming edges.
    let mut ready: BinaryHeap<Reverse<usize>> = (0..n).filter(|&v| indegree[v] == 0).map(Reverse).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(Reverse(u)) = ready.pop() {
        order.push(u);
        for &(v, _) in g.neighbors(u) {
            indegree[v] -= 1;
            if indegree[v] == 0 {
                ready.push(Reverse(v));
            }
        }
    }
    if order.len() != n {
        return Err(ErrorsJSL::RuntimeError("The graph has a cycle, so no topological order exists."));
    }
    Ok(order)
}

/// Strongly connected components by Tarjan's algorithm (iterative, `O(V + E)`). Returns the
/// component label of each vertex; labels are a reverse topological order of the condensation
/// (component 0 has no edges to other components).
pub fn strongly_connected_components(g: &Graph) -> Vec<usize> {
    let n = g.vertex_count();
    let mut index = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack = Vec::new();
    let mut component = vec![usize::MAX; n];
    let (mut counter, mut components) = (0, 0);
    for root in 0..n {
        if index[root] != usize::MAX {
            continue;
        }
        // Frames of (vertex, next neighbour position).
        let mut frames = vec![(root, 0usize)];
        index[root] = counter;
        low[root] = counter;
        counter += 1;
        stack.push(root);
        on_stack[root] = true;
        while let Some(&mut (v, ref mut position)) = frames.last_mut() {
            if let Some(&(w, _)) = g.neighbors(v).get(*position) {
                *position += 1;
                if index[w] == usize::MAX {
                    index[w] = counter;
                    low[w] = counter;
                    counter += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    frames.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
                continue;
            }
            frames.pop();
            if let Some(&(parent, _)) = frames.last() {
                low[parent] = low[parent].min(low[v]);
            }
            if low[v] == index[v] {
                loop {
                    let w = stack.pop().expect("v is on the stack");
                    on_stack[w] = false;
                    component[w] = components;
                    if w == v {
                        break;
                    }
                }
                components += 1;
            }
        }
    }
    component
}

/// The condensation: the directed acyclic graph of strongly connected components.
pub fn condensation(g: &Graph) -> (Vec<usize>, Graph) {
    let labels = strongly_connected_components(g);
    let count = labels.iter().max().map_or(0, |m| m + 1);
    let mut dag = Graph::new(count, true);
    let mut seen = std::collections::HashSet::new();
    for e in g.edges() {
        let (a, b) = (labels[e.from], labels[e.to]);
        if a != b && seen.insert((a, b)) {
            dag.add_edge(a, b, 1.0).expect("labels are in range");
        }
    }
    (labels, dag)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_go_to_the_smallest_ready_vertex() {
        // With edges 0 → 3 and 2 → 1, a first-in-first-out queue gives [0, 2, 3, 1].
        let g = Graph::from_unweighted_edges(4, &[(0, 3), (2, 1)], true).unwrap();
        assert_eq!(topological_sort(&g).unwrap(), vec![0, 2, 1, 3]);
    }

    #[test]
    fn topological_order_respects_every_edge() {
        let edges = [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)];
        let g = Graph::from_unweighted_edges(6, &edges, true).unwrap();
        let order = topological_sort(&g).unwrap();
        let position: Vec<usize> = (0..6).map(|v| order.iter().position(|&o| o == v).unwrap()).collect();
        assert!(edges.iter().all(|&(u, v)| position[u] < position[v]));
        assert_eq!(order, vec![4, 5, 0, 2, 3, 1]);
        assert!(topological_sort(&Graph::from_unweighted_edges(2, &[(0, 1), (1, 0)], true).unwrap()).is_err());
        assert!(topological_sort(&Graph::new(2, false)).is_err());
    }

    #[test]
    fn components_and_condensation() {
        // Components {0, 1, 2}, {3, 4}, {5}, {6, 7}.
        let g = Graph::from_unweighted_edges(8, &[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3), (4, 5), (6, 7), (7, 6), (6, 5)], true).unwrap();
        let labels = strongly_connected_components(&g);
        let same = |a: usize, b: usize| labels[a] == labels[b];
        assert!(same(0, 1) && same(1, 2) && same(3, 4) && same(6, 7));
        assert!(!same(0, 3) && !same(3, 5) && !same(5, 6));
        let (_, dag) = condensation(&g);
        assert_eq!(dag.vertex_count(), 4);
        let order = topological_sort(&dag).unwrap();
        assert_eq!(order.len(), 4);
        // The sink component {5} gets label 0 (reverse topological labelling).
        assert_eq!(labels[5], 0);
        // A long cycle does not overflow the stack.
        let n = 100_000;
        let ring = Graph::from_unweighted_edges(n, &(0..n).map(|i| (i, (i + 1) % n)).collect::<Vec<_>>(), true).unwrap();
        assert!(strongly_connected_components(&ring).iter().all(|&c| c == 0));
    }
}
