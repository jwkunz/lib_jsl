//! Maximum flow and minimum cut: Edmonds-Karp and Dinic.
//!
//! See the book entry [Maximum flow and minimum cut](https://jwkunz.github.io/lib_jsl/book/graph/max-flow.html).

use lib_jsl_core::ErrorsJSL;
use std::collections::VecDeque;

/// A flow network with residual arcs. Each added edge creates a forward arc with its capacity
/// and a reverse arc of capacity zero, stored as consecutive indices `2k`, `2k + 1`.
#[derive(Debug, Clone)]
pub struct FlowNetwork {
    head: Vec<Vec<usize>>,
    to: Vec<usize>,
    capacity: Vec<f64>,
    flow: Vec<f64>,
}

/// The value of a maximum flow and the edge flows (in order of addition).
#[derive(Debug, Clone, PartialEq)]
pub struct MaxFlow {
    pub value: f64,
    pub edge_flows: Vec<f64>,
    /// The source side of a minimum cut (vertices reachable from the source in the residual
    /// graph); its capacity equals the flow value.
    pub source_side: Vec<bool>,
}

const EPS: f64 = 1e-12;

impl FlowNetwork {
    pub fn new(n: usize) -> Self {
        FlowNetwork { head: vec![Vec::new(); n], to: Vec::new(), capacity: Vec::new(), flow: Vec::new() }
    }

    /// Adds a directed edge `from → to` with a non-negative capacity; returns its index.
    pub fn add_edge(&mut self, from: usize, to: usize, capacity: f64) -> Result<usize, ErrorsJSL> {
        let n = self.head.len();
        if from >= n || to >= n || capacity.is_nan() || capacity < 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("Need valid endpoints and a non-negative capacity."));
        }
        let index = self.to.len() / 2;
        for (a, b, c) in [(from, to, capacity), (to, from, 0.0)] {
            self.head[a].push(self.to.len());
            self.to.push(b);
            self.capacity.push(c);
            self.flow.push(0.0);
        }
        Ok(index)
    }

    fn residual(&self, arc: usize) -> f64 {
        self.capacity[arc] - self.flow[arc]
    }

    fn push(&mut self, arc: usize, amount: f64) {
        self.flow[arc] += amount;
        self.flow[arc ^ 1] -= amount;
    }

    fn check(&self, source: usize, sink: usize) -> Result<(), ErrorsJSL> {
        if source >= self.head.len() || sink >= self.head.len() || source == sink {
            return Err(ErrorsJSL::InvalidInputRange("Need distinct source and sink vertices."));
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.flow.iter_mut().for_each(|f| *f = 0.0);
    }

    fn result(&self, source: usize) -> MaxFlow {
        let value = self.head[source].iter().map(|&a| self.flow[a]).sum();
        let edge_flows = (0..self.to.len() / 2).map(|k| self.flow[2 * k]).collect();
        let mut source_side = vec![false; self.head.len()];
        let mut queue = VecDeque::from([source]);
        source_side[source] = true;
        while let Some(u) = queue.pop_front() {
            for &a in &self.head[u] {
                if self.residual(a) > EPS && !source_side[self.to[a]] {
                    source_side[self.to[a]] = true;
                    queue.push_back(self.to[a]);
                }
            }
        }
        MaxFlow { value, edge_flows, source_side }
    }

    /// Edmonds-Karp: augment along shortest (BFS) residual paths, `O(V E²)`.
    pub fn edmonds_karp(&mut self, source: usize, sink: usize) -> Result<MaxFlow, ErrorsJSL> {
        self.check(source, sink)?;
        self.reset();
        loop {
            let mut parent_arc = vec![usize::MAX; self.head.len()];
            let mut queue = VecDeque::from([source]);
            let mut seen = vec![false; self.head.len()];
            seen[source] = true;
            while let Some(u) = queue.pop_front() {
                for &a in &self.head[u] {
                    let v = self.to[a];
                    if !seen[v] && self.residual(a) > EPS {
                        seen[v] = true;
                        parent_arc[v] = a;
                        queue.push_back(v);
                    }
                }
            }
            if !seen[sink] {
                return Ok(self.result(source));
            }
            let mut bottleneck = f64::INFINITY;
            let mut v = sink;
            while v != source {
                let a = parent_arc[v];
                bottleneck = bottleneck.min(self.residual(a));
                v = self.to[a ^ 1];
            }
            let mut v = sink;
            while v != source {
                let a = parent_arc[v];
                self.push(a, bottleneck);
                v = self.to[a ^ 1];
            }
        }
    }

    /// Dinic's algorithm: BFS level graphs and blocking flows by DFS with current-arc pointers,
    /// `O(V² E)` in general and much faster in practice (`O(E √V)` on unit-capacity networks).
    pub fn dinic(&mut self, source: usize, sink: usize) -> Result<MaxFlow, ErrorsJSL> {
        self.check(source, sink)?;
        self.reset();
        let n = self.head.len();
        loop {
            let mut level = vec![usize::MAX; n];
            level[source] = 0;
            let mut queue = VecDeque::from([source]);
            while let Some(u) = queue.pop_front() {
                for &a in &self.head[u] {
                    let v = self.to[a];
                    if level[v] == usize::MAX && self.residual(a) > EPS {
                        level[v] = level[u] + 1;
                        queue.push_back(v);
                    }
                }
            }
            if level[sink] == usize::MAX {
                return Ok(self.result(source));
            }
            // ANCHOR: dinic_phase
            let mut current = vec![0usize; n];
            // Iterative DFS for blocking flow.
            loop {
                let mut path: Vec<usize> = Vec::new();
                let mut u = source;
                loop {
                    if u == sink {
                        break;
                    }
                    let mut advanced = false;
                    while current[u] < self.head[u].len() {
                        let a = self.head[u][current[u]];
                        let v = self.to[a];
                        if self.residual(a) > EPS && level[v] == level[u] + 1 {
                            path.push(a);
                            u = v;
                            advanced = true;
                            break;
                        }
                        current[u] += 1;
                    }
                    if !advanced {
                        // Dead end: retreat and skip the arc that led here.
                        match path.pop() {
                            Some(a) => {
                                u = self.to[a ^ 1];
                                current[u] += 1;
                            }
                            None => break,
                        }
                    }
                }
                if u != sink {
                    break;
                }
                let bottleneck = path.iter().map(|&a| self.residual(a)).fold(f64::INFINITY, f64::min);
                for &a in &path {
                    self.push(a, bottleneck);
                }
            }
            // ANCHOR_END: dinic_phase
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clrs() -> FlowNetwork {
        // The flow network of CLRS figure 26.1: maximum flow 23.
        let mut net = FlowNetwork::new(6);
        for (u, v, c) in [(0, 1, 16.0), (0, 2, 13.0), (1, 3, 12.0), (2, 1, 4.0), (2, 4, 14.0), (3, 2, 9.0), (3, 5, 20.0), (4, 3, 7.0), (4, 5, 4.0)] {
            net.add_edge(u, v, c).unwrap();
        }
        net
    }

    #[test]
    fn both_algorithms_find_the_textbook_maximum_and_its_cut() {
        for use_dinic in [false, true] {
            let mut net = clrs();
            let flow = if use_dinic { net.dinic(0, 5) } else { net.edmonds_karp(0, 5) }.unwrap();
            assert!((flow.value - 23.0).abs() < 1e-12, "dinic={use_dinic}: {}", flow.value);
            // The cut capacity equals the flow (max-flow min-cut theorem).
            let edges = [(0, 1, 16.0), (0, 2, 13.0), (1, 3, 12.0), (2, 1, 4.0), (2, 4, 14.0), (3, 2, 9.0), (3, 5, 20.0), (4, 3, 7.0), (4, 5, 4.0)];
            let cut: f64 = edges.iter().filter(|&&(u, v, _)| flow.source_side[u] && !flow.source_side[v]).map(|e| e.2).sum();
            assert!((cut - 23.0).abs() < 1e-12);
            assert!(flow.edge_flows.iter().zip(&edges).all(|(f, e)| *f >= -1e-12 && *f <= e.2 + 1e-12));
        }
    }

    #[test]
    fn bipartite_matching_and_validation() {
        // Unit capacities: maximum bipartite matching of 4 workers to 4 jobs is 3.
        let mut net = FlowNetwork::new(10);
        let (source, sink) = (8, 9);
        for w in 0..4 {
            net.add_edge(source, w, 1.0).unwrap();
            net.add_edge(4 + w, sink, 1.0).unwrap();
        }
        for (w, j) in [(0, 0), (0, 1), (1, 0), (2, 1), (3, 2), (3, 3), (2, 0)] {
            net.add_edge(w, 4 + j, 1.0).unwrap();
        }
        assert_eq!(net.clone().edmonds_karp(source, sink).unwrap().value, 3.0);
        assert_eq!(net.dinic(source, sink).unwrap().value, 3.0);
        assert!(FlowNetwork::new(2).add_edge(0, 1, -1.0).is_err() && FlowNetwork::new(2).dinic(1, 1).is_err());
    }
}
