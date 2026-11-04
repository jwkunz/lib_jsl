//! Maximum flow and minimum cut.

use lib_jsl_graph::FlowNetwork;

fn main() {
    // The classic six-vertex network (source 0, sink 5).
    let arcs = [(0, 1, 16.0), (0, 2, 13.0), (1, 2, 10.0), (2, 1, 4.0), (1, 3, 12.0), (3, 2, 9.0), (2, 4, 14.0), (4, 3, 7.0), (3, 5, 20.0), (4, 5, 4.0)];
    let mut net = FlowNetwork::new(6);
    for &(a, b, c) in &arcs {
        net.add_edge(a, b, c).unwrap();
    }
    let ek = net.edmonds_karp(0, 5).unwrap();
    let di = net.dinic(0, 5).unwrap();
    println!("maximum flow: Edmonds-Karp {}, Dinic {}", ek.value, di.value);
    println!("arc flows: {:?}", di.edge_flows);
    let cut: Vec<(usize, usize, f64)> = arcs.iter().copied().filter(|&(a, b, _)| di.source_side[a] && !di.source_side[b]).collect();
    println!("minimum cut {:?}, capacity {}", cut, cut.iter().map(|c| c.2).sum::<f64>());
}
