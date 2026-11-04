//! Figures and tables for the chapter on graph algorithms.

use crate::raster::{Colours, Mark, Panel, panels};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_graph::{FlowNetwork, Graph, a_star, bellman_ford, dijkstra, floyd_warshall, kruskal, prim, strongly_connected_components, topological_sort};
use ndarray::Array2;
use std::collections::HashMap;
use std::time::Instant;

const CHAPTER: &str = "graph";

struct Refs {
    graphs: HashMap<String, (usize, bool, Vec<(usize, usize, f64)>)>,
    values: HashMap<String, Vec<f64>>,
}

fn load() -> Refs {
    let (mut graphs, mut values) = (HashMap::new(), HashMap::new());
    for line in include_str!("../data/graph_reference.txt").lines() {
        let (key, rest) = line.split_once('|').unwrap();
        let parse = |v: &str| if v == "inf" { f64::INFINITY } else { v.parse::<f64>().unwrap() };
        if let Some(name) = key.strip_prefix("graph ") {
            let mut parts = rest.split(';');
            let n: usize = parts.next().unwrap().parse().unwrap();
            let directed = parts.next().unwrap() == "1";
            let flat: Vec<f64> = parts.next().unwrap().split(',').map(parse).collect();
            let edges = flat.chunks(3).map(|c| (c[0] as usize, c[1] as usize, c[2])).collect();
            graphs.insert(name.to_string(), (n, directed, edges));
        } else {
            values.insert(key.to_string(), rest.split(',').map(parse).collect());
        }
    }
    Refs { graphs, values }
}

fn build(refs: &Refs, name: &str) -> Graph {
    let (n, directed, edges) = &refs.graphs[name];
    Graph::from_edges(*n, edges, *directed).unwrap()
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| if x.is_infinite() && y.is_infinite() { 0.0 } else { (x - y).abs() }).fold(0.0, f64::max)
}

pub fn generate() {
    let refs = load();
    reference_table(&refs);
    shortest_path_figures();
    spanning_tree_figures();
    flow_figures();
}

fn reference_table(refs: &Refs) {
    let mut rows = Vec::new();
    let g = build(refs, "positive");
    let worst = [0usize, 17, 199].iter().map(|&s| max_diff(&dijkstra(&g, s).unwrap().distance, &refs.values[&format!("dijkstra positive {s}")])).fold(0.0, f64::max);
    rows.push(vec!["Dijkstra, 200 vertices, 1200 arcs, from three sources (`csgraph.dijkstra`)".to_string(), format!("largest difference {worst:.1e}")]);
    let fw = floyd_warshall(&g).unwrap();
    let rows_fw: Vec<f64> = [fw.distance[3].clone(), fw.distance[150].clone()].concat();
    rows.push(vec!["Floyd-Warshall, same graph, two rows (`csgraph.floyd_warshall`)".to_string(), format!("largest difference {:.1e}", max_diff(&rows_fw, &refs.values["floyd positive rows"]))]);
    let s = build(refs, "signed");
    rows.push(vec!["Bellman-Ford with negative weights, 150 vertices (`csgraph.bellman_ford`)".to_string(), format!("largest difference {:.1e}", max_diff(&bellman_ford(&s, 0).unwrap().distance, &refs.values["bellman signed 0"]))]);
    let u = build(refs, "undirected");
    let (k, p) = (kruskal(&u).unwrap(), prim(&u).unwrap());
    let want = refs.values["mst undirected"][0];
    rows.push(vec!["minimum spanning forest weight, 400 vertices (`csgraph.minimum_spanning_tree`)".to_string(), format!("Kruskal {:.3}, Prim {:.3}, SciPy {want:.3}", k.weight, p.weight)]);
    let comps = u.connected_components().iter().max().map_or(0, |m| m + 1);
    rows.push(vec!["connected components (`csgraph.connected_components`)".to_string(), format!("{comps} (SciPy {})", refs.values["components undirected"][0])]);
    let (n, _, edges) = &refs.graphs["flow"];
    let mut net = FlowNetwork::new(*n);
    for &(a, b, c) in edges {
        net.add_edge(a, b, c).unwrap();
    }
    let (ek, di) = (net.edmonds_karp(0, 99).unwrap().value, net.dinic(0, 99).unwrap().value);
    rows.push(vec!["maximum flow, 100 vertices, 700 arcs (`csgraph.maximum_flow`)".to_string(), format!("Edmonds-Karp {ek}, Dinic {di}, SciPy {}", refs.values["maxflow"][0])]);
    let sp = build(refs, "sparse");
    let labels = strongly_connected_components(&sp);
    let count = labels.iter().max().map_or(0, |m| m + 1);
    let mut smallest: HashMap<usize, usize> = HashMap::new();
    let canonical: Vec<f64> = labels.iter().enumerate().map(|(v, &l)| *smallest.entry(l).or_insert(v) as f64).collect();
    rows.push(vec!["strongly connected components, 300 vertices (`connected_components(connection=\"strong\")`)".to_string(), format!("{count} components (SciPy {}); same partition: {}", refs.values["scc sparse"][0], canonical == refs.values["scc sparse partition"])]);
    let dag = build(refs, "dag");
    let order: Vec<f64> = topological_sort(&dag).unwrap().iter().map(|&v| v as f64).collect();
    rows.push(vec!["lexicographically smallest topological order, 60 vertices (NetworkX)".to_string(), format!("identical: {}", order == refs.values["topological dag"])]);
    write(CHAPTER, "reference.md", &md_table(&["computation (reference function)", "result"], &rows));
}

// ------------------------------------------------------------------ shortest paths

fn grid_graph(w: usize, h: usize, blocked: &dyn Fn(usize, usize) -> bool) -> Graph {
    let mut edges = Vec::new();
    for r in 0..h {
        for c in 0..w {
            if blocked(r, c) {
                continue;
            }
            let id = r * w + c;
            if c + 1 < w && !blocked(r, c + 1) {
                edges.push((id, id + 1, 1.0));
            }
            if r + 1 < h && !blocked(r + 1, c) {
                edges.push((id, id + w, 1.0));
            }
        }
    }
    Graph::from_edges(w * h, &edges, false).unwrap()
}

fn shortest_path_figures() {
    // A maze-like grid: walls with gaps.
    let (w, h) = (80usize, 50usize);
    // Vertical walls, each with a five-cell gap at a different height, so a path always exists.
    let blocked = |r: usize, c: usize| c % 16 == 8 && !(((c / 16) * 19 % 40 + 3)..((c / 16) * 19 % 40 + 8)).contains(&r);
    let g = grid_graph(w, h, &blocked);
    let (s, t) = (2 * w + 1, (h - 3) * w + w - 2);
    let t0 = Instant::now();
    let d = dijkstra(&g, s).unwrap();
    let t_d = t0.elapsed().as_secs_f64();
    let manhattan = |v: usize| ((v / w) as f64 - (t / w) as f64).abs() + ((v % w) as f64 - (t % w) as f64).abs();
    let t0 = Instant::now();
    let (cost, path) = a_star(&g, s, t, manhattan).unwrap().unwrap();
    let t_a = t0.elapsed().as_secs_f64();
    // Distance field from Dijkstra, with walls and the A* path.
    let field = Array2::from_shape_fn((h, w), |(r, c)| if blocked(r, c) { -10.0 } else { d.distance[r * w + c].min(200.0) });
    let mut panel = Panel::new("distance from the start (dark: near), walls black, A* path in red", field, Colours::Ramp(0.0, 160.0));
    let mut mask = Array2::from_elem((h, w), false);
    for &v in &path {
        mask[(v / w, v % w)] = true;
    }
    panel.marks.push(Mark::Mask(mask, "#ff3030"));
    let walls = Array2::from_shape_fn((h, w), |(r, c)| blocked(r, c));
    panel.marks.push(Mark::Mask(walls, "#000000"));
    write(CHAPTER, "grid_paths.svg", &panels("Shortest paths on a grid with walls (unit steps)", &[panel], 1, 560.0));
    write(CHAPTER, "grid_facts.md", &format!("Dijkstra's distance to the target is {}, A*'s path costs {cost} over {} steps. Dijkstra settled the whole grid in {:.2} ms; A* with the Manhattan heuristic stopped at the target after {:.2} ms.", d.distance[t], path.len() - 1, 1e3 * t_d, 1e3 * t_a));
    // Scaling on random sparse graphs.
    let mut seed = 7u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    };
    let (mut pd, mut pb, mut pf) = (Vec::new(), Vec::new(), Vec::new());
    for &n in &[100usize, 200, 400, 800, 1600, 3200] {
        let edges: Vec<(usize, usize, f64)> = (0..4 * n).map(|_| ((rnd() * n as f64) as usize, (rnd() * n as f64) as usize, 1.0 + 9.0 * rnd())).collect();
        let g = Graph::from_edges(n, &edges, true).unwrap();
        let time = |f: &dyn Fn()| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_secs_f64() * 1e3
        };
        pd.push((n as f64, time(&|| { dijkstra(&g, 0).unwrap(); })));
        pb.push((n as f64, time(&|| { bellman_ford(&g, 0).unwrap(); })));
        if n <= 800 {
            pf.push((n as f64, time(&|| { floyd_warshall(&g).unwrap(); })));
        }
    }
    let mut plot = Plot::new("Time on random directed graphs with 4V arcs", "vertices V", "time (ms)");
    plot.series.push(Series::new("Dijkstra (one source)", pd, Style::LineMarkers, 0));
    plot.series.push(Series::new("Bellman-Ford (one source)", pb, Style::LineMarkers, 1));
    plot.series.push(Series::new("Floyd-Warshall (all pairs)", pf, Style::LineMarkers, 2));
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "shortest_path_timing.svg", &plot.render());
}

// ------------------------------------------------------------------ spanning trees

fn spanning_tree_figures() {
    let mut seed = 11u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    };
    let n = 250;
    // y spans 0.553 so that both axes have the same scale in the 1.81:1 plot area.
    let pts: Vec<(f64, f64)> = (0..n).map(|_| (rnd(), rnd() * 0.553)).collect();
    // The complete Euclidean graph.
    let mut edges = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            edges.push((i, j, ((pts[i].0 - pts[j].0).powi(2) + (pts[i].1 - pts[j].1).powi(2)).sqrt()));
        }
    }
    let g = Graph::from_edges(n, &edges, false).unwrap();
    let t0 = Instant::now();
    let k = kruskal(&g).unwrap();
    let t_k = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let p = prim(&g).unwrap();
    let t_p = t0.elapsed().as_secs_f64();
    let mut plot = Plot::new(&format!("Euclidean minimum spanning tree of {n} random points (total length {:.3})", k.weight), "x", "y");
    for e in &k.edges {
        plot.series.push(Series::new("", vec![pts[e.from], pts[e.to]], Style::Line, 0));
    }
    plot.series.push(Series::new("points", pts.clone(), Style::Markers, 1));
    plot.x_range = Some((0.0, 1.0));
    plot.y_range = Some((0.0, 0.553));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "euclidean_mst.svg", &plot.render());
    write(CHAPTER, "mst_facts.md", &format!("On this complete graph of {} edges, Kruskal took {:.1} ms and Prim {:.1} ms; both found total length {:.6} and {:.6}.", edges.len(), 1e3 * t_k, 1e3 * t_p, k.weight, p.weight));
}

// ------------------------------------------------------------------ flows

fn flow_figures() {
    let mut seed = 5u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    };
    let mut rows = Vec::new();
    for &(n, m) in &[(200usize, 1600usize), (800, 6400), (3200, 25600)] {
        let mut net = FlowNetwork::new(n);
        let mut arcs = Vec::new();
        for _ in 0..m {
            let (a, b) = ((rnd() * n as f64) as usize, (rnd() * n as f64) as usize);
            if a != b {
                let c = (1.0 + 99.0 * rnd()).round();
                net.add_edge(a, b, c).unwrap();
                arcs.push((a, b, c));
            }
        }
        let t0 = Instant::now();
        let ek = net.edmonds_karp(0, n - 1).unwrap();
        let t_ek = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let di = net.dinic(0, n - 1).unwrap();
        let t_di = t0.elapsed().as_secs_f64();
        // Capacity of the minimum cut found: arcs from the source side to the sink side.
        let cut: f64 = arcs.iter().filter(|&&(a, b, _)| di.source_side[a] && !di.source_side[b]).map(|a| a.2).sum();
        rows.push(vec![format!("{n}"), format!("{}", arcs.len()), format!("{}", ek.value), format!("{}", di.value), format!("{cut}"), format!("{:.1}", 1e3 * t_ek), format!("{:.1}", 1e3 * t_di)]);
    }
    write(CHAPTER, "flow_table.md", &md_table(&["vertices", "arcs", "Edmonds-Karp flow", "Dinic flow", "minimum cut capacity", "Edmonds-Karp (ms)", "Dinic (ms)"], &rows));
}
