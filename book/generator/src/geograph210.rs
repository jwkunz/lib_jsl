//! Tables and figures for release 2.10.0: rotation fitting, Procrustes analysis, geometric
//! slerp, Hausdorff distance and k-d tree extras (geometry chapter), and Johnson's algorithm,
//! Yen's k shortest paths, bipartite matchings and Laplacians (graph chapter), compared with
//! SciPy.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_geometry::algorithms::{KdTree, PeriodicKdTree, Quaternion, Rotation3, align_vectors, directed_hausdorff, geometric_slerp, procrustes, rotation_mean};
use lib_jsl_graph::{Graph, johnson, laplacian, maximum_bipartite_matching, min_weight_full_bipartite_matching, structural_rank, yen};

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/geograph_reference.txt"));
    geometry_table(&refs);
    graph_table(&refs);
    procrustes_figure();
    yen_figure();
}

fn rows3(v: &[f64]) -> Vec<[f64; 3]> {
    v.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn rows_n(v: &[f64], d: usize) -> Vec<Vec<f64>> {
    v.chunks(d).map(|c| c.to_vec()).collect()
}

/// The angle between a rotation and a SciPy quaternion `[x, y, z, w]`.
fn angle_to(r: &Rotation3, q: &[f64]) -> f64 {
    r.angle_to(&Rotation3::from_quaternion(Quaternion::new(q[3], q[0], q[1], q[2])))
}

fn max_abs(a: impl IntoIterator<Item = f64>, b: &[f64]) -> f64 {
    a.into_iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn geometry_table(refs: &Refs) {
    let mut rows = Vec::new();
    let a = rows3(refs.values("align a"));
    let b = rows3(refs.values("align b"));
    let w = refs.values("align w").to_vec();
    let mut winf = w.clone();
    winf[3] = f64::INFINITY;
    for (name, label, al) in [
        ("plain", "`align_vectors`, 12 noisy pairs", align_vectors(&a, &b, None).unwrap()),
        ("weighted", "`align_vectors`, weighted", align_vectors(&a, &b, Some(&w)).unwrap()),
        ("inf", "`align_vectors`, one infinite weight", align_vectors(&a, &b, Some(&winf)).unwrap()),
        ("single", "`align_vectors`, a single pair", align_vectors(&a[..1], &b[..1], None).unwrap()),
    ] {
        let r = refs.get(&format!("align {name}"));
        rows.push(vec![label.into(), fmt_diff(angle_to(&al.rotation, &r[0])), format!("rssd {}", fmt_diff((al.rssd - r[1][0]).abs()))]);
    }
    let rv = rows3(refs.values("mean rotvecs"));
    let rots: Vec<Rotation3> = rv.iter().map(|&v| Rotation3::from_rotvec(v)).collect();
    let m = refs.get("mean");
    let d0 = angle_to(&rotation_mean(&rots, None).unwrap(), &m[0]);
    let d1 = angle_to(&rotation_mean(&rots, Some(refs.values("mean w"))).unwrap(), &m[1]);
    rows.push(vec!["`rotation_mean`, 10 rotations, plain and weighted".into(), fmt_diff(d0.max(d1)), "".into()]);
    let rin = rows3(refs.values("rotvec in"));
    let qref = refs.values("rotvec quat");
    let dq = rin.iter().enumerate().map(|(i, &v)| angle_to(&Rotation3::from_rotvec(v), &qref[4 * i..4 * i + 4])).fold(0.0, f64::max);
    let back: Vec<f64> = rin.iter().flat_map(|&v| Rotation3::from_rotvec(v).as_rotvec()).collect();
    let mrp: Vec<f64> = rin.iter().flat_map(|&v| Rotation3::from_rotvec(v).as_mrp()).collect();
    let fm = refs.values("from mrp");
    let dfm = rin.iter().enumerate().map(|(i, &v)| angle_to(&Rotation3::from_mrp(v.map(|x| x / 3.0)), &fm[4 * i..4 * i + 4])).fold(0.0, f64::max);
    rows.push(vec![
        "`from_rotvec`, `as_rotvec`, `as_mrp`, `from_mrp` (angles 3·10⁻⁶ to π − 10⁻⁹)".into(),
        fmt_diff(dq.max(dfm)),
        format!("components {}", fmt_diff(max_abs(back, refs.values("rotvec back")).max(max_abs(mrp, refs.values("mrp"))))),
    ]);
    let d1 = rows_n(refs.values("procrustes d1"), 3);
    let d2 = rows_n(refs.values("procrustes d2"), 3);
    let p = procrustes(&d1, &d2).unwrap();
    let r = refs.get("procrustes");
    rows.push(vec![
        "`procrustes`, 15 points in 3-D".into(),
        fmt_diff(max_abs(p.mtx2.iter().flatten().copied(), &r[0])),
        format!("disparity {:.6} ({})", p.disparity, fmt_diff((p.disparity - r[1][0]).abs())),
    ]);
    let end = refs.values("slerp end");
    let t: Vec<f64> = (0..7).map(|i| i as f64 / 6.0).collect();
    let s = geometric_slerp(&[1.0, 0.0, 0.0, 0.0], end, &t).unwrap();
    rows.push(vec!["`geometric_slerp`, 4-D, 7 points".into(), fmt_diff(max_abs(s.into_iter().flatten(), refs.values("slerp"))), "".into()]);
    let u = rows_n(refs.values("hausdorff u"), 3);
    let v = rows_n(refs.values("hausdorff v"), 3);
    let h = directed_hausdorff(&u, &v).unwrap();
    let hr = refs.get("hausdorff");
    let hb = directed_hausdorff(&v, &u).unwrap();
    rows.push(vec![
        "`directed_hausdorff`, 200 and 150 points".into(),
        fmt_diff((h.distance - hr[0][0]).abs().max((hb.distance - hr[1][0]).abs())),
        if (h.index_u as f64, h.index_v as f64) == (hr[0][1], hr[0][2]) { "same pair of points".into() } else { "different pair".into() },
    ]);
    let pts = rows3(refs.values("kd pts"));
    let tree = KdTree::new(&pts);
    let pairs: Vec<f64> = tree.query_pairs(0.08).into_iter().flat_map(|(i, j)| [i as f64, j as f64]).collect();
    let same_pairs = pairs == refs.values("kd pairs");
    let other = rows3(refs.values("kd other"));
    let ot = KdTree::new(&other);
    let counts: Vec<f64> = [0.05, 0.1, 0.2].iter().map(|&r| tree.count_neighbors(&ot, r) as f64).collect();
    rows.push(vec![
        format!("`query_pairs` ({} pairs), `count_neighbors` ({:?})", pairs.len() / 2, counts.iter().map(|c| *c as usize).collect::<Vec<_>>()),
        if same_pairs && counts == refs.values("kd count") { "identical".into() } else { "differ".into() },
        "".into(),
    ]);
    let pt = PeriodicKdTree::new(&pts, [1.0; 3]).unwrap();
    let q = rows3(refs.values("kd q"));
    let pr = refs.get("kd periodic");
    let mut dmax = 0.0f64;
    let mut same_idx = true;
    for (i, &qq) in q.iter().enumerate() {
        for (k, (idx, d)) in pt.k_nearest(qq, 4).into_iter().enumerate() {
            dmax = dmax.max((d - pr[0][4 * i + k]).abs());
            same_idx &= idx as f64 == pr[1][4 * i + k];
        }
    }
    let ppairs = pt.query_pairs(0.08).len() as f64;
    rows.push(vec![
        "Periodic box: 4 nearest of 20 queries; pairs".into(),
        fmt_diff(dmax),
        format!("{}; {} pairs ({})", if same_idx { "same neighbours" } else { "different neighbours" }, ppairs, if ppairs == refs.values("kd periodic pairs")[0] { "same" } else { "differ" }),
    ]);
    write("geometry", "geograph_geometry.md", &md_table(&["Function", "Difference from SciPy", "Notes"], &rows));
}

fn graph_from(refs: &Refs, abs_plus: bool, directed: bool) -> Graph {
    let e = refs.values("graph edges");
    let edges: Vec<(usize, usize, f64)> = e.chunks(3).map(|c| (c[0] as usize, c[1] as usize, if abs_plus { c[2].abs() + 0.1 } else { c[2] })).collect();
    if directed {
        Graph::from_edges(40, &edges, true).unwrap()
    } else {
        // SciPy's directed=False treats each stored edge as usable both ways; duplicates keep the
        // smaller weight.
        let mut best: std::collections::HashMap<(usize, usize), f64> = std::collections::HashMap::new();
        for (u, v, w) in edges {
            let k = (u.min(v), u.max(v));
            let x = best.entry(k).or_insert(f64::INFINITY);
            *x = x.min(w);
        }
        let list: Vec<(usize, usize, f64)> = best.into_iter().map(|((u, v), w)| (u, v, w)).collect();
        Graph::from_edges(40, &list, false).unwrap()
    }
}

fn graph_table(refs: &Refs) {
    let mut rows = Vec::new();
    let g = graph_from(refs, false, true);
    let j = johnson(&g).unwrap();
    let want = refs.values("johnson");
    let got: Vec<f64> = j.distance.iter().flatten().copied().collect();
    let d = got.iter().zip(want).fold(0.0f64, |m, (a, b)| if a.is_infinite() && b.is_infinite() { m } else { m.max((a - b).abs()) });
    let negatives = g.edges().iter().filter(|e| e.weight < 0.0).count();
    rows.push(vec![format!("`johnson`, 40 vertices, {} edges ({negatives} negative)", g.edge_count()), fmt_diff(d), "all 1600 distances".into()]);
    for (directed, key) in [(true, "yen"), (false, "yen undirected")] {
        let g = graph_from(refs, true, directed);
        let p = yen(&g, 0, 17, 8).unwrap();
        let lens: Vec<f64> = p.iter().map(|x| x.0).collect();
        rows.push(vec![
            format!("`yen`, 8 shortest paths 0 → 17, {}", if directed { "directed" } else { "undirected" }),
            fmt_diff(max_abs(lens.iter().copied(), refs.values(key))),
            format!("lengths {:.3} … {:.3}", lens[0], lens[lens.len() - 1]),
        ]);
    }
    let gp = graph_from(refs, true, true);
    for (name, normed, out) in [("plain", false, false), ("normed", true, false), ("out", false, true)] {
        let (l, d) = laplacian(&gp, normed, out);
        let r = refs.get(&format!("laplacian {name}"));
        let dl = max_abs(l.into_iter().flatten(), &r[0]).max(max_abs(d, &r[1]));
        let label = match name {
            "plain" => "`laplacian` (in-degrees)",
            "normed" => "`laplacian`, normalised",
            _ => "`laplacian`, out-degrees",
        };
        rows.push(vec![label.into(), fmt_diff(dl), "".into()]);
    }
    let be: Vec<(usize, usize)> = refs.values("bip edges").chunks(2).map(|c| (c[0] as usize, c[1] as usize)).collect();
    let m = maximum_bipartite_matching(30, 40, &be).unwrap();
    let r = refs.get("bip matching");
    let sr = structural_rank(30, 40, &be).unwrap();
    rows.push(vec![
        "`maximum_bipartite_matching`, `structural_rank`, 30 × 40".into(),
        if m.size() as f64 == r[0][0] && sr as f64 == r[1][0] { "same size".into() } else { "differ".into() },
        format!("{} pairs; structural rank {}", m.size(), sr),
    ]);
    let we: Vec<(usize, usize, f64)> = refs.values("mw edges").chunks(3).map(|c| (c[0] as usize, c[1] as usize, c[2])).collect();
    for (maximize, key) in [(false, "mw"), (true, "mw max")] {
        let (ri, ci) = min_weight_full_bipartite_matching(25, 35, &we, maximize).unwrap();
        let r = refs.get(key);
        let wsum: f64 = ri.iter().zip(&ci).map(|(&a, &b)| we.iter().find(|e| e.0 == a && e.1 == b).unwrap().2).sum();
        let same = ci.iter().map(|&c| c as f64).eq(r[1].iter().copied());
        rows.push(vec![
            format!("`min_weight_full_bipartite_matching`, 25 × 35{}", if maximize { ", maximize" } else { "" }),
            fmt_diff((wsum - r[2][0]).abs()),
            format!("total {wsum:.4}; {}", if same { "same assignment" } else { "different assignment" }),
        ]);
    }
    write("graph", "geograph_graph.md", &md_table(&["Function", "Difference from SciPy", "Notes"], &rows));
}

fn procrustes_figure() {
    // An arrow-like outline, and a rotated, scaled, shifted and perturbed copy.
    let shape: Vec<Vec<f64>> = [[0.0, 0.0], [3.0, 0.0], [3.0, -1.0], [5.0, 1.0], [3.0, 3.0], [3.0, 2.0], [0.0, 2.0]].iter().map(|p| p.to_vec()).collect();
    let (s, c) = (2.1f64).sin_cos();
    let wobble = [0.15, -0.1, 0.2, -0.05, 0.1, -0.2, 0.05];
    let moved: Vec<Vec<f64>> = shape.iter().enumerate().map(|(i, p)| vec![1.7 * (c * p[0] - s * p[1]) + 6.0 + wobble[i], 1.7 * (s * p[0] + c * p[1]) - 2.0 - wobble[6 - i]]).collect();
    let r = procrustes(&shape, &moved).unwrap();
    let closed = |pts: &[Vec<f64>]| {
        let mut v: Vec<(f64, f64)> = pts.iter().map(|p| (p[0], p[1])).collect();
        v.push(v[0]);
        v
    };
    // The standardised but unrotated copy, for comparison.
    let mean: Vec<f64> = (0..2).map(|k| moved.iter().map(|p| p[k]).sum::<f64>() / 7.0).collect();
    let centred: Vec<Vec<f64>> = moved.iter().map(|p| vec![p[0] - mean[0], p[1] - mean[1]]).collect();
    let nrm = centred.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
    let unrot: Vec<Vec<f64>> = centred.iter().map(|p| p.iter().map(|v| v / nrm).collect()).collect();
    let mut plot = Plot::new(&format!("Procrustes superposition: disparity {:.4}", r.disparity), "", "");
    plot.legend = Legend::BottomRight;
    plot.x_range = Some((-0.7, 0.7));
    plot.y_range = Some((-0.9, 0.5));
    plot.series.push(Series::new("first shape, centred and scaled", closed(&r.mtx1), Style::LineMarkers, 0));
    plot.series.push(Series::new("second shape, centred and scaled", closed(&unrot), Style::Dashed, 7));
    plot.series.push(Series::new("second shape, rotated onto the first", closed(&r.mtx2), Style::LineMarkers, 1));
    write("geometry", "procrustes.svg", &plot.render());
}

fn yen_figure() {
    // A 7 × 7 grid with deterministic weights; the four shortest paths across it.
    let n = 7;
    let id = |i: usize, j: usize| i * n + j;
    let weight = |a: usize, b: usize| 0.3 + 3.0 * (((a * 37 + b * 101 + a * b) % 23) as f64 / 23.0);
    let mut edges = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if j + 1 < n {
                edges.push((id(i, j), id(i, j + 1), weight(id(i, j), id(i, j + 1))));
            }
            if i + 1 < n {
                edges.push((id(i, j), id(i + 1, j), weight(id(i, j), id(i + 1, j))));
            }
        }
    }
    let g = Graph::from_edges(n * n, &edges, false).unwrap();
    let paths = yen(&g, id(0, 0), id(n - 1, n - 1), 4).unwrap();
    let pos = |v: usize| ((v % n) as f64, (v / n) as f64);
    let mut plot = Plot::new("The four shortest paths across a weighted grid (Yen)", "", "");
    plot.legend = Legend::TopLeft;
    for (k, &(a, b, _)) in edges.iter().enumerate() {
        plot.series.push(Series::new(if k == 0 { "edges" } else { "" }, vec![pos(a), pos(b)], Style::Line, 7));
    }
    for (k, (len, p)) in paths.iter().enumerate() {
        let off = 0.06 * k as f64 - 0.09;
        plot.series.push(Series::new(&format!("path {}: length {len:.2}", k + 1), p.iter().map(|&v| (pos(v).0 + off, pos(v).1 + off)).collect(), Style::Line, k));
    }
    write("graph", "yen_paths.svg", &plot.render());
}
