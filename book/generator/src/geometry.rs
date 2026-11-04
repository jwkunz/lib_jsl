//! Figures and tables for the chapter on computational geometry.

use crate::draw::{Scene, add_mesh, project, render, tint};
use crate::raster::{Colours, Panel, panels};
use lib_jsl_geometry::four_d::{ChristoffelSymbols4D, Metric4, SpaceTimeVector4D};
use ndarray::Array2;
use crate::spectral::md_table;
use crate::svg::{Legend, PALETTE, Plot, Series, Style};
use crate::write;
use lib_jsl_geometry::algorithms::*;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::time::Instant;

const CHAPTER: &str = "geometry";

fn load() -> HashMap<String, Vec<f64>> {
    include_str!("../data/geometry_reference.txt")
        .lines()
        .map(|line| {
            let (key, rest) = line.split_once('|').unwrap();
            (key.to_string(), rest.split(',').map(|v| v.parse::<f64>().unwrap()).collect())
        })
        .collect()
}

fn pts2(v: &[f64]) -> Vec<[f64; 2]> {
    v.chunks(2).map(|c| [c[0], c[1]]).collect()
}

fn pts3(v: &[f64]) -> Vec<[f64; 3]> {
    v.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// A small deterministic generator for the figures' random inputs.
fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn gauss(r: &mut impl FnMut() -> f64) -> f64 {
    (-2.0 * (1.0 - r()).ln()).sqrt() * (2.0 * PI * r()).cos()
}

fn time<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let out = f();
    (out, t0.elapsed().as_secs_f64())
}

fn ms(seconds: f64) -> String {
    if seconds < 1e-3 { format!("{:.0} µs", 1e6 * seconds) } else { format!("{:.1} ms", 1e3 * seconds) }
}

pub fn generate() {
    let refs = load();
    reference_table(&refs);
    delaunay_figures();
    hull_figures();
    boolean_figures();
    tree_figures();
    intersection_figures();
    mesh_figures();
    rotation_figures();
    spacetime_figures();
}

// ------------------------------------------------------------------ references

fn reference_table(refs: &HashMap<String, Vec<f64>>) {
    let mut rows = Vec::new();
    let p = pts2(&refs["points uniform"]);
    let t = delaunay_triangulation(&p).unwrap();
    let mut mine: Vec<[usize; 3]> = t.triangles.iter().map(|t| { let mut s = *t; s.sort_unstable(); s }).collect();
    mine.sort_unstable();
    let want: Vec<[usize; 3]> = refs["delaunay uniform"].chunks(3).map(|c| [c[0] as usize, c[1] as usize, c[2] as usize]).collect();
    rows.push(vec!["Delaunay triangulation, 500 uniform points (`Delaunay`)".into(), format!("{} triangles, {}", mine.len(), if mine == want { "identical to Qhull's".to_string() } else { format!("{} differ", mine.iter().filter(|t| !want.contains(t)).count()) })]);

    let hull = convex_hull_2d(&p);
    let mut hv = hull.clone();
    hv.sort_unstable();
    let h2 = &refs["hull2 uniform"];
    let want: Vec<usize> = h2[1..].iter().map(|&v| v as usize).collect();
    rows.push(vec!["2D convex hull, same points (`ConvexHull`)".into(), format!("{} vertices, same set: {}; area difference {:.1e}", hull.len(), hv == want, (polygon_area_2d(&p, &hull) - h2[0]).abs())]);

    for name in ["ball", "sphere"] {
        let q = pts3(&refs[&format!("points {name}")]);
        let h = convex_hull_3d(&q).unwrap();
        let r = &refs[&format!("hull3 {name}")];
        let want: Vec<usize> = r[2..].iter().map(|&v| v as usize).collect();
        let label = if name == "ball" { "3D convex hull, 1000 points in a ball" } else { "3D convex hull, 400 points on a sphere" };
        rows.push(vec![format!("{label} (`ConvexHull`)"), format!("{} vertices, same set: {}; volume difference {:.1e}, area difference {:.1e}", h.vertices().len(), h.vertices() == want, (h.volume(&q) - r[0]).abs(), (h.surface_area(&q) - r[1]).abs())]);
    }

    let s = pts2(&refs["points sites"]);
    let v = voronoi_diagram(&s, [0.0, 0.0], [1.0, 1.0]).unwrap();
    let worst = v.cells.iter().zip(&refs["voronoi areas"]).map(|(c, a)| (polygon_signed_area(c) - a).abs()).fold(0.0, f64::max);
    rows.push(vec!["Voronoi cells of 300 sites clipped to the unit square (`Voronoi` of the mirrored sites)".into(), format!("largest cell-area difference {worst:.1e}")]);

    let cloud = pts3(&refs["points cloud"]);
    let queries = pts3(&refs["points queries"]);
    let tree = KdTree::new(&cloud);
    let (idx, dist) = (&refs["knn index"], &refs["knn distance"]);
    let (mut same, mut derr) = (0, 0.0f64);
    for (j, q) in queries.iter().enumerate() {
        let got = tree.k_nearest(*q, 8);
        same += usize::from(got.iter().zip(&idx[8 * j..8 * j + 8]).all(|(g, &w)| g.0 == w as usize));
        derr = got.iter().zip(&dist[8 * j..8 * j + 8]).map(|(g, &w)| (g.1 - w).abs()).fold(derr, f64::max);
    }
    rows.push(vec!["8 nearest neighbours of 200 queries among 4000 points in 3D (`cKDTree.query`)".into(), format!("identical for {same} of {}; largest distance difference {derr:.1e}", queries.len())]);
    let quad = QuadTree::from_points(&s);
    let centres = pts2(&refs["points centres"]);
    let agree = centres.iter().zip(&refs["radius counts"]).filter(|(c, n)| quad.query_radius(**c, 0.08).len() == **n as usize).count();
    rows.push(vec!["points within 0.08 of 100 centres, quadtree (`cKDTree.query_ball_point`)".into(), format!("same count for {agree} of {}", centres.len())]);

    let areas = &refs["boolean areas"];
    let mut worst = 0.0f64;
    for trial in 0..200 {
        let a = pts2(&refs[&format!("polygon a{trial}")]);
        let b = pts2(&refs[&format!("polygon b{trial}")]);
        for (k, op) in [BooleanOperation::Intersection, BooleanOperation::Union, BooleanOperation::Difference].into_iter().enumerate() {
            let got: f64 = polygon_boolean(&a, &b, op).iter().map(|p| polygon_signed_area(p)).sum();
            worst = worst.max((got - areas[3 * trial + k]).abs());
        }
    }
    rows.push(vec!["intersection, union and difference of 200 pairs of star polygons (Shapely)".into(), format!("largest area difference {worst:.1e}")]);

    let angles = pts3(&refs["points euler angles"]);
    use Axis::*;
    for (name, seq) in [("XYZ", [X, Y, Z]), ("ZYX", [Z, Y, X]), ("ZXZ", [Z, X, Z]), ("YZY", [Y, Z, Y])] {
        let e = EulerSequence(seq);
        let (m, back) = (&refs[&format!("euler matrix {name}")], &refs[&format!("euler angles {name}")]);
        let (mut merr, mut aerr) = (0.0f64, 0.0f64);
        for (j, a) in angles.iter().enumerate() {
            let mat = e.to_matrix(*a);
            merr = (0..9).map(|k| (mat[k / 3][k % 3] - m[9 * j + k]).abs()).fold(merr, f64::max);
            let got = e.from_matrix(&mat).unwrap();
            aerr = (0..3).map(|k| { let d = (got[k] - back[3 * j + k]).rem_euclid(2.0 * PI); d.min(2.0 * PI - d) }).fold(aerr, f64::max);
        }
        rows.push(vec![format!("intrinsic {name} Euler angles, 100 rotations (`Rotation.from_euler`, `as_euler`)"), format!("matrix difference {merr:.1e}, angle difference {aerr:.1e}")]);
    }
    let ends = &refs["points slerp ends"];
    // SciPy takes quaternions scalar last.
    let q = |k: usize| Quaternion::new(ends[4 * k + 3], ends[4 * k], ends[4 * k + 1], ends[4 * k + 2]).normalized();
    let (q0, q1) = (q(0), q(1));
    let want = &refs["slerp"];
    // Compare components, allowing for q and -q being the same rotation.
    let worst = (0..11)
        .map(|k| {
            let g = q0.slerp(&q1, k as f64 / 10.0);
            let w = &want[4 * k..4 * k + 4];
            let d = |s: f64| (g.w - s * w[0]).abs().max((g.x - s * w[1]).abs()).max((g.y - s * w[2]).abs()).max((g.z - s * w[3]).abs());
            d(1.0).min(d(-1.0))
        })
        .fold(0.0, f64::max);
    rows.push(vec!["spherical linear interpolation at 11 times (`Slerp`)".into(), format!("largest quaternion component difference {worst:.1e}")]);
    write(CHAPTER, "reference.md", &md_table(&["computation (reference)", "result"], &rows));
}

// ------------------------------------------------------------------ Delaunay and Voronoi

fn circumcircle(p: &[[f64; 2]], t: [usize; 3]) -> ([f64; 2], f64) {
    let tri = Triangulation { triangles: vec![t] };
    let c = tri.circumcenter(p, 0);
    (c, ((c[0] - p[t[0]][0]).powi(2) + (c[1] - p[t[0]][1]).powi(2)).sqrt())
}

fn circle(c: [f64; 2], r: f64) -> Vec<[f64; 2]> {
    (0..=96).map(|k| { let a = 2.0 * PI * k as f64 / 96.0; [c[0] + r * a.cos(), c[1] + r * a.sin()] }).collect()
}

fn delaunay_figures() {
    let mut r = rng(4);
    let p: Vec<[f64; 2]> = (0..40).map(|_| [r(), r()]).collect();
    let t = delaunay_triangulation(&p).unwrap();
    let window = [-0.05, 1.05, -0.05, 1.05];
    let mut a = Scene::new("Delaunay triangulation, with two empty circumcircles", window);
    for tri in &t.triangles {
        a.polygon(tri.iter().map(|&i| p[i]).collect(), Some(tint(5, 0.12)), Some(PALETTE[0]), 1.0);
    }
    // The circumcircles of the triangles nearest two points well apart.
    for (n, target) in [[0.3, 0.68], [0.68, 0.3]].into_iter().enumerate() {
        let k = (0..t.triangles.len())
            .min_by(|&x, &y| {
                let d = |k: usize| { let c = t.triangles[k].iter().fold([0.0, 0.0], |s, &i| [s[0] + p[i][0] / 3.0, s[1] + p[i][1] / 3.0]); (c[0] - target[0]).powi(2) + (c[1] - target[1]).powi(2) };
                d(x).total_cmp(&d(y))
            })
            .unwrap();
        let (c, rad) = circumcircle(&p, t.triangles[k]);
        a.polygon(t.triangles[k].iter().map(|&i| p[i]).collect(), Some(tint(4 - 2 * n, 0.45)), Some(PALETTE[0]), 1.0);
        a.dashed(circle(c, rad), PALETTE[1], 1.3);
        a.dot(c, 2.5, PALETTE[1]);
    }
    for &q in &p {
        a.dot(q, 2.8, PALETTE[6]);
    }
    let v = voronoi_diagram(&p, [-0.05, -0.05], [1.05, 1.05]).unwrap();
    let mut b = Scene::new("Voronoi cells, with the Delaunay edges dashed", window);
    for (k, cell) in v.cells.iter().enumerate() {
        b.polygon(cell.clone(), Some(tint(k % 6, 0.28)), Some(PALETTE[6]), 0.8);
    }
    for (i, j) in t.edges() {
        b.dashed(vec![p[i], p[j]], PALETTE[7], 0.7);
    }
    for &q in &p {
        b.dot(q, 2.8, PALETTE[6]);
    }
    write(CHAPTER, "delaunay_voronoi.svg", &render("", &[a, b], 2, 330.0));

    // Running time against the number of points.
    let mut series = Vec::new();
    let mut rows = Vec::new();
    for (label, colour, cloud) in [("uniform in a square", 0usize, 0u8), ("Gaussian cloud", 1, 1), ("points on a circle", 2, 2)] {
        let mut pts_t = Vec::new();
        for &n in &[1000usize, 3000, 10_000, 30_000, 100_000, 300_000] {
            if cloud == 2 && n > 30_000 {
                continue;
            }
            let mut r = rng(n as u64);
            let pts: Vec<[f64; 2]> = (0..n)
                .map(|k| match cloud {
                    0 => [r(), r()],
                    1 => [gauss(&mut r), gauss(&mut r)],
                    _ => { let a = 2.0 * PI * k as f64 / n as f64; [a.cos(), a.sin()] }
                })
                .collect();
            // The best of three runs.
            let (tri, mut secs) = time(|| delaunay_triangulation(&pts).unwrap());
            for _ in 0..2 {
                secs = secs.min(time(|| delaunay_triangulation(&pts).unwrap()).1);
            }
            pts_t.push((n as f64, 1e6 * secs / n as f64));
            if n == 30_000 {
                rows.push(vec![label.to_string(), format!("{}", tri.triangles.len()), ms(secs)]);
            }
        }
        series.push(Series::new(label, pts_t, Style::LineMarkers, colour));
    }
    let mut plot = Plot::new("Delaunay triangulation time per point", "points n", "µs per point");
    plot.log_x = true;
    plot.series = series;
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "delaunay_timing.svg", &plot.render());
    write(CHAPTER, "delaunay_timing.md", &md_table(&["30 000 points", "triangles", "time"], &rows));
}

// ------------------------------------------------------------------ convex hulls

fn hull_figures() {
    let mut r = rng(9);
    let p: Vec<[f64; 2]> = (0..300).map(|_| [gauss(&mut r), 0.6 * gauss(&mut r)]).collect();
    let hull = convex_hull_2d(&p);
    let mut a = Scene::new(&format!("2D hull of 300 Gaussian points: {} vertices", hull.len()), Scene::fit(p.iter().copied(), 0.06));
    a.polygon(hull.iter().map(|&i| p[i]).collect(), Some(tint(5, 0.15)), Some(PALETTE[0]), 1.6);
    for &q in &p {
        a.dot(q, 1.8, PALETTE[6]);
    }
    for &i in &hull {
        a.dot(p[i], 3.2, PALETTE[1]);
    }
    let mut r = rng(10);
    let q: Vec<[f64; 3]> = (0..400).map(|_| [gauss(&mut r), gauss(&mut r), 0.7 * gauss(&mut r)]).collect();
    let h = convex_hull_3d(&q).unwrap();
    let (az, el) = (0.6, 0.45);
    let window = Scene::fit(q.iter().map(|&x| project(x, az, el).0), 0.06);
    let mut b = Scene::new(&format!("3D hull of 400 Gaussian points: {} faces", h.faces.len()), window);
    add_mesh(&mut b, &q, &h.faces, (az, el), 0, true, Some("#333333"));
    write(CHAPTER, "hulls.svg", &render("", &[a, b], 2, 330.0));

    // Running times: 2D monotone chain against 3D incremental insertion.
    let mut rows = Vec::new();
    for &n in &[1000usize, 10_000, 100_000] {
        let mut r = rng(n as u64 + 1);
        let disc: Vec<[f64; 2]> = (0..n).map(|_| [gauss(&mut r), gauss(&mut r)]).collect();
        let (h2, t2) = time(|| convex_hull_2d(&disc));
        let ball: Vec<[f64; 3]> = (0..n).map(|_| [gauss(&mut r), gauss(&mut r), gauss(&mut r)]).collect();
        let (h3, t3) = time(|| convex_hull_3d(&ball).unwrap());
        let sphere: Vec<[f64; 3]> = ball.iter().map(|v| { let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(); [v[0] / l, v[1] / l, v[2] / l] }).collect();
        let (hs, ts) = time(|| convex_hull_3d(&sphere).unwrap());
        rows.push(vec![
            format!("{n}"),
            format!("{} ({} vertices)", ms(t2), h2.len()),
            format!("{} ({} faces)", ms(t3), h3.faces.len()),
            format!("{} ({} faces)", ms(ts), hs.faces.len()),
        ]);
    }
    write(CHAPTER, "hull_timing.md", &md_table(&["points", "2D, Gaussian", "3D, Gaussian", "3D, all on a sphere"], &rows));
}

// ------------------------------------------------------------------ polygon booleans

fn star(r: &mut impl FnMut() -> f64, centre: [f64; 2], m: usize, scale: f64) -> Vec<[f64; 2]> {
    (0..m)
        .map(|i| {
            let t = 2.0 * PI * (i as f64 + 0.6 * r()) / m as f64;
            let rad = scale * (0.3 + 0.7 * r());
            [centre[0] + rad * t.cos(), centre[1] + rad * t.sin()]
        })
        .collect()
}

fn boolean_figures() {
    // The first seed whose pair of polygons makes a several-piece intersection and difference.
    let (a, b) = (1..)
        .map(|seed| {
            let mut r = rng(seed);
            (star(&mut r, [0.0, 0.0], 14, 1.0), star(&mut r, [0.55, 0.2], 12, 0.95))
        })
        .find(|(a, b)| polygon_boolean(a, b, BooleanOperation::Intersection).len() >= 2 && polygon_boolean(a, b, BooleanOperation::Difference).len() >= 2)
        .unwrap();
    let window = Scene::fit(a.iter().chain(&b).copied(), 0.05);
    let mut scenes = Vec::new();
    for (name, op) in [("intersection", BooleanOperation::Intersection), ("union", BooleanOperation::Union), ("difference A − B", BooleanOperation::Difference)] {
        let out = polygon_boolean(&a, &b, op);
        let area: f64 = out.iter().map(|p| polygon_signed_area(p)).sum();
        let holes = out.iter().filter(|p| polygon_signed_area(p) < 0.0).count();
        let pieces = out.len() - holes;
        let plural = |n: usize, w: &str| format!("{n} {w}{}", if n == 1 { "" } else { "s" });
        let caption = if holes > 0 { format!("{name}: {}, {}, area {area:.3}", plural(pieces, "piece"), plural(holes, "hole")) } else { format!("{name}: {}, area {area:.3}", plural(pieces, "piece")) };
        let mut s = Scene::new(&caption, window);
        // Outer boundaries first, then holes painted white over them.
        for piece in out.iter().filter(|p| polygon_signed_area(p) > 0.0) {
            s.polygon(piece.clone(), Some(tint(2, 0.5)), Some(PALETTE[2]), 1.5);
        }
        for hole in out.iter().filter(|p| polygon_signed_area(p) < 0.0) {
            s.polygon(hole.clone(), Some("white".to_string()), Some(PALETTE[2]), 1.5);
        }
        s.dashed([a.clone(), vec![a[0]]].concat(), PALETTE[0], 1.0);
        s.dashed([b.clone(), vec![b[0]]].concat(), PALETTE[1], 1.0);
        scenes.push(s);
    }
    write(CHAPTER, "booleans.svg", &render("", &scenes, 3, 230.0));
}

// ------------------------------------------------------------------ spatial trees

/// The cells of the k-d tree's median splits, replayed: `(axis, split value, cell bounds)`.
fn kd_splits(p: &[[f64; 2]], idx: &mut [usize], depth: usize, cell: [f64; 4], out: &mut Vec<(usize, f64, [f64; 4])>) {
    if idx.len() <= 1 {
        return;
    }
    let axis = depth % 2;
    let mid = idx.len() / 2;
    idx.select_nth_unstable_by(mid, |&a, &b| p[a][axis].total_cmp(&p[b][axis]));
    let v = p[idx[mid]][axis];
    out.push((axis, v, cell));
    let (lo, hi) = if axis == 0 { ([cell[0], v, cell[2], cell[3]], [v, cell[1], cell[2], cell[3]]) } else { ([cell[0], cell[1], cell[2], v], [cell[0], cell[1], v, cell[3]]) };
    let (left, rest) = idx.split_at_mut(mid);
    kd_splits(p, left, depth + 1, lo, out);
    kd_splits(p, &mut rest[1..], depth + 1, hi, out);
}

/// The leaf cells of a quadtree with leaf capacity `cap`, replayed.
fn quad_cells(p: &[[f64; 2]], idx: &[usize], cell: [f64; 4], cap: usize, depth: usize, out: &mut Vec<[f64; 4]>) {
    if idx.len() <= cap || depth >= 20 {
        out.push(cell);
        return;
    }
    let (cx, cy) = (0.5 * (cell[0] + cell[1]), 0.5 * (cell[2] + cell[3]));
    for (qx, qy) in [(false, false), (true, false), (false, true), (true, true)] {
        let sub = [if qx { cx } else { cell[0] }, if qx { cell[1] } else { cx }, if qy { cy } else { cell[2] }, if qy { cell[3] } else { cy }];
        let inside: Vec<usize> = idx.iter().copied().filter(|&i| (p[i][0] >= cx) == qx && (p[i][1] >= cy) == qy).collect();
        quad_cells(p, &inside, sub, cap, depth + 1, out);
    }
}

fn tree_figures() {
    let mut r = rng(31);
    let p: Vec<[f64; 2]> = (0..150).map(|_| [r(), r()]).collect();
    let window = [-0.02, 1.02, -0.02, 1.02];
    let mut a = Scene::new("k-d tree splits and the 6 nearest neighbours", window);
    let mut splits = Vec::new();
    kd_splits(&p, &mut (0..p.len()).collect::<Vec<_>>(), 0, [0.0, 1.0, 0.0, 1.0], &mut splits);
    for (depth_order, (axis, v, c)) in splits.iter().enumerate() {
        let line = if *axis == 0 { vec![[*v, c[2]], [*v, c[3]]] } else { vec![[c[0], *v], [c[1], *v]] };
        a.path(line, if depth_order < 3 { PALETTE[1] } else { PALETTE[0] }, if depth_order < 3 { 1.6 } else { 0.6 });
    }
    let tree = KdTree::new(&p);
    let q = [0.42, 0.58];
    let near = tree.k_nearest(q, 6);
    a.dashed(circle(q, near.last().unwrap().1), PALETTE[2], 1.3);
    for &x in &p {
        a.dot(x, 2.2, PALETTE[6]);
    }
    for &(i, _) in &near {
        a.dot(p[i], 3.6, PALETTE[2]);
    }
    a.dot(q, 3.6, PALETTE[1]);

    // Clustered points show the quadtree adapting its cells to the density.
    let mut r = rng(32);
    let centres = [[0.25, 0.3], [0.7, 0.7], [0.75, 0.2]];
    let c: Vec<[f64; 2]> = (0..400).map(|k| { let m = centres[k % 3]; [(m[0] + 0.08 * gauss(&mut r)).clamp(0.0, 1.0), (m[1] + 0.08 * gauss(&mut r)).clamp(0.0, 1.0)] }).collect();
    let mut cells = Vec::new();
    quad_cells(&c, &(0..c.len()).collect::<Vec<_>>(), [0.0, 1.0, 0.0, 1.0], 8, 0, &mut cells);
    let mut b = Scene::new(&format!("quadtree of 400 clustered points: {} leaves", cells.len()), window);
    for cell in &cells {
        b.polygon(vec![[cell[0], cell[2]], [cell[1], cell[2]], [cell[1], cell[3]], [cell[0], cell[3]]], None, Some(PALETTE[0]), 0.6);
    }
    let mut quad = QuadTree::with_limits([0.0, 0.0], [1.0, 1.0], 8, 20);
    for (i, &x) in c.iter().enumerate() {
        quad.insert(i, x);
    }
    let centre = [0.7, 0.62];
    let inside = quad.query_radius(centre, 0.12);
    b.polygon(circle(centre, 0.12), Some(tint(2, 0.15)), Some(PALETTE[2]), 1.3);
    for &x in &c {
        b.dot(x, 1.8, PALETTE[6]);
    }
    for &i in &inside {
        b.dot(c[i], 3.0, PALETTE[2]);
    }
    write(CHAPTER, "trees.svg", &render("", &[a, b], 2, 330.0));

    // Query time against the number of points, for nearest-neighbour queries in the plane.
    let mut series: Vec<Vec<(f64, f64)>> = vec![Vec::new(); 3];
    let mut rows = Vec::new();
    for &n in &[1000usize, 10_000, 100_000, 1_000_000] {
        let mut r = rng(n as u64 + 7);
        let pts: Vec<[f64; 2]> = (0..n).map(|_| [r(), r()]).collect();
        let queries: Vec<[f64; 2]> = (0..2000).map(|_| [r(), r()]).collect();
        let (kd, tk) = time(|| KdTree::new(&pts));
        let (qt, tq) = time(|| QuadTree::from_points(&pts));
        let (a1, t1) = time(|| queries.iter().map(|&q| kd.nearest(q).unwrap().0).collect::<Vec<_>>());
        let (a2, t2) = time(|| queries.iter().map(|&q| qt.nearest(q).unwrap().0).collect::<Vec<_>>());
        let brute_queries = if n >= 100_000 { 50 } else { 2000 };
        let (a3, t3) = time(|| queries[..brute_queries].iter().map(|&q| (0..n).min_by(|&i, &j| ((pts[i][0] - q[0]).powi(2) + (pts[i][1] - q[1]).powi(2)).total_cmp(&((pts[j][0] - q[0]).powi(2) + (pts[j][1] - q[1]).powi(2)))).unwrap()).collect::<Vec<_>>());
        assert!(a1 == a2 && a1[..brute_queries] == a3[..]);
        series[0].push((n as f64, 1e6 * t1 / 2000.0));
        series[1].push((n as f64, 1e6 * t2 / 2000.0));
        series[2].push((n as f64, 1e6 * t3 / brute_queries as f64));
        rows.push(vec![format!("{n}"), ms(tk), ms(tq), format!("{:.2} µs", 1e6 * t1 / 2000.0), format!("{:.2} µs", 1e6 * t2 / 2000.0), format!("{:.1} µs", 1e6 * t3 / brute_queries as f64)]);
    }
    let mut plot = Plot::new("Nearest-neighbour query time in the plane", "points n", "µs per query");
    plot.log_x = true;
    plot.log_y = true;
    for (k, (label, pts)) in ["k-d tree", "quadtree", "brute force"].iter().zip(series).enumerate() {
        plot.series.push(Series::new(label, pts, Style::LineMarkers, k));
    }
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "tree_timing.svg", &plot.render());
    write(CHAPTER, "tree_timing.md", &md_table(&["points", "k-d build", "quadtree build", "k-d query", "quadtree query", "brute force query"], &rows));
}

// ------------------------------------------------------------------ intersections

fn icosahedron() -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let t = (1.0 + 5f64.sqrt()) / 2.0;
    let v: Vec<[f64; 3]> = [[-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0], [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t], [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0]]
        .iter()
        .map(|p| { let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt(); [p[0] / l, p[1] / l, p[2] / l] })
        .collect();
    let f = vec![[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
    (v, f)
}

/// An icosphere: the icosahedron with each triangle split into four `levels` times.
fn icosphere(levels: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let (mut v, mut f) = icosahedron();
    for _ in 0..levels {
        let mut mid: HashMap<(usize, usize), usize> = HashMap::new();
        let mut next = Vec::with_capacity(4 * f.len());
        for t in &f {
            let m: Vec<usize> = (0..3)
                .map(|k| {
                    let (a, b) = (t[k].min(t[(k + 1) % 3]), t[k].max(t[(k + 1) % 3]));
                    *mid.entry((a, b)).or_insert_with(|| {
                        let p = [0.5 * (v[a][0] + v[b][0]), 0.5 * (v[a][1] + v[b][1]), 0.5 * (v[a][2] + v[b][2])];
                        let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                        v.push([p[0] / l, p[1] / l, p[2] / l]);
                        v.len() - 1
                    })
                })
                .collect();
            next.extend([[t[0], m[0], m[2]], [t[1], m[1], m[0]], [t[2], m[2], m[1]], [m[0], m[1], m[2]]]);
        }
        f = next;
    }
    (v, f)
}

fn intersection_figures() {
    // A small ray-traced scene: a sphere, a box and an icosahedron, lit from one direction.
    let (w, h) = (300usize, 200usize);
    let (ico_v, ico_f) = icosahedron();
    let ico: Vec<[f64; 3]> = ico_v.iter().map(|p| [0.8 * p[0] + 1.4, 0.8 * p[1] + 0.2, 0.8 * p[2] + 0.8]).collect();
    let light = { let l = [-0.6f64, -0.5, 0.7]; let n = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt(); [l[0] / n, l[1] / n, l[2] / n] };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut img = Array2::<f64>::zeros((h, w));
    let eye = [0.0, -4.2, 1.5];
    for row in 0..h {
        for col in 0..w {
            let (u, v) = ((col as f64 + 0.5) / w as f64 - 0.5, 0.5 - (row as f64 + 0.5) / h as f64);
            let dir = [u * 1.5, 1.0, v * 1.0 - 0.12];
            let mut best: (f64, [f64; 3]) = (f64::INFINITY, [0.0; 3]);
            if let Some((t0, _)) = ray_sphere(eye, dir, [-1.3, 0.5, 1.0], 1.0)
                && t0 > 0.0
                && t0 < best.0
            {
                let p = [eye[0] + t0 * dir[0], eye[1] + t0 * dir[1], eye[2] + t0 * dir[2]];
                best = (t0, [p[0] + 1.3, p[1] - 0.5, p[2] - 1.0]);
            }
            if let Some((t0, _)) = ray_aabb(eye, dir, [-0.5, -1.2, 0.0], [0.5, -0.2, 0.7])
                && t0 > 0.0
                && t0 < best.0
            {
                let p = [eye[0] + t0 * dir[0], eye[1] + t0 * dir[1], eye[2] + t0 * dir[2]];
                let c = [0.0, -0.7, 0.35];
                let half = [0.5, 0.5, 0.35];
                let k = (0..3).max_by(|&a, &b| ((p[a] - c[a]) / half[a]).abs().total_cmp(&((p[b] - c[b]) / half[b]).abs())).unwrap();
                let mut n = [0.0; 3];
                n[k] = (p[k] - c[k]).signum();
                best = (t0, n);
            }
            for tri in &ico_f {
                if let Some(hit) = ray_triangle(eye, dir, ico[tri[0]], ico[tri[1]], ico[tri[2]])
                    && hit.t > 0.0
                    && hit.t < best.0
                {
                    let (a, b, c) = (ico[tri[0]], ico[tri[1]], ico[tri[2]]);
                    let (e1, e2) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
                    let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
                    let s = if dot(n, dir) > 0.0 { -1.0 } else { 1.0 };
                    best = (hit.t, [s * n[0], s * n[1], s * n[2]]);
                }
            }
            img[[row, col]] = if best.0.is_finite() {
                let n = best.1;
                let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                0.15 + 0.85 * (dot(n, light) / l).max(0.0)
            } else if dir[2] < 0.0 {
                // The floor z = 0, darkened where the objects shadow it.
                let t = -eye[2] / dir[2];
                let p = [eye[0] + t * dir[0], eye[1] + t * dir[1], 0.0];
                let shadow = ray_sphere(p, light, [-1.3, 0.5, 1.0], 1.0).is_some()
                    || ray_aabb(p, light, [-0.5, -1.2, 0.0], [0.5, -0.2, 0.7]).is_some_and(|(_, far)| far > 1e-9)
                    || ico_f.iter().any(|tri| ray_triangle(p, light, ico[tri[0]], ico[tri[1]], ico[tri[2]]).is_some_and(|x| x.t > 1e-9));
                if shadow { 0.25 } else { 0.55 }
            } else {
                0.85
            };
        }
    }
    let scene = Panel::new("ray casting: sphere, box and 20 triangles", img, Colours::Gray(0.0, 1.0));

    // Which feature of a triangle is closest, for points of a plane just above it.
    let (a, b, c) = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.8, 0.0]);
    let (gw, gh) = (240usize, 200usize);
    let mut regions = Array2::<f64>::zeros((gh, gw));
    for row in 0..gh {
        for col in 0..gw {
            let p = [-0.6 + 2.2 * col as f64 / gw as f64, 1.3 - 1.83 * row as f64 / gh as f64, 0.2];
            let (_, bary) = closest_point_on_triangle(p, a, b, c);
            let zeros = bary.iter().filter(|&&x| x <= 1e-12).count();
            regions[[row, col]] = match zeros {
                0 => 1.0,
                1 => 2.0 + bary.iter().position(|&x| x <= 1e-12).unwrap() as f64,
                _ => 5.0 + bary.iter().position(|&x| x > 1e-12).unwrap() as f64,
            };
        }
    }
    let regions = Panel::new("closest feature of a triangle: face, 3 edges, 3 vertices", regions, Colours::Labels);
    write(CHAPTER, "intersections.svg", &panels("", &[scene, regions], 2, 300.0));

    // Brute-force checks of the closest-point and ray routines.
    let mut r = rng(41);
    let mut v3 = || [2.0 * r() - 1.0, 2.0 * r() - 1.0, 2.0 * r() - 1.0];
    let dist = |a: [f64; 3], b: [f64; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let (mut tri_excess, mut seg_excess, mut ray_disagree) = (0.0f64, 0.0f64, 0usize);
    let trials = 2000;
    for _ in 0..trials {
        let (a, b, c, p) = (v3(), v3(), v3(), v3());
        let (cl, _) = closest_point_on_triangle(p, a, b, c);
        let m = 200;
        let mut best = f64::INFINITY;
        for i in 0..=m {
            for j in 0..=m - i {
                let (u, v) = (i as f64 / m as f64, j as f64 / m as f64);
                best = best.min(dist(p, [0, 1, 2].map(|k| a[k] + u * (b[k] - a[k]) + v * (c[k] - a[k]))));
            }
        }
        tri_excess = tri_excess.max(cl.distance - best);
        let (p2, q2) = (v3(), v3());
        let (_, _, d) = closest_points_between_segments(a, b, p2, q2);
        let best = (0..=2000).map(|i| { let x = [0, 1, 2].map(|k| a[k] + (i as f64 / 2000.0) * (b[k] - a[k])); closest_point_on_segment(x, p2, q2).0.distance }).fold(f64::INFINITY, f64::min);
        seg_excess = seg_excess.max(d - best);
        let dir = v3();
        let normal = { let (e1, e2) = ([0, 1, 2].map(|k| b[k] - a[k]), [0, 1, 2].map(|k| c[k] - a[k])); [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]] };
        let through = line_plane_parameter(p, dir, a, normal).filter(|&t| t >= 0.0).is_some_and(|t| closest_point_on_triangle([0, 1, 2].map(|k| p[k] + t * dir[k]), a, b, c).0.distance < 1e-9);
        ray_disagree += usize::from(through != ray_triangle(p, dir, a, b, c).is_some());
    }
    let rows = vec![
        vec![format!("closest point on a triangle, {trials} random cases"), "a 20 301-point grid of the triangle".into(), format!("never farther than the grid's best (excess {tri_excess:.1e})")],
        vec![format!("closest points between segments, {trials} cases"), "2001 points along one segment".into(), format!("excess {seg_excess:.1e}")],
        vec![format!("ray-triangle hits, {trials} cases"), "ray-plane hit, then point-in-triangle by distance".into(), format!("{ray_disagree} disagreements")],
    ];
    write(CHAPTER, "intersection_checks.md", &md_table(&["routine", "brute-force check", "result"], &rows));
}

// ------------------------------------------------------------------ meshes

fn mesh_volume(v: &[[f64; 3]], f: &[[usize; 3]]) -> f64 {
    f.iter().map(|t| { let [a, b, c] = t.map(|i| v[i]); (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0 }).sum()
}

/// A torus of radii `big` and `small` around the z axis, as an `m × k` grid of quads split into triangles.
fn torus(big: f64, small: f64, m: usize, k: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let v = (0..m).flat_map(|i| (0..k).map(move |j| { let (u, w) = (2.0 * PI * i as f64 / m as f64, 2.0 * PI * j as f64 / k as f64); [(big + small * w.cos()) * u.cos(), (big + small * w.cos()) * u.sin(), small * w.sin()] })).collect();
    let id = |i: usize, j: usize| (i % m) * k + (j % k);
    let f = (0..m).flat_map(|i| (0..k).flat_map(move |j| [[id(i, j), id(i + 1, j), id(i + 1, j + 1)], [id(i, j), id(i + 1, j + 1), id(i, j + 1)]])).collect();
    (v, f)
}

fn mesh_figures() {
    // Smoothing a noisy sphere.
    let (v, f) = icosphere(4);
    let mut r = rng(51);
    let noisy: Vec<[f64; 3]> = v.iter().map(|p| { let s = 1.0 + 0.04 * gauss(&mut r); [s * p[0], s * p[1], s * p[2]] }).collect();
    let radius_stats = |v: &[[f64; 3]]| {
        let radii: Vec<f64> = v.iter().map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()).collect();
        let mean = radii.iter().sum::<f64>() / radii.len() as f64;
        let rough = (radii.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / radii.len() as f64).sqrt();
        (mean, rough)
    };
    let (az, el) = (0.7, 0.4);
    let window = [-1.15, 1.15, -1.15, 1.15];
    let mut scenes = Vec::new();
    let lap = laplacian_smoothing(&noisy, &f, 20, 0.5, false);
    let tau = taubin_smoothing(&noisy, &f, 20, 0.5, -0.53, false);
    let mut rows = Vec::new();
    for (label, verts) in [("noisy (4% radial noise)", &noisy), ("Laplacian, 20 steps", &lap), ("Taubin, 20 steps", &tau)] {
        let (mean, rough) = radius_stats(verts);
        rows.push(vec![label.to_string(), format!("{mean:.4}"), format!("{rough:.4}")]);
        let mut s = Scene::new(label, window);
        add_mesh(&mut s, verts, &f, (az, el), 0, true, None);
        scenes.push(s);
    }
    for (label, verts) in [("Laplacian, 60 steps", laplacian_smoothing(&noisy, &f, 60, 0.5, false)), ("Taubin, 60 steps", taubin_smoothing(&noisy, &f, 60, 0.5, -0.53, false))] {
        let (mean, rough) = radius_stats(&verts);
        rows.push(vec![label.to_string(), format!("{mean:.4}"), format!("{rough:.4}")]);
    }
    write(CHAPTER, "smoothing.svg", &render("", &scenes, 3, 240.0));
    write(CHAPTER, "smoothing.md", &md_table(&["sphere", "mean radius", "rms deviation from it"], &rows));
    let mut plot = Plot::new("Smoothing a noisy unit sphere (5120 triangles)", "iterations", "shrinkage and roughness");
    let (mut lm, mut lr, mut tm, mut tr) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for it in 1..=60 {
        let l = laplacian_smoothing(&noisy, &f, it, 0.5, false);
        let t = taubin_smoothing(&noisy, &f, it, 0.5, -0.53, false);
        let ((a, b), (c, d)) = (radius_stats(&l), radius_stats(&t));
        lm.push((it as f64, (1.0 - a).abs()));
        lr.push((it as f64, b));
        tm.push((it as f64, (1.0 - c).abs()));
        tr.push((it as f64, d));
    }
    plot.series.push(Series::new("Laplacian: shrinkage |1 − mean radius|", lm, Style::Line, 0));
    plot.series.push(Series::new("Taubin: shrinkage", tm, Style::Line, 1));
    plot.series.push(Series::new("Laplacian: roughness (rms)", lr, Style::Dashed, 0));
    plot.series.push(Series::new("Taubin: roughness", tr, Style::Dashed, 1));
    plot.log_y = true;
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "smoothing_curves.svg", &plot.render());

    // Decimating a torus.
    let (big, small) = (1.0, 0.4);
    let (tv, tf) = torus(big, small, 96, 48);
    let exact_volume = 2.0 * PI * PI * big * small * small;
    let deviation = |v: &[[f64; 3]]| v.iter().map(|p| (((p[0] * p[0] + p[1] * p[1]).sqrt() - big).hypot(p[2]) - small).abs()).fold(0.0, f64::max);
    let (az, el) = (0.5, 0.75);
    let window = Scene::fit(tv.iter().map(|&x| project(x, az, el).0), 0.04);
    let mut scenes = Vec::new();
    let mut rows = Vec::new();
    for target in [tf.len(), 2000, 500, 150] {
        let ((dv, df), secs) = if target == tf.len() { ((tv.clone(), tf.clone()), 0.0) } else { time(|| decimate(&tv, &tf, target)) };
        // The decimated mesh's own error: sample each triangle's centroid against the true torus.
        let centroids: Vec<[f64; 3]> = df.iter().map(|t| { let [a, b, c] = t.map(|i| dv[i]); [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0] }).collect();
        rows.push(vec![format!("{}", df.len()), format!("{}", dv.len()), format!("{:.1e}", deviation(&dv)), format!("{:.1e}", deviation(&centroids)), format!("{:+.2}%", 100.0 * (mesh_volume(&dv, &df) / exact_volume - 1.0)), if secs > 0.0 { ms(secs) } else { "—".into() }]);
        let mut s = Scene::new(&format!("{} triangles", df.len()), window);
        add_mesh(&mut s, &dv, &df, (az, el), 2, true, Some("#1d4f3f"));
        scenes.push(s);
    }
    write(CHAPTER, "decimation.svg", &render("", &scenes, 4, 200.0));
    write(CHAPTER, "decimation.md", &md_table(&["triangles", "vertices", "largest vertex distance from the torus", "largest centroid distance", "volume error", "time"], &rows));

    // Slicing a torus: at every height the cut is two circles of total length 4πR.
    let (sv, sf) = torus(big, small, 192, 96);
    let mut rows = Vec::new();
    let mut slice_scene = Scene::new("slices of the torus at six heights, seen from above", [-1.5, 1.5, -1.5, 1.5]);
    for (k, &z) in [-0.35, -0.2, -0.05, 0.1, 0.25, 0.38].iter().enumerate() {
        let contours = slice_mesh(&sv, &sf, [0.0, 0.0, z], [0.0, 0.0, 1.0], 1e-12);
        let total: f64 = contours.iter().map(|c| c.length()).sum();
        rows.push(vec![format!("{z:+.2}"), format!("{}", contours.len()), format!("{}", contours.iter().all(|c| c.closed)), format!("{total:.5}"), format!("{:.2e}", total / (4.0 * PI * big) - 1.0)]);
        for c in &contours {
            let mut pts: Vec<[f64; 2]> = c.points.iter().map(|p| [p[0], p[1]]).collect();
            pts.push(pts[0]);
            slice_scene.path(pts, PALETTE[k % 6], 1.2);
        }
    }
    write(CHAPTER, "slices.svg", &render("", &[slice_scene], 1, 300.0));
    write(CHAPTER, "slicing.md", &md_table(&["height z", "contours", "all closed", "total length", "relative to 4πR"], &rows));
}

// ------------------------------------------------------------------ rotations

fn rotation_figures() {
    // Interpolating a 160° turn: slerp sweeps the angle uniformly, nlerp does not.
    let a = Quaternion::identity();
    let b = Quaternion::from_axis_angle([1.0, 2.0, 2.0], 160f64.to_radians());
    let (mut s, mut nl) = (Vec::new(), Vec::new());
    for k in 0..=100 {
        let t = k as f64 / 100.0;
        s.push((t, a.angle_to(&a.slerp(&b, t)).to_degrees()));
        nl.push((t, a.angle_to(&a.nlerp(&b, t)).to_degrees()));
    }
    let mut plot = Plot::new("Interpolating a 160° rotation", "interpolation parameter t", "angle turned from the start (degrees)");
    plot.series.push(Series::new("slerp", s, Style::Line, 0));
    plot.series.push(Series::new("nlerp", nl, Style::Dashed, 1));
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "slerp.svg", &plot.render());

    // Composing a million small rotations: drift away from a true rotation.
    let step = Rotation3::from_axis_angle([0.3, -0.5, 0.8], 0.001);
    let step_m = step.matrix();
    let (mut m, mut q) = (rotations::IDENTITY3, Quaternion::identity());
    let (mut ms_, mut qs, mut cleaned) = (Vec::new(), Vec::new(), Vec::new());
    let mut mc = rotations::IDENTITY3;
    for k in 1..=1_000_000usize {
        m = rotations::mat3_mul(&step_m, &m);
        mc = rotations::mat3_mul(&step_m, &mc);
        q = step.quaternion() * q;
        if k % 1000 == 0 {
            mc = rotations::nearest_rotation_matrix(&mc).unwrap();
        }
        if k % 10_000 == 0 {
            let orth = |m: &Matrix3| { let p = rotations::mat3_mul(m, &rotations::mat3_transpose(m)); (0..9).map(|i| (p[i / 3][i % 3] - if i % 4 == 0 { 1.0 } else { 0.0 }).abs()).fold(0.0, f64::max) };
            ms_.push((k as f64, orth(&m).max(1e-17)));
            qs.push((k as f64, (q.norm() - 1.0).abs().max(1e-17)));
            cleaned.push((k as f64, orth(&mc).max(1e-17)));
        }
    }
    let mut plot = Plot::new("Accumulated rounding when composing rotations", "rotations composed", "departure from a rotation");
    plot.log_y = true;
    plot.series.push(Series::new("matrix product: max |RRᵀ − I|", ms_, Style::Line, 0));
    plot.series.push(Series::new("quaternion product: |‖q‖ − 1|", qs, Style::Line, 1));
    plot.series.push(Series::new("matrix, re-orthogonalised every 1000", cleaned, Style::Line, 2));
    plot.legend = Legend::TopLeft;
    plot.y_range = Some((1e-17, 1e-10));
    write(CHAPTER, "drift.svg", &plot.render());
}

// ------------------------------------------------------------------ space-time

fn schwarzschild_metric(m: f64) -> impl Fn(&[f64; 4]) -> Metric4 {
    move |x: &[f64; 4]| {
        let (r, th) = (x[1], x[2]);
        let f = 1.0 - 2.0 * m / r;
        let mut g = [[0.0; 4]; 4];
        g[0][0] = -f;
        g[1][1] = 1.0 / f;
        g[2][2] = r * r;
        g[3][3] = r * r * th.sin().powi(2);
        g
    }
}

fn spacetime_figures() {
    // Christoffel symbols by finite differences against the exact ones.
    let mut rows = Vec::new();
    for (r, th) in [(10.0, 1.0), (3.0, 0.4), (50.0, 2.0)] {
        let exact = ChristoffelSymbols4D::schwarzschild(1.0, r, th).unwrap();
        let numeric = ChristoffelSymbols4D::from_metric_function(schwarzschild_metric(1.0), [0.0, r, th, 0.0], 1e-3 * r).unwrap();
        let mut worst = 0.0f64;
        let mut largest = 0.0f64;
        for l in 0..4 {
            for a in 0..4 {
                for b in 0..4 {
                    worst = worst.max((exact.gamma[l][a][b] - numeric.gamma[l][a][b]).abs());
                    largest = largest.max(exact.gamma[l][a][b].abs());
                }
            }
        }
        rows.push(vec![format!("r = {r} M, θ = {th}"), format!("{largest:.3}"), format!("{worst:.1e}")]);
    }
    write(CHAPTER, "christoffel.md", &md_table(&["point", "largest symbol", "finite-difference error"], &rows));

    // A bound orbit between r = 20M and r = 60M, integrated along the geodesic equation.
    let mass = 1.0f64;
    let (r1, r2) = (20.0f64, 60.0f64);
    let l2 = 2.0 * mass * (1.0 / r1 - 1.0 / r2) / ((1.0 - 2.0 * mass / r1) / (r1 * r1) - (1.0 - 2.0 * mass / r2) / (r2 * r2));
    let ang = l2.sqrt();
    let energy = ((1.0 - 2.0 * mass / r1) * (1.0 + l2 / (r1 * r1))).sqrt();
    let mut x = [0.0, r1, PI / 2.0, 0.0];
    let mut u = [energy / (1.0 - 2.0 * mass / r1), 0.0, 0.0, ang / (r1 * r1)];
    let deriv = |x: &[f64; 4], u: &[f64; 4]| -> ([f64; 4], [f64; 4]) { (*u, ChristoffelSymbols4D::schwarzschild(mass, x[1], x[2]).unwrap().geodesic_acceleration(*u)) };
    let h = 0.5;
    let mut path = Vec::new();
    let mut periapses = Vec::new();
    let mut last_ur = 0.0;
    let mut last_phi = 0.0;
    let mut max_drift = 0.0f64;
    for _ in 0..200_000 {
        let (k1x, k1u) = deriv(&x, &u);
        let mid = |a: &[f64; 4], k: &[f64; 4], s: f64| std::array::from_fn::<f64, 4, _>(|i| a[i] + s * k[i]);
        let (k2x, k2u) = deriv(&mid(&x, &k1x, h / 2.0), &mid(&u, &k1u, h / 2.0));
        let (k3x, k3u) = deriv(&mid(&x, &k2x, h / 2.0), &mid(&u, &k2u, h / 2.0));
        let (k4x, k4u) = deriv(&mid(&x, &k3x, h), &mid(&u, &k3u, h));
        x = std::array::from_fn(|i| x[i] + h / 6.0 * (k1x[i] + 2.0 * k2x[i] + 2.0 * k3x[i] + k4x[i]));
        u = std::array::from_fn(|i| u[i] + h / 6.0 * (k1u[i] + 2.0 * k2u[i] + 2.0 * k3u[i] + k4u[i]));
        // The normalisation g(u, u) = −1 is conserved along a timelike geodesic.
        let g = schwarzschild_metric(mass)(&x);
        let norm: f64 = (0..4).map(|i| g[i][i] * u[i] * u[i]).sum();
        max_drift = max_drift.max((norm + 1.0).abs());
        if last_ur < 0.0 && u[1] >= 0.0 {
            // Interpolate the angle at which the radial velocity changes sign within the step.
            periapses.push(last_phi + (x[3] - last_phi) * (-last_ur) / (u[1] - last_ur));
        }
        last_ur = u[1];
        last_phi = x[3];
        path.push([x[1] * x[3].cos(), x[1] * x[3].sin()]);
        if periapses.len() == 4 {
            break;
        }
    }
    let measured = (periapses[3] - periapses[0]) / 3.0 - 2.0 * PI;
    // The exact advance: with w = 1/r, (dw/dφ)² = 2M (w − w1)(w − w2)(w − w3), so one radial
    // period sweeps 2∫ 2 dχ / sqrt(2M (w3 − w(χ))) over χ in [0, π/2], with w = w2 + (w1 − w2) sin²χ.
    let (w1, w2) = (1.0 / r1, 1.0 / r2);
    let w3 = 1.0 / (2.0 * mass) - w1 - w2;
    let n = 20_000;
    let integral: f64 = (0..n).map(|k| { let chi = (k as f64 + 0.5) / n as f64 * PI / 2.0; let w = w2 + (w1 - w2) * chi.sin().powi(2); 2.0 / (2.0 * mass * (w3 - w)).sqrt() }).sum::<f64>() * (PI / 2.0) / n as f64;
    let exact = 2.0 * integral - 2.0 * PI;
    let semi_latus = 2.0 * r1 * r2 / (r1 + r2);
    let weak = 6.0 * PI * mass / semi_latus;
    write(
        CHAPTER,
        "precession.md",
        &md_table(
            &["perihelion advance per orbit", "radians", "degrees"],
            &[
                vec!["geodesic equation, RK4 with the library's symbols".into(), format!("{measured:.5}"), format!("{:.3}", measured.to_degrees())],
                vec!["exact (elliptic integral)".into(), format!("{exact:.5}"), format!("{:.3}", exact.to_degrees())],
                vec!["weak-field formula 6πM/p".into(), format!("{weak:.5}"), format!("{:.3}", weak.to_degrees())],
            ],
        ),
    );
    write(CHAPTER, "precession_facts.md", &format!("Over the three orbits the conserved normalisation $g_{{\\mu\\nu}} u^\\mu u^\\nu = -1$ drifted by at most {max_drift:.0e}."));
    let mut scene = Scene::new(&format!("orbit between r = 20M and 60M: advance {:.1}° per turn", measured.to_degrees()), Scene::fit(path.iter().copied(), 0.05));
    scene.path(path.clone(), PALETTE[0], 1.0);
    // The first orbit's major axis, and the same axis three turns later.
    for (k, &phi) in [periapses[0], periapses[3]].iter().enumerate() {
        scene.dashed(vec![[r1 * phi.cos(), r1 * phi.sin()], [-r2 * phi.cos(), -r2 * phi.sin()]], PALETTE[1 + k], 1.2);
    }
    scene.polygon(circle([0.0, 0.0], 2.0 * mass), Some("black".into()), None, 0.0);
    write(CHAPTER, "orbit.svg", &render("", &[scene], 1, 360.0));

    // Special relativity checks.
    let v = SpaceTimeVector4D::new(5.0, 1.0, -2.0, 0.5);
    let boosted = v.boosted([0.6, 0.2, -0.3]).unwrap();
    let rest = SpaceTimeVector4D::four_velocity([0.6, 0.2, -0.3]).unwrap().boosted([0.6, 0.2, -0.3]).unwrap();
    let (u, w) = (0.6, 0.7);
    let twice = SpaceTimeVector4D::new(1.0, 0.0, 0.0, 0.0).boosted([u, 0.0, 0.0]).unwrap().boosted([w, 0.0, 0.0]).unwrap();
    let once = SpaceTimeVector4D::new(1.0, 0.0, 0.0, 0.0).boosted([SpaceTimeVector4D::add_collinear_velocities(u, w), 0.0, 0.0]).unwrap();
    let rows = vec![
        vec!["interval −t² + x² + y² + z² before and after a boost".into(), format!("{:.12} and {:.12}", v.interval_squared(), boosted.interval_squared())],
        vec!["four-velocity seen from its own rest frame".into(), format!("({:.3}, {:.1e}, {:.1e}, {:.1e})", rest.t, rest.x, rest.y, rest.z)],
        vec!["boosts by 0.6c then 0.7c along x, against one boost by their relativistic sum".into(), format!("{:.3}c; largest component difference {:.1e}", SpaceTimeVector4D::add_collinear_velocities(u, w), (twice.t - once.t).abs().max((twice.x - once.x).abs()))],
    ];
    write(CHAPTER, "relativity_checks.md", &md_table(&["check", "result"], &rows));
}
