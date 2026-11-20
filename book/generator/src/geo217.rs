//! Tables and figures for release 2.17.0 (geometry chapter): convex hulls, Delaunay
//! triangulations and Voronoi diagrams in any dimension, half-space intersections, spherical
//! Voronoi diagrams and linear interpolation of scattered data in 3-D and 4-D, against SciPy
//! (Qhull).

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_geometry::algorithms::hull_nd::{ConvexHullNd, DelaunayNd, HalfspaceIntersection, SphericalVoronoi, VoronoiNd};
use lib_jsl_interpolation::nd::linear_nd::LinearNdInterpolator;

const CHAPTER: &str = "geometry";

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/geo217_reference.txt"));
    table(&refs);
    figure(&refs);
}

fn rows_of(v: &[f64], d: usize) -> Vec<Vec<f64>> {
    v.chunks(d).map(|c| c.to_vec()).collect()
}

/// The largest distance from each point of `a` to the nearest point of `b` (sets of equal size).
fn matched(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter().map(|p| b.iter().map(|q| p.iter().zip(q).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()).fold(f64::INFINITY, f64::min)).fold(0.0, f64::max)
}

fn sorted_simplices(s: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = s.iter().map(|x| {
        let mut x = x.clone();
        x.sort_unstable();
        x
    }).collect();
    out.sort();
    out
}

fn table(refs: &Refs) {
    let mut rows = Vec::new();
    for d in [2usize, 3, 4, 5, 6] {
        let w = refs.get(&format!("hull {d}"));
        let pts = rows_of(&w[0], d);
        let h = ConvexHullNd::new(&pts).unwrap();
        let verts: Vec<f64> = h.vertices.iter().map(|&v| v as f64).collect();
        rows.push(vec![
            format!("`ConvexHullNd`, {} normal points in {d}-D", pts.len()),
            format!(
                "volume {}, area {}; {} facets vs {}; vertices {}",
                fmt_diff((h.volume - w[1][0]).abs() / w[1][0]),
                fmt_diff((h.area - w[1][1]).abs() / w[1][1]),
                h.simplices.len(),
                w[1][2],
                if verts == w[2] { "identical" } else { "differ" }
            ),
        ]);
    }
    for d in [2usize, 3, 4] {
        let w = refs.get(&format!("delaunay {d}"));
        let pts = rows_of(&w[0], d);
        let t = DelaunayNd::new(&pts).unwrap();
        let ours = sorted_simplices(&t.simplices);
        let theirs: Vec<Vec<usize>> = sorted_simplices(&w[1].chunks(d + 1).map(|c| c.iter().map(|&v| v as usize).collect()).collect::<Vec<_>>());
        let q = rows_of(&w[2], d);
        let found: Vec<Vec<f64>> = w[3].chunks(d + 1).map(|c| c.to_vec()).collect();
        let same_found = q.iter().zip(&found).filter(|(x, f)| match t.find_simplex(x) {
            None => f[0] < 0.0,
            Some(i) => {
                let mut s = t.simplices[i].clone();
                s.sort_unstable();
                s.iter().map(|&v| v as f64).collect::<Vec<_>>() == **f
            }
        }).count();
        rows.push(vec![
            format!("`DelaunayNd`, {} uniform points in {d}-D", pts.len()),
            format!("{} simplices vs {}, {}; `find_simplex` agrees for {same_found} of {} points", ours.len(), theirs.len(), if ours == theirs { "identical" } else { "differ" }, q.len()),
        ]);
    }
    for d in [2usize, 3] {
        let w = refs.get(&format!("voronoi {d}"));
        let pts = rows_of(&w[0], d);
        let v = VoronoiNd::new(&pts).unwrap();
        let theirs_v = rows_of(&w[1], d);
        let mut rp: Vec<Vec<usize>> = v.ridge_points.iter().map(|r| r.to_vec()).collect();
        rp.sort();
        let theirs_r: Vec<Vec<usize>> = w[2].chunks(2).map(|c| vec![c[0] as usize, c[1] as usize]).collect();
        rows.push(vec![
            format!("`VoronoiNd`, {} uniform points in {d}-D", pts.len()),
            format!("{} vertices vs {}, matched to {}; ridges {}", v.vertices.len(), theirs_v.len(), fmt_diff(matched(&v.vertices, &theirs_v)), if rp == theirs_r { "identical" } else { "differ" }),
        ]);
    }
    let w = refs.get("halfspace 3");
    let hs = rows_of(&w[0], 4);
    let h = HalfspaceIntersection::new(&hs, &[0.05, -0.02, 0.01]).unwrap();
    let dv: Vec<f64> = h.dual_vertices.iter().map(|&v| v as f64).collect();
    rows.push(vec![
        "`HalfspaceIntersection`, 30 tangent planes of the unit sphere and 5 redundant ones (3-D)".into(),
        format!("{} vertices vs {}, matched to {}; non-redundant half-spaces {}", h.intersections.len(), w[1].len() / 3, fmt_diff(matched(&h.intersections, &rows_of(&w[1], 3))), if dv == w[2] { "identical" } else { "differ" }),
    ]);
    let w = refs.get("spherical");
    let pts: Vec<[f64; 3]> = w[0].chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let sv = SphericalVoronoi::new(&pts, 1.0, [0.0; 3]).unwrap();
    let verts: Vec<Vec<f64>> = sv.vertices.iter().map(|v| v.to_vec()).collect();
    let areas = sv.calculate_areas(&pts);
    rows.push(vec![
        "`SphericalVoronoi`, 60 random points on the unit sphere".into(),
        format!("{} vertices vs {}, matched to {}; region areas {}", verts.len(), w[1].len() / 3, fmt_diff(matched(&verts, &rows_of(&w[1], 3))), fmt_diff(areas.iter().zip(&w[2]).fold(0.0f64, |m, (a, b)| m.max((a - b).abs())))),
    ]);
    for d in [3usize, 4] {
        let w = refs.get(&format!("linearnd {d}"));
        let it = LinearNdInterpolator::new(&rows_of(&w[0], d), &w[1]).unwrap();
        let q = rows_of(&w[2], d);
        let (mut diff, mut same_nan) = (0.0f64, true);
        for (x, &want) in q.iter().zip(&w[3]) {
            let got = it.evaluate(x);
            if want >= 1e299 || got.is_nan() {
                same_nan &= want >= 1e299 && got.is_nan();
            } else {
                diff = diff.max((got - want).abs());
            }
        }
        rows.push(vec![
            format!("`LinearNdInterpolator`, 80 points in {d}-D, 100 queries"),
            format!("{}; outside the hull (NaN) {}", fmt_diff(diff), if same_nan { "at the same queries" } else { "differs" }),
        ]);
    }
    write(CHAPTER, "geo217.md", &md_table(&["Function and data", "Library against SciPy (Qhull)"], &rows));
}

fn figure(refs: &Refs) {
    let hs = rows_of(refs.values("halfspace 2"), 3);
    let h = HalfspaceIntersection::new(&hs, &[0.0, 0.0]).unwrap();
    let mut plot = Plot::new("HalfspaceIntersection: 12 half-planes and the polygon they bound", "x", "y");
    plot.x_range = Some((-1.6, 1.6));
    plot.y_range = Some((-1.6, 1.6));
    plot.legend = Legend::TopRight;
    for (k, l) in hs.iter().enumerate() {
        // The line a·x + b = 0, drawn as a segment around its closest point to the origin.
        let (a, b) = ([l[0], l[1]], l[2]);
        let foot = [-b * a[0], -b * a[1]];
        let dir = [-a[1], a[0]];
        let pts = vec![(foot[0] - 3.0 * dir[0], foot[1] - 3.0 * dir[1]), (foot[0] + 3.0 * dir[0], foot[1] + 3.0 * dir[1])];
        plot.series.push(Series::new(if k == 0 { "half-plane boundaries" } else { "" }, pts, Style::Dashed, 7));
    }
    // Order the vertices by angle to draw the polygon.
    let mut v = h.intersections.clone();
    v.sort_by(|p, q| p[1].atan2(p[0]).total_cmp(&q[1].atan2(q[0])));
    let mut poly: Vec<(f64, f64)> = v.iter().map(|p| (p[0], p[1])).collect();
    poly.push(poly[0]);
    plot.series.push(Series::new(&format!("intersection ({} vertices)", v.len()), poly, Style::LineMarkers, 0));
    plot.series.push(Series::new("interior point", vec![(0.0, 0.0)], Style::Markers, 1));
    write(CHAPTER, "halfspace_polygon.svg", &plot.render());
}
