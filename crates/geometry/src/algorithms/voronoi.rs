//! Voronoi diagrams of 2D point sets, as the dual of the Delaunay triangulation.

use crate::algorithms::delaunay_triangulation::{Triangulation, delaunay_triangulation};
use std::collections::BTreeSet;

/// A Voronoi diagram clipped to a bounding box: the cell of each site is the convex polygon of
/// box points closer to that site than to any other.
#[derive(Debug, Clone)]

pub struct VoronoiDiagram {
    /// Counter-clockwise cell polygons, one per input site (empty for duplicate or non-finite
    /// sites, which have no cell of their own).
    pub cells: Vec<Vec<[f64; 2]>>,
    /// The Delaunay triangulation it was built from.
    pub triangulation: Triangulation,
    /// The Voronoi vertices: circumcentres of the Delaunay triangles (possibly outside the box).
    pub vertices: Vec<[f64; 2]>,
}

/// Clips a convex polygon to the half-plane `{x : (x − m)·d ≤ 0}`.
fn clip(polygon: &[[f64; 2]], m: [f64; 2], d: [f64; 2]) -> Vec<[f64; 2]> {
    let side = |p: [f64; 2]| (p[0] - m[0]) * d[0] + (p[1] - m[1]) * d[1];
    let mut out = Vec::with_capacity(polygon.len() + 1);
    for k in 0..polygon.len() {
        let (a, b) = (polygon[k], polygon[(k + 1) % polygon.len()]);
        let (sa, sb) = (side(a), side(b));
        if sa <= 0.0 {
            out.push(a);
        }
        if (sa < 0.0 && sb > 0.0) || (sa > 0.0 && sb < 0.0) {
            let t = sa / (sa - sb);
            out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
    out
}

/// The Voronoi diagram of `sites` within the box `[lo, hi]`. Each cell is the box clipped by the
/// perpendicular bisectors to the site's Delaunay neighbours, which are exactly the sites whose
/// cells it shares an edge with. Returns `None` when the sites cannot be triangulated (fewer than
/// three distinct points, or all collinear).
pub fn voronoi_diagram(sites: &[[f64; 2]], lo: [f64; 2], hi: [f64; 2]) -> Option<VoronoiDiagram> {
    let triangulation = delaunay_triangulation(sites)?;
    let mut neighbours: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); sites.len()];
    for (a, b) in triangulation.edges() {
        neighbours[a].insert(b);
        neighbours[b].insert(a);
    }
    let bounding = vec![lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]];
    let cells = (0..sites.len())
        .map(|i| {
            if neighbours[i].is_empty() {
                return Vec::new();
            }
            neighbours[i].iter().fold(bounding.clone(), |cell, &j| {
                let (s, t) = (sites[i], sites[j]);
                clip(&cell, [(s[0] + t[0]) / 2.0, (s[1] + t[1]) / 2.0], [t[0] - s[0], t[1] - s[1]])
            })
        })
        .collect();
    let vertices = (0..triangulation.triangles.len()).map(|t| triangulation.circumcenter(sites, t)).collect();
    Some(VoronoiDiagram { cells, triangulation, vertices })
}

/// The area of a simple polygon (positive for counter-clockwise order).
pub fn polygon_signed_area(polygon: &[[f64; 2]]) -> f64 {
    (0..polygon.len()).map(|k| { let (a, b) = (polygon[k], polygon[(k + 1) % polygon.len()]); a[0] * b[1] - b[0] * a[1] }).sum::<f64>() / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn cells_tile_the_box_and_contain_their_nearest_points() {
        let mut r = lcg(3);
        let sites: Vec<[f64; 2]> = (0..60).map(|_| [r() * 4.0, r() * 2.0]).collect();
        let v = voronoi_diagram(&sites, [0.0, 0.0], [4.0, 2.0]).unwrap();
        let total: f64 = v.cells.iter().map(|c| polygon_signed_area(c)).sum();
        assert!((total - 8.0).abs() < 1e-9, "{total}");
        assert!(v.cells.iter().all(|c| polygon_signed_area(c) > 0.0));
        // Random probes: the nearest site's cell contains the probe (convex point-in-polygon).
        for _ in 0..500 {
            let p = [r() * 4.0, r() * 2.0];
            let nearest = (0..sites.len()).min_by(|&a, &b| {
                let d = |s: [f64; 2]| (s[0] - p[0]).powi(2) + (s[1] - p[1]).powi(2);
                d(sites[a]).total_cmp(&d(sites[b]))
            }).unwrap();
            let cell = &v.cells[nearest];
            let inside = (0..cell.len()).all(|k| {
                let (a, b) = (cell[k], cell[(k + 1) % cell.len()]);
                (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12
            });
            assert!(inside);
        }
    }

    #[test]
    fn a_regular_grid_has_square_cells() {
        let sites: Vec<[f64; 2]> = (0..4).flat_map(|i| (0..4).map(move |j| [i as f64 + 0.5, j as f64 + 0.5])).collect();
        let v = voronoi_diagram(&sites, [0.0, 0.0], [4.0, 4.0]).unwrap();
        assert!(v.cells.iter().all(|c| (polygon_signed_area(c) - 1.0).abs() < 1e-9));
        assert!(voronoi_diagram(&[[0.0, 0.0], [1.0, 1.0]], [0.0, 0.0], [1.0, 1.0]).is_none());
    }
}
