//! Slicing triangle meshes with planes into contours.
//!
//! Each triangle crossing the plane contributes one segment ([`triangle_plane`]); the segments are
//! then chained end to end into polylines, closed where the cut goes all the way round. This is
//! the core of layer-by-layer slicing for 3D printing and of cross-section views.

use super::intersection::triangle_plane;
use std::collections::HashMap;

/// A chain of points where a plane cuts a mesh.
#[derive(Debug, Clone, PartialEq)]

pub struct Contour {
    /// Vertices in order. For a closed contour the first vertex is *not* repeated at the end.
    pub points: Vec<[f64; 3]>,
    /// Whether the last vertex connects back to the first.
    pub closed: bool,
}

impl Contour {
    /// Total length (including the closing edge if closed).
    pub fn length(&self) -> f64 {
        let n = self.points.len();
        let edges = if self.closed { n } else { n.saturating_sub(1) };
        (0..edges)
            .map(|i| {
                let (a, b) = (self.points[i], self.points[(i + 1) % n]);
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
            })
            .sum()
    }

    /// Removes interior vertices that lie on the straight line between their neighbours (within
    /// `tol`), e.g. the extra points a cut makes where it crosses a flat face's diagonal.
    pub fn simplified(&self, tol: f64) -> Contour {
        let mut pts = self.points.clone();
        let mut i = 0;
        while pts.len() > 2 && i < pts.len() {
            let n = pts.len();
            if !self.closed && (i == 0 || i == n - 1) {
                i += 1;
                continue;
            }
            let (a, b, c) = (pts[(i + n - 1) % n], pts[i], pts[(i + 1) % n]);
            let (ab, ac) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
            let cr = [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]];
            let area2 = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
            let base = (ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2]).sqrt();
            if base == 0.0 || area2 / base <= tol {
                pts.remove(i);
                if self.closed && pts.len() < 3 {
                    break;
                }
            } else {
                i += 1;
            }
        }
        let closed = self.closed && pts.len() >= 3;
        Contour { points: pts, closed }
    }
}

/// Slices the triangle mesh with the plane through `plane_point` with `normal`, returning the
/// cut as contours. `tol` is the distance within which vertices count as on the plane and cut
/// endpoints count as coincident; choose it relative to the mesh size (e.g. `1e-9 * extent`).
///
/// Triangles lying exactly in the plane contribute nothing (their boundary appears from the
/// neighbouring triangles), and an edge lying in the plane is reported once even though both of
/// its triangles cut there, so a plane through a ring of vertices gives one loop. Triangles with
/// out-of-range indices are skipped.
pub fn slice_mesh(vertices: &[[f64; 3]], triangles: &[[usize; 3]], plane_point: [f64; 3], normal: [f64; 3], tol: f64) -> Vec<Contour> {
    let tol = tol.max(f64::MIN_POSITIVE);
    // A mesh edge lying in the plane is reported by both triangles that share it; keep one copy.
    let key = |p: [f64; 3]| -> (i64, i64, i64) { ((p[0] / tol).round() as i64, (p[1] / tol).round() as i64, (p[2] / tol).round() as i64) };
    let mut seen = std::collections::HashSet::new();
    let segments: Vec<([f64; 3], [f64; 3])> = triangles
        .iter()
        .filter_map(|t| {
            let (a, b, c) = (*vertices.get(t[0])?, *vertices.get(t[1])?, *vertices.get(t[2])?);
            triangle_plane(a, b, c, plane_point, normal, tol)
        })
        .filter(|&(p, q)| {
            let (kp, kq) = (key(p), key(q));
            seen.insert(if kp <= kq { (kp, kq) } else { (kq, kp) })
        })
        .collect();
    chain_segments(&segments, tol)
}

/// Slices the mesh at each of `levels` along a coordinate axis (`0` = x, `1` = y, `2` = z),
/// returning `(level, contours)` for every level, in the order given.
pub fn slice_mesh_levels(vertices: &[[f64; 3]], triangles: &[[usize; 3]], axis: usize, levels: &[f64], tol: f64) -> Vec<(f64, Vec<Contour>)> {
    assert!(axis < 3, "axis must be 0, 1 or 2");
    let mut normal = [0.0; 3];
    normal[axis] = 1.0;
    levels
        .iter()
        .map(|&level| {
            let mut p = [0.0; 3];
            p[axis] = level;
            (level, slice_mesh(vertices, triangles, p, normal, tol))
        })
        .collect()
}

fn chain_segments(segments: &[([f64; 3], [f64; 3])], tol: f64) -> Vec<Contour> {
    // Snap endpoints to a grid of cell size `tol` so coincident ends share a key. Neighbouring
    // cells are also probed so points straddling a cell boundary still match.
    let key = |p: [f64; 3]| -> (i64, i64, i64) { ((p[0] / tol).round() as i64, (p[1] / tol).round() as i64, (p[2] / tol).round() as i64) };
    let mut ends: HashMap<(i64, i64, i64), Vec<(usize, bool)>> = HashMap::new(); // (segment, is_second_end)
    for (i, &(p, q)) in segments.iter().enumerate() {
        ends.entry(key(p)).or_default().push((i, false));
        ends.entry(key(q)).or_default().push((i, true));
    }
    let find_next = |pt: [f64; 3], used: &[bool]| -> Option<(usize, bool)> {
        let (kx, ky, kz) = key(pt);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(list) = ends.get(&(kx + dx, ky + dy, kz + dz)) {
                        for &(s, second) in list {
                            let e = if second { segments[s].1 } else { segments[s].0 };
                            let d2 = (e[0] - pt[0]).powi(2) + (e[1] - pt[1]).powi(2) + (e[2] - pt[2]).powi(2);
                            if !used[s] && d2 <= (2.0 * tol) * (2.0 * tol) {
                                return Some((s, second));
                            }
                        }
                    }
                }
            }
        }
        None
    };

    let mut used = vec![false; segments.len()];
    let mut out = Vec::new();
    for start in 0..segments.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let mut chain: std::collections::VecDeque<[f64; 3]> = [segments[start].0, segments[start].1].into();
        // Grow forward from the tail, then backward from the head.
        for forward in [true, false] {
            loop {
                let end = if forward { *chain.back().expect("non-empty") } else { *chain.front().expect("non-empty") };
                let Some((s, second)) = find_next(end, &used) else { break };
                used[s] = true;
                let far = if second { segments[s].0 } else { segments[s].1 };
                if forward { chain.push_back(far) } else { chain.push_front(far) }
            }
        }
        let mut points: Vec<[f64; 3]> = chain.into();
        let dist2 = |a: [f64; 3], b: [f64; 3]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
        let closed = points.len() >= 4 && dist2(points[0], points[points.len() - 1]) <= (2.0 * tol) * (2.0 * tol);
        if closed {
            points.pop();
        }
        out.push(Contour { points, closed });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube at the origin with 12 triangles.
    fn cube() -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
        let v: Vec<[f64; 3]> = (0..8).map(|i| [(i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64]).collect();
        let quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]];
        let t = quads.iter().flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]).collect();
        (v, t)
    }

    #[test]
    fn cube_cross_section_is_one_closed_square() {
        let (v, t) = cube();
        let cuts = slice_mesh(&v, &t, [0.0, 0.0, 0.5], [0.0, 0.0, 1.0], 1e-9);
        assert_eq!(cuts.len(), 1);
        assert!(cuts[0].closed);
        assert!((cuts[0].length() - 4.0).abs() < 1e-9);
        assert!(cuts[0].points.iter().all(|p| (p[2] - 0.5).abs() < 1e-12));
        let s = cuts[0].simplified(1e-9);
        assert_eq!(s.points.len(), 4);
        assert!(s.closed && (s.length() - 4.0).abs() < 1e-9);
        // An oblique cut through the cube's centre is a hexagon-or-larger loop, still closed.
        let oblique = slice_mesh(&v, &t, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], 1e-9);
        assert_eq!(oblique.len(), 1);
        assert!(oblique[0].closed);
        assert_eq!(oblique[0].simplified(1e-9).points.len(), 6);
    }

    #[test]
    fn plane_misses_and_stacked_levels() {
        let (v, t) = cube();
        assert!(slice_mesh(&v, &t, [0.0, 0.0, 2.0], [0.0, 0.0, 1.0], 1e-9).is_empty());
        let layers = slice_mesh_levels(&v, &t, 2, &[0.25, 0.5, 0.75, 1.5], 1e-9);
        let counts: Vec<usize> = layers.iter().map(|(_, c)| c.len()).collect();
        assert_eq!(counts, vec![1, 1, 1, 0]);
        assert!(layers[..3].iter().all(|(_, c)| (c[0].length() - 4.0).abs() < 1e-9));
        // Slicing exactly on a face leaves the face's own triangles silent but the side walls
        // still report their edge, so the boundary comes out as a loop.
        let on_face = slice_mesh(&v, &t, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1e-9);
        assert!(!on_face.is_empty());
    }

    #[test]
    fn a_plane_through_a_ring_of_vertices_gives_one_loop() {
        // A square tube from z = 0 to 2 with a ring of vertices at z = 1: each ring edge belongs
        // to a triangle above and one below, and used to be reported twice.
        let corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let v: Vec<[f64; 3]> = (0..3).flat_map(|z| corners.iter().map(move |c| [c[0], c[1], z as f64])).collect();
        let mut t = Vec::new();
        for level in 0..2 {
            for k in 0..4 {
                let (a, b) = (4 * level + k, 4 * level + (k + 1) % 4);
                t.push([a, b, b + 4]);
                t.push([a, b + 4, a + 4]);
            }
        }
        let cuts = slice_mesh(&v, &t, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 1e-9);
        assert_eq!(cuts.len(), 1);
        assert!(cuts[0].closed && (cuts[0].length() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn two_separate_solids_give_two_contours_and_open_meshes_give_open_chains() {
        let (mut v, mut t) = cube();
        let base = v.len();
        v.extend(cube().0.into_iter().map(|p| [p[0] + 5.0, p[1], p[2]]));
        t.extend(cube().1.into_iter().map(|tr| [tr[0] + base, tr[1] + base, tr[2] + base]));
        let cuts = slice_mesh(&v, &t, [0.0, 0.0, 0.5], [0.0, 0.0, 1.0], 1e-9);
        assert_eq!(cuts.len(), 2);
        assert!(cuts.iter().all(|c| c.closed));
        // Drop the top face (two triangles): cutting just under the open top still closes via the
        // walls, but cutting a lone wall quad yields an open chain.
        let wall: Vec<[usize; 3]> = vec![[0, 1, 5], [0, 5, 4]]; // the y=0 wall
        let open = slice_mesh(&v, &wall, [0.0, 0.0, 0.5], [0.0, 0.0, 1.0], 1e-9);
        assert_eq!(open.len(), 1);
        assert!(!open[0].closed && (open[0].length() - 1.0).abs() < 1e-9);
        // Bad indices are skipped rather than panicking.
        assert!(slice_mesh(&v, &[[0, 1, 99]], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0], 1e-9).is_empty());
    }
}
