//! Boolean operations on simple polygons by the Greiner-Hormann algorithm.
//!
//! Both polygons are walked as cyclic vertex lists with their mutual edge intersections
//! inserted; each intersection is classified as an entry into or exit from the other polygon,
//! and the result is traced by following one polygon until the next intersection, then
//! switching to the other. Degenerate configurations (a vertex on the other polygon's boundary,
//! overlapping collinear edges) are resolved by perturbing the clip polygon by a relative 1e-9,
//! so results are exact up to that tolerance. Results are counter-clockwise outer boundaries
//! and clockwise holes. Holes arise in a difference when the clip polygon lies strictly inside
//! the subject, and in a union when the two polygons enclose a region between them; areas add
//! up correctly when the holes' negative signed areas are included.
//!
//! See the book entry [Convex hulls and polygon booleans](https://jwkunz.github.io/lib_jsl/book/geometry/hulls-booleans.html#the-greinerhormann-algorithm).

/// The boolean operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]

pub enum BooleanOperation {
    Intersection,
    Union,
    /// The subject minus the clip polygon.
    Difference,
}

type P = [f64; 2];

fn signed_area(poly: &[P]) -> f64 {
    (0..poly.len()).map(|k| { let (a, b) = (poly[k], poly[(k + 1) % poly.len()]); a[0] * b[1] - b[0] * a[1] }).sum::<f64>() / 2.0
}

/// Whether `p` is inside the polygon (even-odd rule).
pub fn point_in_polygon(p: P, poly: &[P]) -> bool {
    let mut inside = false;
    for k in 0..poly.len() {
        let (a, b) = (poly[k], poly[(k + 1) % poly.len()]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]) {
            inside = !inside;
        }
    }
    inside
}

#[derive(Clone)]

struct Node {
    p: P,
    /// Index of the matching node in the other list, for intersections.
    neighbour: Option<usize>,
    entry: bool,
    visited: bool,
}

enum Outcome {
    Degenerate,
    Crossing(Vec<Node>, Vec<Node>),
    Disjoint,
}

/// Inserts the edge intersections into both vertex lists.
fn build(s: &[P], c: &[P]) -> Outcome {
    // Must stay well below the perturbation applied on degeneracy.
    let eps = 1e-12;
    let mut on_s: Vec<Vec<(f64, usize)>> = vec![Vec::new(); s.len()];
    let mut on_c: Vec<Vec<(f64, usize)>> = vec![Vec::new(); c.len()];
    let mut points = Vec::new();
    for i in 0..s.len() {
        let (a, b) = (s[i], s[(i + 1) % s.len()]);
        for j in 0..c.len() {
            let (p, q) = (c[j], c[(j + 1) % c.len()]);
            let r = [b[0] - a[0], b[1] - a[1]];
            let d = [q[0] - p[0], q[1] - p[1]];
            let denominator = r[0] * d[1] - r[1] * d[0];
            let scale = (r[0].abs() + r[1].abs()) * (d[0].abs() + d[1].abs());
            let w = [p[0] - a[0], p[1] - a[1]];
            if denominator.abs() <= 1e-14 * scale {
                // Parallel: degenerate only if collinear and overlapping.
                if (w[0] * r[1] - w[1] * r[0]).abs() <= 1e-12 * scale.max(1e-300) {
                    let rr = r[0] * r[0] + r[1] * r[1];
                    let t0 = (w[0] * r[0] + w[1] * r[1]) / rr;
                    let t1 = ((q[0] - a[0]) * r[0] + (q[1] - a[1]) * r[1]) / rr;
                    if t0.max(t1) >= -eps && t0.min(t1) <= 1.0 + eps {
                        return Outcome::Degenerate;
                    }
                }
                continue;
            }
            let t = (w[0] * d[1] - w[1] * d[0]) / denominator;
            let u = (w[0] * r[1] - w[1] * r[0]) / denominator;
            if t < -eps || t > 1.0 + eps || u < -eps || u > 1.0 + eps {
                continue;
            }
            if t < eps || t > 1.0 - eps || u < eps || u > 1.0 - eps {
                return Outcome::Degenerate;
            }
            let id = points.len();
            points.push([a[0] + t * r[0], a[1] + t * r[1]]);
            on_s[i].push((t, id));
            on_c[j].push((u, id));
        }
    }
    if points.is_empty() {
        return Outcome::Disjoint;
    }
    let make = |poly: &[P], on: &mut [Vec<(f64, usize)>]| -> (Vec<Node>, Vec<Option<usize>>) {
        let mut nodes = Vec::new();
        let mut position = vec![None; points.len()];
        for (k, list) in on.iter_mut().enumerate() {
            nodes.push(Node { p: poly[k], neighbour: None, entry: false, visited: false });
            list.sort_by(|x, y| x.0.total_cmp(&y.0));
            for &(_, id) in list.iter() {
                position[id] = Some(nodes.len());
                nodes.push(Node { p: points[id], neighbour: Some(usize::MAX), entry: false, visited: false });
            }
        }
        (nodes, position)
    };
    let (mut ns, ps) = make(s, &mut on_s);
    let (mut nc, pc) = make(c, &mut on_c);
    for id in 0..points.len() {
        let (a, b) = (ps[id].expect("inserted"), pc[id].expect("inserted"));
        ns[a].neighbour = Some(b);
        nc[b].neighbour = Some(a);
    }
    Outcome::Crossing(ns, nc)
}

fn mark_entries(nodes: &mut [Node], other: &[P], flip: bool) {
    let mut entry = !point_in_polygon(nodes[0].p, other);
    for n in nodes.iter_mut().filter(|n| n.neighbour.is_some()) {
        n.entry = entry != flip;
        entry = !entry;
    }
}

fn trace(mut ns: Vec<Node>, mut nc: Vec<Node>) -> Vec<Vec<P>> {
    let mut result = Vec::new();
    while let Some(start) = (0..ns.len()).find(|&i| ns[i].neighbour.is_some() && !ns[i].visited) {
        let mut polygon = vec![ns[start].p];
        let (mut on_subject, mut i) = (true, start);
        loop {
            let (list, other) = if on_subject { (&mut ns, &mut nc) } else { (&mut nc, &mut ns) };
            list[i].visited = true;
            let j = list[i].neighbour.expect("intersection");
            other[j].visited = true;
            let forward = list[i].entry;
            let len = list.len();
            loop {
                i = if forward { (i + 1) % len } else { (i + len - 1) % len };
                polygon.push(list[i].p);
                if list[i].neighbour.is_some() {
                    break;
                }
            }
            let j = list[i].neighbour.expect("intersection");
            on_subject = !on_subject;
            i = j;
            let back = if on_subject { &ns } else { &nc };
            if back[i].visited {
                break;
            }
        }
        polygon.pop(); // the closing point repeats the start
        if polygon.len() >= 3 && signed_area(&polygon).abs() > 0.0 {
            result.push(polygon);
        }
    }
    result
}

/// Orients traced loops by nesting depth: loops inside an even number of other loops are outer
/// boundaries (counter-clockwise), the others holes (clockwise).
fn orient_by_nesting(mut loops: Vec<Vec<P>>) -> Vec<Vec<P>> {
    let probes: Vec<P> = loops.iter().map(|l| [(l[0][0] + l[1][0]) / 2.0, (l[0][1] + l[1][1]) / 2.0]).collect();
    let depth: Vec<usize> = (0..loops.len()).map(|i| (0..loops.len()).filter(|&j| j != i && point_in_polygon(probes[i], &loops[j])).count()).collect();
    for (l, d) in loops.iter_mut().zip(depth) {
        if (signed_area(l) > 0.0) != d.is_multiple_of(2) {
            l.reverse();
        }
    }
    loops
}

/// Applies `operation` to the simple polygons `subject` and `clip` (either orientation).
pub fn polygon_boolean(subject: &[P], clip: &[P], operation: BooleanOperation) -> Vec<Vec<P>> {
    if subject.len() < 3 || clip.len() < 3 {
        return match operation {
            BooleanOperation::Intersection => Vec::new(),
            _ if subject.len() >= 3 => vec![subject.to_vec()],
            BooleanOperation::Union if clip.len() >= 3 => vec![clip.to_vec()],
            _ => Vec::new(),
        };
    }
    let ccw = |p: &[P]| -> Vec<P> { if signed_area(p) < 0.0 { p.iter().rev().copied().collect() } else { p.to_vec() } };
    let s = ccw(subject);
    let mut c = ccw(clip);
    let scale = s.iter().chain(&c).flat_map(|p| p.iter()).fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    for _attempt in 0..20 {
        match build(&s, &c) {
            Outcome::Degenerate => {
                // Nudge the clip polygon off the degeneracy.
                for p in c.iter_mut() {
                    for v in p.iter_mut() {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        *v += 1e-9 * scale * ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
                    }
                }
            }
            Outcome::Disjoint => {
                let s_in_c = point_in_polygon(s[0], &c);
                let c_in_s = point_in_polygon(c[0], &s);
                return match operation {
                    BooleanOperation::Intersection if s_in_c => vec![s],
                    BooleanOperation::Intersection if c_in_s => vec![c],
                    BooleanOperation::Intersection => Vec::new(),
                    BooleanOperation::Union if s_in_c => vec![c],
                    BooleanOperation::Union if c_in_s => vec![s],
                    BooleanOperation::Union => vec![s, c],
                    BooleanOperation::Difference if s_in_c => Vec::new(),
                    BooleanOperation::Difference if c_in_s => vec![s, c.into_iter().rev().collect()],
                    BooleanOperation::Difference => vec![s],
                };
            }
            Outcome::Crossing(mut ns, mut nc) => {
                let (flip_s, flip_c) = match operation {
                    BooleanOperation::Intersection => (false, false),
                    BooleanOperation::Union => (true, true),
                    BooleanOperation::Difference => (true, false),
                };
                mark_entries(&mut ns, &c, flip_s);
                mark_entries(&mut nc, &s, flip_c);
                return orient_by_nesting(trace(ns, nc));
            }
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use BooleanOperation::*;

    fn area(polys: &[Vec<P>]) -> f64 {
        polys.iter().map(|p| signed_area(p)).sum()
    }

    fn square(x: f64, y: f64, side: f64) -> Vec<P> {
        vec![[x, y], [x + side, y], [x + side, y + side], [x, y + side]]
    }

    #[test]
    fn overlapping_squares_satisfy_the_area_identities() {
        let (a, b) = (square(0.0, 0.0, 2.0), square(1.0, 1.0, 2.0));
        let inter = polygon_boolean(&a, &b, Intersection);
        assert_eq!(inter.len(), 1);
        assert!((area(&inter) - 1.0).abs() < 1e-12);
        assert!((area(&polygon_boolean(&a, &b, Union)) - 7.0).abs() < 1e-12);
        assert!((area(&polygon_boolean(&a, &b, Difference)) - 3.0).abs() < 1e-12);
        assert!((area(&polygon_boolean(&b, &a, Difference)) - 3.0).abs() < 1e-12);
        // Orientation of the inputs does not matter.
        let reversed: Vec<P> = b.iter().rev().copied().collect();
        assert!((area(&polygon_boolean(&a, &reversed, Intersection)) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_union_can_enclose_a_hole() {
        // A "C" open to the right, closed off by a rectangle: the gap between them is a hole.
        let c = vec![[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [1.0, 1.0], [1.0, 2.0], [3.0, 2.0], [3.0, 3.0], [0.0, 3.0]];
        let bar = vec![[2.0, 0.5], [4.0, 0.5], [4.0, 2.5], [2.0, 2.5]];
        let u = polygon_boolean(&c, &bar, Union);
        let holes: Vec<&Vec<P>> = u.iter().filter(|p| signed_area(p) < 0.0).collect();
        assert_eq!((u.len(), holes.len()), (2, 1));
        assert!((signed_area(holes[0]) + 1.0).abs() < 1e-9);
        assert!((area(&u) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn concave_shapes_split_into_several_pieces() {
        // A U shape cut by a horizontal bar through both arms: two pieces.
        let u = vec![[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [2.0, 3.0], [2.0, 1.0], [1.0, 1.0], [1.0, 3.0], [0.0, 3.0]];
        let bar = vec![[-1.0, 2.0], [4.0, 2.0], [4.0, 2.5], [-1.0, 2.5]];
        let pieces = polygon_boolean(&u, &bar, Intersection);
        assert_eq!(pieces.len(), 2);
        assert!((area(&pieces) - 1.0).abs() < 1e-12);
        let (au, ab) = (signed_area(&u), signed_area(&bar));
        let union = area(&polygon_boolean(&u, &bar, Union));
        assert!((union - (au + ab - 1.0)).abs() < 1e-9, "{union}");
        assert!((area(&polygon_boolean(&u, &bar, Difference)) - (au - 1.0)).abs() < 1e-9);
    }

    #[test]
    fn nested_disjoint_and_degenerate_cases() {
        let (outer, inner, far) = (square(0.0, 0.0, 4.0), square(1.0, 1.0, 1.0), square(10.0, 10.0, 1.0));
        assert!((area(&polygon_boolean(&outer, &inner, Intersection)) - 1.0).abs() < 1e-12);
        assert!((area(&polygon_boolean(&outer, &inner, Union)) - 16.0).abs() < 1e-12);
        let holed = polygon_boolean(&outer, &inner, Difference);
        assert!(holed.len() == 2 && (area(&holed) - 15.0).abs() < 1e-12);
        assert!(polygon_boolean(&outer, &far, Intersection).is_empty());
        assert!((area(&polygon_boolean(&outer, &far, Union)) - 17.0).abs() < 1e-12);
        // Shared edges and vertices are handled by perturbation.
        let (left, right) = (square(0.0, 0.0, 1.0), square(1.0, 0.0, 1.0));
        assert!((area(&polygon_boolean(&left, &right, Union)) - 2.0).abs() < 1e-8);
        assert!(area(&polygon_boolean(&left, &right, Intersection)).abs() < 1e-8);
        let corner = square(1.0, 1.0, 1.0);
        assert!((area(&polygon_boolean(&left, &corner, Union)) - 2.0).abs() < 1e-8);
        let same = polygon_boolean(&left, &left, Intersection);
        assert!((area(&same) - 1.0).abs() < 1e-8);
    }
}
