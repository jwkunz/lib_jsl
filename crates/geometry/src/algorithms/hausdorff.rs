//! The Hausdorff distance between point sets (SciPy's `scipy.spatial.distance.directed_hausdorff`).
//!
//! See the book entry [Rotation fitting, Procrustes analysis and Hausdorff distance](https://jwkunz.github.io/lib_jsl/book/geometry/alignment.html#the-hausdorff-distance).

use lib_jsl_core::ErrorsJSL;

/// The result of [`directed_hausdorff`]: the distance and the pair of points that realise it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct DirectedHausdorff {
    pub distance: f64,
    /// The point of `u` farthest from `v`.
    pub index_u: usize,
    /// Its nearest point in `v`.
    pub index_v: usize,
}

// ANCHOR: directed_hausdorff
/// The directed Hausdorff distance `h(u, v) = max_{a ∈ u} min_{b ∈ v} ‖a − b‖` between point sets
/// in any dimension (SciPy's `directed_hausdorff`), by Taha and Hanbury's early-break algorithm:
/// the inner search over `v` stops as soon as some point is closer than the largest minimum found
/// so far, since that `a` can no longer raise the maximum. SciPy shuffles the points first (with
/// a fixed seed) to make early breaks likely; the library keeps the given order, which gives the
/// same distance and, without ties, the same pair.
pub fn directed_hausdorff(u: &[Vec<f64>], v: &[Vec<f64>]) -> Result<DirectedHausdorff, ErrorsJSL> {
    let d = u.first().map_or(0, |p| p.len());
    if u.is_empty() || v.is_empty() || d == 0 || u.iter().chain(v).any(|p| p.len() != d) {
        return Err(ErrorsJSL::InvalidInputRange("directed_hausdorff: need non-empty point sets of one dimension"));
    }
    let (mut cmax, mut iu, mut iv) = (0.0f64, 0usize, 0usize);
    for (i, a) in u.iter().enumerate() {
        let (mut cmin, mut jmin, mut broke) = (f64::INFINITY, 0usize, false);
        for (j, b) in v.iter().enumerate() {
            let dist: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
            if dist < cmax {
                broke = true;
                break;
            }
            if dist < cmin {
                cmin = dist;
                jmin = j;
            }
        }
        if !broke && cmin != f64::INFINITY && cmin > cmax {
            cmax = cmin;
            iu = i;
            iv = jmin;
        }
    }
    Ok(DirectedHausdorff { distance: cmax.sqrt(), index_u: iu, index_v: iv })
}
// ANCHOR_END: directed_hausdorff

/// The (symmetric) Hausdorff distance `max(h(u, v), h(v, u))`.
pub fn hausdorff_distance(u: &[Vec<f64>], v: &[Vec<f64>]) -> Result<f64, ErrorsJSL> {
    Ok(directed_hausdorff(u, v)?.distance.max(directed_hausdorff(v, u)?.distance))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_definition() {
        let u: Vec<Vec<f64>> = (0..30).map(|i| vec![(i as f64 * 0.37).sin(), (i as f64 * 0.91).cos(), i as f64 * 0.05]).collect();
        let v: Vec<Vec<f64>> = (0..25).map(|i| vec![(i as f64 * 0.53).cos(), (i as f64 * 0.29).sin(), 0.7]).collect();
        let r = directed_hausdorff(&u, &v).unwrap();
        let dist = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt();
        let brute = u.iter().map(|a| v.iter().map(|b| dist(a, b)).fold(f64::INFINITY, f64::min)).fold(0.0, f64::max);
        assert!((r.distance - brute).abs() < 1e-15);
        assert!((dist(&u[r.index_u], &v[r.index_v]) - brute).abs() < 1e-15);
        assert!(hausdorff_distance(&u, &v).unwrap() >= brute);
    }
}
