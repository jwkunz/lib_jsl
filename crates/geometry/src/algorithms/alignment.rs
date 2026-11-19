//! Fitting rotations to data, following `scipy.spatial`: the rotation that best aligns two sets
//! of vectors (`Rotation.align_vectors`, the Kabsch or Wahba problem), the mean of rotations
//! (`Rotation.mean`), orthogonal Procrustes analysis (`procrustes`, `orthogonal_procrustes`),
//! and interpolation along great circles (`geometric_slerp`).
//!
//! See the book entry [Rotation fitting, Procrustes analysis and Hausdorff distance](https://jwkunz.github.io/lib_jsl/book/geometry/alignment.html).

use super::quaternions::Quaternion;
use super::rotations::{Matrix3, Rotation3};
use super::small_svd::{determinant, svd_square};
use lib_jsl_core::ErrorsJSL;

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

/// The result of [`align_vectors`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Alignment {
    /// The rotation `C` with `a ≈ C b`.
    pub rotation: Rotation3,
    /// The root of the weighted sum of squared distances `√Σ wᵢ ‖aᵢ − C bᵢ‖²` (without the
    /// primary pair when its weight is infinite).
    pub rssd: f64,
}

// ANCHOR: align_vectors
/// The rotation `C` minimising `Σ wᵢ ‖aᵢ − C bᵢ‖²` (SciPy's `Rotation.align_vectors`; Wahba's
/// problem). With `B = Σ wᵢ aᵢ bᵢᵀ = U S Vᵀ`, the answer is `C = U diag(1, 1, det(UVᵀ)) Vᵀ` (Kabsch's
/// algorithm); the determinant correction keeps it a rotation rather than a reflection. With a
/// single pair, or one infinite weight, that pair (the primary) is aligned exactly by the
/// smallest rotation, and the remaining freedom — a rotation about the primary — best aligns
/// the others.
pub fn align_vectors(a: &[[f64; 3]], b: &[[f64; 3]], weights: Option<&[f64]>) -> Result<Alignment, ErrorsJSL> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return Err(ErrorsJSL::IncompatibleArraySizes((a.len(), b.len())));
    }
    let w: Vec<f64> = match weights {
        Some(w) if w.len() != n => return Err(ErrorsJSL::IncompatibleArraySizes((w.len(), n))),
        Some(w) if w.iter().any(|v| v.is_nan() || *v < 0.0) => return Err(ErrorsJSL::InvalidInputRange("align_vectors: weights must be non-negative")),
        Some(w) => w.to_vec(),
        None => vec![1.0; n],
    };
    let n_inf = w.iter().filter(|v| v.is_infinite()).count();
    if n_inf > 1 {
        return Err(ErrorsJSL::InvalidInputRange("align_vectors: only one infinite weight is allowed"));
    }
    if n == 1 || n_inf == 1 {
        let p = if n == 1 { 0 } else { w.iter().position(|v| v.is_infinite()).expect("one infinite weight") };
        return align_with_primary(a, b, &w, p);
    }
    let mut bm = vec![vec![0.0; 3]; 3];
    for i in 0..n {
        for r in 0..3 {
            for c in 0..3 {
                bm[r][c] += w[i] * a[i][r] * b[i][c];
            }
        }
    }
    let (mut u, mut s, v) = svd_square(&bm);
    let uv: Vec<Vec<f64>> = (0..3).map(|r| (0..3).map(|c| (0..3).map(|k| u[r][k] * v[c][k]).sum()).collect()).collect();
    if determinant(&uv) < 0.0 {
        s[2] = -s[2];
        for row in u.iter_mut() {
            row[2] = -row[2];
        }
    }
    let c: Matrix3 = std::array::from_fn(|r| std::array::from_fn(|col| (0..3).map(|k| u[r][k] * v[col][k]).sum()));
    let ssd: f64 = (0..n).map(|i| w[i] * (dot3(a[i], a[i]) + dot3(b[i], b[i]))).sum::<f64>() - 2.0 * s.iter().sum::<f64>();
    Ok(Alignment { rotation: Rotation3::from_matrix(&c), rssd: ssd.max(0.0).sqrt() })
}
// ANCHOR_END: align_vectors

/// Align `a[p] ← b[p]` exactly with the smallest rotation, then rotate about `â[p]` by the angle
/// that best aligns the other pairs.
fn align_with_primary(a: &[[f64; 3]], b: &[[f64; 3]], w: &[f64], p: usize) -> Result<Alignment, ErrorsJSL> {
    let (na, nb) = (norm3(a[p]), norm3(b[p]));
    if na == 0.0 || nb == 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("align_vectors: cannot align zero-length primary vectors"));
    }
    let ap = a[p].map(|v| v / na);
    let bp = b[p].map(|v| v / nb);
    let cross = cross3(bp, ap);
    let primary = if norm3(cross) == 0.0 && dot3(ap, bp) < 0.0 {
        // Exactly opposite: half a turn about an axis perpendicular to â, as SciPy picks it.
        let i = (0..3).min_by(|&x, &y| ap[x].abs().total_cmp(&ap[y].abs())).expect("three components");
        let axis = match i {
            0 => [0.0, -ap[2], ap[1]],
            1 => [ap[2], 0.0, -ap[0]],
            _ => [-ap[1], ap[0], 0.0],
        };
        Rotation3::from_axis_angle(axis, std::f64::consts::PI)
    } else {
        Rotation3::from_two_vectors(bp, ap)
    };
    let (mut sn, mut cs) = (0.0, 0.0);
    for i in (0..a.len()).filter(|&i| i != p) {
        let c = primary.rotate(b[i]);
        sn += w[i] * dot3(cross3(c, a[i]), ap);
        cs += w[i] * (dot3(c, a[i]) - dot3(c, ap) * dot3(a[i], ap));
    }
    let phi = sn.atan2(cs);
    let rotation = Rotation3::from_axis_angle(ap, phi) * primary;
    // The primary counts only when its weight is finite (the single-pair case).
    let ssd: f64 = (0..a.len())
        .filter(|&i| i != p || w[p].is_finite())
        .map(|i| {
            let c = rotation.rotate(b[i]);
            w[i] * (0..3).map(|k| (a[i][k] - c[k]).powi(2)).sum::<f64>()
        })
        .sum();
    Ok(Alignment { rotation, rssd: ssd.sqrt() })
}

// ANCHOR: mean
/// The mean of rotations (SciPy's `Rotation.mean`): the unit quaternion maximising
/// `Σ wᵢ (q · qᵢ)²`, the eigenvector of the largest eigenvalue of `Σ wᵢ qᵢ qᵢᵀ` (Markley et al.).
/// It minimises the weighted sum of squared chordal distances `‖R − Rᵢ‖²`, and is unaffected by
/// the sign ambiguity `q ~ −q`.
pub fn rotation_mean(rotations: &[Rotation3], weights: Option<&[f64]>) -> Result<Rotation3, ErrorsJSL> {
    if rotations.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("rotation_mean: the mean of no rotations is undefined"));
    }
    if let Some(w) = weights {
        if w.len() != rotations.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((w.len(), rotations.len())));
        }
        if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("rotation_mean: weights must be non-negative"));
        }
    }
    let mut k = vec![vec![0.0; 4]; 4];
    for (i, r) in rotations.iter().enumerate() {
        let q = r.quaternion();
        let v = [q.x, q.y, q.z, q.w];
        let wi = weights.map_or(1.0, |w| w[i]);
        for p in 0..4 {
            for c in 0..4 {
                k[p][c] += wi * v[p] * v[c];
            }
        }
    }
    // K is symmetric positive semi-definite: its leading singular vector is the eigenvector.
    let (u, _, _) = svd_square(&k);
    Ok(Rotation3::from_quaternion(Quaternion::new(u[3][0], u[0][0], u[1][0], u[2][0])))
}
// ANCHOR_END: mean

// ANCHOR: procrustes
/// The orthogonal matrix `R` minimising `‖A R − B‖_F` and the scale `Σ σᵢ` (SciPy's
/// `orthogonal_procrustes`): with `AᵀB = U S Vᵀ`, `R = U Vᵀ`. `A` and `B` are `m × d` (rows are
/// points); reflections are allowed.
pub fn orthogonal_procrustes(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, f64), ErrorsJSL> {
    let (m, d) = shape(a)?;
    if shape(b)? != (m, d) {
        return Err(ErrorsJSL::IncompatibleArraySizes((b.len(), m)));
    }
    let atb: Vec<Vec<f64>> = (0..d).map(|r| (0..d).map(|c| (0..m).map(|i| a[i][r] * b[i][c]).sum()).collect()).collect();
    let (u, s, v) = svd_square(&atb);
    let r = (0..d).map(|i| (0..d).map(|j| (0..d).map(|k| u[i][k] * v[j][k]).sum()).collect()).collect();
    Ok((r, s.iter().sum()))
}

/// The result of [`procrustes`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ProcrustesResult {
    /// The first data set, centred and scaled to unit Frobenius norm.
    pub mtx1: Vec<Vec<f64>>,
    /// The second, centred, scaled, and rotated (or reflected) and rescaled onto the first.
    pub mtx2: Vec<Vec<f64>>,
    /// `Σ (mtx1 − mtx2)²`, between 0 (identical shapes) and 1.
    pub disparity: f64,
}

/// Procrustes analysis of two configurations of `m` points in `d` dimensions (SciPy's
/// `scipy.spatial.procrustes`): both are centred and scaled to unit norm, then the second is
/// rotated (reflections allowed) and scaled to best match the first; the disparity is the
/// remaining sum of squared differences, a measure of difference in shape.
pub fn procrustes(data1: &[Vec<f64>], data2: &[Vec<f64>]) -> Result<ProcrustesResult, ErrorsJSL> {
    let (m, d) = shape(data1)?;
    if shape(data2)? != (m, d) {
        return Err(ErrorsJSL::IncompatibleArraySizes((data2.len(), m)));
    }
    let standardise = |x: &[Vec<f64>]| -> Result<Vec<Vec<f64>>, ErrorsJSL> {
        let mean: Vec<f64> = (0..d).map(|j| x.iter().map(|r| r[j]).sum::<f64>() / m as f64).collect();
        let c: Vec<Vec<f64>> = x.iter().map(|r| r.iter().zip(&mean).map(|(v, mu)| v - mu).collect()).collect();
        let norm = c.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("procrustes: the input must contain more than one distinct point"));
        }
        Ok(c.into_iter().map(|r| r.into_iter().map(|v| v / norm).collect()).collect())
    };
    let mtx1 = standardise(data1)?;
    let mtx2 = standardise(data2)?;
    let (r, s) = orthogonal_procrustes(&mtx1, &mtx2)?;
    // mtx2 ← mtx2 Rᵀ s.
    let mtx2: Vec<Vec<f64>> = mtx2.iter().map(|row| (0..d).map(|j| s * (0..d).map(|k| row[k] * r[j][k]).sum::<f64>()).collect()).collect();
    let disparity = mtx1.iter().zip(&mtx2).flat_map(|(p, q)| p.iter().zip(q).map(|(a, b)| (a - b).powi(2))).sum();
    Ok(ProcrustesResult { mtx1, mtx2, disparity })
}
// ANCHOR_END: procrustes

fn shape(x: &[Vec<f64>]) -> Result<(usize, usize), ErrorsJSL> {
    let m = x.len();
    let d = x.first().map_or(0, |r| r.len());
    if m == 0 || d == 0 || x.iter().any(|r| r.len() != d) {
        return Err(ErrorsJSL::InvalidInputRange("expected a non-empty matrix with rows of equal length"));
    }
    Ok((m, d))
}

// ANCHOR: slerp
/// Points along the great circle from `start` to `end` on the unit sphere in any dimension, at
/// the fractions `t` of the way (SciPy's `geometric_slerp`):
/// `(sin((1 − t)ω) start + sin(tω) end) / sin ω`, `ω` the angle between them. Both must be unit
/// vectors; antipodal points have no unique great circle and are an error, and equal points give
/// that point.
pub fn geometric_slerp(start: &[f64], end: &[f64], t: &[f64]) -> Result<Vec<Vec<f64>>, ErrorsJSL> {
    let tol = 1e-7;
    if start.len() != end.len() || start.len() < 2 {
        return Err(ErrorsJSL::IncompatibleArraySizes((start.len(), end.len())));
    }
    let unit = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>().sqrt() - 1.0).abs() < tol;
    if !unit(start) || !unit(end) {
        return Err(ErrorsJSL::InvalidInputRange("geometric_slerp: start and end must be unit vectors"));
    }
    if t.iter().any(|v| !(0.0..=1.0).contains(v)) {
        return Err(ErrorsJSL::InvalidInputRange("geometric_slerp: interpolation parameters must lie in [0, 1]"));
    }
    let d: f64 = start.iter().zip(end).map(|(a, b)| a * b).sum::<f64>().clamp(-1.0, 1.0);
    if start.iter().zip(end).all(|(a, b)| (a - b).abs() <= tol) {
        return Ok(t.iter().map(|_| start.to_vec()).collect());
    }
    if start.iter().zip(end).all(|(a, b)| (a + b).abs() <= tol) {
        return Err(ErrorsJSL::InvalidInputRange("geometric_slerp: antipodal points have no unique great circle"));
    }
    let omega = d.acos();
    let so = omega.sin();
    Ok(t.iter()
        .map(|&tt| {
            let (p, q) = (((1.0 - tt) * omega).sin() / so, (tt * omega).sin() / so);
            start.iter().zip(end).map(|(a, b)| p * a + q * b).collect()
        })
        .collect())
}
// ANCHOR_END: slerp

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_a_rotation_and_handles_the_special_cases() {
        let truth = Rotation3::from_axis_angle([0.3, -1.0, 0.5], 2.2);
        let b = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.5], [0.3, -0.4, 1.0], [1.0, 1.0, 1.0]];
        let a: Vec<[f64; 3]> = b.iter().map(|&v| truth.rotate(v)).collect();
        let r = align_vectors(&a, &b, None).unwrap();
        assert!(r.rotation.angle_to(&truth) < 1e-12 && r.rssd < 1e-7);
        // Mirrored data cannot be matched by a rotation: the determinant correction.
        let mirror: Vec<[f64; 3]> = a.iter().map(|v| [-v[0], v[1], v[2]]).collect();
        let r = align_vectors(&mirror, &b, None).unwrap();
        assert!(crate::algorithms::rotations::is_rotation_matrix(&r.rotation.matrix(), 1e-12) && r.rssd > 0.5);
        // A primary vector with infinite weight is matched exactly.
        let noisy: Vec<[f64; 3]> = a.iter().enumerate().map(|(i, v)| [v[0] + 0.01 * i as f64, v[1], v[2]]).collect();
        let r = align_vectors(&noisy, &b, Some(&[f64::INFINITY, 1.0, 1.0, 1.0])).unwrap();
        let c = r.rotation.rotate(b[0]);
        assert!((0..3).all(|k| (c[k] - noisy[0][k]).abs() < 1e-12));
        // The mean of rotations about one axis is the rotation by the mean angle.
        let rs: Vec<Rotation3> = [0.1, 0.4, 0.7].iter().map(|&t| Rotation3::from_axis_angle([0.0, 0.0, 1.0], t)).collect();
        let m = rotation_mean(&rs, None).unwrap();
        assert!(m.angle_to(&Rotation3::from_axis_angle([0.0, 0.0, 1.0], 0.4)) < 1e-12);
    }

    #[test]
    fn procrustes_ignores_similarity_transforms() {
        let a = vec![vec![1.0, 3.0], vec![1.0, 2.0], vec![1.0, 1.0], vec![2.0, 1.0]];
        let b = vec![vec![4.0, -2.0], vec![4.0, -4.0], vec![4.0, -6.0], vec![2.0, -6.0]];
        let r = procrustes(&a, &b).unwrap();
        assert!(r.disparity < 1e-24, "{}", r.disparity);
        let p = geometric_slerp(&[1.0, 0.0], &[0.0, 1.0], &[0.0, 0.5, 1.0]).unwrap();
        let h = 0.5f64.sqrt();
        assert!((p[1][0] - h).abs() < 1e-15 && (p[1][1] - h).abs() < 1e-15 && (p[2][1] - 1.0).abs() < 1e-15);
        assert!(geometric_slerp(&[1.0, 0.0], &[-1.0, 0.0], &[0.5]).is_err());
    }
}
