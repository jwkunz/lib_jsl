//! Rotations in two and three dimensions: matrices, axis-angle, Euler sequences, and composable
//! rotation types.
//!
//! [`Rotation3`] stores a unit [`Quaternion`], so composition is cheap and stays normalised, and
//! converts to and from rotation matrices, axis-angle and any of the twelve Euler sequences.
//! Matrices are row-major `[[f64; 3]; 3]` acting on column vectors. Euler sequences are
//! *intrinsic*: `Rotation3::from_euler(XYZ, [a, b, c])` rotates by `a` about X, then by `b` about
//! the new Y, then by `c` about the newest Z, i.e. `R = Rx(a) Ry(b) Rz(c)`.

use super::quaternions::Quaternion;
use std::f64::consts::PI;
use std::ops::Mul;

/// A 3×3 matrix in row-major order.
pub type Matrix3 = [[f64; 3]; 3];

pub const IDENTITY3: Matrix3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Matrix product `a * b`.
pub fn mat3_mul(a: &Matrix3, b: &Matrix3) -> Matrix3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

pub fn mat3_transpose(m: &Matrix3) -> Matrix3 {
    std::array::from_fn(|i| std::array::from_fn(|j| m[j][i]))
}

pub fn mat3_determinant(m: &Matrix3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Matrix-vector product.
pub fn mat3_apply(m: &Matrix3, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// Whether `m` is orthogonal with determinant +1 within `tol`.
pub fn is_rotation_matrix(m: &Matrix3, tol: f64) -> bool {
    let p = mat3_mul(m, &mat3_transpose(m));
    (0..3).all(|i| (0..3).all(|j| (p[i][j] - IDENTITY3[i][j]).abs() <= tol)) && (mat3_determinant(m) - 1.0).abs() <= tol
}

/// The rotation matrix closest (in Frobenius norm) to `m`, by Newton iteration for the polar
/// factor. Useful to clean up a matrix that has drifted after many multiplications. Returns `None`
/// if `m` is singular or has a negative determinant (is not near a proper rotation).
pub fn nearest_rotation_matrix(m: &Matrix3) -> Option<Matrix3> {
    if mat3_determinant(m) <= 0.0 {
        return None;
    }
    let mut r = *m;
    for _ in 0..50 {
        let det = mat3_determinant(&r);
        // inverse transpose = cofactor matrix / det
        let cof: Matrix3 = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let (i1, i2, j1, j2) = ((i + 1) % 3, (i + 2) % 3, (j + 1) % 3, (j + 2) % 3);
                (r[i1][j1] * r[i2][j2] - r[i1][j2] * r[i2][j1]) / det
            })
        });
        let next: Matrix3 = std::array::from_fn(|i| std::array::from_fn(|j| 0.5 * (r[i][j] + cof[i][j])));
        let delta: f64 = (0..3).flat_map(|i| (0..3).map(move |j| (i, j))).map(|(i, j)| (next[i][j] - r[i][j]).abs()).sum();
        r = next;
        if delta < 1e-15 {
            break;
        }
    }
    Some(r)
}

/// Rotation about the X axis by `angle` radians.
pub fn rotation_matrix_x(angle: f64) -> Matrix3 {
    let (s, c) = angle.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

/// Rotation about the Y axis by `angle` radians.
pub fn rotation_matrix_y(angle: f64) -> Matrix3 {
    let (s, c) = angle.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

/// Rotation about the Z axis by `angle` radians.
pub fn rotation_matrix_z(angle: f64) -> Matrix3 {
    let (s, c) = angle.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// Rodrigues' formula: the matrix of a rotation by `angle` about `axis` (normalised internally;
/// a zero axis gives the identity).
pub fn rotation_matrix_axis_angle(axis: [f64; 3], angle: f64) -> Matrix3 {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if n == 0.0 {
        return IDENTITY3;
    }
    let (x, y, z) = (axis[0] / n, axis[1] / n, axis[2] / n);
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
}

/// A coordinate axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    fn index(self) -> usize {
        self as usize
    }
    fn matrix(self, angle: f64) -> Matrix3 {
        match self {
            Axis::X => rotation_matrix_x(angle),
            Axis::Y => rotation_matrix_y(angle),
            Axis::Z => rotation_matrix_z(angle),
        }
    }
}

/// An intrinsic Euler-angle sequence: three axes where consecutive axes differ. Sequences whose
/// first and last axes coincide (XYX, ...) are "proper" Euler angles; the others (XYZ, ...) are
/// Tait-Bryan angles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EulerSequence(pub [Axis; 3]);

impl EulerSequence {
    pub const XYZ: EulerSequence = EulerSequence([Axis::X, Axis::Y, Axis::Z]);
    pub const XZY: EulerSequence = EulerSequence([Axis::X, Axis::Z, Axis::Y]);
    pub const YXZ: EulerSequence = EulerSequence([Axis::Y, Axis::X, Axis::Z]);
    pub const YZX: EulerSequence = EulerSequence([Axis::Y, Axis::Z, Axis::X]);
    pub const ZXY: EulerSequence = EulerSequence([Axis::Z, Axis::X, Axis::Y]);
    pub const ZYX: EulerSequence = EulerSequence([Axis::Z, Axis::Y, Axis::X]);
    pub const XYX: EulerSequence = EulerSequence([Axis::X, Axis::Y, Axis::X]);
    pub const XZX: EulerSequence = EulerSequence([Axis::X, Axis::Z, Axis::X]);
    pub const YXY: EulerSequence = EulerSequence([Axis::Y, Axis::X, Axis::Y]);
    pub const YZY: EulerSequence = EulerSequence([Axis::Y, Axis::Z, Axis::Y]);
    pub const ZXZ: EulerSequence = EulerSequence([Axis::Z, Axis::X, Axis::Z]);
    pub const ZYZ: EulerSequence = EulerSequence([Axis::Z, Axis::Y, Axis::Z]);

    /// Whether consecutive axes differ (the only valid sequences).
    pub fn is_valid(&self) -> bool {
        self.0[0] != self.0[1] && self.0[1] != self.0[2]
    }

    /// The matrix `R = R_a0(α) R_a1(β) R_a2(γ)` for `angles = [α, β, γ]`.
    pub fn to_matrix(&self, angles: [f64; 3]) -> Matrix3 {
        let m = mat3_mul(&self.0[0].matrix(angles[0]), &self.0[1].matrix(angles[1]));
        mat3_mul(&m, &self.0[2].matrix(angles[2]))
    }

    /// Angles `[α, β, γ]` reproducing the rotation matrix `m` (inverse of [`EulerSequence::to_matrix`]).
    /// The decomposition is not unique; the result has the middle angle in `[0, π]` for proper
    /// sequences and in `[-π/2, π/2]` for Tait-Bryan ones. At gimbal lock the third angle is
    /// folded into the first. `None` for an invalid sequence.
    pub fn from_matrix(&self, m: &Matrix3) -> Option<[f64; 3]> {
        if !self.is_valid() {
            return None;
        }
        let [a0, a1, a2] = self.0.map(Axis::index);
        let odd = if (a0 + 1) % 3 == a1 { 0 } else { 1 };
        let i = a0;
        let j = (a0 + 1 + odd) % 3;
        let k = (a0 + 2 - odd) % 3;
        let mut res = [0.0; 3];
        if a0 == a2 {
            res[0] = m[j][i].atan2(m[k][i]);
            let s2 = m[j][i].hypot(m[k][i]);
            if (odd == 1 && res[0] < 0.0) || (odd == 0 && res[0] > 0.0) {
                res[0] += if res[0] > 0.0 { -PI } else { PI };
                res[1] = -s2.atan2(m[i][i]);
            } else {
                res[1] = s2.atan2(m[i][i]);
            }
            let (s1, c1) = res[0].sin_cos();
            res[2] = (c1 * m[j][k] - s1 * m[k][k]).atan2(c1 * m[j][j] - s1 * m[k][j]);
        } else {
            res[0] = m[j][k].atan2(m[k][k]);
            let c2 = m[i][i].hypot(m[i][j]);
            if (odd == 1 && res[0] < 0.0) || (odd == 0 && res[0] > 0.0) {
                res[0] += if res[0] > 0.0 { -PI } else { PI };
                res[1] = (-m[i][k]).atan2(-c2);
            } else {
                res[1] = (-m[i][k]).atan2(c2);
            }
            let (s1, c1) = res[0].sin_cos();
            res[2] = (s1 * m[k][i] - c1 * m[j][i]).atan2(c1 * m[j][j] - s1 * m[k][j]);
        }
        if odd == 0 {
            res = res.map(|a| -a);
        }
        Some(res)
    }
}

/// A rotation in the plane about the origin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotation2 {
    pub angle: f64,
}

impl Rotation2 {
    pub fn new(angle: f64) -> Self {
        Rotation2 { angle }
    }
    pub fn identity() -> Self {
        Rotation2 { angle: 0.0 }
    }
    /// Counter-clockwise rotation taking direction `from` to direction `to`.
    pub fn from_two_vectors(from: [f64; 2], to: [f64; 2]) -> Self {
        let cross = from[0] * to[1] - from[1] * to[0];
        let dot = from[0] * to[0] + from[1] * to[1];
        Rotation2 { angle: cross.atan2(dot) }
    }
    pub fn matrix(&self) -> [[f64; 2]; 2] {
        let (s, c) = self.angle.sin_cos();
        [[c, -s], [s, c]]
    }
    pub fn rotate(&self, p: [f64; 2]) -> [f64; 2] {
        let (s, c) = self.angle.sin_cos();
        [c * p[0] - s * p[1], s * p[0] + c * p[1]]
    }
    /// Rotates `p` about `pivot` instead of the origin.
    pub fn rotate_about(&self, p: [f64; 2], pivot: [f64; 2]) -> [f64; 2] {
        let r = self.rotate([p[0] - pivot[0], p[1] - pivot[1]]);
        [r[0] + pivot[0], r[1] + pivot[1]]
    }
    pub fn inverse(&self) -> Self {
        Rotation2 { angle: -self.angle }
    }
    /// The angle wrapped into `(-π, π]`.
    pub fn normalized(&self) -> Self {
        let mut a = self.angle.rem_euclid(2.0 * PI);
        if a > PI {
            a -= 2.0 * PI;
        }
        Rotation2 { angle: a }
    }
}

impl Mul for Rotation2 {
    type Output = Rotation2;
    /// `a * b` rotates by `b` first, then by `a`.
    fn mul(self, o: Rotation2) -> Rotation2 {
        Rotation2 { angle: self.angle + o.angle }
    }
}

/// A rotation in three dimensions, stored as a unit quaternion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotation3 {
    q: Quaternion,
}

impl Rotation3 {
    pub fn identity() -> Self {
        Rotation3 { q: Quaternion::identity() }
    }
    /// Wraps a quaternion, normalising it.
    pub fn from_quaternion(q: Quaternion) -> Self {
        Rotation3 { q: q.normalized() }
    }
    pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> Self {
        Rotation3 { q: Quaternion::from_axis_angle(axis, angle) }
    }
    /// From a rotation matrix; use [`nearest_rotation_matrix`] first if it may be noisy.
    pub fn from_matrix(m: &Matrix3) -> Self {
        Rotation3 { q: Quaternion::from_rotation_matrix(m).normalized() }
    }
    pub fn from_euler(sequence: EulerSequence, angles: [f64; 3]) -> Self {
        Self::from_matrix(&sequence.to_matrix(angles))
    }
    /// The shortest rotation taking direction `from` to direction `to`.
    pub fn from_two_vectors(from: [f64; 3], to: [f64; 3]) -> Self {
        Rotation3 { q: Quaternion::from_two_vectors(from, to) }
    }
    pub fn quaternion(&self) -> Quaternion {
        self.q
    }
    pub fn matrix(&self) -> Matrix3 {
        self.q.to_rotation_matrix()
    }
    /// Unit axis and angle in `[0, π]`.
    pub fn axis_angle(&self) -> ([f64; 3], f64) {
        self.q.to_axis_angle()
    }
    pub fn to_euler(&self, sequence: EulerSequence) -> Option<[f64; 3]> {
        sequence.from_matrix(&self.matrix())
    }
    pub fn inverse(&self) -> Self {
        Rotation3 { q: self.q.conjugate() }
    }
    /// Applies the rotation to a vector.
    pub fn rotate(&self, v: [f64; 3]) -> [f64; 3] {
        self.q.rotate_vector(v)
    }
    /// Rotates a point about `pivot` instead of the origin.
    pub fn rotate_about(&self, p: [f64; 3], pivot: [f64; 3]) -> [f64; 3] {
        let r = self.rotate([p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2]]);
        [r[0] + pivot[0], r[1] + pivot[1], r[2] + pivot[2]]
    }
    /// The angle of the rotation that takes `self` to `other`.
    pub fn angle_to(&self, other: &Rotation3) -> f64 {
        self.q.angle_to(&other.q)
    }
    pub fn slerp(&self, other: &Rotation3, t: f64) -> Self {
        Rotation3 { q: self.q.slerp(&other.q, t) }
    }
}

impl Mul for Rotation3 {
    type Output = Rotation3;
    /// `a * b` applies `b` first, then `a`.
    fn mul(self, o: Rotation3) -> Rotation3 {
        Rotation3 { q: (self.q * o.q).normalized() }
    }
}

impl From<Quaternion> for Rotation3 {
    fn from(q: Quaternion) -> Self {
        Rotation3::from_quaternion(q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    fn close(a: &Matrix3, b: &Matrix3, tol: f64) -> bool {
        (0..3).all(|i| (0..3).all(|j| (a[i][j] - b[i][j]).abs() < tol))
    }

    #[test]
    fn axis_matrices_and_rodrigues_agree() {
        assert!(close(&rotation_matrix_axis_angle([1.0, 0.0, 0.0], 0.7), &rotation_matrix_x(0.7), 1e-14));
        assert!(close(&rotation_matrix_axis_angle([0.0, 2.0, 0.0], -1.3), &rotation_matrix_y(-1.3), 1e-14));
        assert!(close(&rotation_matrix_axis_angle([0.0, 0.0, 5.0], 2.9), &rotation_matrix_z(2.9), 1e-14));
        assert!(is_rotation_matrix(&rotation_matrix_axis_angle([1.0, 2.0, 3.0], 1.1), 1e-12));
        assert_eq!(rotation_matrix_axis_angle([0.0; 3], 1.0), IDENTITY3);
        let q = Quaternion::from_axis_angle([1.0, 2.0, 3.0], 1.1);
        assert!(close(&q.to_rotation_matrix(), &rotation_matrix_axis_angle([1.0, 2.0, 3.0], 1.1), 1e-12));
    }

    #[test]
    fn every_euler_sequence_round_trips_through_the_matrix() {
        let seqs = [
            EulerSequence::XYZ, EulerSequence::XZY, EulerSequence::YXZ, EulerSequence::YZX, EulerSequence::ZXY, EulerSequence::ZYX,
            EulerSequence::XYX, EulerSequence::XZX, EulerSequence::YXY, EulerSequence::YZY, EulerSequence::ZXZ, EulerSequence::ZYZ,
        ];
        for seq in seqs {
            for angles in [[0.3, 0.5, -1.2], [-2.0, 1.2, 2.5], [3.0, -0.7, 0.1], [0.0, 0.0, 0.0], [1.0, 2.5, -3.0]] {
                let m = seq.to_matrix(angles);
                let back = seq.from_matrix(&m).unwrap();
                assert!(close(&seq.to_matrix(back), &m, 1e-9), "{seq:?} {angles:?} -> {back:?}");
            }
        }
        assert!(EulerSequence([Axis::X, Axis::X, Axis::Z]).from_matrix(&IDENTITY3).is_none());
    }

    #[test]
    fn euler_gimbal_lock_still_reproduces_the_matrix() {
        let seq = EulerSequence::ZYX;
        for pitch in [FRAC_PI_2, -FRAC_PI_2] {
            let m = seq.to_matrix([0.4, pitch, 0.9]);
            assert!(close(&seq.to_matrix(seq.from_matrix(&m).unwrap()), &m, 1e-9));
        }
        let m = EulerSequence::ZXZ.to_matrix([0.4, 0.0, 0.9]); // proper-Euler singularity at beta = 0
        assert!(close(&EulerSequence::ZXZ.to_matrix(EulerSequence::ZXZ.from_matrix(&m).unwrap()), &m, 1e-9));
    }

    #[test]
    fn rotation3_composes_inverts_and_converts() {
        let a = Rotation3::from_axis_angle([1.0, 0.0, 0.0], 0.8);
        let b = Rotation3::from_euler(EulerSequence::ZYX, [0.3, -0.4, 1.0]);
        let v = [1.0, -2.0, 0.5];
        let ab = a * b;
        let want = a.rotate(b.rotate(v));
        let got = ab.rotate(v);
        assert!((0..3).all(|i| (want[i] - got[i]).abs() < 1e-12));
        assert!((ab * ab.inverse()).angle_to(&Rotation3::identity()) < 1e-12);
        assert!(close(&Rotation3::from_matrix(&b.matrix()).matrix(), &b.matrix(), 1e-12));
        let e = b.to_euler(EulerSequence::ZYX).unwrap();
        assert!(Rotation3::from_euler(EulerSequence::ZYX, e).angle_to(&b) < 1e-9);
        let pivoted = Rotation3::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2).rotate_about([2.0, 1.0, 0.0], [1.0, 1.0, 0.0]);
        assert!((pivoted[0] - 1.0).abs() < 1e-12 && (pivoted[1] - 2.0).abs() < 1e-12);
        assert!(Rotation3::identity().slerp(&a, 0.5).angle_to(&Rotation3::from_axis_angle([1.0, 0.0, 0.0], 0.4)) < 1e-12);
    }

    #[test]
    fn rotation2_and_matrix_cleanup() {
        let r = Rotation2::new(FRAC_PI_2);
        let p = r.rotate([1.0, 0.0]);
        assert!(p[0].abs() < 1e-15 && (p[1] - 1.0).abs() < 1e-15);
        assert_eq!(r.matrix()[1][0], 1.0);
        let about = r.rotate_about([2.0, 1.0], [1.0, 1.0]);
        assert!((about[0] - 1.0).abs() < 1e-12 && (about[1] - 2.0).abs() < 1e-12);
        assert!((Rotation2::from_two_vectors([1.0, 0.0], [0.0, -3.0]).angle + FRAC_PI_2).abs() < 1e-12);
        assert!((Rotation2::new(7.0).normalized().angle - (7.0 - 2.0 * PI)).abs() < 1e-12);
        assert!(((r * r.inverse()).angle).abs() < 1e-15);

        let mut m = rotation_matrix_axis_angle([1.0, 1.0, 0.0], 0.6);
        m[0][0] += 1e-3;
        m[1][2] -= 2e-3;
        assert!(!is_rotation_matrix(&m, 1e-6));
        let fixed = nearest_rotation_matrix(&m).unwrap();
        assert!(is_rotation_matrix(&fixed, 1e-12));
        assert!(nearest_rotation_matrix(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]).is_none());
    }
}
