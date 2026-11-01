//! Unit quaternions for representing and composing 3D rotations.
//!
//! A [`Quaternion`] `w + xi + yj + zk` with unit norm encodes a rotation by an angle θ about a unit
//! axis `n` as `cos(θ/2) + sin(θ/2) (n_x i + n_y j + n_z k)`. Composition is the Hamilton product,
//! and `q` and `-q` describe the same rotation. Arithmetic is in `f64`.

use crate::three_d::{CoordinateVector3D, FreeVector3D};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A quaternion `w + xi + yj + zk` (scalar part first).
#[derive(Debug, Clone, Copy, PartialEq)]

pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Quaternion { w, x, y, z }
    }

    /// The multiplicative identity, i.e. no rotation.
    pub const fn identity() -> Self {
        Quaternion::new(1.0, 0.0, 0.0, 0.0)
    }

    /// A pure quaternion (zero scalar part) holding a vector.
    pub const fn from_vector(v: [f64; 3]) -> Self {
        Quaternion::new(0.0, v[0], v[1], v[2])
    }

    /// The vector (imaginary) part.
    pub const fn vector_part(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Rotation by `angle` radians about `axis` (which need not be normalised; a zero axis gives the
    /// identity).
    pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> Self {
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if n == 0.0 {
            return Self::identity();
        }
        let (s, c) = (0.5 * angle).sin_cos();
        Quaternion::new(c, s * axis[0] / n, s * axis[1] / n, s * axis[2] / n)
    }

    /// The axis and angle in `[0, π]` of a (unit) rotation quaternion; the axis is `[1, 0, 0]` for
    /// the identity, where it is arbitrary.
    pub fn to_axis_angle(&self) -> ([f64; 3], f64) {
        let q = if self.w < 0.0 { -*self } else { *self }.normalized();
        let s = (q.x * q.x + q.y * q.y + q.z * q.z).sqrt();
        if s < 1e-15 {
            return ([1.0, 0.0, 0.0], 0.0);
        }
        ([q.x / s, q.y / s, q.z / s], 2.0 * s.atan2(q.w))
    }

    /// Rotation from intrinsic Z-Y-X (yaw, pitch, roll) Euler angles in radians: first yaw about
    /// Z, then pitch about the new Y, then roll about the new X.
    pub fn from_euler(roll: f64, pitch: f64, yaw: f64) -> Self {
        let (sr, cr) = (0.5 * roll).sin_cos();
        let (sp, cp) = (0.5 * pitch).sin_cos();
        let (sy, cy) = (0.5 * yaw).sin_cos();
        Quaternion::new(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    }

    /// `(roll, pitch, yaw)` for the Z-Y-X convention of [`Quaternion::from_euler`]; pitch is in
    /// `[-π/2, π/2]` (at the ±90° gimbal lock yaw is taken as zero).
    pub fn to_euler(&self) -> (f64, f64, f64) {
        let q = self.normalized();
        let sinp = 2.0 * (q.w * q.y - q.z * q.x);
        if sinp.abs() >= 1.0 - 1e-12 {
            let pitch = std::f64::consts::FRAC_PI_2.copysign(sinp);
            let roll = 2.0 * q.x.atan2(q.w) * sinp.signum();
            return (roll, pitch, 0.0);
        }
        let roll = (2.0 * (q.w * q.x + q.y * q.z)).atan2(1.0 - 2.0 * (q.x * q.x + q.y * q.y));
        let yaw = (2.0 * (q.w * q.z + q.x * q.y)).atan2(1.0 - 2.0 * (q.y * q.y + q.z * q.z));
        (roll, sinp.asin(), yaw)
    }

    /// The 3×3 rotation matrix (row-major) of the normalised quaternion.
    pub fn to_rotation_matrix(&self) -> [[f64; 3]; 3] {
        let q = self.normalized();
        let (w, x, y, z) = (q.w, q.x, q.y, q.z);
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    }

    /// The quaternion (with `w >= 0`) of a proper rotation matrix, by Shepperd's method, which is
    /// numerically stable for every rotation.
    pub fn from_rotation_matrix(m: &[[f64; 3]; 3]) -> Self {
        let trace = m[0][0] + m[1][1] + m[2][2];
        let q = if trace > 0.0 {
            let s = 2.0 * (trace + 1.0).sqrt();
            Quaternion::new(0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
        } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
            let s = 2.0 * (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt();
            Quaternion::new((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
        } else if m[1][1] > m[2][2] {
            let s = 2.0 * (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt();
            Quaternion::new((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
        } else {
            let s = 2.0 * (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt();
            Quaternion::new((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)
        };
        if q.w < 0.0 { -q } else { q }
    }

    /// The shortest rotation taking direction `from` onto direction `to` (neither need be unit).
    /// Returns the identity if either is zero.
    pub fn from_two_vectors(from: [f64; 3], to: [f64; 3]) -> Self {
        let (nf, nt) = (norm3(from), norm3(to));
        if nf == 0.0 || nt == 0.0 {
            return Self::identity();
        }
        let (a, b) = (scale3(from, 1.0 / nf), scale3(to, 1.0 / nt));
        let d = dot3(a, b);
        if d < -1.0 + 1e-12 {
            // Opposite directions: rotate half a turn about any axis perpendicular to `a`.
            let helper = if a[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
            return Self::from_axis_angle(cross3(a, helper), std::f64::consts::PI);
        }
        let c = cross3(a, b);
        Quaternion::new(1.0 + d, c[0], c[1], c[2]).normalized()
    }

    pub fn norm_squared(&self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Scaled to unit length; the identity if the norm is zero.
    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n == 0.0 { Self::identity() } else { *self / n }
    }

    pub fn conjugate(&self) -> Self {
        Quaternion::new(self.w, -self.x, -self.y, -self.z)
    }

    /// The multiplicative inverse `q* / |q|²`; `None` for the zero quaternion.
    pub fn inverse(&self) -> Option<Self> {
        let n2 = self.norm_squared();
        (n2 > 0.0).then(|| self.conjugate() / n2)
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Rotates `v` by this (normalised) quaternion: `q v q*`.
    pub fn rotate_vector(&self, v: [f64; 3]) -> [f64; 3] {
        let q = self.normalized();
        let u = [q.x, q.y, q.z];
        // v' = v + 2 w (u × v) + 2 u × (u × v)
        let t = scale3(cross3(u, v), 2.0);
        let c = cross3(u, t);
        [v[0] + q.w * t[0] + c[0], v[1] + q.w * t[1] + c[1], v[2] + q.w * t[2] + c[2]]
    }

    /// Rotates a [`FreeVector3D`] (computed in `f64`, stored back as `f32`).
    pub fn rotate_free_vector(&self, v: &FreeVector3D) -> FreeVector3D {
        let r = self.rotate_vector(v.cartesian_components().map(f64::from));
        FreeVector3D::new(r[0] as f32, r[1] as f32, r[2] as f32)
    }

    /// Rotates a point about the origin.
    pub fn rotate_point(&self, p: &CoordinateVector3D) -> CoordinateVector3D {
        let r = self.rotate_vector([p.x() as f64, p.y() as f64, p.z() as f64]);
        CoordinateVector3D::new(r[0] as f32, r[1] as f32, r[2] as f32)
    }

    /// The quaternion exponential.
    pub fn exp(&self) -> Self {
        let v = norm3(self.vector_part());
        let e = self.w.exp();
        if v < 1e-300 {
            return Quaternion::new(e, 0.0, 0.0, 0.0);
        }
        let s = e * v.sin() / v;
        Quaternion::new(e * v.cos(), s * self.x, s * self.y, s * self.z)
    }

    /// The principal logarithm (zero quaternion has no logarithm: returns `-∞` scalar).
    pub fn ln(&self) -> Self {
        let n = self.norm();
        let v = norm3(self.vector_part());
        if v < 1e-300 {
            return Quaternion::new(n.ln(), 0.0, 0.0, 0.0);
        }
        let k = (self.w / n).acos() / v;
        Quaternion::new(n.ln(), k * self.x, k * self.y, k * self.z)
    }

    /// Real power `q^t` through `exp(t ln q)`; for a unit rotation this scales the angle by `t`.
    pub fn powf(&self, t: f64) -> Self {
        (self.ln() * t).exp()
    }

    /// Spherical linear interpolation along the shortest arc from `self` (t = 0) to `other` (t = 1),
    /// for unit quaternions. Constant angular velocity.
    pub fn slerp(&self, other: &Self, t: f64) -> Self {
        let mut b = *other;
        let mut d = self.dot(&b);
        if d < 0.0 {
            b = -b;
            d = -d;
        }
        if d > 1.0 - 1e-10 {
            return self.nlerp(&b, t);
        }
        let theta = d.acos();
        let s = theta.sin();
        (*self * (((1.0 - t) * theta).sin() / s) + b * ((t * theta).sin() / s)).normalized()
    }

    /// Normalised linear interpolation: cheaper than [`Quaternion::slerp`], not constant-speed.
    pub fn nlerp(&self, other: &Self, t: f64) -> Self {
        let b = if self.dot(other) < 0.0 { -*other } else { *other };
        (*self * (1.0 - t) + b * t).normalized()
    }

    /// The angle in radians (0 to π) of the rotation taking `self` to `other`.
    pub fn angle_to(&self, other: &Self) -> f64 {
        2.0 * self.normalized().dot(&other.normalized()).abs().min(1.0).acos()
    }

    /// Whether the two quaternions are within `tol` componentwise (not modulo sign).
    pub fn approx_eq(&self, other: &Self, tol: f64) -> bool {
        (self.w - other.w).abs() <= tol && (self.x - other.x).abs() <= tol && (self.y - other.y).abs() <= tol && (self.z - other.z).abs() <= tol
    }
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

impl Default for Quaternion {
    fn default() -> Self {
        Self::identity()
    }
}

impl Add for Quaternion {
    type Output = Quaternion;
    fn add(self, o: Quaternion) -> Quaternion {
        Quaternion::new(self.w + o.w, self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl Sub for Quaternion {
    type Output = Quaternion;
    fn sub(self, o: Quaternion) -> Quaternion {
        Quaternion::new(self.w - o.w, self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Neg for Quaternion {
    type Output = Quaternion;
    fn neg(self) -> Quaternion {
        Quaternion::new(-self.w, -self.x, -self.y, -self.z)
    }
}
impl Mul<f64> for Quaternion {
    type Output = Quaternion;
    fn mul(self, s: f64) -> Quaternion {
        Quaternion::new(self.w * s, self.x * s, self.y * s, self.z * s)
    }
}
impl Div<f64> for Quaternion {
    type Output = Quaternion;
    fn div(self, s: f64) -> Quaternion {
        Quaternion::new(self.w / s, self.x / s, self.y / s, self.z / s)
    }
}
/// The Hamilton product: `a * b` applies `b` first, then `a`, when used as rotations.
impl Mul for Quaternion {
    type Output = Quaternion;
    fn mul(self, o: Quaternion) -> Quaternion {
        Quaternion::new(
            self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        )
    }
}

impl From<FreeVector3D> for Quaternion {
    /// The pure quaternion of a vector.
    fn from(v: FreeVector3D) -> Self {
        Quaternion::from_vector(v.cartesian_components().map(f64::from))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, PI};

    fn close3(a: [f64; 3], b: [f64; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-12)
    }

    #[test]
    fn axis_angle_rotates_vectors() {
        let q = Quaternion::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
        assert!(close3(q.rotate_vector([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
        let q = Quaternion::from_axis_angle([1.0, 1.0, 1.0], 2.0 * FRAC_PI_3); // cycles the axes
        assert!(close3(q.rotate_vector([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
        let (axis, angle) = q.to_axis_angle();
        assert!(close3(axis, [1.0 / 3f64.sqrt(); 3]) && (angle - 2.0 * FRAC_PI_3).abs() < 1e-12);
        assert_eq!(Quaternion::from_axis_angle([0.0; 3], 1.0), Quaternion::identity());
    }

    #[test]
    fn hamilton_product_composes_rotations() {
        let a = Quaternion::from_axis_angle([1.0, 0.0, 0.0], 0.7);
        let b = Quaternion::from_axis_angle([0.0, 1.0, 0.0], -1.1);
        let v = [0.3, -0.5, 2.0];
        assert!(close3((a * b).rotate_vector(v), a.rotate_vector(b.rotate_vector(v))));
        let inv = a.inverse().unwrap();
        assert!((a * inv).approx_eq(&Quaternion::identity(), 1e-14));
        assert!(Quaternion::new(0.0, 0.0, 0.0, 0.0).inverse().is_none());
        // i j = k
        assert_eq!(Quaternion::new(0.0, 1.0, 0.0, 0.0) * Quaternion::new(0.0, 0.0, 1.0, 0.0), Quaternion::new(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn rotation_matrix_round_trip_for_all_branches() {
        for (axis, angle) in [([1.0, 2.0, 3.0], 0.4), ([1.0, 0.0, 0.0], 3.0), ([0.0, 1.0, 0.0], 3.0), ([0.0, 0.0, 1.0], 3.1), ([1.0, -1.0, 2.0], PI)] {
            let q = Quaternion::from_axis_angle(axis, angle);
            let m = q.to_rotation_matrix();
            let back = Quaternion::from_rotation_matrix(&m);
            assert!(q.angle_to(&back) < 1e-7, "{axis:?} {angle}");
            let v = [0.2, 0.9, -0.4];
            let mv = [dot3(m[0], v), dot3(m[1], v), dot3(m[2], v)];
            assert!(close3(mv, q.rotate_vector(v)));
        }
    }

    #[test]
    fn euler_round_trip_and_gimbal_lock() {
        let (r, p, y) = (0.3, -0.8, 2.0);
        let (r2, p2, y2) = Quaternion::from_euler(r, p, y).to_euler();
        assert!((r - r2).abs() < 1e-12 && (p - p2).abs() < 1e-12 && (y - y2).abs() < 1e-12);
        // Pure yaw of 90 degrees maps x to y.
        assert!(close3(Quaternion::from_euler(0.0, 0.0, FRAC_PI_2).rotate_vector([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
        let q = Quaternion::from_euler(0.4, FRAC_PI_2, 0.0);
        let (_, pitch, _) = q.to_euler();
        assert!((pitch - FRAC_PI_2).abs() < 1e-9);
    }

    #[test]
    fn slerp_has_constant_angular_velocity() {
        let a = Quaternion::identity();
        let b = Quaternion::from_axis_angle([0.0, 0.0, 1.0], 1.2);
        for k in 0..=10 {
            let t = k as f64 / 10.0;
            let (_, angle) = a.slerp(&b, t).to_axis_angle();
            assert!((angle - 1.2 * t).abs() < 1e-12, "t={t}");
        }
        // Takes the short way round even if the operands have opposite sign.
        assert!(a.slerp(&-b, 0.5).angle_to(&Quaternion::from_axis_angle([0.0, 0.0, 1.0], 0.6)) < 1e-12);
        assert!(a.nlerp(&b, 0.5).angle_to(&a.slerp(&b, 0.5)) < 1e-3);
    }

    #[test]
    fn exp_ln_powf_and_two_vector_rotation() {
        let q = Quaternion::from_axis_angle([0.0, 1.0, 0.0], 0.9);
        assert!(q.ln().exp().approx_eq(&q, 1e-12));
        assert!(q.powf(2.0).angle_to(&Quaternion::from_axis_angle([0.0, 1.0, 0.0], 1.8)) < 1e-12);
        let r = Quaternion::from_two_vectors([1.0, 0.0, 0.0], [0.0, 2.0, 0.0]);
        assert!(close3(r.rotate_vector([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
        let flip = Quaternion::from_two_vectors([0.0, 0.0, 1.0], [0.0, 0.0, -3.0]);
        assert!(close3(flip.rotate_vector([0.0, 0.0, 1.0]), [0.0, 0.0, -1.0]));
        assert_eq!(Quaternion::from_two_vectors([0.0; 3], [1.0, 0.0, 0.0]), Quaternion::identity());
    }

    #[test]
    fn works_with_crate_vector_types() {
        let q = Quaternion::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
        let v = q.rotate_free_vector(&FreeVector3D::new(1.0, 0.0, 0.0));
        assert!((v.x()).abs() < 1e-6 && (v.y() - 1.0).abs() < 1e-6);
        let p = q.rotate_point(&CoordinateVector3D::new(0.0, 2.0, 5.0));
        assert!((p.x() + 2.0).abs() < 1e-6 && (p.z() - 5.0).abs() < 1e-6);
        let pure: Quaternion = FreeVector3D::new(1.0, 2.0, 3.0).into();
        assert_eq!(pure, Quaternion::new(0.0, 1.0, 2.0, 3.0));
    }
}
