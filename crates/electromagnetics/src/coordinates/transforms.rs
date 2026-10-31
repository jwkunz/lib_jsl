//! Rigid transforms of positions and field vectors: rotation followed by translation.

use super::systems::Position;
use crate::conventions::C64;
use lib_jsl_geometry::algorithms::rotations::Rotation3;

/// `x ↦ R x + t`. Positions are rotated and translated; field vectors (directions, polarizations)
/// are only rotated. Complex vectors are rotated component-wise in their real and imaginary parts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidTransform {
    pub rotation: Rotation3,
    pub translation: [f64; 3],
}

impl Default for RigidTransform {
    fn default() -> Self {
        Self::identity()
    }
}

impl RigidTransform {
    pub fn identity() -> Self {
        RigidTransform { rotation: Rotation3::identity(), translation: [0.0; 3] }
    }

    pub fn translation(t: [f64; 3]) -> Self {
        RigidTransform { rotation: Rotation3::identity(), translation: t }
    }

    pub fn rotation(r: Rotation3) -> Self {
        RigidTransform { rotation: r, translation: [0.0; 3] }
    }

    /// Rotation about `axis` by `angle` radians (through the origin).
    pub fn rotation_about(axis: [f64; 3], angle: f64) -> Self {
        Self::rotation(Rotation3::from_axis_angle(axis, angle))
    }

    /// Applies the transform to a point given in Cartesian coordinates.
    pub fn apply_point(&self, p: [f64; 3]) -> [f64; 3] {
        let r = self.rotation.rotate(p);
        [r[0] + self.translation[0], r[1] + self.translation[1], r[2] + self.translation[2]]
    }

    /// Applies the transform to a [`Position`], returning it in the same coordinate system.
    pub fn apply_position(&self, p: &Position) -> Position {
        let c = self.apply_point(p.to_cartesian());
        Position::cartesian(c[0], c[1], c[2]).in_system(p.system)
    }

    /// Rotates a real vector given in Cartesian components (translation does not apply).
    pub fn apply_vector(&self, v: [f64; 3]) -> [f64; 3] {
        self.rotation.rotate(v)
    }

    /// Rotates a complex vector given in Cartesian components.
    pub fn apply_complex_vector(&self, v: [C64; 3]) -> [C64; 3] {
        let re = self.rotation.rotate(v.map(|c| c.re));
        let im = self.rotation.rotate(v.map(|c| c.im));
        std::array::from_fn(|k| C64::new(re[k], im[k]))
    }

    /// `self ∘ other`: apply `other` first, then `self`.
    pub fn then_after(&self, other: &RigidTransform) -> RigidTransform {
        RigidTransform { rotation: self.rotation * other.rotation, translation: self.apply_point(other.translation) }
    }

    /// The inverse transform.
    pub fn inverse(&self) -> RigidTransform {
        let inv = self.rotation.inverse();
        let t = inv.rotate(self.translation);
        RigidTransform { rotation: inv, translation: [-t[0], -t[1], -t[2]] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    fn close(a: [f64; 3], b: [f64; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-12)
    }

    #[test]
    fn points_move_vectors_only_rotate() {
        let t = RigidTransform { rotation: Rotation3::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2), translation: [1.0, 2.0, 3.0] };
        assert!(close(t.apply_point([1.0, 0.0, 0.0]), [1.0, 3.0, 3.0]));
        assert!(close(t.apply_vector([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
        let cv = t.apply_complex_vector([C64::new(1.0, 2.0), C64::new(0.0, 0.0), C64::new(0.0, -1.0)]);
        assert!((cv[1] - C64::new(1.0, 2.0)).norm() < 1e-12 && (cv[2] - C64::new(0.0, -1.0)).norm() < 1e-12);
        // Positions keep their coordinate system.
        let p = t.apply_position(&Position::spherical(1.0, FRAC_PI_2, 0.0));
        assert_eq!(p.system, crate::coordinates::systems::CoordinateSystem::Spherical);
        assert!(close(p.to_cartesian(), [1.0, 3.0, 3.0]));
    }

    #[test]
    fn composition_and_inverse() {
        let a = RigidTransform { rotation: Rotation3::from_axis_angle([1.0, 1.0, 0.0], 0.7), translation: [0.5, -1.0, 2.0] };
        let b = RigidTransform { rotation: Rotation3::from_axis_angle([0.0, 1.0, 1.0], -1.2), translation: [3.0, 0.0, -0.5] };
        let p = [0.3, -0.8, 1.5];
        assert!(close(a.then_after(&b).apply_point(p), a.apply_point(b.apply_point(p))));
        assert!(close(a.inverse().apply_point(a.apply_point(p)), p));
        assert!(close(a.then_after(&a.inverse()).apply_point(p), p));
        assert!(close(RigidTransform::identity().apply_point(p), p));
        assert!(close(RigidTransform::translation([1.0, 1.0, 1.0]).apply_vector(p), p));
        let r = RigidTransform::rotation_about([0.0, 0.0, 1.0], 0.3);
        let v = r.apply_vector(p);
        assert!(((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() - (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()).abs() < 1e-12);
    }
}
