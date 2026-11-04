//! Traits describing in-place geometric transformations.

use crate::common::{GeometricPrimitive, GeometryMeasure, IsPlane, IsUnitVector};
use crate::one_d::IsLine;
use crate::zero_d::IsPoint;

/// Translates a primitive along a line-defined direction.
pub trait CanTranslate: GeometricPrimitive + Sized {
    /// Point type used by the translation line.
    type Point: IsPoint;

    /// Mutates `self` by applying the translation vector.
    fn translate<'a, L>(&mut self, translation_vector: &L)
    where
        L: IsLine<'a, Self::Point>;
}

/// Rotates a primitive around a line by an angle in radians.
pub trait CanRotate: GeometricPrimitive + Sized {
    /// Point type used by the rotation axis.
    type Point: IsPoint;

    /// Mutates `self` by rotating about `axis`.
    fn rotate<'a, L>(&mut self, axis: &L, angle_radians: GeometryMeasure)
    where
        L: IsLine<'a, Self::Point>;
}

/// Shears a primitive parallel to the x axis: every vertex moves by `x ← x + k·y`, where the shear
/// factor `k` is the length of the supplied line. The line's position and direction are ignored.
pub trait CanShear: GeometricPrimitive + Sized {
    /// Point type used by the shear line.
    type Point: IsPoint;

    /// Mutates `self` by the shear `x ← x + k·y`, with `k = shear_line.length()`.
    fn shear<'a, L>(&mut self, shear_line: &L)
    where
        L: IsLine<'a, Self::Point>;
}

/// Mirrors a primitive across a compatible plane.
pub trait CanMirror: GeometricPrimitive + Sized {
    /// Point type used by the mirror plane.
    type Point: IsPoint;
    /// Unit normal type used by the mirror plane.
    type Normal: IsUnitVector;

    /// Reflects `self` across `mirror_plane`.
    fn mirror<P>(&mut self, mirror_plane: &P)
    where
        P: IsPlane<Point = Self::Point, Normal = Self::Normal>;
}
