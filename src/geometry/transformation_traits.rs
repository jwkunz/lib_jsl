//! Traits describing in-place geometric transformations.

use crate::geometry::common::{GeometricPrimitive, GeometryMeasure, IsUnitVector};
use crate::geometry::one_d::IsLine;
use crate::geometry::three_d::IsPlane;
use crate::geometry::zero_d::IsPoint;

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

/// Shears a primitive using a line-defined shear direction.
pub trait CanShear: GeometricPrimitive + Sized {
    /// Point type used by the shear line.
    type Point: IsPoint;

    /// Mutates `self` according to the supplied shear line.
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
