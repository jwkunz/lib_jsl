//! Zero-dimensional geometry traits.

use crate::geometry::common::{
    CanScale, CanScaleNonUniform, CoordinatePrimitive, ScalarOperable, SelfAddition,
};

/// A point primitive supporting the core arithmetic traits.
pub trait IsPoint: CoordinatePrimitive + ScalarOperable + SelfAddition {}

/// A stronger convenience trait for point types that support the library's common transforms.
pub trait TransformablePoint:
    IsPoint
    + CanScale
    + CanScaleNonUniform
    + crate::geometry::transformation_traits::CanTranslate<Point = Self>
    + crate::geometry::transformation_traits::CanRotate<Point = Self>
    + crate::geometry::transformation_traits::CanMirror<Point = Self>
{
}

impl<T> TransformablePoint for T
where
    T: IsPoint
        + CanScale
        + CanScaleNonUniform
        + crate::geometry::transformation_traits::CanTranslate<Point = T>
        + crate::geometry::transformation_traits::CanRotate<Point = T>
        + crate::geometry::transformation_traits::CanMirror<Point = T>,
{
}
