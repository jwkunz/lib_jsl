//! Zero-dimensional geometry traits.

use crate::common::{
    CanScale, CanScaleNonUniform, CoordinatePrimitive, ScalarOperable, SelfAddition,
};

/// A point primitive supporting the core arithmetic traits.
pub trait IsPoint: CoordinatePrimitive + ScalarOperable + SelfAddition {}

/// A stronger convenience trait for point types that support the library's common transforms.
pub trait TransformablePoint:
    IsPoint
    + CanScale
    + CanScaleNonUniform
    + crate::transformation_traits::CanTranslate<Point = Self>
    + crate::transformation_traits::CanRotate<Point = Self>
    + crate::transformation_traits::CanMirror<Point = Self>
{
}

impl<T> TransformablePoint for T
where
    T: IsPoint
        + CanScale
        + CanScaleNonUniform
        + crate::transformation_traits::CanTranslate<Point = T>
        + crate::transformation_traits::CanRotate<Point = T>
        + crate::transformation_traits::CanMirror<Point = T>,
{
}
