//! Zero-dimensional geometry traits.

use crate::geometry::common::{
    CanScale, CanScaleNonUniform, CoordinatePrimitive, ScalarOperable, SelfAddition,
};
use crate::geometry::transformation_traits::{CanMirror, CanRotate, CanTranslate};

/// A point primitive supporting the core arithmetic and transformation traits.
pub trait IsPoint:
    CoordinatePrimitive
    + ScalarOperable
    + SelfAddition
    + CanTranslate<Point = Self>
    + CanScale
    + CanScaleNonUniform
    + CanRotate<Point = Self>
    + CanMirror<Point = Self>
{
}
