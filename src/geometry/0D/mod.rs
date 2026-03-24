//! Zero-dimensional geometry traits.

/// Concrete three-dimensional point implementation.
pub mod point3d;

use crate::geometry::common::{
    CanScale, CanScaleNonUniform, CoordinatePrimitive, ScalarOperable, SelfAddition,
};
use crate::geometry::transformation_traits::{CanMirror, CanRotate, CanTranslate};
pub use point3d::Point3D;

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
