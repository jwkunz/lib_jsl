//! Internal transformation helpers shared by concrete two-dimensional primitives.

use crate::geometry::common::GeometryMeasure;
use crate::geometry::two_d::{Point2D, UnitVector2D};

/// Rotates a point around an anchor in 2D space.
pub(crate) fn rotate_point_around_anchor_2d(
    point: Point2D,
    origin: Point2D,
    angle: GeometryMeasure,
) -> Point2D {
    let translated = point - origin;
    let translated = translated.cartesian_components();
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    let rotated = Point2D::new(
        translated[0] * cos_theta - translated[1] * sin_theta,
        translated[0] * sin_theta + translated[1] * cos_theta,
    );
    origin + rotated
}

/// Reflects a point across a point-normal line in 2D.
pub(crate) fn reflect_point_across_plane_2d(
    point: Point2D,
    plane_point: Point2D,
    normal: UnitVector2D,
) -> Point2D {
    let offset = point - plane_point;
    let offset = offset.cartesian_components();
    let distance = offset[0] * normal.x() + offset[1] * normal.y();
    point - Point2D::new(normal.x(), normal.y()) * (2.0 * distance)
}
