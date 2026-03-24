//! Internal 3D transformation helpers shared by concrete primitives.

use crate::geometry::common::{GeometryMeasure, IsUnitVector};
use crate::geometry::three_d::{Point3D, UnitVector3D};

/// Rotates a point around an axis using Rodrigues' rotation formula.
pub(crate) fn rotate_point_around_axis(
    point: Point3D,
    origin: Point3D,
    axis: &impl IsUnitVector,
    angle: GeometryMeasure,
) -> Point3D {
    let axis = Point3D::new(axis[0], axis[1], axis[2]);
    let translated = point - origin;
    let translated = translated.cartesian_components();
    let translated = Point3D::new(translated[0], translated[1], translated[2]);
    let axis_components = axis.cartesian_components();
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    let cross = Point3D::new(
        axis_components[1] * translated.z() - axis_components[2] * translated.y(),
        axis_components[2] * translated.x() - axis_components[0] * translated.z(),
        axis_components[0] * translated.y() - axis_components[1] * translated.x(),
    );
    let dot = axis_components[0] * translated.x()
        + axis_components[1] * translated.y()
        + axis_components[2] * translated.z();
    let rotated = translated * cos_theta + cross * sin_theta + axis * (dot * (1.0 - cos_theta));
    origin + rotated
}

/// Reflects a point across a plane defined by a point and unit normal.
pub(crate) fn reflect_point_across_plane(
    point: Point3D,
    plane_point: Point3D,
    normal: UnitVector3D,
) -> Point3D {
    let offset = point - plane_point;
    let normal_point = normal.as_point();
    let distance = offset * normal_point;
    point - normal_point * (2.0 * distance)
}
