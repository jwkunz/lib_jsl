//! Internal 3D transformation helpers shared by concrete primitives.

use crate::common::{GeometryMeasure, IsUnitVector};
use crate::three_d::{FreeVector3D, CoordinateVector3D, UnitVector3D};

/// Rotates a point around an axis using Rodrigues' rotation formula.
pub(crate) fn rotate_point_around_axis(
    point: CoordinateVector3D,
    origin: CoordinateVector3D,
    axis: &impl IsUnitVector,
    angle: GeometryMeasure,
) -> CoordinateVector3D {
    let axis = FreeVector3D::new(axis[0], axis[1], axis[2]);
    let translated = FreeVector3D::new(point.x() - origin.x(), point.y() - origin.y(), point.z() - origin.z());
    let axis_components = axis.cartesian_components();
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    let cross = FreeVector3D::new(
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
    point: CoordinateVector3D,
    plane_point: CoordinateVector3D,
    normal: UnitVector3D,
) -> CoordinateVector3D {
    let offset = FreeVector3D::new(
        point.x() - plane_point.x(),
        point.y() - plane_point.y(),
        point.z() - plane_point.z(),
    );
    let normal_vector = normal.as_vector();
    let distance = offset * normal_vector;
    point - normal_vector * (2.0 * distance)
}
