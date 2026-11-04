//! Rotation matrices, Euler angles, quaternions and interpolation.

use lib_jsl_geometry::algorithms::rotations::{is_rotation_matrix, mat3_apply};
use lib_jsl_geometry::algorithms::{EulerSequence, Quaternion, Rotation3};
use std::f64::consts::FRAC_PI_2;

fn main() {
    // A quarter turn about z, then one about the new x (intrinsic Z-X-Z with a zero last angle).
    let r = Rotation3::from_euler(EulerSequence::ZXZ, [FRAC_PI_2, FRAC_PI_2, 0.0]);
    let m = r.matrix();
    println!("matrix rows {:.3?}", m.map(|row| row.map(|v| v + 0.0)));
    println!("is a rotation: {}, x axis goes to {:.3?}", is_rotation_matrix(&m, 1e-12), mat3_apply(&m, [1.0, 0.0, 0.0]).map(|v| v + 0.0));
    let (axis, angle) = r.axis_angle();
    println!("axis {:.4?}, angle {:.4} rad", axis, angle);
    println!("as intrinsic X-Y-Z angles: {:.4?}", r.to_euler(EulerSequence::XYZ).unwrap());

    // Quaternions compose by the Hamilton product: a then b is b * a.
    let a = Quaternion::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
    let b = Quaternion::from_axis_angle([1.0, 0.0, 0.0], FRAC_PI_2);
    let ab = b * a;
    println!("quaternion of a then b: {:.4} {:+.4}i {:+.4}j {:+.4}k", ab.w, ab.x, ab.y, ab.z);
    println!("rotates (1, 0, 0) to {:.3?}", ab.rotate_vector([1.0, 0.0, 0.0]).map(|v| v + 0.0));

    // Halfway along the 120 degree rotation r is a 60 degree turn.
    let half = Rotation3::identity().slerp(&r, 0.5);
    println!("slerp halfway: {:.4} rad from the start, {:.4} from the end", Rotation3::identity().angle_to(&half), half.angle_to(&r));
    println!("roll, pitch, yaw of a: {:.4?}", a.to_euler());
}
