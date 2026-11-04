//! Points, vectors and primitives sharing a keyed point table.

use lib_jsl_geometry::common::{IsGeometryTable, IsGeometryTableBase, PointId, TriangleId};
use lib_jsl_geometry::concrete::{CoordinateVector3D, FreeVector3D, GeometryTableRegistry, Line3D, Plane3D, Triangle3D, UnitVector3D};
use lib_jsl_geometry::coordinate_systems::CoordinateSystem3D;
use lib_jsl_geometry::one_d::IsLine;
use lib_jsl_geometry::transformation_traits::{CanMirror, CanTranslate};
use lib_jsl_geometry::two_d::IsPolygon;

fn main() {
    // Points live once, in the registry's point table, and are referred to by id.
    let mut registry = GeometryTableRegistry::new();
    for (id, p) in [(1, [0.0, 0.0, 0.0]), (2, [2.0, 0.0, 0.0]), (3, [0.0, 1.0, 0.0]), (4, [0.0, 0.0, 3.0])] {
        registry.point_table_mut().insert(PointId(id), CoordinateVector3D::new(p[0], p[1], p[2])).unwrap();
    }
    let points = registry.point_table().clone();
    let mut triangle = Triangle3D::new(PointId(1), PointId(2), PointId(3), points.clone());
    registry.triangle_table_mut().insert(TriangleId(1), triangle.clone()).unwrap();
    let edge = Line3D::new(PointId(1), PointId(2), points.clone());
    println!("triangle area {}, perimeter {:.4}, normal {}", triangle.area(), triangle.perimeter(), triangle.normal());
    println!("edge length {}", edge.length());

    // Moving the triangle moves its points in the shared table, so the edge follows.
    let up = Line3D::new(PointId(1), PointId(4), points.clone());
    triangle.translate(&up);
    println!("after translating by (0, 0, 3): edge from {} to {}", edge.head().unwrap(), edge.tail().unwrap());
    triangle.mirror(&Plane3D::new(CoordinateVector3D::new(0.0, 0.0, 1.0), UnitVector3D::new(0.0, 0.0, 1.0)));
    println!("after mirroring in z = 1: edge from {} to {}", edge.head().unwrap(), edge.tail().unwrap());

    // Coordinates can be held in another system; arithmetic works on the Cartesian values.
    let mut p = CoordinateVector3D::new(1.0, 1.0, 1.0);
    p.set_coordinate_system(CoordinateSystem3D::Spherical);
    println!("(1, 1, 1) in spherical coordinates: {:?}", p.raw_components());
    let q = p + FreeVector3D::new(0.0, 0.0, 1.0);
    println!("plus (0, 0, 1): Cartesian {:?}", q.cartesian_components());
}
