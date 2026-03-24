//! Shared foundational traits used by all geometry primitives.

use crate::geometry::one_d::IsLine;
use crate::geometry::two_d::{IsPolygon, IsTriangle};
use crate::geometry::three_d::IsTetrahedron;
use crate::geometry::zero_d::IsPoint;
use serde::Serialize;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

/// Canonical scalar type used for geometric measurements.
pub type GeometryMeasure = f32;

/// Stable identifier for entries in the root point table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct PointId(pub u64);

/// Stable identifier for entries in the root line table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct LineId(pub u64);

/// Stable identifier for entries in the root polygon-face table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct FaceId(pub u64);

/// Stable identifier for entries in the root triangle table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct TriangleId(pub u64);

/// Stable identifier for entries in the root tetrahedron table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct TetrahedronId(pub u64);

/// Base trait for stable geometry table keys.
pub trait IsGeometryKey: Debug + Clone + Copy + PartialEq + Eq + Hash + Serialize + Sized {}

impl IsGeometryKey for PointId {}
impl IsGeometryKey for LineId {}
impl IsGeometryKey for FaceId {}
impl IsGeometryKey for TriangleId {}
impl IsGeometryKey for TetrahedronId {}

/// Base trait implemented by all geometry primitives in this module tree.
pub trait GeometricPrimitive:
    Debug
    + Clone
    + Display
    + PartialEq
    + Serialize
    + Hash
    + Sized
{
}

/// A coordinate-bearing primitive that supports indexed scalar access.
///
/// This trait is intended for point- and vector-like values. Composite shapes such as lines,
/// polygons, planes, and meshes should generally implement [`GeometricPrimitive`] without also
/// implementing this trait unless they are intentionally represented as flat coordinate vectors.
pub trait CoordinatePrimitive:
    GeometricPrimitive
    + AsRef<GeometryMeasure>
    + AsMut<GeometryMeasure>
    + Index<usize, Output = GeometryMeasure>
    + IndexMut<usize, Output = GeometryMeasure>
{
}

/// Marker trait for primitives that live in a 2D space.
pub trait GeometricPrimitive2D: GeometricPrimitive {}

/// Marker trait for primitives that live in a 3D space.
pub trait GeometricPrimitive3D: GeometricPrimitive {}

/// Marker trait for primitives that live in a 4D space.
pub trait GeometricPrimitive4D: GeometricPrimitive {}

/// A plane-like primitive represented by a point and a unit normal.
///
/// This abstraction is dimension-neutral: in 3D it corresponds to a geometric plane, while in 2D
/// and 1D it can serve as the mirror/reference hyperplane for transformation traits.
pub trait IsPlane: GeometricPrimitive {
    /// Point type used to anchor the plane.
    type Point: IsPoint;
    /// Unit normal type used to orient the plane.
    type Normal: IsUnitVector;

    /// Returns a point on the plane.
    fn point(&self) -> Self::Point;
    /// Returns a mutable reference to a point on the plane.
    fn point_mut(&mut self) -> &mut Self::Point;
    /// Returns the plane normal.
    fn normal(&self) -> Self::Normal;
    /// Returns a mutable reference to the plane normal.
    fn normal_mut(&mut self) -> &mut Self::Normal;
}

/// Exposes the dimensionality of a primitive as a compile-time constant.
pub trait HasDimension {
    /// Number of coordinates used to represent the primitive.
    const DIM: usize;
}

/// Supports scalar arithmetic against the geometry measurement type.
pub trait ScalarOperable:
    CoordinatePrimitive
    + Add<GeometryMeasure, Output = Self>
    + Sub<GeometryMeasure, Output = Self>
    + Mul<GeometryMeasure, Output = Self>
    + Div<GeometryMeasure, Output = Self>
{
}

/// Supports primitive-to-primitive addition and subtraction.
pub trait SelfAddition:
    CoordinatePrimitive + Add<Output = Self> + Sub<Output = Self>
{
}

/// Supports an inner-product-like multiplication producing a scalar measure.
pub trait SelfProductInner:
    CoordinatePrimitive + Mul<Self, Output = GeometryMeasure>
{
}

/// Abstraction over a keyed mutable collection of geometry primitives.
pub trait IsGeometryTable {
    /// Stable key used to address items in the table.
    type Key: IsGeometryKey;
    /// Item stored by the table.
    type Item: GeometricPrimitive;

    /// Returns the item stored at `key`, if present.
    fn get(&self, key: &Self::Key) -> Option<Self::Item>;
    /// Inserts or replaces the item stored at `key`.
    fn insert(&mut self, key: Self::Key, value: Self::Item) -> Result<(), String>;
    /// Removes and returns the item stored at `key`, if present.
    fn remove(&mut self, key: &Self::Key) -> Option<Self::Item>;
    /// Returns `true` if `key` exists in the table.
    fn contains_key(&self, key: &Self::Key) -> bool;
    /// Returns the number of items held by the table.
    fn size(&self) -> usize;
    /// Computes the union of two tables.
    fn union(&self, other: &Self) -> Self
    where
        Self: Sized;
    /// Computes the intersection of two tables.
    fn intersection(&self, other: &Self) -> Self
    where
        Self: Sized;
    /// Iterates immutably over key/value pairs in the table.
    fn iter(&self) -> Box<dyn Iterator<Item = (Self::Key, Self::Item)> + '_>;
    /// Serializes the table to a file.
    fn to_file(&self, path: &str) -> Result<(), String>;
    /// Loads a table instance from a file.
    fn from_file(path: &str) -> Result<Self, String>
    where
        Self: Sized;
}

/// Root registry that owns or exposes the current core geometry tables.
pub trait IsGeometryTableBase<'a>
where
    <Self::Tetrahedron as HasEdges>::Edge: IsLine<'a, Self::Point>,
{
    /// Point primitive stored in the point table.
    type Point: IsPoint;
    /// Unit normal type used by polygonal faces and triangles.
    type Normal: IsUnitVector;
    /// Line primitive stored in the line table.
    type Line: IsLine<'a, Self::Point>;
    /// Polygon face primitive stored in the face table.
    type Face: IsPolygon<'a, Self::Point, Self::Normal>;
    /// Triangle primitive stored in the triangle table.
    type Triangle: IsTriangle<'a, Self::Point, Self::Normal>;
    /// Tetrahedron primitive stored in the tetrahedron table.
    type Tetrahedron: IsTetrahedron<'a, Self::Point, Self::Normal>;

    /// Concrete point table type.
    type PointTable: IsGeometryTable<Key = PointId, Item = Self::Point> + ?Sized + 'a;
    /// Concrete line table type.
    type LineTable: IsGeometryTable<Key = LineId, Item = Self::Line> + ?Sized + 'a;
    /// Concrete face table type.
    type FaceTable: IsGeometryTable<Key = FaceId, Item = Self::Face> + ?Sized + 'a;
    /// Concrete triangle table type.
    type TriangleTable: IsGeometryTable<Key = TriangleId, Item = Self::Triangle> + ?Sized + 'a;
    /// Concrete tetrahedron table type.
    type TetrahedronTable: IsGeometryTable<Key = TetrahedronId, Item = Self::Tetrahedron> + ?Sized + 'a;

    /// Returns an immutable reference to the point table.
    fn point_table(&self) -> &Self::PointTable;
    /// Returns a mutable reference to the point table.
    fn point_table_mut(&mut self) -> &mut Self::PointTable;
    /// Returns an immutable reference to the line table.
    fn line_table(&self) -> &Self::LineTable;
    /// Returns a mutable reference to the line table.
    fn line_table_mut(&mut self) -> &mut Self::LineTable;
    /// Returns an immutable reference to the face table.
    fn face_table(&self) -> &Self::FaceTable;
    /// Returns a mutable reference to the face table.
    fn face_table_mut(&mut self) -> &mut Self::FaceTable;
    /// Returns an immutable reference to the triangle table.
    fn triangle_table(&self) -> &Self::TriangleTable;
    /// Returns a mutable reference to the triangle table.
    fn triangle_table_mut(&mut self) -> &mut Self::TriangleTable;
    /// Returns an immutable reference to the tetrahedron table.
    fn tetrahedron_table(&self) -> &Self::TetrahedronTable;
    /// Returns a mutable reference to the tetrahedron table.
    fn tetrahedron_table_mut(&mut self) -> &mut Self::TetrahedronTable;
}

/// Provides access to a borrowed point table.
pub trait HasVertices<'a>: Sized {
    /// Point primitive stored in the vertex table.
    type Vertex: IsPoint;
    /// Borrowed point-table type.
    type VertexTable: IsGeometryTable<Key = PointId, Item = Self::Vertex> + ?Sized + 'a;

    /// Returns an immutable reference to the vertex table.
    fn vertex_table(&self) -> &Self::VertexTable;
    /// Returns a mutable reference to the vertex table.
    fn vertex_table_mut(&mut self) -> &mut Self::VertexTable;
    /// Rebinds the implementor to a different borrowed vertex table.
    fn set_vertex_table(&mut self, table: &'a mut Self::VertexTable);

    /// Returns the number of entries in the vertex table.
    fn vertex_table_size(&self) -> usize {
        self.vertex_table().size()
    }

    /// Returns the vertex stored at `id`.
    fn get_vertex(&self, id: &PointId) -> Option<Self::Vertex> {
        self.vertex_table().get(id)
    }

    /// Returns `true` if the vertex table contains `id`.
    fn contains_vertex(&self, id: &PointId) -> bool {
        self.vertex_table().contains_key(id)
    }

    /// Inserts or replaces the vertex stored at `id`.
    fn insert_vertex(&mut self, id: PointId, value: Self::Vertex) -> Result<(), String> {
        self.vertex_table_mut().insert(id, value)
    }

    /// Removes and returns the vertex stored at `id`, if present.
    fn remove_vertex(&mut self, id: &PointId) -> Option<Self::Vertex> {
        self.vertex_table_mut().remove(id)
    }
}

/// Provides access to a borrowed line table.
pub trait HasLines<'a>: Sized {
    /// Point primitive referenced by lines in the line table.
    type Point: IsPoint;
    /// Line primitive stored in the line table.
    type Line: IsLine<'a, Self::Point>;
    /// Borrowed line-table type.
    type LineTable: IsGeometryTable<Key = LineId, Item = Self::Line> + ?Sized + 'a;

    /// Returns an immutable reference to the line table.
    fn line_table(&self) -> &Self::LineTable;
    /// Returns a mutable reference to the line table.
    fn line_table_mut(&mut self) -> &mut Self::LineTable;
    /// Rebinds the implementor to a different borrowed line table.
    fn set_line_table(&mut self, table: &'a mut Self::LineTable);

    /// Returns the number of entries in the line table.
    fn line_table_size(&self) -> usize {
        self.line_table().size()
    }

    /// Returns the line stored at `id`.
    fn get_line(&self, id: &LineId) -> Option<Self::Line> {
        self.line_table().get(id)
    }

    /// Returns `true` if the line table contains `id`.
    fn contains_line(&self, id: &LineId) -> bool {
        self.line_table().contains_key(id)
    }

    /// Inserts or replaces the line stored at `id`.
    fn insert_line(&mut self, id: LineId, value: Self::Line) -> Result<(), String> {
        self.line_table_mut().insert(id, value)
    }

    /// Removes and returns the line stored at `id`, if present.
    fn remove_line(&mut self, id: &LineId) -> Option<Self::Line> {
        self.line_table_mut().remove(id)
    }
}

/// Provides access to a borrowed triangle table.
pub trait HasTriangles<'a>: Sized {
    /// Point primitive referenced by the stored triangles.
    type Point: IsPoint;
    /// Unit normal type used by the stored triangles.
    type Normal: IsUnitVector;
    /// Triangle primitive stored in the triangle table.
    type Triangle: IsTriangle<'a, Self::Point, Self::Normal>;
    /// Borrowed triangle-table type.
    type TriangleTable: IsGeometryTable<Key = TriangleId, Item = Self::Triangle> + ?Sized + 'a;

    /// Returns an immutable reference to the triangle table.
    fn triangle_table(&self) -> &Self::TriangleTable;
    /// Returns a mutable reference to the triangle table.
    fn triangle_table_mut(&mut self) -> &mut Self::TriangleTable;
    /// Rebinds the implementor to a different borrowed triangle table.
    fn set_triangle_table(&mut self, table: &'a mut Self::TriangleTable);

    /// Returns the number of entries in the triangle table.
    fn triangle_table_size(&self) -> usize {
        self.triangle_table().size()
    }

    /// Returns the triangle stored at `id`.
    fn get_triangle(&self, id: &TriangleId) -> Option<Self::Triangle> {
        self.triangle_table().get(id)
    }

    /// Returns `true` if the triangle table contains `id`.
    fn contains_triangle(&self, id: &TriangleId) -> bool {
        self.triangle_table().contains_key(id)
    }

    /// Inserts or replaces the triangle stored at `id`.
    fn insert_triangle(&mut self, id: TriangleId, value: Self::Triangle) -> Result<(), String> {
        self.triangle_table_mut().insert(id, value)
    }

    /// Removes and returns the triangle stored at `id`, if present.
    fn remove_triangle(&mut self, id: &TriangleId) -> Option<Self::Triangle> {
        self.triangle_table_mut().remove(id)
    }
}

/// Provides access to a borrowed tetrahedron table.
///
/// ```compile_fail
/// use lib_jsl::geometry::common::{HasTetrahedra, IsGeometryTable, PointId, TetrahedronId};
/// use lib_jsl::geometry::two_d::{Point2D, UnitVector2D};
///
/// #[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize)]
/// struct FakeTable;
///
/// impl IsGeometryTable for FakeTable {
///     type Key = TetrahedronId;
///     type Item = Point2D;
///
///     fn get(&self, _: &Self::Key) -> Option<Self::Item> { None }
///     fn insert(&mut self, _: Self::Key, _: Self::Item) -> Result<(), String> { Ok(()) }
///     fn remove(&mut self, _: &Self::Key) -> Option<Self::Item> { None }
///     fn contains_key(&self, _: &Self::Key) -> bool { false }
///     fn size(&self) -> usize { 0 }
///     fn union(&self, _: &Self) -> Self where Self: Sized { FakeTable }
///     fn intersection(&self, _: &Self) -> Self where Self: Sized { FakeTable }
///     fn iter(&self) -> Box<dyn Iterator<Item = (Self::Key, Self::Item)> + '_> { Box::new(std::iter::empty()) }
///     fn to_file(&self, _: &str) -> Result<(), String> { Ok(()) }
///     fn from_file(_: &str) -> Result<Self, String> where Self: Sized { Ok(FakeTable) }
/// }
///
/// struct BadCells<'a> {
///     table: &'a mut FakeTable,
/// }
///
/// impl<'a> HasTetrahedra<'a> for BadCells<'a> {
///     type Point = Point2D;
///     type Normal = UnitVector2D;
///     type Tetrahedron = Point2D;
///     type TetrahedronTable = FakeTable;
///
///     fn tetrahedron_table(&self) -> &Self::TetrahedronTable { self.table }
///     fn tetrahedron_table_mut(&mut self) -> &mut Self::TetrahedronTable { self.table }
///     fn set_tetrahedron_table(&mut self, table: &'a mut Self::TetrahedronTable) { self.table = table; }
/// }
/// ```
pub trait HasTetrahedra<'a>: Sized
where
    <Self::Tetrahedron as HasEdges>::Edge: IsLine<'a, Self::Point>,
{
    /// Point primitive referenced by the stored tetrahedra.
    type Point: IsPoint;
    /// Unit normal type used by the stored tetrahedra.
    type Normal: IsUnitVector;
    /// Tetrahedron primitive stored in the tetrahedron table.
    type Tetrahedron: IsTetrahedron<'a, Self::Point, Self::Normal>;
    /// Borrowed tetrahedron-table type.
    type TetrahedronTable: IsGeometryTable<Key = TetrahedronId, Item = Self::Tetrahedron> + ?Sized + 'a;

    /// Returns an immutable reference to the tetrahedron table.
    fn tetrahedron_table(&self) -> &Self::TetrahedronTable;
    /// Returns a mutable reference to the tetrahedron table.
    fn tetrahedron_table_mut(&mut self) -> &mut Self::TetrahedronTable;
    /// Rebinds the implementor to a different borrowed tetrahedron table.
    fn set_tetrahedron_table(&mut self, table: &'a mut Self::TetrahedronTable);

    /// Returns the number of entries in the tetrahedron table.
    fn tetrahedron_table_size(&self) -> usize {
        self.tetrahedron_table().size()
    }

    /// Returns the tetrahedron stored at `id`.
    fn get_tetrahedron(&self, id: &TetrahedronId) -> Option<Self::Tetrahedron> {
        self.tetrahedron_table().get(id)
    }

    /// Returns `true` if the tetrahedron table contains `id`.
    fn contains_tetrahedron(&self, id: &TetrahedronId) -> bool {
        self.tetrahedron_table().contains_key(id)
    }

    /// Inserts or replaces the tetrahedron stored at `id`.
    fn insert_tetrahedron(
        &mut self,
        id: TetrahedronId,
        value: Self::Tetrahedron,
    ) -> Result<(), String> {
        self.tetrahedron_table_mut().insert(id, value)
    }

    /// Removes and returns the tetrahedron stored at `id`, if present.
    fn remove_tetrahedron(&mut self, id: &TetrahedronId) -> Option<Self::Tetrahedron> {
        self.tetrahedron_table_mut().remove(id)
    }
}

/// Provides access to a borrowed polygon-face table.
pub trait HasFaces<'a>: Sized {
    /// Point primitive referenced by the stored faces.
    type Point: IsPoint;
    /// Unit normal type used by the stored faces.
    type Normal: IsUnitVector;
    /// Polygon face primitive stored in the face table.
    type Face: IsPolygon<'a, Self::Point, Self::Normal>;
    /// Borrowed face-table type.
    type FaceTable: IsGeometryTable<Key = FaceId, Item = Self::Face> + ?Sized + 'a;

    /// Returns an immutable reference to the face table.
    fn face_table(&self) -> &Self::FaceTable;
    /// Returns a mutable reference to the face table.
    fn face_table_mut(&mut self) -> &mut Self::FaceTable;
    /// Rebinds the implementor to a different borrowed face table.
    fn set_face_table(&mut self, table: &'a mut Self::FaceTable);

    /// Returns the number of entries in the face table.
    fn face_table_size(&self) -> usize {
        self.face_table().size()
    }

    /// Returns the face stored at `id`.
    fn get_face(&self, id: &FaceId) -> Option<Self::Face> {
        self.face_table().get(id)
    }

    /// Returns `true` if the face table contains `id`.
    fn contains_face(&self, id: &FaceId) -> bool {
        self.face_table().contains_key(id)
    }

    /// Inserts or replaces the face stored at `id`.
    fn insert_face(&mut self, id: FaceId, value: Self::Face) -> Result<(), String> {
        self.face_table_mut().insert(id, value)
    }

    /// Removes and returns the face stored at `id`, if present.
    fn remove_face(&mut self, id: &FaceId) -> Option<Self::Face> {
        self.face_table_mut().remove(id)
    }
}

/// Provides indexed access to a shape's derived geometric edges.
///
/// Unlike [`HasLines`], this trait does not imply that edges are stored in a table. Edges may be
/// computed views over a higher-level primitive boundary.
pub trait HasEdges {
    /// Edge type returned by the implementor.
    type Edge: GeometricPrimitive;

    /// Number of edges available on the shape.
    fn edge_count(&self) -> usize;
    /// Returns an edge by index.
    fn edge(&self, index: usize) -> Option<Self::Edge>;
}

/// Provides a canonical scalar measure for a primitive.
pub trait HasMeasure {
    /// Returns the primitive's primary measure, such as length or area.
    fn measure(&self) -> GeometryMeasure;
}

/// Exposes a geometric center point.
pub trait HasCenter {
    /// Point type used to represent the center.
    type Point: IsPoint;

    /// Returns the center point.
    fn center(&self) -> Self::Point;
}

/// Exposes a centroid point for the primitive.
pub trait HasCentroid {
    /// Point type used to represent the centroid.
    type Point: IsPoint;

    /// Returns the centroid.
    fn centroid(&self) -> Self::Point;
}

/// Provides access to a primitive's bounding box.
pub trait HasBoundingBox {
    /// Bounding box type associated with the primitive.
    type BoundingBox: GeometricPrimitive;

    /// Computes the bounding box.
    fn bounding_box(&self) -> Self::BoundingBox;
}

/// Computes a distance from `self` to another value.
pub trait HasDistanceTo<Rhs = Self> {
    /// Returns the distance to `other`.
    fn distance_to(&self, other: &Rhs) -> GeometryMeasure;
}

/// Tests whether a primitive contains another value.
pub trait Contains<T> {
    /// Returns `true` when `other` is contained within `self`.
    fn contains(&self, other: &T) -> bool;
}

/// Tests whether two values intersect.
pub trait Intersects<T> {
    /// Returns `true` when `self` intersects `other`.
    fn intersects(&self, other: &T) -> bool;
}

/// Supplies a default epsilon for approximate comparisons.
pub trait HasEpsilon {
    /// Returns the epsilon value to use for fuzzy comparisons.
    fn epsilon() -> GeometryMeasure;
}

/// Provides approximate equality comparisons with a caller-supplied epsilon.
pub trait ApproxEq<Rhs = Self> {
    /// Returns `true` if `self` and `other` are approximately equal.
    fn approx_eq(&self, other: &Rhs, epsilon: GeometryMeasure) -> bool;
}

/// Indicates whether a primitive satisfies its invariants.
pub trait IsValid {
    /// Returns `true` when the primitive is valid.
    fn is_valid(&self) -> bool;
}

/// Attempts to repair an invalid or non-canonical primitive in place.
pub trait Repair {
    /// Repairs the primitive or returns an error.
    fn repair(&mut self) -> Result<(), String>;
}

/// Reorders or normalizes an in-memory representation in place.
pub trait Canonicalize {
    /// Converts the primitive to a canonical representation.
    fn canonicalize(&mut self);
}

/// Uniformly scales a primitive in place.
pub trait CanScale: GeometricPrimitive + Sized {
    /// Applies a uniform scaling factor.
    fn scale(&mut self, factor: GeometryMeasure);
}

/// Applies non-uniform scaling using a separate scale vector.
pub trait CanScaleNonUniform: GeometricPrimitive + Sized {
    /// Primitive used to represent non-uniform scale factors.
    type ScaleVector: CoordinatePrimitive;

    /// Applies non-uniform scaling in place.
    fn scale_non_uniform(&mut self, factors: &Self::ScaleVector);
}

/// Projects a primitive onto another target type.
pub trait CanProject: GeometricPrimitive + Sized {
    /// Result type produced by the projection.
    type Projection;

    /// Projects `self` onto `target`.
    fn project_onto<T>(&self, target: &T) -> Self::Projection;
}

/// Normalizes a primitive in place.
pub trait CanNormalize: CoordinatePrimitive + Sized {
    /// Mutates the primitive into a normalized form.
    fn normalize(&mut self);
}

/// Marker trait for vectors guaranteed to have unit magnitude.
pub trait IsUnitVector: CoordinatePrimitive {}

/// Computes a dot product between two values.
pub trait DotProduct<Rhs = Self>: CoordinatePrimitive {
    /// Output type produced by the dot product.
    type Output;

    /// Computes the dot product.
    fn dot(&self, rhs: &Rhs) -> <Self as DotProduct<Rhs>>::Output;
}

/// Computes a cross product between two values.
pub trait CrossProduct<Rhs = Self>: CoordinatePrimitive {
    /// Output type produced by the cross product.
    type Output;

    /// Computes the cross product.
    fn cross(&self, rhs: &Rhs) -> <Self as CrossProduct<Rhs>>::Output;
}

/// Computes the norm or magnitude of a value.
pub trait HasNorm: CoordinatePrimitive {
    /// Returns the norm.
    fn norm(&self) -> GeometryMeasure;
}

/// Produces a normalized copy without mutating the original value.
pub trait Normalize: CoordinatePrimitive + Sized {
    /// Returns a normalized copy of `self`.
    fn normalized(&self) -> Self;
}
