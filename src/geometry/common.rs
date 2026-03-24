//! Shared foundational traits used by all geometry primitives.

use crate::geometry::one_d::IsLine;
use crate::geometry::two_d::{IsPolygon, IsTriangle};
use crate::geometry::zero_d::IsPoint;
use serde::Serialize;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

/// Canonical scalar type used for geometric measurements.
pub type GeometryMeasure = f32;

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

/// Abstraction over a mutable collection of geometry primitives.
pub trait IsGeometryTable {
    /// Item stored by the table.
    type Item: GeometricPrimitive;

    /// Reads an item by index.
    fn read(&self, index: usize) -> Option<Self::Item>;
    /// Writes an item at the given index.
    fn write(&mut self, index: usize, value: Self::Item) -> Result<(), String>;
    /// Deletes the item at the given index.
    fn delete(&mut self, index: usize) -> Result<(), String>;
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
    /// Iterates immutably over the table contents.
    fn iter(&self) -> Box<dyn Iterator<Item = Self::Item> + '_>;
    /// Iterates mutably over the table contents.
    fn iter_mut(&mut self) -> Box<dyn Iterator<Item = &mut Self::Item> + '_>;
    /// Serializes the table to a file.
    fn to_file(&self, path: &str) -> Result<(), String>;
    /// Loads a table instance from a file.
    fn from_file(path: &str) -> Result<Self, String>
    where
        Self: Sized;
}

/// Indicates that a type borrows and exposes a geometry table.
pub trait UsesTable<'a>: Sized {
    /// Item type stored in the referenced table.
    type Item: GeometricPrimitive;
    /// Concrete or dynamic table type used by the implementor.
    type Table: IsGeometryTable<Item = Self::Item> + ?Sized + 'a;

    /// Returns an immutable reference to the configured table.
    fn table(&self) -> &Self::Table;
    /// Returns a mutable reference to the configured table.
    fn table_mut(&mut self) -> &mut Self::Table;
    /// Rebinds the implementor to a different borrowed table.
    fn set_table(&mut self, table: &'a mut Self::Table);
}

/// Provides vertex-oriented access for primitives backed by point tables.
pub trait HasVertices<'a>: UsesTable<'a>
where
    Self::Item: IsPoint,
{
    /// Returns the number of vertices stored by the primitive.
    fn vertex_count(&self) -> usize {
        self.table().size()
    }

    /// Returns a vertex by index.
    fn vertex(&self, index: usize) -> Option<Self::Item> {
        self.table().read(index)
    }

    /// Replaces the vertex at `index`.
    fn set_vertex(&mut self, index: usize, value: Self::Item) -> Result<(), String> {
        self.table_mut().write(index, value)
    }

    /// Removes the vertex at `index`.
    fn remove_vertex(&mut self, index: usize) -> Result<(), String> {
        self.table_mut().delete(index)
    }
}

/// Provides line-oriented access for primitives backed by stored line tables.
pub trait HasLines<'a>: UsesTable<'a, Item = Self::Line> {
    /// Point type used by the stored lines.
    type Point: IsPoint;
    /// Line type stored by the primitive.
    type Line: IsLine<'a, Self::Point>;

    /// Returns the number of stored lines.
    fn line_count(&self) -> usize {
        self.table().size()
    }

    /// Returns a line by index.
    fn line(&self, index: usize) -> Option<Self::Line> {
        self.table().read(index)
    }

    /// Replaces the line at `index`.
    fn set_line(&mut self, index: usize, value: Self::Line) -> Result<(), String> {
        self.table_mut().write(index, value)
    }

    /// Removes the line at `index`.
    fn remove_line(&mut self, index: usize) -> Result<(), String> {
        self.table_mut().delete(index)
    }
}

/// Provides triangle-oriented access for primitives backed by stored triangle tables.
pub trait HasTriangles<'a>: UsesTable<'a, Item = Self::Triangle> {
    /// Point type used by the stored triangles.
    type Point: IsPoint;
    /// Unit normal type used by the stored triangles.
    type Normal: IsUnitVector;
    /// Triangle type stored by the primitive.
    type Triangle: IsTriangle<'a, Self::Point, Self::Normal>;

    /// Returns the number of stored triangles.
    fn triangle_count(&self) -> usize {
        self.table().size()
    }

    /// Returns a triangle by index.
    fn triangle(&self, index: usize) -> Option<Self::Triangle> {
        self.table().read(index)
    }

    /// Replaces the triangle at `index`.
    fn set_triangle(&mut self, index: usize, value: Self::Triangle) -> Result<(), String> {
        self.table_mut().write(index, value)
    }

    /// Removes the triangle at `index`.
    fn remove_triangle(&mut self, index: usize) -> Result<(), String> {
        self.table_mut().delete(index)
    }
}

/// Provides polygon-face access for primitives backed by stored face tables.
pub trait HasFaces<'a>: UsesTable<'a, Item = Self::Face> {
    /// Point type used by the stored faces.
    type Point: IsPoint;
    /// Unit normal type used by the stored faces.
    type Normal: IsUnitVector;
    /// Polygon face type stored by the primitive.
    type Face: IsPolygon<'a, Self::Point, Self::Normal>;

    /// Returns the number of stored faces.
    fn face_count(&self) -> usize {
        self.table().size()
    }

    /// Returns a face by index.
    fn face(&self, index: usize) -> Option<Self::Face> {
        self.table().read(index)
    }

    /// Replaces the face at `index`.
    fn set_face(&mut self, index: usize, value: Self::Face) -> Result<(), String> {
        self.table_mut().write(index, value)
    }

    /// Removes the face at `index`.
    fn remove_face(&mut self, index: usize) -> Result<(), String> {
        self.table_mut().delete(index)
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
