//! Computational geometry: primitive types in one to four dimensions and the algorithms that work
//! on points and meshes.
//!
//! - [`algorithms`]: Delaunay triangulation and Voronoi diagrams, convex hulls in 2D and 3D,
//!   polygon boolean operations, k-d trees, quadtrees and octrees, intersection tests and closest
//!   points, rotations and quaternions, and mesh smoothing, decimation and slicing. These take
//!   plain `f64` arrays such as `[f64; 3]` and index triples.
//! - The primitive types: coordinate, free and unit vectors, lines, planes, triangles, polygon
//!   faces, tetrahedra and meshes in [`one_d`], [`two_d`] and [`three_d`], whose shapes refer to
//!   shared points by id through the keyed tables of [`registry`] and [`tables`] (start with
//!   [`crate::concrete`] and a [`GeometryTableRegistry`](crate::registry::GeometryTableRegistry)).
//!   Their components are `f32`.
//! - [`four_d`]: Minkowski four-vectors with Lorentz boosts, and Christoffel symbols for geodesics
//!   in curved space-time.
//! - [`coordinate_systems`], [`geometric_traits`] and [`transformation_traits`]: coordinate
//!   systems and the shared trait surface.
//!
//! See the book entry [Computational geometry](https://jwkunz.github.io/lib_jsl/book/geometry/index.html).

pub mod algorithms;
/// Shared foundational traits used across all geometry dimensions.
pub mod common;
/// Coordinate-system enums and conversion traits.
pub mod coordinate_systems;
/// Compatibility re-exports for the concrete geometry implementation surface.
pub mod concrete;
/// Concrete root registry for geometry tables.
pub mod registry;
/// Concrete keyed geometry table implementations.
pub mod tables;
/// Traits for geometric transformations such as translation and rotation.
pub mod transformation_traits;
/// Zero-dimensional geometry traits centered on points.
pub mod dim_0;
/// Alias for [`dim_0`].
pub use dim_0 as zero_d;
/// One-dimensional geometry traits such as lines, rays, and paths.
pub mod dim_1;
/// Alias for [`dim_1`].
pub use dim_1 as one_d;
/// Two-dimensional geometry traits and orientation helpers.
pub mod dim_2;
/// Alias for [`dim_2`].
pub use dim_2 as two_d;
/// Three-dimensional geometry traits such as planes, spheres, and meshes.
pub mod dim_3;
/// Alias for [`dim_3`].
pub use dim_3 as three_d;
/// Four-dimensional geometry extension points.
pub mod dim_4;
/// Alias for [`dim_4`].
pub use dim_4 as four_d;
/// Compatibility re-exports for the geometry trait surface.
pub mod geometric_traits;
