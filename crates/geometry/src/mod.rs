//! Geometry primitives, concrete implementations, and coordinate-system abstractions.
//!
//! The module tree is split into two main layers:
//!
//! - trait modules such as [`zero_d`], [`one_d`], [`two_d`], and [`three_d`]
//! - concrete implementations and storage helpers such as [`concrete`], [`tables`], and
//!   [`registry`]
//!
//! For most users, the quickest entry points are:
//!
//! - [`crate::concrete`] for concrete types like `CoordinateVector3D`, `Line3D`, and `SurfaceMesh3D`
//! - [`crate::geometric_traits`] for the broad trait surface

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
#[path = "dim_0/mod.rs"]
/// Zero-dimensional geometry traits centered on points.
pub mod zero_d;
#[path = "dim_1/mod.rs"]
/// One-dimensional geometry traits such as lines, rays, and paths.
pub mod one_d;
#[path = "dim_2/mod.rs"]
/// Two-dimensional geometry traits and orientation helpers.
pub mod two_d;
#[path = "dim_3/mod.rs"]
/// Three-dimensional geometry traits such as planes, spheres, and meshes.
pub mod three_d;
#[path = "dim_4/mod.rs"]
/// Four-dimensional geometry extension points.
pub mod four_d;
/// Compatibility re-exports for the geometry trait surface.
pub mod geometric_traits;
