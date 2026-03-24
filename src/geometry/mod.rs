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
//! - [`crate::geometry::concrete`] for concrete types like `Point3D`, `Line3D`, and `Mesh3D`
//! - [`crate::geometry::geometric_traits`] for the broad trait surface

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
#[path = "0D/mod.rs"]
/// Zero-dimensional geometry traits centered on points.
pub mod zero_d;
#[path = "1D/mod.rs"]
/// One-dimensional geometry traits such as lines, rays, and paths.
pub mod one_d;
#[path = "2D/mod.rs"]
/// Two-dimensional geometry traits and orientation helpers.
pub mod two_d;
#[path = "3D/mod.rs"]
/// Three-dimensional geometry traits such as planes, spheres, and meshes.
pub mod three_d;
#[path = "4D/mod.rs"]
/// Four-dimensional geometry extension points.
pub mod four_d;
/// Compatibility re-exports for the geometry trait surface.
pub mod geometric_traits;
