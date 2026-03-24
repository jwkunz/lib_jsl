//! Geometry primitives, traits, and coordinate-system abstractions.

/// Shared foundational traits used across all geometry dimensions.
pub mod common;
/// Coordinate-system enums and conversion traits.
pub mod coordinate_systems;
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
