//! Compatibility re-exports for the public geometry API surface.
//!
//! This module is intended as a convenience import for users who want a broad view of the
//! geometry system without pulling items from each dimension-specific module by hand.

pub use crate::geometry::common::*;
pub use crate::geometry::concrete::*;
pub use crate::geometry::coordinate_systems::*;
pub use crate::geometry::four_d::*;
pub use crate::geometry::one_d::*;
pub use crate::geometry::registry::*;
pub use crate::geometry::tables::*;
pub use crate::geometry::three_d::*;
pub use crate::geometry::transformation_traits::*;
pub use crate::geometry::two_d::*;
pub use crate::geometry::zero_d::*;
