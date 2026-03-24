//! Compatibility re-exports for the public geometry API surface.
//!
//! This module is intended as a convenience import for users who want a broad view of the
//! geometry system without pulling items from each dimension-specific module by hand.

pub use crate::common::*;
pub use crate::concrete::*;
pub use crate::coordinate_systems::*;
pub use crate::four_d::*;
pub use crate::one_d::*;
pub use crate::registry::*;
pub use crate::tables::*;
pub use crate::three_d::*;
pub use crate::transformation_traits::*;
pub use crate::two_d::*;
pub use crate::zero_d::*;
