//! Electromagnetics algorithms.

pub mod overlap_integral;
pub mod cfm_beamformer;
pub mod lcmv_beamformer;

pub use overlap_integral::overlap_integral;
pub use cfm_beamformer::cfm_beamformer;
pub use lcmv_beamformer::lcmv_beamformer;
