//! Electromagnetics algorithms: the overlap integral and the beamformer solvers.

pub mod beamforming;
pub mod cfm_beamformer;
pub mod lcmv_beamformer;
pub mod max_snr_beamformer;
pub mod overlap_integral;

pub use beamforming::{normalize, output_power, response, sinr};
pub use cfm_beamformer::{cfm_beamformer, cfm_for_direction, cfm_weights};
pub use lcmv_beamformer::{lcmv_beamformer, mvdr_beamformer};
pub use max_snr_beamformer::max_snr_beamformer;
pub use overlap_integral::{Overlap, overlap_integral};
