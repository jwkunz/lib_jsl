//! Electromagnetics algorithms: the overlap integral and the beamformer solvers.

pub mod beamforming;
pub mod cfm_beamformer;
pub mod direction_of_arrival;
pub mod fdtd;
pub mod lcmv_beamformer;
pub mod max_snr_beamformer;
pub mod method_of_moments;
pub mod near_field_far_field;
pub mod overlap_integral;

pub use beamforming::{normalize, output_power, response, sinr};
pub use cfm_beamformer::{cfm_beamformer, cfm_for_direction, cfm_weights};
pub use lcmv_beamformer::{lcmv_beamformer, mvdr_beamformer};
pub use max_snr_beamformer::max_snr_beamformer;
pub use overlap_integral::{Overlap, overlap_integral};
pub use direction_of_arrival::{bartlett_spectrum, capon_spectrum, esprit_ula, music_spectrum, spectrum_peaks};
pub use fdtd::{Fdtd1D, FdtdTmz, gaussian_pulse};
pub use method_of_moments::{WireDipole, WireSolution};
pub use near_field_far_field::{planar_near_to_far, scan_taper, spherical_near_to_far};
