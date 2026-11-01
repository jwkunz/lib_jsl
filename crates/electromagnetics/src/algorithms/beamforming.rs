//! Shared beamformer conventions and figures of merit.
//!
//! A beamformer combines array snapshots `x` (length `N`) as `y = wᴴ x`, so a weight vector `w`
//! passes a signal with steering vector `a` with gain `wᴴ a` and outputs the power `wᴴ R w` for a
//! covariance `R`.

use crate::conventions::C64;
use crate::linalg::quadratic_form;
use lib_jsl_core::ErrorsJSL;
use ndarray::{Array1, Array2};

/// The output power `wᴴ R w` (W when `R` is in W).
pub fn output_power(weights: &Array1<C64>, covariance: &Array2<C64>) -> Result<f64, ErrorsJSL> {
    if covariance.dim() != (weights.len(), weights.len()) {
        return Err(ErrorsJSL::IncompatibleArraySizes((covariance.nrows(), weights.len())));
    }
    Ok(quadratic_form(weights, covariance))
}

/// The signal-to-interference-plus-noise ratio `wᴴ R_s w / wᴴ R_n w`.
pub fn sinr(weights: &Array1<C64>, signal: &Array2<C64>, interference_plus_noise: &Array2<C64>) -> Result<f64, ErrorsJSL> {
    let n = output_power(weights, interference_plus_noise)?;
    if n <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("The interference-plus-noise output power is not positive."));
    }
    Ok(output_power(weights, signal)? / n)
}

/// The complex gain `wᴴ a` of the beamformer for a steering vector.
pub fn response(weights: &Array1<C64>, steering: &Array1<C64>) -> Result<C64, ErrorsJSL> {
    if weights.len() != steering.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((weights.len(), steering.len())));
    }
    Ok(weights.iter().zip(steering.iter()).map(|(w, a)| w.conj() * a).sum())
}

/// `w / ‖w‖`, with the phase fixed so the largest-magnitude weight is real and positive (making
/// eigenvector solutions reproducible).
pub fn normalize(weights: &Array1<C64>) -> Result<Array1<C64>, ErrorsJSL> {
    let norm = weights.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    if norm.is_nan() || norm <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("Cannot normalise a zero weight vector."));
    }
    // The reference element is the first whose magnitude is within a relative 1e-9 of the
    // largest, so equal-magnitude weights (steering vectors) get a phase reference that does not
    // depend on rounding noise.
    let largest = weights.iter().fold(0.0f64, |m, v| m.max(v.norm()));
    let big = weights.iter().copied().find(|v| v.norm() >= largest * (1.0 - 1e-9)).expect("non-zero weights");
    let phase = big.conj() / big.norm();
    Ok(weights.mapv(|v| v * phase / norm))
}
