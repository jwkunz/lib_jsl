//! Direction-of-arrival estimation on sensor arrays.
//!
//! Given an array covariance `R` (estimated from snapshots, see
//! [`crate::signal_models::covariance::sample_covariance`]), these scan steering vectors
//! `a(Ω) = A(Ω) p` of any [`SensorArray`] (any geometry, embedded element patterns, a chosen
//! polarization `p`) over candidate directions:
//!
//! * Bartlett (conventional beamforming) `aᴴRa / aᴴa`: robust, resolution limited to a beamwidth;
//! * Capon (MVDR) `1 / (aᴴR⁻¹a)`: adaptive nulling sharpens the peaks;
//! * MUSIC `aᴴa / (aᴴ E_n E_nᴴ a)` with the noise subspace `E_n`: super-resolution for a known
//!   number of sources;
//!
//! and ESPRIT estimates the direction cosines of a uniform linear array in closed form from the
//! rotational invariance between its two overlapping subarrays.

use crate::arrays::SensorArray;
use crate::conventions::{C64, TimeConvention};
use crate::linalg::{check_hermitian, hermitian_eigen, hermitian_transpose};
use lib_jsl_core::ErrorsJSL;
use ndarray::{Array1, Array2, s};
use ndarray_linalg::{Eig, Inverse};

fn steering(array: &SensorArray, direction: (f64, f64), polarization: [C64; 2]) -> Result<Array1<C64>, ErrorsJSL> {
    array.steering_vector(direction.0, direction.1, polarization)
}

fn hermitian_product(a: &Array1<C64>, m: &Array2<C64>) -> f64 {
    a.iter().zip(m.dot(a).iter()).map(|(x, y)| (x.conj() * y).re).sum()
}

fn norm_squared(a: &Array1<C64>) -> f64 {
    a.iter().map(|v| v.norm_sqr()).sum()
}

/// The Bartlett spectrum `aᴴRa / aᴴa` at each `(θ, φ)`.
pub fn bartlett_spectrum(array: &SensorArray, covariance: &Array2<C64>, directions: &[(f64, f64)], polarization: [C64; 2]) -> Result<Vec<f64>, ErrorsJSL> {
    check_hermitian(covariance, array.len())?;
    directions.iter().map(|&d| { let a = steering(array, d, polarization)?; Ok(hermitian_product(&a, covariance) / norm_squared(&a)) }).collect()
}

/// The Capon (minimum variance) spectrum `1 / (aᴴR⁻¹a)`, with `a` normalised to unit norm.
pub fn capon_spectrum(array: &SensorArray, covariance: &Array2<C64>, directions: &[(f64, f64)], polarization: [C64; 2]) -> Result<Vec<f64>, ErrorsJSL> {
    check_hermitian(covariance, array.len())?;
    let inverse = covariance.inv().map_err(|_| ErrorsJSL::RuntimeError("The covariance is singular."))?;
    directions.iter().map(|&d| { let a = steering(array, d, polarization)?; Ok(norm_squared(&a) / hermitian_product(&a, &inverse)) }).collect()
}

/// The MUSIC pseudospectrum `aᴴa / (aᴴ E_n E_nᴴ a)` for `sources` signals (fewer than the
/// number of elements).
pub fn music_spectrum(array: &SensorArray, covariance: &Array2<C64>, sources: usize, directions: &[(f64, f64)], polarization: [C64; 2]) -> Result<Vec<f64>, ErrorsJSL> {
    let n = array.len();
    check_hermitian(covariance, n)?;
    if sources == 0 || sources >= n {
        return Err(ErrorsJSL::InvalidInputRange("Need 0 < sources < elements."));
    }
    let (_, vectors) = hermitian_eigen(covariance)?;
    let noise = vectors.slice(s![.., sources..]).to_owned();
    let projector = noise.dot(&hermitian_transpose(&noise));
    directions.iter().map(|&d| { let a = steering(array, d, polarization)?; Ok(norm_squared(&a) / hermitian_product(&a, &projector).max(1e-300)) }).collect()
}

/// The indices of the `count` largest local maxima of a spectrum sampled on a 1D grid (ends
/// count when they exceed their single neighbour), strongest first.
pub fn spectrum_peaks(spectrum: &[f64], count: usize) -> Vec<usize> {
    let n = spectrum.len();
    let mut peaks: Vec<usize> = (0..n)
        .filter(|&i| (i == 0 || spectrum[i] > spectrum[i - 1]) && (i + 1 == n || spectrum[i] >= spectrum[i + 1]))
        .collect();
    peaks.sort_by(|&a, &b| spectrum[b].total_cmp(&spectrum[a]));
    peaks.truncate(count);
    peaks
}

/// ESPRIT for a uniform linear array of elements spaced `spacing` metres apart along its axis
/// (elements ordered along the axis, identical patterns): returns the `sources` direction
/// cosines `u = r̂ · axis` (for an array on the x axis in the xy plane, `u = cos φ`).
pub fn esprit_ula(covariance: &Array2<C64>, sources: usize, spacing: f64, wavenumber: f64, convention: TimeConvention) -> Result<Vec<f64>, ErrorsJSL> {
    let n = covariance.nrows();
    check_hermitian(covariance, n)?;
    if sources == 0 || sources >= n {
        return Err(ErrorsJSL::InvalidInputRange("Need 0 < sources < elements."));
    }
    let (_, vectors) = hermitian_eigen(covariance)?;
    let signal = vectors.slice(s![.., ..sources]).to_owned();
    let (first, second) = (signal.slice(s![..n - 1, ..]).to_owned(), signal.slice(s![1.., ..]).to_owned());
    // Least squares Ψ = (E₁ᴴE₁)⁻¹ E₁ᴴ E₂; its eigenvalues are e^{±jkd u}.
    let e1h = hermitian_transpose(&first);
    let psi = e1h.dot(&first).inv().map_err(|_| ErrorsJSL::RuntimeError("Degenerate signal subspace."))?.dot(&e1h.dot(&second));
    let (eigenvalues, _) = psi.eig().map_err(|_| ErrorsJSL::RuntimeError("Eigendecomposition failed."))?;
    let mut u: Vec<f64> = eigenvalues.iter().map(|l| (l.arg() / (convention.sign() * wavenumber * spacing)).clamp(-1.0, 1.0)).collect();
    u.sort_by(f64::total_cmp);
    Ok(u)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::antenna_models::HertzianDipole;
    use crate::coordinates::transforms::RigidTransform;
    use crate::signal_models::covariance::{sample_covariance, synthesize_snapshots};
    use std::f64::consts::FRAC_PI_2;
    use std::sync::Arc;

    const F: f64 = 1e9;

    /// An 8-element half-wave-spaced line of z dipoles on the x axis, the scene's true
    /// covariance for two unit sources at the given azimuths with noise σ², and a sample estimate.
    fn scene(azimuths_deg: [f64; 2], noise: f64) -> (SensorArray, Array2<C64>) {
        let lambda = lib_jsl_core::constants::em::C0 / F;
        let positions = SensorArray::linear_positions(8, lambda / 2.0, [1.0, 0.0, 0.0]);
        let array = SensorArray::from_antenna(&positions, Arc::new(HertzianDipole::new(0.01)), &RigidTransform::identity(), F, TimeConvention::PlusJ, 1.0).unwrap();
        let pol = [C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
        let mut r = Array2::<C64>::eye(8) * C64::new(noise, 0.0);
        for az in azimuths_deg {
            let a = array.steering_vector(FRAC_PI_2, az.to_radians(), pol).unwrap();
            let a = &a / C64::new(norm_squared(&a).sqrt() / 8f64.sqrt(), 0.0);
            for i in 0..8 {
                for j in 0..8 {
                    r[[i, j]] += a[i] * a[j].conj();
                }
            }
        }
        let snapshots = synthesize_snapshots(&r, 400, 3).unwrap();
        (array, sample_covariance(&snapshots).unwrap())
    }

    fn grid() -> (Vec<f64>, Vec<(f64, f64)>) {
        let az: Vec<f64> = (0..=1800).map(|i| i as f64 * 0.1).collect();
        let dirs = az.iter().map(|a| (FRAC_PI_2, a.to_radians())).collect();
        (az, dirs)
    }

    #[test]
    fn music_resolves_sources_inside_a_beamwidth_where_bartlett_cannot() {
        // 8 half-wave-spaced elements have a broadside beamwidth near 13°; the sources are 7° apart.
        let (array, r) = scene([85.0, 92.0], 0.1);
        let (az, dirs) = grid();
        let pol = [C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
        let music = music_spectrum(&array, &r, 2, &dirs, pol).unwrap();
        let mut found: Vec<f64> = spectrum_peaks(&music, 2).iter().map(|&i| az[i]).collect();
        found.sort_by(f64::total_cmp);
        assert!((found[0] - 85.0).abs() < 0.5 && (found[1] - 92.0).abs() < 0.5, "{found:?}");
        let bartlett = bartlett_spectrum(&array, &r, &dirs, pol).unwrap();
        let bartlett_peaks = spectrum_peaks(&bartlett[800..1000], 3);
        assert_eq!(bartlett_peaks.len(), 1, "Bartlett merges the two sources");
        let capon = capon_spectrum(&array, &r, &dirs, pol).unwrap();
        let mut capon_found: Vec<f64> = spectrum_peaks(&capon, 2).iter().map(|&i| az[i]).collect();
        capon_found.sort_by(f64::total_cmp);
        assert!((capon_found[0] - 85.0).abs() < 1.5 && (capon_found[1] - 92.0).abs() < 1.5, "{capon_found:?}");
        assert!(music_spectrum(&array, &r, 8, &dirs, pol).is_err());
    }

    #[test]
    fn esprit_estimates_direction_cosines_in_closed_form() {
        let (array, r) = scene([40.0, 120.0], 0.05);
        let u = esprit_ula(&r, 2, array.wavelength() / 2.0, array.wavenumber(), TimeConvention::PlusJ).unwrap();
        let mut angles: Vec<f64> = u.iter().map(|v| v.acos().to_degrees()).collect();
        angles.sort_by(f64::total_cmp);
        assert!((angles[0] - 40.0).abs() < 0.3 && (angles[1] - 120.0).abs() < 0.3, "{angles:?}");
        assert!(spectrum_peaks(&[1.0, 3.0, 2.0, 5.0, 4.0], 5) == vec![3, 1]);
    }
}
