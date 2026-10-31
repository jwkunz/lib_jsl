//! Circular aperture (reflector dish) antennas.

use super::{Antenna, Excitation};
use crate::conventions::{C64, Medium, TimeConvention};
use lib_jsl_core::constants::em::C0;
use lib_jsl_special_functions::bessel_functions_integer::bessel_j_integer;
use std::f64::consts::{PI, TAU};

/// A circular aperture of diameter `D` in the `xy` plane, radiating along `+z`, modelled as a
/// Huygens source with a `ŷ`-polarized aperture field `E₀ (1 - ρ²/a²)^p` (`a = D/2`; `p = 0`
/// uniform, larger `p` tapers the edge and lowers the sidelobes). Far field per unit `E₀`, in
/// the `e^{+jωt}` convention:
///
/// `[E_θ, E_φ] = j k a² 2^p p! J_{p+1}(u)/u^{p+1} · (1 + cos θ)/2 · [sin φ, cos φ]`, `u = ka sin θ`.
///
/// Directivity `(πD/λ)² (2p+1)/(p+1)²`, i.e. aperture efficiency 1, 0.75, 0.556, ... for
/// `p = 0, 1, 2`. Radiated power is computed from the aperture field, `∫ |E_a|² dA / (2η)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dish {
    pub diameter: f64,
    pub taper: u32,
}

impl Dish {
    /// A uniformly illuminated aperture.
    pub fn new(diameter: f64) -> Self {
        Dish { diameter, taper: 0 }
    }

    /// An aperture with the parabolic-on-pedestal taper `(1 - ρ²/a²)^p`.
    pub fn with_taper(diameter: f64, taper: u32) -> Self {
        Dish { diameter, taper }
    }

    /// The aperture efficiency `(2p+1)/(p+1)²` of the taper.
    pub fn aperture_efficiency(&self) -> f64 {
        let p = f64::from(self.taper);
        (2.0 * p + 1.0) / ((p + 1.0) * (p + 1.0))
    }

    /// The exact boresight directivity `(πD/λ)² × efficiency`.
    pub fn boresight_directivity(&self, frequency: f64) -> f64 {
        (PI * self.diameter * frequency / C0).powi(2) * self.aperture_efficiency()
    }

    /// The half-power beamwidth (radians) found from the pattern in the `xz` plane.
    pub fn half_power_beamwidth(&self, frequency: f64) -> f64 {
        let peak = self.radiation_intensity(0.0, 0.0, frequency);
        let (mut lo, mut hi) = (0.0, PI / 2.0);
        // The main lobe falls monotonically to the first null; bisect the -3 dB point inside it.
        let first_null = 3.8317 / (TAU * frequency / C0 * self.diameter / 2.0);
        hi = f64::min(hi, first_null * (1.0 + f64::from(self.taper)));
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if self.radiation_intensity(mid, 0.0, frequency) > 0.5 * peak {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        2.0 * lo
    }
}

impl Antenna for Dish {
    fn far_field(&self, theta: f64, phi: f64, frequency: f64, convention: TimeConvention) -> [C64; 2] {
        let k = TAU * frequency / C0;
        let a = self.diameter / 2.0;
        let u = k * a * theta.sin();
        let p = self.taper as i32;
        let factorial: f64 = (1..=self.taper).map(f64::from).product();
        // 2^p p! J_{p+1}(u) / u^{p+1}, with its limit 1/(2(p+1)) at u = 0.
        let lambda = if u.abs() < 1e-6 { 1.0 / (2.0 * (f64::from(self.taper) + 1.0)) } else { 2f64.powi(p) * factorial * bessel_j_integer(p + 1, u) / u.powi(p + 1) };
        let amplitude = convention.unit() * (k * a * a * lambda * (1.0 + theta.cos()) / 2.0);
        [amplitude * phi.sin(), amplitude * phi.cos()]
    }

    fn excitation(&self) -> Excitation {
        Excitation::ApertureField
    }

    fn radiated_power(&self, _frequency: f64) -> f64 {
        let a = self.diameter / 2.0;
        let eta = Medium::VACUUM.impedance(TimeConvention::PlusJ).re;
        PI * a * a / (2.0 * eta * (2.0 * f64::from(self.taper) + 1.0))
    }

    fn peak_directivity(&self, frequency: f64) -> f64 {
        self.boresight_directivity(frequency)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::domain::{Axis, Domain};
    use crate::fields::physics::radiated_power;
    use std::sync::Arc;

    #[test]
    fn boresight_directivity_matches_the_aperture_formula() {
        let f = 10e9;
        let lambda = C0 / f;
        for taper in [0, 1, 2] {
            let dish = Dish::with_taper(10.0 * lambda, taper);
            let d = dish.directivity(0.0, 0.0, f);
            assert!((d - dish.boresight_directivity(f)).abs() < 1e-9 * d, "taper {taper}");
        }
        assert!((Dish::new(1.0).aperture_efficiency() - 1.0).abs() < 1e-15);
        assert!((Dish::with_taper(1.0, 1).aperture_efficiency() - 0.75).abs() < 1e-15);
        // (πD/λ)² for D = 10λ: ≈ 987 (29.9 dBi).
        assert!((Dish::new(10.0 * lambda).boresight_directivity(f) - 986.96).abs() < 0.01);
    }

    #[test]
    fn analytic_power_agrees_with_the_far_field_integral_and_tapers_widen_the_beam() {
        let f = 10e9;
        let lambda = C0 / f;
        let dish = Dish::with_taper(4.0 * lambda, 1);
        // Integrate the far field over the sphere (fine θ grid near the narrow main beam).
        let d = Arc::new(Domain::far_field_sphere(Axis::gauss_legendre_polar(160).unwrap(), Axis::periodic(0.0, TAU, 16).unwrap()).unwrap());
        let e = dish.far_field_pattern(d, f, TimeConvention::PlusJ).unwrap();
        let numeric = radiated_power(&e, &crate::conventions::Medium::VACUUM).unwrap();
        // The Huygens far field is approximate off boresight; agreement is close for D ≫ λ.
        assert!((numeric - dish.radiated_power(f)).abs() < 0.03 * numeric, "{numeric} vs {}", dish.radiated_power(f));
        // Uniform illumination: HPBW ≈ 1.02 λ/D. Tapering widens it.
        let uniform = Dish::new(20.0 * lambda);
        let hpbw = uniform.half_power_beamwidth(f);
        assert!((hpbw - 1.029 * lambda / (20.0 * lambda)).abs() < 2e-3 * hpbw.max(1.0), "{hpbw}");
        assert!(Dish::with_taper(20.0 * lambda, 2).half_power_beamwidth(f) > hpbw);
        // Polarization: ŷ aperture field gives E_θ in the yz plane and E_φ in the xz plane.
        let [et, ep] = uniform.far_field(0.01, PI / 2.0, f, TimeConvention::PlusJ);
        assert!(et.norm() > 0.0 && ep.norm() < 1e-12 * et.norm());
    }
}
