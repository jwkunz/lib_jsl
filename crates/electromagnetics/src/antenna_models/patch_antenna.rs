//! Rectangular microstrip patch antennas (transmission-line / cavity model).
//!
//! See the book entry [Antenna models](https://jwkunz.github.io/lib_jsl/book/em-arrays/antennas.html).

use super::{Antenna, Excitation};
use crate::conventions::{C64, Medium, TimeConvention};
use crate::fields::domain::{Axis, Domain};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_core::constants::em::C0;
use lib_jsl_special_functions::bessel_functions_integer::bessel_j_integer;
use std::f64::consts::{PI, TAU};

/// A rectangular patch of width `W` (along `y`) and length `L` (along `x`, the resonant
/// dimension) on a grounded substrate of height `h` and relative permittivity `ε_r`, centred at
/// the origin with the ground plane at `z = 0` and radiation into `z > 0`.
///
/// The dominant TM₀₁₀ mode is modelled by two radiating slots of width `W` at `x = ±L_e/2`
/// (`L_e = L + 2ΔL`), each an equivalent magnetic line current of `2V₀` doubled by the ground
/// plane; the far field per unit edge voltage `V₀` (in `e^{+jωt}`) is
///
/// `E_θ = -jk/(4π) · S · cos φ · A`,  `E_φ = jk/(4π) · S · cos θ sin φ · A`,
///
/// with `S = 2W sinc(kW sin θ sin φ / 2)` the slot factor and `A = 2 cos(k L_e sin θ cos φ / 2)`
/// the two-slot array factor, and zero below the (infinite) ground plane. Design and impedance
/// formulas follow Balanis, *Antenna Theory*, §14.2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchAntenna {
    pub width: f64,
    pub length: f64,
    pub height: f64,
    pub eps_r: f64,
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 { 1.0 } else { x.sin() / x }
}

impl PatchAntenna {
    pub fn new(width: f64, length: f64, height: f64, eps_r: f64) -> Result<Self, ErrorsJSL> {
        if [width, length, height].iter().any(|v| !(v.is_finite() && *v > 0.0)) || eps_r.is_nan() || eps_r < 1.0 {
            return Err(ErrorsJSL::InvalidInputRange("Need positive dimensions and eps_r >= 1."));
        }
        Ok(PatchAntenna { width, length, height, eps_r })
    }

    /// The standard design for resonance at `frequency`: `W = c/(2f) √(2/(ε_r+1))` and
    /// `L = c/(2f√ε_eff) - 2ΔL`.
    pub fn design(frequency: f64, eps_r: f64, height: f64) -> Result<Self, ErrorsJSL> {
        let width = C0 / (2.0 * frequency) * (2.0 / (eps_r + 1.0)).sqrt();
        let probe = PatchAntenna::new(width, 1.0, height, eps_r)?;
        let length = C0 / (2.0 * frequency * probe.effective_permittivity().sqrt()) - 2.0 * probe.length_extension();
        PatchAntenna::new(width, length, height, eps_r)
    }

    /// The effective permittivity `(ε_r+1)/2 + (ε_r-1)/2 (1 + 12h/W)^{-1/2}`.
    pub fn effective_permittivity(&self) -> f64 {
        (self.eps_r + 1.0) / 2.0 + (self.eps_r - 1.0) / 2.0 / (1.0 + 12.0 * self.height / self.width).sqrt()
    }

    /// The fringing length extension ΔL at each radiating edge.
    pub fn length_extension(&self) -> f64 {
        let e = self.effective_permittivity();
        let wh = self.width / self.height;
        0.412 * self.height * (e + 0.3) * (wh + 0.264) / ((e - 0.258) * (wh + 0.8))
    }

    /// The effective length `L + 2ΔL` (the slot separation).
    pub fn effective_length(&self) -> f64 {
        self.length + 2.0 * self.length_extension()
    }

    /// The TM₀₁₀ resonant frequency `c / (2 L_e √ε_eff)`.
    pub fn resonant_frequency(&self) -> f64 {
        C0 / (2.0 * self.effective_length() * self.effective_permittivity().sqrt())
    }

    fn slot_integral(&self, frequency: f64, coupled: bool) -> f64 {
        let k = TAU * frequency / C0;
        let axis = Axis::gauss_legendre(0.0, PI, 200).expect("valid");
        axis.values()
            .iter()
            .zip(axis.weights())
            .map(|(&t, &w)| {
                let c = t.cos();
                let x = 0.5 * k * self.width * c;
                // [sin(kW cos θ / 2) / cos θ]², with its finite limit at cos θ = 0.
                let base = if c.abs() < 1e-12 { (0.5 * k * self.width).powi(2) } else { (x.sin() / c).powi(2) };
                let j0 = if coupled { bessel_j_integer(0, k * self.length * t.sin()) } else { 1.0 };
                w * base * j0 * t.sin().powi(3)
            })
            .sum()
    }

    /// The self conductance `G₁` of one radiating slot (S).
    pub fn slot_conductance(&self, frequency: f64) -> f64 {
        self.slot_integral(frequency, false) / (120.0 * PI * PI)
    }

    /// The mutual conductance `G₁₂` between the two slots (S).
    pub fn mutual_conductance(&self, frequency: f64) -> f64 {
        self.slot_integral(frequency, true) / (120.0 * PI * PI)
    }

    /// The resonant input resistance at a radiating edge, `1 / (2 (G₁ + G₁₂))`.
    pub fn edge_resistance(&self, frequency: f64) -> f64 {
        1.0 / (2.0 * (self.slot_conductance(frequency) + self.mutual_conductance(frequency)))
    }

    /// The resistance of an inset feed `y₀` in from the edge: `R_edge cos²(π y₀ / L)`.
    pub fn inset_resistance(&self, frequency: f64, inset: f64) -> f64 {
        self.edge_resistance(frequency) * (PI * inset / self.length).cos().powi(2)
    }

    /// The inset depth giving the input resistance `target` (Ω), if it is reachable.
    pub fn inset_for_resistance(&self, frequency: f64, target: f64) -> Option<f64> {
        let edge = self.edge_resistance(frequency);
        (target > 0.0 && target <= edge).then(|| self.length / PI * (target / edge).sqrt().acos())
    }
}

impl Antenna for PatchAntenna {
    fn far_field(&self, theta: f64, phi: f64, frequency: f64, convention: TimeConvention) -> [C64; 2] {
        if theta > PI / 2.0 {
            return [C64::new(0.0, 0.0); 2];
        }
        let k = TAU * frequency / C0;
        let (st, ct) = theta.sin_cos();
        let (sp, cp) = phi.sin_cos();
        let slot = 2.0 * self.width * sinc(0.5 * k * self.width * st * sp);
        let array = 2.0 * (0.5 * k * self.effective_length() * st * cp).cos();
        let j = convention.unit();
        let common = k / (4.0 * PI) * slot * array;
        [-j * (common * cp), j * (common * ct * sp)]
    }

    fn excitation(&self) -> Excitation {
        Excitation::Voltage
    }

    fn radiated_power(&self, frequency: f64) -> f64 {
        // Integrate over the upper hemisphere only (the pattern is cut off at the ground plane).
        let d = Domain::far_field_sphere(Axis::gauss_legendre(0.0, PI / 2.0, 64).expect("valid"), Axis::periodic(0.0, TAU, 128).expect("valid")).expect("valid");
        let eta = Medium::VACUUM.impedance(TimeConvention::PlusJ).re;
        d.integrate_fn(|p| {
            let [a, b] = self.far_field(p.coords[1], p.coords[2], frequency, TimeConvention::PlusJ);
            (a.norm_sqr() + b.norm_sqr()) / (2.0 * eta)
        })
    }

    fn peak_directivity(&self, frequency: f64) -> f64 {
        // Broadside radiation: the maximum is at θ = 0.
        self.directivity(0.0, 0.0, frequency)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn design_reproduces_the_textbook_example() {
        // Balanis Example 14.1: ε_r = 2.2, h = 0.1588 cm, f_r = 10 GHz.
        let p = PatchAntenna::design(10e9, 2.2, 0.1588e-2).unwrap();
        assert!((p.width - 1.186e-2).abs() < 0.005e-2, "W = {}", p.width);
        assert!((p.effective_permittivity() - 1.972).abs() < 0.002);
        assert!((p.length_extension() - 0.081e-2).abs() < 0.001e-2);
        assert!((p.length - 0.906e-2).abs() < 0.005e-2, "L = {}", p.length);
        assert!((p.resonant_frequency() - 10e9).abs() < 1e3);
    }

    #[test]
    fn impedance_and_inset_feed() {
        let f = 10e9;
        let p = PatchAntenna::design(f, 2.2, 0.1588e-2).unwrap();
        let (g1, g12) = (p.slot_conductance(f), p.mutual_conductance(f));
        assert!((g1 - 1.57e-3).abs() < 0.05e-3, "G1 = {g1}"); // Balanis: G₁ ≈ 0.00157 S
        assert!(g12 > 0.0 && g12 < g1);
        let edge = p.edge_resistance(f);
        assert!((150.0..300.0).contains(&edge), "{edge}");
        let y0 = p.inset_for_resistance(f, 50.0).unwrap();
        assert!((p.inset_resistance(f, y0) - 50.0).abs() < 1e-9 && y0 > 0.0 && y0 < p.length / 2.0);
        assert!(p.inset_for_resistance(f, edge * 2.0).is_none());
        assert!(PatchAntenna::new(1.0, 1.0, 0.0, 2.0).is_err() && PatchAntenna::new(1.0, 1.0, 1.0, 0.5).is_err());
    }

    #[test]
    fn pattern_is_broadside_with_the_usual_directivity_and_no_back_radiation() {
        let f = 10e9;
        let p = PatchAntenna::design(f, 2.2, 0.1588e-2).unwrap();
        let d = p.peak_directivity(f);
        assert!((5.0..9.0).contains(&d), "D = {d} ({:.2} dBi)", 10.0 * d.log10());
        assert_eq!(p.far_field(2.0, 0.3, f, TimeConvention::PlusJ), [C64::new(0.0, 0.0); 2]);
        // E-plane (φ = 0) is θ-polarized and does not vanish at grazing; the H-plane (φ = 90°)
        // is φ-polarized and vanishes at grazing.
        let [et, ep] = p.far_field(PI / 2.0 - 1e-9, 0.0, f, TimeConvention::PlusJ);
        assert!(et.norm() > 0.0 && ep.norm() < 1e-12);
        let [_, eph] = p.far_field(PI / 2.0 - 1e-9, PI / 2.0, f, TimeConvention::PlusJ);
        assert!(eph.norm() < 1e-6 * et.norm());
    }
}
