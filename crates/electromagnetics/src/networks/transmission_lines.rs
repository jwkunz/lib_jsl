//! Transmission lines: impedance transformation, reflection, standing waves and matching.
//!
//! Phasors use the engineering convention `e^{+jωt}`: a wave travelling towards the load is
//! `e^{−γz}` with propagation constant `γ = α + jβ`, and lines are characterised per unit length
//! by `R, L, G, C`, giving `Z₀ = √((R + jωL)/(G + jωC))` and `γ = √((R + jωL)(G + jωC))`.

use crate::conventions::C64;
use lib_jsl_core::ErrorsJSL;
use lib_jsl_core::constants::em::C0;
use std::f64::consts::{PI, TAU};

/// A uniform transmission line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransmissionLine {
    /// A lossless line of real characteristic impedance with phase velocity `v = factor · c`.
    Lossless { impedance: f64, velocity_factor: f64 },
    /// A line with per-unit-length resistance (Ω/m), inductance (H/m), conductance (S/m) and
    /// capacitance (F/m).
    Distributed { r: f64, l: f64, g: f64, c: f64 },
}

impl TransmissionLine {
    pub fn lossless(impedance: f64, velocity_factor: f64) -> Result<Self, ErrorsJSL> {
        if !(impedance > 0.0) || !(velocity_factor > 0.0 && velocity_factor <= 1.0) {
            return Err(ErrorsJSL::InvalidInputRange("Need a positive impedance and 0 < velocity factor <= 1."));
        }
        Ok(TransmissionLine::Lossless { impedance, velocity_factor })
    }

    pub fn distributed(r: f64, l: f64, g: f64, c: f64) -> Result<Self, ErrorsJSL> {
        if r < 0.0 || g < 0.0 || !(l > 0.0) || !(c > 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("Need R, G >= 0 and L, C > 0."));
        }
        Ok(TransmissionLine::Distributed { r, l, g, c })
    }

    /// The characteristic impedance `Z₀` at `frequency`.
    pub fn characteristic_impedance(&self, frequency: f64) -> C64 {
        match *self {
            TransmissionLine::Lossless { impedance, .. } => C64::new(impedance, 0.0),
            TransmissionLine::Distributed { r, l, g, c } => {
                let w = TAU * frequency;
                (C64::new(r, w * l) / C64::new(g, w * c)).sqrt()
            }
        }
    }

    /// The propagation constant `γ = α + jβ` (Np/m, rad/m).
    pub fn propagation_constant(&self, frequency: f64) -> C64 {
        match *self {
            TransmissionLine::Lossless { velocity_factor, .. } => C64::new(0.0, TAU * frequency / (velocity_factor * C0)),
            TransmissionLine::Distributed { r, l, g, c } => {
                let w = TAU * frequency;
                let gamma = (C64::new(r, w * l) * C64::new(g, w * c)).sqrt();
                if gamma.re < 0.0 { -gamma } else { gamma }
            }
        }
    }

    /// The guided wavelength `2π/β`.
    pub fn wavelength(&self, frequency: f64) -> f64 {
        TAU / self.propagation_constant(frequency).im
    }

    /// The impedance seen at the input of a section of `length` metres terminated in `load`
    /// (`Z_in = Z₀ (Z_L + Z₀ tanh γl) / (Z₀ + Z_L tanh γl)`; an infinite load is an open circuit).
    pub fn input_impedance(&self, load: C64, length: f64, frequency: f64) -> C64 {
        // The cosh/sinh form avoids the poles of tanh at odd multiples of a quarter wave.
        let z0 = self.characteristic_impedance(frequency);
        let gl = self.propagation_constant(frequency) * length;
        let (ch, sh) = (gl.cosh(), gl.sinh());
        if load.re.is_infinite() || load.im.is_infinite() {
            return z0 * ch / sh;
        }
        z0 * (load * ch + z0 * sh) / (z0 * ch + load * sh)
    }

    /// The ABCD (chain) matrix of a section: `[[cosh γl, Z₀ sinh γl], [sinh γl / Z₀, cosh γl]]`.
    pub fn abcd(&self, length: f64, frequency: f64) -> [[C64; 2]; 2] {
        let z0 = self.characteristic_impedance(frequency);
        let gl = self.propagation_constant(frequency) * length;
        [[gl.cosh(), z0 * gl.sinh()], [gl.sinh() / z0, gl.cosh()]]
    }

    /// `|V(d)| / |V⁺|` at distances `d` from the load: `|e^{γd} + Γ_L e^{−γd}|`, the
    /// standing-wave envelope.
    pub fn standing_wave(&self, load: C64, distances: &[f64], frequency: f64) -> Vec<f64> {
        let gamma_l = reflection_coefficient(load, self.characteristic_impedance(frequency));
        let g = self.propagation_constant(frequency);
        distances.iter().map(|&d| ((g * d).exp() + gamma_l * (-g * d).exp()).norm()).collect()
    }
}

/// The reflection coefficient `Γ = (Z − Z₀)/(Z + Z₀)` (`1` for an open circuit).
pub fn reflection_coefficient(load: C64, z0: C64) -> C64 {
    if load.re.is_infinite() || load.im.is_infinite() {
        return C64::new(1.0, 0.0);
    }
    (load - z0) / (load + z0)
}

/// The voltage standing-wave ratio `(1 + |Γ|)/(1 − |Γ|)` (infinite for total reflection).
pub fn vswr(gamma: C64) -> f64 {
    let m = gamma.norm();
    if m >= 1.0 { f64::INFINITY } else { (1.0 + m) / (1.0 - m) }
}

/// The return loss `−20 log₁₀ |Γ|` in dB.
pub fn return_loss_db(gamma: C64) -> f64 {
    -20.0 * gamma.norm().log10()
}

/// The mismatch loss `−10 log₁₀(1 − |Γ|²)` in dB: the power lost to reflection.
pub fn mismatch_loss_db(gamma: C64) -> f64 {
    -10.0 * (1.0 - gamma.norm_sqr()).log10()
}

/// The characteristic impedance `√(Z_in Z_L)` of a quarter-wave transformer matching real
/// impedances.
pub fn quarter_wave_transformer(source: f64, load: f64) -> f64 {
    (source * load).sqrt()
}

/// Single shunt short-circuited stub matching of `load` to a lossless line of impedance `z0`:
/// returns both solutions as `(distance from the load, stub length)`, in wavelengths.
pub fn single_stub_match(load: C64, z0: f64) -> Result<[(f64, f64); 2], ErrorsJSL> {
    if load.re <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("Stub matching needs a load with positive resistance."));
    }
    let gamma = reflection_coefficient(load, C64::new(z0, 0.0));
    if gamma.norm() < 1e-12 {
        return Ok([(0.0, 0.25), (0.0, 0.25)]);
    }
    // The normalised admittance y(d) = (1 − Γe^{−j2βd})/(1 + Γe^{−j2βd}) must have real part 1,
    // i.e. Γe^{−j2βd} on the circle g = 1 of the admittance chart: cos(θ_Γ − 2βd) = −|Γ|.
    let (m, phase) = (gamma.norm(), gamma.arg());
    let theta = (-m).acos();
    let mut out = [(0.0, 0.0); 2];
    for (k, target) in [theta, -theta].into_iter().enumerate() {
        // 2βd = phase − target (mod 2π), d in wavelengths: βd = 2π d.
        let d = ((phase - target).rem_euclid(TAU)) / (4.0 * PI);
        let rotated = gamma * C64::from_polar(1.0, -4.0 * PI * d);
        let y = (C64::new(1.0, 0.0) - rotated) / (C64::new(1.0, 0.0) + rotated);
        // A shorted stub of length s has admittance −j cot(2πs); cancel the susceptance b.
        let b = y.im;
        let s = ((1.0 / b).atan() / TAU).rem_euclid(0.5);
        out[k] = (d, s);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const F: f64 = 1e9;

    #[test]
    fn quarter_and_half_wave_sections_transform_impedances() {
        let line = TransmissionLine::lossless(50.0, 0.66).unwrap();
        let lambda = line.wavelength(F);
        assert!((lambda - 0.66 * C0 / F).abs() < 1e-12);
        let load = C64::new(100.0, 25.0);
        let quarter = line.input_impedance(load, lambda / 4.0, F);
        assert!((quarter - C64::new(2500.0, 0.0) / load).norm() < 1e-9);
        assert!((line.input_impedance(load, lambda / 2.0, F) - load).norm() < 1e-8);
        // Stubs: a shorted λ/8 stub is +j Z₀, an open λ/8 stub −j Z₀.
        assert!((line.input_impedance(C64::new(0.0, 0.0), lambda / 8.0, F) - C64::new(0.0, 50.0)).norm() < 1e-9);
        assert!((line.input_impedance(C64::new(f64::INFINITY, 0.0), lambda / 8.0, F) - C64::new(0.0, -50.0)).norm() < 1e-9);
        assert_eq!(quarter_wave_transformer(50.0, 200.0), 100.0);
    }

    #[test]
    fn reflection_metrics_and_standing_waves() {
        let gamma = reflection_coefficient(C64::new(150.0, 0.0), C64::new(50.0, 0.0));
        assert!((gamma.re - 0.5).abs() < 1e-15 && (vswr(gamma) - 3.0).abs() < 1e-12);
        assert!((return_loss_db(gamma) - 6.0206).abs() < 1e-4 && (mismatch_loss_db(gamma) - 1.2494).abs() < 1e-4);
        let line = TransmissionLine::lossless(50.0, 1.0).unwrap();
        let lambda = line.wavelength(F);
        let d: Vec<f64> = (0..200).map(|i| lambda * i as f64 / 200.0).collect();
        let v = line.standing_wave(C64::new(150.0, 0.0), &d, F);
        let (vmax, vmin) = v.iter().fold((0.0f64, f64::INFINITY), |(a, b), &x| (a.max(x), b.min(x)));
        assert!((vmax / vmin - 3.0).abs() < 1e-3);
        // A resistive load above Z₀ puts a voltage maximum at the load.
        assert!((v[0] - vmax).abs() < 1e-9);
    }

    #[test]
    fn lossy_lines_and_chain_matrices() {
        // RG-58-like parameters: Z₀ near 50 Ω, a long line looks like Z₀ from the input.
        let line = TransmissionLine::distributed(0.5, 250e-9, 1e-6, 100e-12).unwrap();
        let z0 = line.characteristic_impedance(F);
        assert!((z0.re - 50.0).abs() < 0.1 && z0.im.abs() < 0.1);
        let far = line.input_impedance(C64::new(0.0, 0.0), 2000.0, F);
        assert!((far - z0).norm() < 1e-3 * z0.norm());
        assert!(line.propagation_constant(F).re > 0.0);
        // The chain matrix reproduces the input impedance: Z_in = (A Z_L + B)/(C Z_L + D).
        let load = C64::new(30.0, -20.0);
        let m = line.abcd(0.37, F);
        let zin = (m[0][0] * load + m[0][1]) / (m[1][0] * load + m[1][1]);
        assert!((zin - line.input_impedance(load, 0.37, F)).norm() < 1e-9);
        assert!(TransmissionLine::lossless(-1.0, 0.5).is_err());
    }

    #[test]
    fn single_stub_matches_cancel_the_reflection() {
        let z0 = 50.0;
        let load = C64::new(25.0, 40.0);
        let line = TransmissionLine::lossless(z0, 1.0).unwrap();
        let lambda = line.wavelength(F);
        for (d, s) in single_stub_match(load, z0).unwrap() {
            let at_stub = line.input_impedance(load, d * lambda, F);
            let stub = line.input_impedance(C64::new(0.0, 0.0), s * lambda, F);
            let total = 1.0 / (1.0 / at_stub + 1.0 / stub);
            assert!(reflection_coefficient(total, C64::new(z0, 0.0)).norm() < 1e-9, "d={d} s={s}");
        }
        assert!(single_stub_match(C64::new(-1.0, 0.0), z0).is_err());
    }
}
