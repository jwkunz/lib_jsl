//! Test waveforms as functions of time, following `scipy.signal`: `sawtooth`, `square` (with a
//! duty cycle), `gausspulse`, `chirp` (linear, quadratic, logarithmic, hyperbolic), `sweep_poly`
//! and `unit_impulse`.
//!
//! See the book entry [Partial fractions, zoom FFT and waveforms](https://jwkunz.github.io/lib_jsl/book/spectral/partial-fractions-waveforms.html#waveforms).

use lib_jsl_core::ErrorsJSL;
use std::f64::consts::{PI, TAU};

/// A sawtooth of period `2π` rising from −1 to 1 over the first `width · 2π` and falling back
/// over the rest (SciPy's `sawtooth`); `width = 0.5` gives a triangle wave. `NaN` for a width
/// outside `[0, 1]`.
pub fn sawtooth(t: &[f64], width: f64) -> Vec<f64> {
    t.iter()
        .map(|&ti| {
            if !(0.0..=1.0).contains(&width) {
                return f64::NAN;
            }
            let m = ti.rem_euclid(TAU);
            if m < width * TAU { m / (PI * width) - 1.0 } else { (PI * (width + 1.0) - m) / (PI * (1.0 - width)) }
        })
        .collect()
}

/// A square wave of period `2π`: +1 for the first `duty · 2π`, −1 after (SciPy's `square`).
pub fn square_wave(t: &[f64], duty: f64) -> Vec<f64> {
    t.iter()
        .map(|&ti| {
            if !(0.0..=1.0).contains(&duty) {
                return f64::NAN;
            }
            if ti.rem_euclid(TAU) < duty * TAU { 1.0 } else { -1.0 }
        })
        .collect()
}

/// The in-phase part, quadrature part and envelope of a Gaussian-modulated sinusoid.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussPulse {
    pub in_phase: Vec<f64>,
    pub quadrature: Vec<f64>,
    pub envelope: Vec<f64>,
}

fn gauss_a(fc: f64, bw: f64, bwr: f64) -> Result<f64, ErrorsJSL> {
    if fc < 0.0 || bw <= 0.0 || bwr >= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("need fc ≥ 0, bw > 0 and bwr < 0 dB"));
    }
    let reference = 10f64.powf(bwr / 20.0);
    Ok(-(PI * fc * bw).powi(2) / (4.0 * reference.ln()))
}

// ANCHOR: gausspulse
/// `exp(−a t²) exp(2πi f_c t)` (SciPy's `gausspulse`), with `a` chosen so that the spectrum's
/// width at `bwr` dB (−6 by default) is the fraction `bw` of `f_c`.
pub fn gausspulse(t: &[f64], fc: f64, bw: f64, bwr: f64) -> Result<GaussPulse, ErrorsJSL> {
    let a = gauss_a(fc, bw, bwr)?;
    let envelope: Vec<f64> = t.iter().map(|&ti| (-a * ti * ti).exp()).collect();
    Ok(GaussPulse {
        in_phase: t.iter().zip(&envelope).map(|(&ti, e)| e * (TAU * fc * ti).cos()).collect(),
        quadrature: t.iter().zip(&envelope).map(|(&ti, e)| e * (TAU * fc * ti).sin()).collect(),
        envelope,
    })
}
// ANCHOR_END: gausspulse

/// The time after which the envelope of [`gausspulse`] stays below `tpr` dB (SciPy's
/// `gausspulse('cutoff', …)`).
pub fn gausspulse_cutoff(fc: f64, bw: f64, bwr: f64, tpr: f64) -> Result<f64, ErrorsJSL> {
    if tpr >= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("the cutoff level must be below 0 dB"));
    }
    let a = gauss_a(fc, bw, bwr)?;
    Ok((-(10f64.powf(tpr / 20.0)).ln() / a).sqrt())
}

/// How the frequency of a [`chirp`] moves from `f0` at `t = 0` to `f1` at `t = t1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChirpMethod {
    Linear,
    /// Quadratic, with the parabola's vertex at `t = 0` (`vertex_zero = true`) or at `t1`.
    Quadratic { vertex_zero: bool },
    /// Geometric: `f(t) = f0 (f1/f0)^{t/t1}`.
    Logarithmic,
    /// `f(t) = f0 f1 t1 / ((f0 − f1) t + f1 t1)`.
    Hyperbolic,
}

// ANCHOR: chirp
/// A swept-frequency cosine `cos(φ(t) + φ₀)` (SciPy's `chirp`, `phi` in degrees), where the phase
/// `φ(t) = 2π ∫₀ᵗ f(τ) dτ` integrates the instantaneous frequency of the chosen sweep.
pub fn chirp(t: &[f64], f0: f64, t1: f64, f1: f64, method: ChirpMethod, phi: f64) -> Result<Vec<f64>, ErrorsJSL> {
    let phase = |ti: f64| -> Result<f64, ErrorsJSL> {
        Ok(match method {
            ChirpMethod::Linear => TAU * (f0 * ti + 0.5 * (f1 - f0) / t1 * ti * ti),
            ChirpMethod::Quadratic { vertex_zero } => {
                let beta = (f1 - f0) / (t1 * t1);
                if vertex_zero { TAU * (f0 * ti + beta * ti.powi(3) / 3.0) } else { TAU * (f1 * ti + beta * ((t1 - ti).powi(3) - t1.powi(3)) / 3.0) }
            }
            ChirpMethod::Logarithmic => {
                if f0 * f1 <= 0.0 {
                    return Err(ErrorsJSL::InvalidInputRange("a logarithmic chirp needs f0 and f1 of the same sign"));
                }
                if f0 == f1 { TAU * f0 * ti } else { TAU * t1 / (f1 / f0).ln() * f0 * ((f1 / f0).powf(ti / t1) - 1.0) }
            }
            ChirpMethod::Hyperbolic => {
                if f0 == 0.0 || f1 == 0.0 {
                    return Err(ErrorsJSL::InvalidInputRange("a hyperbolic chirp needs non-zero f0 and f1"));
                }
                if f0 == f1 {
                    TAU * f0 * ti
                } else {
                    let sing = -f1 * t1 / (f0 - f1);
                    TAU * (-sing * f0) * (1.0 - ti / sing).abs().ln()
                }
            }
        })
    };
    t.iter().map(|&ti| Ok((phase(ti)? + phi.to_radians()).cos())).collect()
}
// ANCHOR_END: chirp

/// A cosine whose instantaneous frequency is the polynomial `poly(t)` (highest power first;
/// SciPy's `sweep_poly`): `cos(2π ∫₀ᵗ poly + φ₀)`, `phi` in degrees.
pub fn sweep_poly(t: &[f64], poly: &[f64], phi: f64) -> Vec<f64> {
    let n = poly.len();
    // The antiderivative, highest power first, with zero constant term.
    let integral: Vec<f64> = poly.iter().enumerate().map(|(i, c)| c / (n - i) as f64).chain(std::iter::once(0.0)).collect();
    t.iter().map(|&ti| (TAU * integral.iter().fold(0.0, |acc, &c| acc * ti + c) + phi.to_radians()).cos()).collect()
}

/// A unit impulse of length `n` at index `idx` (SciPy's `unit_impulse`).
pub fn unit_impulse(n: usize, idx: usize) -> Vec<f64> {
    let mut v = vec![0.0; n];
    if idx < n {
        v[idx] = 1.0;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes() {
        let t: Vec<f64> = (0..8).map(|i| i as f64 * PI / 4.0).collect();
        assert_eq!(sawtooth(&t, 1.0)[0], -1.0);
        assert!((sawtooth(&t, 0.5)[2] - 0.0).abs() < 1e-15);
        assert_eq!(square_wave(&t, 0.5), vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]);
        let g = gausspulse(&[0.0], 1000.0, 0.5, -6.0).unwrap();
        assert_eq!((g.in_phase[0], g.envelope[0]), (1.0, 1.0));
        // At the cutoff time the envelope is exactly tpr dB.
        let tc = gausspulse_cutoff(1000.0, 0.5, -6.0, -60.0).unwrap();
        let e = gausspulse(&[tc], 1000.0, 0.5, -6.0).unwrap().envelope[0];
        assert!((20.0 * e.log10() + 60.0).abs() < 1e-10);
    }

    #[test]
    fn a_linear_chirp_is_a_linear_sweep_poly() {
        // f(t) = 10 + 15 t sweeps from 10 Hz to 40 Hz over 2 s.
        let lin = chirp(&[0.3, 1.1], 10.0, 2.0, 40.0, ChirpMethod::Linear, 0.0).unwrap();
        let poly = sweep_poly(&[0.3, 1.1], &[15.0, 10.0], 0.0);
        assert!(lin.iter().zip(&poly).all(|(a, b)| (a - b).abs() < 1e-12));
        for method in [ChirpMethod::Quadratic { vertex_zero: false }, ChirpMethod::Logarithmic, ChirpMethod::Hyperbolic] {
            assert!(chirp(&[0.0, 0.7, 2.0], 10.0, 2.0, 40.0, method, 0.0).unwrap().iter().all(|v| v.abs() <= 1.0));
        }
    }
}
