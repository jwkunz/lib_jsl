//! Quadrature (complex) source: cosine on the real axis, sine on the imaginary axis.
//!
//! See the book entry [Signal sources: oscillators, CORDIC and waveforms](https://jwkunz.github.io/lib_jsl/book/comms/sources.html#waveforms-and-noise).

use super::nco::{Nco, NcoSource, Phase, Waveform};
use super::sinusoid::SineKernel;
use super::cordic::{Cordic32, Cordic64};
use lib_jsl_core::ErrorsJSL;
use num_complex::Complex;

/// The sense in which the quadrature output rotates, i.e. the sign of its imaginary (sine) part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Polarity {
    /// `cos θ + j sin θ = e^{+jθ}`: rotates counter-clockwise, so a positive oscillator frequency
    /// appears at a positive frequency in the spectrum.
    Positive,
    /// `cos θ − j sin θ = e^{−jθ}`: the complex conjugate; rotates clockwise, which mirrors the
    /// spectrum (the same as running the oscillator at the negative frequency).
    Negative,
}

impl Polarity {
    /// `+1.0` for [`Positive`](Polarity::Positive), `-1.0` for [`Negative`](Polarity::Negative).
    pub fn sign(self) -> f64 {
        match self {
            Polarity::Positive => 1.0,
            Polarity::Negative => -1.0,
        }
    }
}

/// The quadrature waveform: `cos(2π·phase) + j·polarity·sin(2π·phase)`, from one shared kernel
/// evaluation (a single CORDIC pass yields both components).
#[derive(Clone, Debug)]
pub struct QuadratureWave {
    kernel: SineKernel,
    polarity: Polarity,
}

impl QuadratureWave {
    pub fn new(kernel: SineKernel, polarity: Polarity) -> QuadratureWave {
        QuadratureWave { kernel, polarity }
    }

    pub fn polarity(&self) -> Polarity {
        self.polarity
    }

    /// Changes the polarity; takes effect from the next sample, with the phase unbroken.
    pub fn set_polarity(&mut self, polarity: Polarity) {
        self.polarity = polarity;
    }
}

impl Waveform for QuadratureWave {
    type Output = Complex<f64>;

    fn sample(&mut self, phase: Phase) -> Complex<f64> {
        let (s, c) = self.kernel.sin_cos(phase);
        Complex::new(c, self.polarity.sign() * s)
    }
}

impl NcoSource<QuadratureWave> {
    /// A quadrature source at `frequency_hz` using floating-point `sin`/`cos`.
    pub fn quadrature(frequency_hz: f64, sample_rate: f64, polarity: Polarity) -> Result<Self, ErrorsJSL> {
        Ok(NcoSource::from_parts(Nco::new(frequency_hz, sample_rate)?, QuadratureWave::new(SineKernel::Direct, polarity)))
    }

    /// A quadrature source whose sine and cosine come from a single 32-bit integer CORDIC pass
    /// with `iterations` (1 to 30) micro-rotations (accurate to about 1e-8).
    pub fn quadrature_cordic(frequency_hz: f64, sample_rate: f64, polarity: Polarity, iterations: u32) -> Result<Self, ErrorsJSL> {
        let kernel = SineKernel::Cordic(Cordic32::new(iterations)?);
        Ok(NcoSource::from_parts(Nco::new(frequency_hz, sample_rate)?, QuadratureWave::new(kernel, polarity)))
    }

    /// Like [`quadrature_cordic`](Self::quadrature_cordic) but with the 64-bit CORDIC
    /// (`iterations` 1 to 62; 52 or more reaches double-precision accuracy) driven by the
    /// oscillator's full 64-bit phase.
    pub fn quadrature_cordic64(frequency_hz: f64, sample_rate: f64, polarity: Polarity, iterations: u32) -> Result<Self, ErrorsJSL> {
        let kernel = SineKernel::Cordic64(Cordic64::new(iterations)?);
        Ok(NcoSource::from_parts(Nco::new(frequency_hz, sample_rate)?, QuadratureWave::new(kernel, polarity)))
    }

    /// The current polarity.
    pub fn polarity(&self) -> Polarity {
        self.waveform().polarity()
    }

    /// Changes the polarity; the phase continues without a jump.
    pub fn set_polarity(&mut self, polarity: Polarity) {
        self.waveform_mut().set_polarity(polarity);
    }
}

/// Convenience: `length` samples of a unit complex tone at `frequency_hz`.
pub fn quadrature(frequency_hz: f64, sample_rate: f64, length: usize, polarity: Polarity) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
    Ok(NcoSource::quadrature(frequency_hz, sample_rate, polarity)?.generate(length))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::stream_operator::{StreamOperator, StreamOperatorManagement};
    use std::f64::consts::TAU;

    /// Magnitude of the DFT of `x` at normalized frequency `f` (cycles per sample).
    fn dft_at(x: &[Complex<f64>], f: f64) -> f64 {
        let sum: Complex<f64> = x.iter().enumerate().map(|(n, v)| v * Complex::from_polar(1.0, -TAU * f * n as f64)).sum();
        sum.norm() / x.len() as f64
    }

    #[test]
    fn positive_polarity_is_the_complex_exponential() {
        let x = quadrature(1000.0, 8000.0, 100, Polarity::Positive).unwrap();
        for (n, v) in x.iter().enumerate() {
            let want = Complex::from_polar(1.0, TAU * 0.125 * n as f64);
            assert!((v - want).norm() < 1e-9, "n = {n}");
        }
        assert!((x[0] - Complex::new(1.0, 0.0)).norm() < 1e-12);
        assert!((x[2] - Complex::new(0.0, 1.0)).norm() < 1e-9, "a quarter turn is +j");
    }

    #[test]
    fn negative_polarity_is_the_conjugate() {
        let pos = quadrature(1234.0, 10_000.0, 200, Polarity::Positive).unwrap();
        let neg = quadrature(1234.0, 10_000.0, 200, Polarity::Negative).unwrap();
        for (p, n) in pos.iter().zip(&neg) {
            assert_eq!(p.re, n.re);
            assert_eq!(p.im, -n.im);
        }
        assert_eq!(Polarity::Positive.sign(), 1.0);
        assert_eq!(Polarity::Negative.sign(), -1.0);
    }

    #[test]
    fn spectrum_is_single_sided_and_polarity_mirrors_it() {
        let n = 800;
        let f = 0.1;
        let pos = quadrature(f * 8000.0, 8000.0, n, Polarity::Positive).unwrap();
        let neg = quadrature(f * 8000.0, 8000.0, n, Polarity::Negative).unwrap();
        assert!((dft_at(&pos, f) - 1.0).abs() < 1e-9, "all the energy at +f");
        assert!(dft_at(&pos, -f) < 1e-9, "no image at -f");
        assert!((dft_at(&neg, -f) - 1.0).abs() < 1e-9);
        assert!(dft_at(&neg, f) < 1e-9);
        // Negative polarity equals negative frequency.
        let backwards = quadrature(-f * 8000.0, 8000.0, n, Polarity::Positive).unwrap();
        for (a, b) in neg.iter().zip(&backwards) {
            assert!((a - b).norm() < 1e-9);
        }
    }

    #[test]
    fn cordic_kernel_gives_both_components_with_unit_magnitude() {
        let mut direct = NcoSource::quadrature(777.7, 48_000.0, Polarity::Positive).unwrap();
        let mut cordic = NcoSource::quadrature_cordic(777.7, 48_000.0, Polarity::Positive, 30).unwrap();
        for _ in 0..5000 {
            let (d, c) = (direct.next_sample(), cordic.next_sample());
            assert!((d - c).norm() < 5e-8);
            assert!((c.norm() - 1.0).abs() < 2e-8);
        }
        // The single-sided spectrum survives the fixed-point arithmetic (image below ~ -150 dBc).
        // (800 Hz at 8 kHz: 36-degree steps and exactly 80 cycles in 800 samples, so no leakage.)
        let x = NcoSource::quadrature_cordic(800.0, 8000.0, Polarity::Positive, 30).unwrap().generate(800);
        assert!(dft_at(&x, -0.1) < 5e-8 && (dft_at(&x, 0.1) - 1.0).abs() < 5e-8);
        assert!(NcoSource::quadrature_cordic(1.0, 10.0, Polarity::Positive, 0).is_err());
        // A coarse rotator is visibly worse.
        let coarse = NcoSource::quadrature_cordic(800.0, 8000.0, Polarity::Positive, 8).unwrap().generate(800);
        assert!(dft_at(&coarse, -0.1) > 1e-4, "image {}", dft_at(&coarse, -0.1));
    }

    #[test]
    fn cordic64_quadrature_is_double_precision_clean() {
        // 800 Hz at 8 kHz: exactly 80 cycles in 800 samples, so the image bin has no leakage.
        let x = NcoSource::quadrature_cordic64(800.0, 8000.0, Polarity::Positive, 52).unwrap().generate(800);
        assert!(dft_at(&x, -0.1) < 5e-15, "image {}", dft_at(&x, -0.1));
        assert!((dft_at(&x, 0.1) - 1.0).abs() < 5e-15);
        for v in &x {
            assert!((v.norm() - 1.0).abs() < 3e-15);
        }
        let mut direct = NcoSource::quadrature(333.3, 8000.0, Polarity::Negative).unwrap();
        let mut wide = NcoSource::quadrature_cordic64(333.3, 8000.0, Polarity::Negative, 52).unwrap();
        for _ in 0..3000 {
            assert!((direct.next_sample() - wide.next_sample()).norm() < 5e-15);
        }
        assert!(NcoSource::quadrature_cordic64(1.0, 10.0, Polarity::Positive, 63).is_err());
    }

    #[test]
    fn polarity_can_be_switched_mid_stream_without_a_phase_jump() {
        let mut s = NcoSource::quadrature(1000.0, 8000.0, Polarity::Positive).unwrap();
        let before = s.generate(5);
        s.set_polarity(Polarity::Negative);
        assert_eq!(s.polarity(), Polarity::Negative);
        let after = s.generate(5);
        // Real part continues the cosine; the imaginary part flips sign relative to the sine.
        for (k, v) in after.iter().enumerate() {
            let n = (5 + k) as f64;
            assert!((v.re - (TAU * 0.125 * n).cos()).abs() < 1e-9);
            assert!((v.im + (TAU * 0.125 * n).sin()).abs() < 1e-9);
        }
        assert!((before[4].re - (TAU * 0.5).cos()).abs() < 1e-9);
    }

    #[test]
    fn stream_operator_amplitude_offset_and_modulation() {
        let mut s = NcoSource::quadrature(500.0, 8000.0, Polarity::Positive).unwrap();
        let a = StreamOperator::<(), Complex<f64>>::process(&mut s, &[(); 64]).unwrap().unwrap();
        assert_eq!(a, quadrature(500.0, 8000.0, 64, Polarity::Positive).unwrap());
        StreamOperatorManagement::reset(&mut s).unwrap();
        assert_eq!(StreamOperator::<(), Complex<f64>>::process(&mut s, &[(); 64]).unwrap().unwrap(), a);
        assert!(StreamOperator::<(), Complex<f64>>::flush(&mut s).unwrap().is_none());
        assert!(StreamOperator::<(), Complex<f64>>::process(&mut s, &[]).unwrap().is_none());

        // Scaling and a complex DC offset.
        let mut scaled = NcoSource::quadrature(1000.0, 8000.0, Polarity::Positive).unwrap().with_amplitude(2.0).with_offset(Complex::new(1.0, -3.0));
        let v = scaled.generate(3);
        assert!((v[0] - Complex::new(3.0, -3.0)).norm() < 1e-12);
        assert!((v[2] - Complex::new(1.0, -1.0)).norm() < 1e-9);

        // FM: +250 Hz of offset on a 750 Hz oscillator equals a 1 kHz tone.
        let mut fm = NcoSource::quadrature(750.0, 8000.0, Polarity::Positive).unwrap();
        let out = StreamOperator::<f64, Complex<f64>>::process(&mut fm, &[250.0; 32]).unwrap().unwrap();
        let reference = quadrature(1000.0, 8000.0, 32, Polarity::Positive).unwrap();
        for (a, b) in out.iter().zip(&reference) {
            assert!((a - b).norm() < 1e-9);
        }
    }

    #[test]
    fn quarter_turn_start_swaps_the_components() {
        // Starting a quarter turn ahead turns (cos, sin) into (-sin, cos): multiplication by +j.
        let base = quadrature(900.0, 8000.0, 20, Polarity::Positive).unwrap();
        let ahead = NcoSource::quadrature(900.0, 8000.0, Polarity::Positive).unwrap().with_phase_turns(0.25).generate(20);
        for (b, a) in base.iter().zip(&ahead) {
            assert!((a - b * Complex::new(0.0, 1.0)).norm() < 1e-9);
        }
    }
}
