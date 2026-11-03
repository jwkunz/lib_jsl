//! The Bessel (Thomson) analogue prototype.
//!
//! See the book entry [Analogue prototypes: Butterworth, Chebyshev, elliptic and Bessel](https://jwkunz.github.io/lib_jsl/book/filters/analog-prototypes.html#where-the-poles-are).

use ndarray::Array1;
use num_complex::Complex;

use crate::{number_theory::polynomial::polynomial_roots, prelude::ErrorsJSL};

use super::design::Zpk;

/// Normalization choices for Bessel/Thomson prototypes.
///
/// These mirror SciPy's options:
/// `Phase` is the MATLAB-compatible default,
/// `Delay` is the natural delay-normalized prototype,
/// and `Mag` shifts the prototype so the `-3 dB` point lands at `1 rad/s`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BesselNorm {
    Phase,
    Delay,
    Mag,
}

fn falling_factorial(x: usize, n: usize) -> usize {
    let mut value = 1usize;
    for k in (x - n + 1)..=x {
        value *= k;
    }
    value
}

fn bessel_poly(order: usize, reverse: bool) -> Vec<usize> {
    let mut coeffs = Vec::with_capacity(order + 1);
    for k in 0..=order {
        let num = falling_factorial(2 * order - k, order);
        let den = (1usize << (order - k)) * (1..=k).product::<usize>().max(1);
        coeffs.push(num / den);
    }
    if reverse {
        coeffs.reverse();
    }
    coeffs
}

fn bessel_mag_norm_factor(poles: &[Complex<f64>], gain: f64) -> Result<f64, ErrorsJSL> {
    // The magnitude-normalized Bessel prototype is obtained by starting from
    // the delay-normalized version and then finding the frequency scale that
    // places the -3 dB point at 1 rad/s.
    let gain_mag = |w: f64| -> f64 {
        let s = Complex::new(0.0, w);
        let den = poles
            .iter()
            .fold(Complex::new(1.0, 0.0), |acc, &p| acc * (s - p));
        gain / den.norm()
    };

    let target = std::f64::consts::FRAC_1_SQRT_2;
    let mut lo = 1e-6;
    let mut hi = 1.5;
    while gain_mag(hi) > target && hi < 1e6 {
        hi *= 2.0;
    }
    if hi >= 1e6 {
        return Err(ErrorsJSL::RuntimeError(
            "failed to bracket Bessel magnitude normalization factor",
        ));
    }

    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if gain_mag(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Bessel/Thomson analog lowpass prototype (`besselap`) with selectable
/// normalization.
///
/// The Bessel family is all-pole and is valued for nearly linear phase and
/// maximally flat group delay in the passband. The underlying prototype comes
/// from the reverse Bessel polynomial. Different normalizations simply rescale
/// the same pole pattern for different cutoff conventions.
pub fn besselap(order: usize, norm: Option<BesselNorm>) -> Result<Zpk, ErrorsJSL> {
    let norm = norm.unwrap_or(BesselNorm::Phase);
    if order == 0 {
        return Ok(Zpk {
            z: Array1::from_vec(vec![]),
            p: Array1::from_vec(vec![]),
            k: Complex::new(1.0, 0.0),
        });
    }

    let reverse_coeffs = bessel_poly(order, true)
        .into_iter()
        .map(|x| Complex::new(x as f64, 0.0))
        .collect::<Vec<_>>();
    let mut poles = polynomial_roots(&reverse_coeffs)?;
    let a_last = (falling_factorial(2 * order, order) / (1usize << order)) as f64;

    // `Delay` uses the natural reverse-Bessel scaling directly. `Phase` rescales
    // the poles so the phase midpoint sits at 1 rad/s, matching SciPy/MATLAB.
    // `Mag` starts from the delay-normalized prototype and applies an extra
    // frequency shift so the magnitude is -3 dB at 1 rad/s.
    let gain = match norm {
        BesselNorm::Delay => Complex::new(a_last, 0.0),
        BesselNorm::Mag => {
            let norm_factor = bessel_mag_norm_factor(poles.as_slice().unwrap_or(&[]), a_last)?;
            poles = poles.mapv(|p| p / norm_factor);
            Complex::new(a_last / norm_factor.powi(order as i32), 0.0)
        }
        BesselNorm::Phase => {
            let phase_scale = a_last.powf(-1.0 / order as f64);
            poles = poles.mapv(|p| p * phase_scale);
            Complex::new(1.0, 0.0)
        }
    };

    Ok(Zpk {
        z: Array1::from_vec(vec![]),
        p: poles,
        k: gain,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continuous::filter::design::zpk_to_tf;
    use lib_jsl_number_theory::polynomial::polynomial_eval;

    #[test]
    fn test_besselap_phase_norm_has_unity_gain() {
        let zpk = besselap(3, Some(BesselNorm::Phase)).unwrap();
        assert!((zpk.k - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_besselap_delay_norm_has_expected_gain_constant() {
        let zpk = besselap(3, Some(BesselNorm::Delay)).unwrap();
        assert!((zpk.k - Complex::new(15.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_besselap_mag_norm_hits_minus_three_db_at_unity_frequency() {
        let zpk = besselap(4, Some(BesselNorm::Mag)).unwrap();
        let tf = zpk_to_tf(&zpk).unwrap();
        let s = Complex::new(0.0, 1.0);
        let h = polynomial_eval(tf.b.as_slice().unwrap_or(&[]), s)
            / polynomial_eval(tf.a.as_slice().unwrap_or(&[]), s);
        let attenuation_db = -20.0 * h.norm().log10();
        let expected_db = 10.0 * 2.0_f64.log10();
        assert!((attenuation_db - expected_db).abs() < 1e-6);
    }
}
