/// Fractional Fourier transform (FRFT) utilities.
///
/// This module provides a fast one-dimensional FRFT implementation for slices
/// of data implementing [`IsAnalytic`]. The implementation follows a chirp-FFT-
/// chirp scheme, so the nontrivial cases run in `O(N log N)` time rather than
/// the `O(N^2)` cost of a direct fractional-transform matrix multiply.
///
/// The order parameter `alpha` is measured in quarter-turns:
/// - `alpha = 0` gives the identity transform
/// - `alpha = 1` gives the unitary forward DFT
/// - `alpha = 2` gives time reversal
/// - `alpha = 3` gives the unitary inverse DFT
///
/// Intermediate values of `alpha` interpolate continuously between these cases.
use std::f64::consts::PI;

use num::Complex;

use crate::{
    ffts::{
        bluestein_fft::BluesteinFft,
        fft_engine_trait::{FfftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
    },
    prelude::{C1D, ErrorsJSL, IsAnalytic},
};

const ALPHA_TOL: f64 = 1e-12;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= ALPHA_TOL
}

fn unitary_fft(
    engine: &mut BluesteinFft,
    data: &[Complex<f64>],
) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
    engine.execute(data)
}

fn wrapped_center_coordinate(index: usize, size: usize) -> f64 {
    if index < size / 2 {
        index as f64
    } else {
        index as f64 - size as f64
    }
}

/// Reusable FRFT planner/executor for a fixed transform size.
pub struct FractionalFft {
    size: usize,
    chirp_argument: Vec<Complex<f64>>,
    fft_engine: BluesteinFft,
    ifft_engine: BluesteinFft,
}

impl FractionalFft {
    pub fn new() -> Self {
        Self {
            size: 0,
            chirp_argument: Vec::new(),
            fft_engine: BluesteinFft::new(),
            ifft_engine: BluesteinFft::new(),
        }
    }

    /// Plan the internal chirp tables and unitary FFT engines for a given size.
    pub fn plan(&mut self, size: usize) -> Result<(), ErrorsJSL> {
        if size == 0 {
            return Err(ErrorsJSL::InvalidInputRange("size must be > 0"));
        }

        self.size = size;
        self.chirp_argument = (0..size)
            .map(|k| {
                let coord = wrapped_center_coordinate(k, size);
                Complex::new(0.0, PI * coord * coord / size as f64)
            })
            .collect();

        self.fft_engine.plan(
            size,
            FftScaleFactor::SqrtN,
            FftDirection::Forward,
            FftOrdering::Standard,
        )?;
        self.ifft_engine.plan(
            size,
            FftScaleFactor::SqrtN,
            FftDirection::Inverse,
            FftOrdering::Standard,
        )?;
        Ok(())
    }

    fn ensure_size(&mut self, size: usize) -> Result<(), ErrorsJSL> {
        if self.size != size {
            self.plan(size)?;
        }
        Ok(())
    }

    fn execute_quarter_turn_case(
        &mut self,
        data: &[Complex<f64>],
        alpha_mod: f64,
    ) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        if approx_eq(alpha_mod, 0.0) || approx_eq(alpha_mod, 4.0) {
            return Ok(Some(data.to_vec()));
        }
        if approx_eq(alpha_mod, 1.0) {
            return Ok(Some(unitary_fft(&mut self.fft_engine, data)?));
        }
        if approx_eq(alpha_mod, 2.0) {
            return Ok(Some(data.iter().rev().copied().collect()));
        }
        if approx_eq(alpha_mod, 3.0) {
            return Ok(Some(unitary_fft(&mut self.ifft_engine, data)?));
        }
        Ok(None)
    }

    fn normalize_alpha(
        &mut self,
        mut data: Vec<Complex<f64>>,
        alpha_mod: f64,
    ) -> Result<(Vec<Complex<f64>>, f64), ErrorsJSL> {
        if alpha_mod < 0.5 {
            data = unitary_fft(&mut self.ifft_engine, &data)?;
            Ok((data, alpha_mod + 1.0))
        } else if alpha_mod < 1.5 {
            Ok((data, alpha_mod))
        } else if alpha_mod < 2.5 {
            data = unitary_fft(&mut self.fft_engine, &data)?;
            Ok((data, alpha_mod - 1.0))
        } else if alpha_mod < 3.5 {
            data.reverse();
            Ok((data, alpha_mod - 2.0))
        } else {
            data = unitary_fft(&mut self.ifft_engine, &data)?;
            Ok((data, alpha_mod - 3.0))
        }
    }

    /// Execute the fractional Fourier transform of order `alpha`.
    ///
    /// `alpha` is periodic with period `4`.
    pub fn execute<T: IsAnalytic>(&mut self, input: &[T], alpha: f64) -> Result<C1D, ErrorsJSL> {
        if input.is_empty() {
            return Err(ErrorsJSL::InvalidInputRange("input must be non-empty"));
        }
        if !alpha.is_finite() {
            return Err(ErrorsJSL::InvalidInputRange("alpha must be finite"));
        }

        self.ensure_size(input.len())?;
        let data = input.iter().map(|x| x.to_complex()).collect::<Vec<_>>();
        let alpha_mod = alpha.rem_euclid(4.0);

        if let Some(result) = self.execute_quarter_turn_case(&data, alpha_mod)? {
            return Ok(C1D::from_vec(result));
        }

        let (data, alpha_core) = self.normalize_alpha(data, alpha_mod)?;

        let phi = alpha_core * PI / 2.0;
        let cot_phi = 1.0 / phi.tan();
        let sq_cot_phi = (1.0 + cot_phi * cot_phi).sqrt();
        let scale =
            Complex::new(1.0, -cot_phi).sqrt() / Complex::new((self.size as f64).sqrt(), 0.0);

        let chirp1 = self
            .chirp_argument
            .iter()
            .map(|arg| (*arg * (cot_phi - sq_cot_phi)).exp())
            .collect::<Vec<_>>();
        let chirp2 = self
            .chirp_argument
            .iter()
            .map(|arg| (*arg * sq_cot_phi).exp())
            .collect::<Vec<_>>();

        let weighted = chirp1
            .iter()
            .zip(data.iter())
            .map(|(c, x)| *c * *x)
            .collect::<Vec<_>>();
        let fft1 = unitary_fft(&mut self.fft_engine, &chirp2)?;
        let fft2 = unitary_fft(&mut self.fft_engine, &weighted)?;
        let product = fft1
            .iter()
            .zip(fft2.iter())
            .map(|(a, b)| *a * *b)
            .collect::<Vec<_>>();
        let ifft = unitary_fft(&mut self.ifft_engine, &product)?;

        Ok(C1D::from_vec(
            chirp1
                .iter()
                .zip(ifft.iter())
                .map(|(c, y)| scale * *c * *y)
                .collect(),
        ))
    }
}

/// Convenience wrapper for one-shot FRFT evaluation.
pub fn frft<T: IsAnalytic>(input: &[T], alpha: f64) -> Result<C1D, ErrorsJSL> {
    let mut frft_engine = FractionalFft::new();
    frft_engine.execute(input, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_complex_close(actual: Complex<f64>, expected: Complex<f64>, tol: f64) {
        assert!(
            (actual - expected).norm() < tol,
            "actual={actual}, expected={expected}, tol={tol}"
        );
    }

    #[test]
    fn test_frft_alpha_zero_is_identity() {
        let x = [
            Complex::new(1.0, 0.2),
            Complex::new(-0.5, 1.0),
            Complex::new(0.25, -0.75),
            Complex::new(2.0, 0.0),
            Complex::new(-1.5, 0.4),
        ];
        let y = frft(&x, 0.0).unwrap();
        assert_eq!(y.len(), x.len());
        for (actual, expected) in y.iter().zip(x.iter()) {
            assert_complex_close(*actual, *expected, 1e-12);
        }
    }

    #[test]
    fn test_frft_alpha_one_matches_unitary_fft() {
        let x = [
            Complex::new(1.0, 0.0),
            Complex::new(-0.5, 0.75),
            Complex::new(0.25, -0.25),
            Complex::new(2.0, 1.0),
            Complex::new(-1.0, 0.5),
        ];
        let y = frft(&x, 1.0).unwrap();

        let mut fft = BluesteinFft::new();
        fft.plan(
            x.len(),
            FftScaleFactor::SqrtN,
            FftDirection::Forward,
            FftOrdering::Standard,
        )
        .unwrap();
        let expected = fft.execute(&x).unwrap();

        for (actual, expected) in y.iter().zip(expected.iter()) {
            assert_complex_close(*actual, *expected, 1e-12);
        }
    }

    #[test]
    fn test_frft_alpha_two_is_reversal() {
        let x = [1.0, -0.5, 0.25, 2.0, -1.25, 0.4];
        let y = frft(&x, 2.0).unwrap();
        let expected = x
            .iter()
            .rev()
            .map(|&v| Complex::new(v, 0.0))
            .collect::<Vec<_>>();
        for (actual, expected) in y.iter().zip(expected.iter()) {
            assert_complex_close(*actual, *expected, 1e-12);
        }
    }

    #[test]
    fn test_frft_is_periodic_mod_four() {
        let x = [
            Complex::new(1.0, 0.1),
            Complex::new(-0.5, 0.75),
            Complex::new(0.25, -0.25),
            Complex::new(2.0, 1.0),
            Complex::new(-1.0, 0.5),
            Complex::new(0.3, -0.2),
        ];
        let alpha = 0.73;
        let y0 = frft(&x, alpha).unwrap();
        let y1 = frft(&x, alpha + 4.0).unwrap();

        for (actual, expected) in y0.iter().zip(y1.iter()) {
            assert_complex_close(*actual, *expected, 1e-6);
        }
    }
}
