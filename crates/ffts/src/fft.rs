//! [`Fft`]: the default FFT engine. Plan it once, then call [`FftEngine1D::execute`] repeatedly.

use crate::best_fft::BestFft;
use crate::bluestein_fft::BluesteinFft;
use crate::fft_engine_trait::{FftDirection, FftEngine1D, FftOrdering, FftScaleFactor};
use lib_jsl_core::ErrorsJSL;
use num::Complex;

#[derive(Debug)]
enum Backend {
    Unplanned,
    PowerOfTwo(Box<BestFft>),
    Arbitrary(Box<BluesteinFft>),
}

/// A one-dimensional complex FFT of any length that picks the best algorithm for the size:
///
/// - powers of two use [`BestFft`] (radix-4 passes, AVX2/FMA kernels where available, cache
///   blocking), which is competitive with the fastest Rust FFT libraries;
/// - every other length uses [`BluesteinFft`], which costs a few power-of-two FFTs of roughly twice
///   the length but is still O(n log n).
///
/// ```
/// use lib_jsl_ffts::{Fft, FftDirection, FftEngine1D, FftOrdering, FftScaleFactor};
/// use num::Complex;
///
/// let mut fft = Fft::new();
/// fft.plan(6, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).unwrap();
/// let spectrum = fft.execute(&[Complex::new(1.0, 0.0); 6]).unwrap();
/// assert!((spectrum[0].re - 6.0).abs() < 1e-12); // a constant signal is all DC
/// ```
///
/// `FftOrdering::BitReversed` input is only available for power-of-two sizes.
#[derive(Debug)]
pub struct Fft {
    backend: Backend,
    size: usize,
    scale: FftScaleFactor,
    direction: FftDirection,
    ordering: FftOrdering,
}

impl Fft {
    pub fn new() -> Self {
        Fft {
            backend: Backend::Unplanned,
            size: 0,
            scale: FftScaleFactor::None,
            direction: FftDirection::Forward,
            ordering: FftOrdering::Standard,
        }
    }

    /// Creates and plans an engine in one step.
    pub fn with_plan(size: usize, scale: FftScaleFactor, direction: FftDirection, ordering: FftOrdering) -> Result<Self, ErrorsJSL> {
        let mut fft = Self::new();
        fft.plan(size, scale, direction, ordering)?;
        Ok(fft)
    }
}

impl Default for Fft {
    fn default() -> Self {
        Self::new()
    }
}

impl FftEngine1D for Fft {
    fn execute(&mut self, input: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
        match &mut self.backend {
            Backend::Unplanned => Err(ErrorsJSL::InvalidInputRange("FFT must be planned before it is executed")),
            Backend::PowerOfTwo(engine) => engine.execute(input),
            Backend::Arbitrary(engine) => engine.execute(input),
        }
    }

    fn plan(&mut self, size: usize, scale: FftScaleFactor, direction: FftDirection, ordering: FftOrdering) -> Result<(), ErrorsJSL> {
        if size == 0 {
            return Err(ErrorsJSL::InvalidInputRange("Size must be greater than 0"));
        }
        let backend = if size.is_power_of_two() {
            let mut engine = BestFft::new();
            engine.plan(size, scale, direction, ordering)?;
            Backend::PowerOfTwo(Box::new(engine))
        } else {
            if matches!(ordering, FftOrdering::BitReversed) {
                return Err(ErrorsJSL::InvalidInputRange("BitReversed ordering requires a power-of-two transform size"));
            }
            let mut engine = BluesteinFft::new();
            engine.plan(size, scale, direction, ordering)?;
            Backend::Arbitrary(Box::new(engine))
        };
        // Commit only after the chosen engine planned successfully.
        self.backend = backend;
        (self.size, self.scale, self.direction, self.ordering) = (size, scale, direction, ordering);
        Ok(())
    }

    fn get_size(&self) -> usize {
        self.size
    }
    fn get_scale_factor(&self) -> FftScaleFactor {
        self.scale
    }
    fn get_direction(&self) -> FftDirection {
        self.direction
    }
    fn get_ordering(&self) -> FftOrdering {
        self.ordering
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn naive(input: &[Complex<f64>], inverse: bool) -> Vec<Complex<f64>> {
        let n = input.len();
        let sign = if inverse { 1.0 } else { -1.0 };
        (0..n)
            .map(|k| input.iter().enumerate().map(|(j, &x)| x * Complex::from_polar(1.0, sign * 2.0 * PI * ((k * j) % n) as f64 / n as f64)).sum())
            .collect()
    }

    fn signal(n: usize) -> Vec<Complex<f64>> {
        (0..n).map(|i| Complex::new((i as f64 * 0.7).sin() + 0.1 * i as f64, (i as f64 * 1.3).cos())).collect()
    }

    #[test]
    fn matches_the_dft_for_power_of_two_and_arbitrary_sizes() {
        for n in [1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16, 17, 31, 32, 63, 64, 100, 128, 255] {
            for (direction, inverse) in [(FftDirection::Forward, false), (FftDirection::Inverse, true)] {
                let x = signal(n);
                let mut fft = Fft::with_plan(n, FftScaleFactor::None, direction, FftOrdering::Standard).unwrap();
                let got = fft.execute(&x).unwrap();
                let want = naive(&x, inverse);
                let worst = got.iter().zip(&want).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
                assert!(worst < 1e-9 * n as f64, "n={n} inverse={inverse} worst={worst}");
            }
        }
    }

    #[test]
    fn scaling_round_trip_and_replanning() {
        let mut fwd = Fft::new();
        let mut inv = Fft::new();
        for n in [12usize, 64, 45, 1024] {
            fwd.plan(n, FftScaleFactor::SqrtN, FftDirection::Forward, FftOrdering::Standard).unwrap();
            inv.plan(n, FftScaleFactor::SqrtN, FftDirection::Inverse, FftOrdering::Standard).unwrap();
            let x = signal(n);
            let back = inv.execute(&fwd.execute(&x).unwrap()).unwrap();
            assert!(x.iter().zip(&back).all(|(a, b)| (a - b).norm() < 1e-9), "n={n}");
            assert_eq!(fwd.get_size(), n);
        }
    }

    #[test]
    fn errors_leave_the_previous_plan_intact() {
        let mut fft = Fft::new();
        assert!(fft.execute(&[Complex::new(1.0, 0.0)]).is_err());
        fft.plan(8, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).unwrap();
        assert!(fft.plan(0, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).is_err());
        assert!(fft.plan(12, FftScaleFactor::None, FftDirection::Forward, FftOrdering::BitReversed).is_err());
        assert_eq!(fft.get_size(), 8);
        assert!(fft.execute(&signal(8)).is_ok());
        assert!(fft.execute(&signal(9)).is_err());
    }
}
