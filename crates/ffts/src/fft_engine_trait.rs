//! The interface shared by the one-dimensional FFT engines: plan once for a size, scaling,
//! direction and input ordering, then execute repeatedly.

/// The sign of the exponent: `Forward` computes `X_k = Σ x_n e^{−2πi kn/N}`, `Inverse` uses `e^{+2πi kn/N}`.
#[derive(Clone,Copy,Debug)]
pub enum FftDirection {
    Forward, // exp(-2*pi*i*k*n/N)
    Inverse, // exp(2*pi*i*k*n/N)
}

/// The factor applied to the output: none, `1/√N` (the unitary transform) or `1/N`.
#[derive(Clone,Copy,Debug)]
pub enum FftScaleFactor {
    None,
    SqrtN,
    N,
}

/// The order of the input samples (the output is always in natural order).
#[derive(Clone,Copy,Debug)]
pub enum FftOrdering {
    Standard, // k = 0, 1, ..., N-1
    BitReversed,  // input samples arrive in bit-reversed index order (output is in natural order)
}


use num_complex::Complex;

use lib_jsl_core::{ErrorsJSL};
/// A planned one-dimensional complex FFT.
pub trait FftEngine1D {
    fn execute(&mut self, input: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, ErrorsJSL>;
    fn plan(&mut self, size: usize, scale: FftScaleFactor, direction: FftDirection, ordering: FftOrdering) -> Result<(), ErrorsJSL>;
    fn get_size(&self) -> usize;
    fn get_scale_factor(&self) -> FftScaleFactor;
    fn get_direction(&self) -> FftDirection;
    fn get_ordering(&self) -> FftOrdering;
}