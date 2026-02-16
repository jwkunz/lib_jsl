/// This file defines the trait for FFT engines, which are used to perform FFTs in the spectral methods.
/// The trait defines the required methods for an FFT engine, such as forward and inverse FFTs, and the size of the FFT. This allows for different implementations of FFT engines to be used interchangeably in the spectral methods, as long as they implement the trait.
/// The trait is defined in a separate file to keep the code organized and modular, and to allow for easy extension in the future if new FFT engines are added.
/// This trait can be implemented by different FFT engines, such as the RustFFT library, or a custom implementation. The spectral methods can then use any FFT engine that implements this trait, allowing for flexibility and modularity in the codebase.  
/// The trait is defined in the `fft_enginer_trait.rs` file, and is used in the spectral methods to perform FFTs on the input data. The spectral methods can then be used for various applications, such as solving partial differential equations, performing signal processing, and more.     

#[derive(Clone,Copy,Debug)]
pub enum FftDirection {
    Forward, // exp(-2*pi*i*k*n/N)
    Inverse, // exp(2*pi*i*k*n/N)
}

#[derive(Clone,Copy,Debug)]
pub enum FftScaleFactor {
    None,
    SqrtN,
    N,
}

#[derive(Clone,Copy,Debug)]
pub enum FftOrdering {
    Standard, // k = 0, 1, ..., N-1
    BitReversed,  // k = -N/2, ..., -1, 0, 1, ..., N/2-1
}


use num::Complex;

use crate::prelude::{ErrorsJSL};
pub trait FfftEngine1D {
    fn execute(&mut self, input: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, ErrorsJSL>;
    fn plan(&mut self, size: usize, scale: FftScaleFactor, direction: FftDirection, ordering: FftOrdering) -> Result<(), ErrorsJSL>;
    fn get_size(&self) -> usize;
    fn get_scale_factor(&self) -> FftScaleFactor;
    fn get_direction(&self) -> FftDirection;
    fn get_ordering(&self) -> FftOrdering;
}