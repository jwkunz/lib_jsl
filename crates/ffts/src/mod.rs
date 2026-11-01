//! Fast Fourier transforms.
//!
//! # Which FFT should I use?
//!
//! Use [`Fft`]. It handles any length, picks the right algorithm for it, and is fast:
//!
//! ```
//! use lib_jsl_ffts::{Fft, FftDirection, FftEngine1D, FftOrdering, FftScaleFactor};
//! use num::Complex;
//!
//! let mut fft = Fft::new();
//! fft.plan(1024, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).unwrap();
//! let spectrum = fft.execute(&vec![Complex::new(0.0, 0.0); 1024]).unwrap();
//! # assert_eq!(spectrum.len(), 1024);
//! ```
//!
//! Plan once and reuse the engine; planning builds the twiddle tables.
//!
//! # What else is here
//!
//! | Item | Use it when |
//! |------|-------------|
//! | [`Fft`] | You want a transform and don't care how it is computed (default) |
//! | [`best_fft::BestFft`] | You know the size is a power of two and want that engine directly |
//! | [`bluestein_fft::BluesteinFft`] | You want the arbitrary-length algorithm directly |
//! | [`simd_fft::SimdFft`] | You need vectorised kernels on ARM or RISC-V as well as x86 |
//! | [`fractional_fft::FractionalFft`] | You need the fractional Fourier transform |
//! | [`fft_engine_trait::FftEngine1D`] | You want to write code generic over FFT engines |
//!
//! Readable textbook implementations (recursive radix-2, iterative radix-2, split radix) live in
//! the separate `lib_jsl_ffts_reference` crate, so they stay out of the way here.

pub mod fft;
pub mod fft_engine_trait;

pub mod best_fft;
pub mod bluestein_fft;
pub mod dct;
pub mod fractional_fft;
pub mod multidimensional_fft;
pub mod real_fft;
pub mod simd_fft;

pub use best_fft::BestFft;
pub use bluestein_fft::BluesteinFft;
pub use dct::{TrigNormalization, TrigTransformType, dct, dst, idct, idst};
pub use fft::Fft;
pub use fft_engine_trait::{FftDirection, FftEngine1D, FftOrdering, FftScaleFactor};
pub use fractional_fft::{FractionalFft, frft};
pub use multidimensional_fft::{fft2, fft3, fft_axis, fftn, fftshift, ifft2, ifft3};
pub use real_fft::{RealFft, irfft, rfft};
pub use simd_fft::SimdFft;

#[cfg(test)]
pub mod test_bench_data;
