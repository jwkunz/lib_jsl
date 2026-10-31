//! Readable reference FFTs, kept for study and as benchmark baselines.
//!
//! **If you just want an FFT, use [`lib_jsl_ffts::Fft`] instead.** The engines here trade speed
//! for clarity, are limited to power-of-two sizes, and exist to show how the algorithms work:
//!
//! 1. [`simple_cooley_tukey`]: the textbook recursive radix-2 decimation-in-time algorithm.
//! 2. [`optimized_radix2`]: the same algorithm made iterative, with precomputed twiddle factors
//!    and bit-reversal indexing.
//! 3. [`optimized_split_radix`]: the split-radix variant, which needs fewer multiplications.
//!
//! They all implement [`lib_jsl_ffts::fft_engine_trait::FftEngine1D`], so they are drop-in
//! interchangeable with the production engines. Run `cargo bench -p lib_jsl_ffts_reference` to
//! see how much the production engines gain from vectorization, radix-4 passes and cache blocking.

pub mod optimized_radix2;
pub mod optimized_split_radix;
pub mod simple_cooley_tukey;
#[cfg(test)]
mod test_bench_data;
