//! Random number generation: uniform generators, distribution samplers, quasi-random sequences and Markov chain Monte Carlo.
//!
//! See the book chapter [Random numbers](https://jwkunz.github.io/lib_jsl/book/random/index.html).

pub mod distributions;
pub mod histogram;
pub mod mcmc;
pub mod quasi_random;
pub mod split_mix_64;
pub mod uniform_generator;
pub mod xoshiro256plusplus;
