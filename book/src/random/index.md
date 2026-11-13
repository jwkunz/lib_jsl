# Random numbers

Simulation, statistics and machine learning all consume randomness: Monte Carlo integration,
stochastic simulation of queues and molecules, the bootstrap, random initialisation and
mini-batching in training, randomised algorithms, cryptographic keys. Computers are deterministic,
so in practice the randomness comes from **pseudo-random number generators**: deterministic
sequences designed to be statistically indistinguishable from independent uniform random numbers.
Everything else is built on them: samples from other **distributions** by transformation, and
**Markov chain Monte Carlo** for distributions known only up to a constant. For integration, where
evenness matters more than independence, **quasi-random** sequences do better than random ones.

## The central questions

- **Is the generator good enough?** A generator must pass demanding statistical test batteries,
  have a long period, and be fast. For simulation, not cryptography, modern scrambled-linear
  generators such as xoshiro256++ are the standard; they are not secure against an adversary.
- **How to sample a given distribution?** By **inversion** of its CDF when that is cheap; by a special
  **transformation** (Box-Muller for the normal); by **rejection** from a simpler distribution; or by
  **composition** of other samples (beta from gammas, Student's $t$ from a normal and a chi-square).
- **Reproducibility and parallelism.** A seeded generator reproduces a simulation exactly, and
  **jumping** ahead gives independent streams for parallel workers.
- **Random or quasi-random?** Monte Carlo error falls like $N^{-1/2}$ whatever the dimension; low-discrepancy
  points can do almost as well as $N^{-1}$ for smooth integrands, but are deterministic and need random shifts for
  error estimates.

## Testing the library

Every sampler is checked in the [distributions entry](distributions.md) by comparing 100 000 samples
with the exact distribution function. 

### Against SciPy

The references come from SciPy 1.17.1, written by `book/generator/data/make_random_refs.py`. The unscrambled quasi-random sequences reproduce `scipy.stats.qmc` exactly (Sobol from the same Joe-Kuo direction numbers; the library's Halton sequence starts after the origin, which SciPy includes):

{{#include ../generated/random/scipy_qmc.md}}

As an independent check of the samplers, which does not use the library's own distribution functions, 100 000 samples of each are counted in bins whose edges are SciPy's quantiles (20 equally likely bins) or, for discrete distributions, whose probabilities come from SciPy's `pmf` (grouped so that each bin expects at least 50 samples). The chi-square statistic then tests the counts:

{{#include ../generated/random/scipy_samplers.md}}

Under correct sampling the p-values are uniformly distributed, so among 17 tests one or two below 0.1 are expected. Repeating the lowest here (the normal sampler) with a million samples and five seeds gives p-values from 0.3 to 0.8.

## In the library

The crate is `lib_jsl_random` ([API reference](../../api/lib_jsl_random/index.html)). Generators implement the
`UniformGenerator` trait (`next_u64`, `next_f64`); `UniformRNG` is xoshiro256++. Distribution samplers are
functions taking `&mut impl UniformGenerator`, so any generator can drive them.

| You need | Use | Entry |
|---|---|---|
| Uniform random numbers, reproducible streams | `UniformRNG` (xoshiro256++), `jump` | [Pseudo-random generators](generators.md) |
| Samples from a named distribution | `distributions::*`, `MultivariateNormal` | [Sampling distributions](distributions.md) |
| A histogram of samples | `Histogram` | [Pseudo-random generators](generators.md#histograms) |
| Even points for integration | `SobolSequence`, `HaltonSequence`, `random_shift` | [Quasi-random sequences](quasi-random.md) |
| Samples from an unnormalised density | `metropolis_hastings` | [Markov chain Monte Carlo](mcmc.md) |
