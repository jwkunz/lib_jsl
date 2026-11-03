# Pseudo-random number generators

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Produce a long, fast, reproducible sequence of 64-bit integers (and uniform numbers in $[0, 1)$) that passes statistical tests of randomness |
| **Devised by** | John von Neumann (1946), middle-square; Derrick Lehmer (1949), linear congruential generators; George Marsaglia (2003), xorshift; Guy Steele, Doug Lea and Christine Flood (2014), SplitMix; David Blackman and Sebastiano Vigna (2018), xoshiro256++ |
| **Accuracy** | Period $2^{256} - 1$; passes the BigCrush and PractRand test batteries; 53 random bits per floating-point number |
| **Cost** | About 2 ns per 64-bit number; 32 bytes of state |
| **API** | [`UniformGenerator`](../../api/lib_jsl_random/uniform_generator/trait.UniformGenerator.html), [`UniformRNG`](../../api/lib_jsl_random/uniform_generator/type.UniformRNG.html), [`Xoshiro256PlusPlus`](../../api/lib_jsl_random/xoshiro256plusplus/struct.Xoshiro256PlusPlus.html), [`SplitMix64`](../../api/lib_jsl_random/split_mix_64/struct.SplitMix64.html), [`Histogram`](../../api/lib_jsl_random/histogram/struct.Histogram.html) |

</div>

## Overview

A pseudo-random number generator (PRNG) is a deterministic state machine: a state, a function that
advances it, and a function that outputs a number from it. A good one has a period far longer than any
computation, outputs that pass every practical statistical test, and is fast. Because it is deterministic,
the same seed reproduces the same sequence, which makes simulations debuggable and repeatable.

The library uses **xoshiro256++** (the `UniformRNG` alias): 256 bits of state advanced by shifts, rotations
and exclusive-ors, scrambled by an addition and a rotation. It is seeded through **SplitMix64**, a simple
generator that spreads any 64-bit seed over the larger state. Neither is cryptographically secure: an
observer can reconstruct the state from a few outputs. For keys and tokens, use an operating-system source.

## History

John von Neumann's middle-square method (1946), squaring a number and keeping its middle digits, was
quickly found to fall into short cycles; he remarked that "anyone who considers arithmetical methods of
producing random digits is, of course, in a state of sin". Derrick Lehmer's linear congruential generators
(1949), $x_{k+1} = (ax_k + c) \bmod m$, dominated for decades, despite infamous failures such as IBM's RANDU, whose
consecutive triples fall on 15 planes. Makoto Matsumoto and Takuji Nishimura's Mersenne Twister (1998) brought
enormous periods and became the default of many languages. George Marsaglia (2003) introduced fast xorshift
generators, and Pierre L'Ecuyer and Richard Simard's TestU01 (2007) set the standard of statistical testing.
David Blackman and Sebastiano Vigna (2018) combined xorshift-style linear engines with nonlinear scramblers
in the xoshiro and xoroshiro families, small, fast and statistically excellent.

## The idea

### xoshiro256++

The state is four 64-bit words $s_0, \ldots, s_3$. Each step outputs $\operatorname{rotl}(s_0 + s_3, 23) + s_0$ and updates the state by
a linear map over $\mathbb F_2$ (shifts, exclusive-ors and a rotation), whose characteristic polynomial is primitive: every
non-zero state lies on a single cycle of length $2^{256} - 1$. The linear engine alone would fail some tests (its low bits are
linear); the "++" scrambler, an addition and rotation, breaks the linearity. The state must not be all zero, the one fixed
point.

A floating-point number in $[0, 1)$ uses the top 53 bits of the output, $\lfloor x/2^{11}\rfloor \cdot 2^{-53}$, so that every value is an exact
multiple of $2^{-53}$ and all are equally likely. It can be exactly 0, which matters for samplers that take logarithms (they use $1 - u$).

### Seeding with SplitMix64

A 64-bit seed must be expanded to 256 well-mixed bits. **SplitMix64** adds the golden-ratio constant $\lfloor2^{64}/\varphi\rfloor$ to a
counter and passes it through a strong mixing function (two multiply-xorshift rounds); its outputs for consecutive counters are
effectively independent. The library's version also skips the value 0, so that a seed can never produce the forbidden all-zero
state. That check is recursive, which keeps it from being inlined, and makes it slower than xoshiro in the table below; it is used
only for seeding.

### Independent streams

Parallel workers need non-overlapping streams. Because the update is linear, advancing the state by $2^{128}$ steps is itself a
linear map, computed by the `jump` function from a precomputed polynomial: $2^{128}$ jumps fit in the period, each stream $2^{128}$ long.
The alternative of seeding each worker with a different seed works in practice but carries no guarantee against overlap.

### Histograms

`Histogram` counts samples in equal bins over $[\text{min}, \text{max}]$, the quickest visual check of a sampler, and the basis of
chi-square goodness-of-fit tests. Writing this chapter revealed that it placed its bins from 0 rather than from `bin_min`, and
panicked at the upper edge; both are fixed.

### Speed

{{#include ../generated/random/speed.md}}

## Algorithm

`next_u64`: `result = rotl(s0 + s3, 23) + s0`; `t = s1 << 17`; `s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t; s3 = rotl(s3, 45)`.

## Implementation

```rust,ignore
{{#include ../../../crates/random/src/xoshiro256plusplus.rs:xoshiro}}
```

```rust,ignore
{{#include ../../../crates/random/src/split_mix_64.rs:splitmix}}
```

- `UniformRNG::from_seed(u64)` seeds through SplitMix64; `from_array_seed` takes the state directly (rejecting all zeros);
  `from_random_seed` uses the clock, convenient but not reproducible.

## In practice

```rust,ignore
{{#include ../../../crates/random/examples/random_generators.rs}}
```

**Strengths**

- Fast, small, statistically excellent, with a period of $2^{256} - 1$ and cheap independent streams.

**Weaknesses**

- Not cryptographically secure.
- Floating-point numbers have 53 random bits, so values below $2^{-53}$ never occur except 0.

## Exercises

1. Show that a linear congruential generator with modulus $2^{32}$ has a lowest bit that alternates.
2. Find the 15 planes of RANDU, $x_{k+1} = 65539x_k \bmod 2^{31}$, by showing $x_{k+2} = 6x_{k+1} - 9x_k \pmod{2^{31}}$.
3. Estimate $\pi$ by the fraction of uniform points in the unit quarter disc, and the error, for $N = 10^2, \ldots, 10^7$.
4. Run a chi-square test on 10 bins of 100 000 uniforms using `Histogram`, and on the low 3 bits of `next_u64`.
5. Verify experimentally that streams separated by `jump` show no correlation over $10^7$ values.

## Further reading

- D. Blackman and S. Vigna, "Scrambled linear pseudorandom number generators", *ACM Transactions on Mathematical Software* 47
  (2021) 36:1–36:32.
- G. L. Steele, D. Lea and C. H. Flood, "Fast splittable pseudorandom number generators", *OOPSLA* (2014) 453–472.
- P. L'Ecuyer and R. Simard, "TestU01: A C library for empirical testing of random number generators", *ACM Transactions on
  Mathematical Software* 33 (2007) 22:1–22:40.
- D. E. Knuth, *The Art of Computer Programming, Volume 2: Seminumerical Algorithms*, 3rd ed., Addison-Wesley, 1997, chapter 3.
