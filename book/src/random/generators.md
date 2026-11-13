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

A pseudo-random number generator has three parts: a **state** (a block of memory), an **update**
that maps each state to the next, and an **output function** that turns the state into a number.
Everything it produces is determined by the initial state, so the "randomness" is a matter of
the outputs being statistically indistinguishable from random ones. Since there are finitely
many states, the sequence must eventually repeat; the length of the cycle is the **period**.

**The state and its update.** In xoshiro256++ the state is four 64-bit words $s_0, \ldots, s_3$,
256 bits in all. The update uses only shifts, rotations and exclusive-ors, which are fast on
every processor. Such an update is **linear over $\mathbb F_2$**: regard the state as a vector of
256 bits and do arithmetic modulo 2 (where addition is exclusive-or), and the update is
multiplication by a fixed $256 \times 256$ matrix of zeros and ones.

Linearity is what allows the period to be *proved* instead of hoped for. The matrix has been
chosen so that its characteristic polynomial is **primitive**, and that guarantees that every
non-zero state lies on one single cycle of length $2^{256} - 1$, the longest possible. The
all-zero state is the one exception: it maps to itself, so it must never be used.

**The output function.** Linearity is also a weakness. If the output were simply a word of the
state, its bits would satisfy linear relations that statistical tests can detect, the lowest
bits especially. The output is therefore passed through a nonlinear **scrambler**. Each step
returns

$$
\operatorname{rotl}(s_0 + s_3,\ 23) + s_0,
$$

where the additions are ordinary 64-bit integer additions (whose carries are *not* linear over
$\mathbb F_2$) and rotl rotates the bits left. This is the "++" of the name.

**From integers to floating point.** A uniform number in $[0, 1)$ is made from the top 53 bits
of the output, $\lfloor x/2^{11}\rfloor \cdot 2^{-53}$. A double holds 53 significant bits, so every
value produced is an exact multiple of $2^{-53}$ and all $2^{53}$ of them are equally likely. The
result can be exactly 0 but never 1, which matters for samplers that take logarithms: they use
$\ln(1 - u)$ instead of $\ln u$.

### Seeding with SplitMix64

A user supplies a seed of 64 bits, but the generator needs 256 bits of state, and they should
look unrelated to each other even when the seeds are simple numbers such as 1, 2, 3. A second,
simpler generator does the expansion. **SplitMix64** keeps a counter, adds the constant
$\lfloor2^{64}/\varphi\rfloor$ (with $\varphi$ the golden ratio) to it at each call, and passes the result
through a strong **mixing function**, two rounds of multiplying and xor-shifting, in which every
input bit affects every output bit. Its outputs for consecutive counters are effectively
independent, and four of them fill the state.

The library's version also skips the value 0, so that a seed can never produce the forbidden
all-zero state. That check is recursive, which keeps it from being inlined, and makes it slower
than xoshiro in the table below; it is used only for seeding.

### Independent streams

A parallel computation needs one stream of random numbers per worker, and the streams must not
overlap, or the workers would be using the same "random" numbers. Linearity provides a
guarantee. Advancing the state by one step is multiplication by the matrix $M$, so advancing by
$2^{128}$ steps is multiplication by $M^{2^{128}}$, another fixed linear map. The `jump` function
applies it in one call, using a precomputed polynomial. Calling `jump` repeatedly yields
starting states that are $2^{128}$ steps apart: since the period is about $2^{256}$, there is room
for $2^{128}$ such streams, each $2^{128}$ numbers long, and they provably do not overlap.

The alternative of seeding each worker with a different seed works in practice, because the
period is so vast that two random starting points are very unlikely to be close, but it carries
no guarantee.

### Histograms

`Histogram` counts samples in equal bins over $[\text{min}, \text{max}]$, the quickest visual check of a sampler, and the basis of
chi-square goodness-of-fit tests. Bins start at `bin_min`, and the upper edge belongs to the last bin.

### Speed

<p class="caption" id="tab-9-3"><strong>Table 9.3.</strong> Time per sample of the generators and samplers.</p>

{{#include ../generated/random/speed.md}}

## Algorithm

**`next_u64`**, one step of xoshiro256++, on the state words $s_0, s_1, s_2, s_3$:

1. **Output.** Compute `result = rotl(s0 + s3, 23) + s0`, with wrapping 64-bit additions, from
   the state as it stands.
2. **Update the state**, in this order:
   `t = s1 << 17`; `s2 ^= s0`; `s3 ^= s1`; `s1 ^= s2`; `s0 ^= s3`; `s2 ^= t`;
   `s3 = rotl(s3, 45)`.
3. Return `result`.

The whole step is about ten machine instructions, with no multiplication and no branch.

**`next_f64`**: take `next_u64`, shift it right by 11 bits to keep the top 53, and multiply by
$2^{-53}$.

**Seeding** from a 64-bit seed: run SplitMix64 four times from the seed, and use the four
outputs as $s_0, \ldots, s_3$.

**`jump`**: start a new state at zero. For each bit of a fixed 256-bit constant, in order: if
the bit is 1, exclusive-or the current state into the new state; then advance the generator one
step. Finally replace the state by the new state. The constant encodes the polynomial that
makes this equivalent to $2^{128}$ steps.

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

<details>
<summary>Hints and answers</summary>

1. If $a$ and $c$ are odd, $x_{k+1} = ax_k + c \equiv x_k + 1 \pmod 2$, so the lowest bit alternates; the second bit has period 4, and so on.
2. $65539 = 2^{16} + 3$, so $65539^2 = 2^{32} + 6\cdot2^{16} + 9 \equiv 6\cdot2^{16} + 9 \pmod{2^{31}}$. Then $x_{k+2} = 65539^2x_k = 6(65539 - 3)x_k + 9x_k = 6x_{k+1} - 9x_k$. Consequently $9x_k - 6x_{k+1} + x_{k+2} \equiv 0$: every triple lies on one of at most 15 planes.
3. The standard error of $\hat\pi$ is about $1.64/\sqrt N$, so the error falls from about $0.2$ at $N = 10^2$ to about $5\times10^{-4}$ at $N = 10^7$.
4. For a good generator, the chi-square statistic has mean equal to the degrees of freedom (9 for 10 bins) and should be above the 5% critical value (16.9) only about one time in twenty; repeat with several seeds.
5. Compare the sample correlation of the two streams at several lags; for $10^7$ values expect values of order $3\times10^{-4}$.

</details>

## Further reading

- D. Blackman and S. Vigna, "Scrambled linear pseudorandom number generators", *ACM Transactions on Mathematical Software* 47
  (2021) 36:1–36:32.
- G. L. Steele, D. Lea and C. H. Flood, "Fast splittable pseudorandom number generators", *OOPSLA* (2014) 453–472.
- P. L'Ecuyer and R. Simard, "TestU01: A C library for empirical testing of random number generators", *ACM Transactions on
  Mathematical Software* 33 (2007) 22:1–22:40.
- D. E. Knuth, *The Art of Computer Programming, Volume 2: Seminumerical Algorithms*, 3rd ed., Addison-Wesley, 1997, chapter 3.
