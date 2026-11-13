# Quasi-random sequences

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Generate points in the unit cube $[0, 1)^d$ that fill it as evenly as possible, for numerical integration and sampling |
| **Devised by** | Johannes van der Corput (1935), the radical inverse; John Halton (1960); Ilya Sobol (1967); Ivan Antonov and Vladimir Saleev (1979), Gray-code generation; Roy Cranley and Thomas Patterson (1976), random shifts; Stephen Joe and Frances Kuo (2008), direction numbers |
| **Accuracy** | Star discrepancy $O((\log N)^d/N)$, so quasi-Monte Carlo errors approach $O(N^{-1})$ for smooth integrands, against $O(N^{-1/2})$ for Monte Carlo |
| **Cost** | Sobol: one exclusive-or per coordinate per point. Halton: $O(\log N)$ digit operations per coordinate |
| **API** | [`SobolSequence`](../../api/lib_jsl_random/quasi_random/struct.SobolSequence.html) (up to 21 dimensions), [`HaltonSequence`](../../api/lib_jsl_random/quasi_random/struct.HaltonSequence.html) (up to 100), [`radical_inverse`](../../api/lib_jsl_random/quasi_random/fn.radical_inverse.html), [`random_shift`](../../api/lib_jsl_random/quasi_random/fn.random_shift.html) |

</div>

## Overview

Random points cluster and leave gaps: that is what independence means. For integration, gaps waste
evaluations, and the Monte Carlo error falls only like $N^{-1/2}$. **Low-discrepancy** or **quasi-random**
sequences are deterministic and designed to fill space evenly at every stage: each new point goes where the
earlier ones left room. The error of an average over them is bounded by the **Koksma-Hlawka inequality**,
$|\text{error}| \le V(f)\,D^*_N$, the variation of the integrand times the **star discrepancy** of the points (the largest
difference between the fraction of points in a box anchored at the origin and the box's volume), and good sequences
achieve $D^*_N = O((\log N)^d/N)$.

The [Monte Carlo integration](../integration/monte-carlo.md) entry uses these sequences; this entry describes how
they are made and where they fail.

| | | |
|:---:|:---:|:---:|
| ![Random points](../generated/integration/points_random.svg) | ![Sobol points](../generated/integration/points_sobol.svg) | ![Halton points](../generated/integration/points_halton.svg) |

## History

Johannes van der Corput (1935) introduced the radical-inverse sequence in base 2 in his work on the uniform
distribution of sequences, John Halton (1960) extended it to several dimensions with coprime bases, and Ilya Sobol
(1967), in Moscow, constructed sequences from binary "direction numbers" with far better uniformity in high dimensions.
Ivan Antonov and Vladimir Saleev (1979) showed how to generate Sobol points in Gray-code order with a single exclusive-or
per coordinate. Roy Cranley and Thomas Patterson (1976) proposed random shifts to obtain error estimates, the start of
randomised quasi-Monte Carlo. Stephen Joe and Frances Kuo (2008) computed direction numbers that give good
two-dimensional projections in thousands of dimensions; quasi-Monte Carlo is now standard in financial mathematics,
where integrals over hundreds of dimensions price derivatives.

## The idea

### The radical inverse and the Halton sequence

Random points are not as evenly spread as one might expect: by chance they form clusters and
leave gaps. For [numerical integration](../integration/monte-carlo.md) evenness is what
matters, and a **quasi-random** (or **low-discrepancy**) sequence is a deterministic sequence
designed to fill space as evenly as possible, at every stage as points are added.

**One dimension.** The basic construction is the **radical inverse**: write the index $n$ in
base $b$ and mirror its digits about the radix point. If $n = \sum_ja_jb^j$ then

$$
\phi_b(n) = \sum_ja_jb^{-j-1} .
$$

In base 2, for example, $n = 6$ is $110_2$, and mirroring gives $0.011_2 = \tfrac38$. The
sequence $n = 1, 2, 3, 4, \ldots$ gives $\tfrac12, \tfrac14, \tfrac34, \tfrac18, \tfrac58, \tfrac38, \tfrac78, \ldots$. The
fastest-changing digit of $n$ becomes the most significant digit of the point, so consecutive
points jump far apart, and each new point lands in the middle of one of the largest remaining
gaps. This is the **van der Corput sequence**.

**Several dimensions.** A point in $d$ dimensions needs $d$ coordinates that are evenly spread
*jointly*, not just one at a time. Using the same base for two coordinates would put every point
on the diagonal. The **Halton sequence** uses a different prime base for each coordinate, the
first $d$ primes, so that the digit patterns of the coordinates never fall into step.

The Halton sequence is simple and can be extended to any number of points. Its weakness appears
in high dimensions, where the bases are large primes. In base 113, say, the first 112 points of
that coordinate simply climb in steps of $1/113$, and a neighbouring coordinate with a similar
base climbs at almost the same rate. The first points in the two coordinates therefore rise
together in long correlated runs:

| | |
|:---:|:---:|
| ![Halton dimensions 29 and 30](../generated/random/halton_29_30.svg) | ![Sobol dimensions 20 and 21](../generated/random/sobol_20_21.svg) |

The first 256 Halton points in dimensions 29 and 30 lie on three lines; Sobol points in its highest dimensions also show
structure at this size, but cover the square far better. Scrambling the digits, or skipping the first points, mitigates the
Halton problem.

### The Sobol sequence

The Sobol sequence avoids large bases by working in base 2 for every coordinate, and makes the
coordinates differ in another way. In the base-2 radical inverse, bit $j$ of the index
contributes the fixed amount $2^{-j-1}$ to the point. Sobol's construction replaces these
amounts by a table of **direction numbers** $v_{k,j}$, a different set for each coordinate $k$,
and combines them by exclusive-or instead of addition: coordinate $k$ of a point is the
exclusive-or of the $v_{k,j}$ for the bits $j$ that are set in its index. The direction numbers
of each coordinate are derived from a primitive polynomial over $\mathbb F_2$ and a few initial
values, chosen so that the coordinates are evenly spread jointly. The library uses Joe and Kuo's
direction numbers.

**Gray-code order.** Computing a point from scratch needs one exclusive-or per set bit. If the
points are generated in **Gray-code order**, an ordering of the integers in which consecutive
values differ in exactly one bit, each point differs from the previous one by a single direction
number, and costs one exclusive-or per coordinate. The same set of points is produced, in a
different order.

**The guarantee.** The evenness of Sobol points can be stated precisely. Every block of $2^m$
consecutive points starting at a multiple of $2^m$ is a **$(t, m, d)$-net**: divide the unit cube
into equal boxes of volume $2^{t-m}$ by cutting each axis into a power-of-two number of pieces,
in any combination, and every box contains exactly $2^t$ points. No box is over- or
under-filled. The smaller the quality parameter $t$, the finer the boxes for which this holds.

### Error estimates by random shifts

A Monte Carlo estimate comes with an error bar, computed from the scatter of the samples.
Quasi-random points are deterministic, so an average over them has no scatter to measure, and no
error estimate. Randomness can be reintroduced without losing the evenness. A
**Cranley-Patterson rotation** shifts every point by the same random vector $\mathbf u$ and wraps
around the edges of the unit cube, $\mathbf x \mapsto (\mathbf x + \mathbf u) \bmod 1$. The shifted set is just
as evenly spread as the original, and because $\mathbf u$ is uniform, the average over it is an
unbiased random estimate of the integral. Repeating with several independent shifts gives
independent estimates, whose spread measures the error.

### Integration error

![QMC integration error](../generated/random/qmc_error.svg)

<p class="caption" id="fig-9-2"><strong>Figure 9.2.</strong> QMC integration error.</p>

For a smooth product integrand in six dimensions, pseudo-random points follow $N^{-1/2}$, while Sobol and Halton points follow
roughly $N^{-1}$, two to four hundred times more accurate at $10^5$ points, with the irregular decrease typical of quasi-random errors (Sobol is
best at powers of two).

## Algorithm

**Halton point $n$ in $d$ dimensions.** For each coordinate $k = 1, \ldots, d$, with $b$ the
$k$-th prime, compute the radical inverse of $n$ in base $b$:

1. set $r = 0$ and $f = 1/b$;
2. while $n > 0$: add $f \times (n \bmod b)$ to $r$; replace $n$ by $\lfloor n/b\rfloor$; divide $f$
   by $b$;
3. coordinate $k$ is $r$.

The cost is $O(\log n)$ per coordinate. Points are usually numbered from $n = 1$, since $n = 0$
gives the corner of the cube.

**Sobol points** are generated in sequence, each from the one before. The generator keeps the
index $n$ and, for each coordinate $k$, an integer state $x_k$, initially zero.

1. Output the current point: each $x_k$, read as a binary fraction, is a coordinate in $[0, 1)$.
2. Find $c$, the position of the lowest zero bit of $n$. This is the bit that changes between
   the Gray codes of $n$ and $n + 1$.
3. For each coordinate, exclusive-or the direction number $v_{k,c}$ into $x_k$.
4. Increment $n$.

The cost is one exclusive-or per coordinate per point.

**A randomly shifted point set**: draw one uniform vector $\mathbf u$; replace every point
$\mathbf x$ by the fractional part of $\mathbf x + \mathbf u$.

## Implementation

```rust,ignore
{{#include ../../../crates/random/src/quasi_random.rs:radical_inverse}}
```

```rust,ignore
{{#include ../../../crates/random/src/quasi_random.rs:sobol_next}}
```

- The Sobol sequence starts at the origin; the Halton sequence starts at index 1, skipping it.
- `skip(count)` jumps ahead, for distributing a sequence across workers.

## In practice

```rust,ignore
{{#include ../../../crates/random/examples/random_quasi.rs}}
```

**Strengths**

- Errors close to $N^{-1}$ for smooth integrands, against $N^{-1/2}$ for Monte Carlo.
- Deterministic and reproducible, with random shifts available for error estimates.

**Weaknesses**

- The advantage fades for non-smooth integrands and very high effective dimension.
- Halton degrades in high dimensions; this Sobol implementation has 21 dimensions.
- Not random: never use quasi-random points where independence matters (simulation of random processes, statistical tests).

## Exercises

1. Compute the first 16 points of the van der Corput sequence in base 2 and show that each block of $2^m$ points has one point in each
   interval $[k2^{-m}, (k+1)2^{-m})$.
2. Show that the radical inverse in base $b$ of $n = 0, \ldots, b^m - 1$ is a permutation of the points $k/b^m$.
3. Integrate $\prod_i(1 + (x_i - \tfrac12)/i)$ over $[0, 1]^{20}$ with Sobol points and with pseudo-random points, and compare the errors.
4. Estimate the error of a Sobol integration with 10 random shifts, and compare with the true error.
5. Plot the first 1000 Halton points in dimensions 1 and 2 and in dimensions 40 and 41, and explain the difference.

## Further reading

- J. Dick and F. Pillichshammer, *Digital Nets and Sequences*, Cambridge University Press, 2010.
- S. Joe and F. Y. Kuo, "Constructing Sobol sequences with better two-dimensional projections", *SIAM Journal on Scientific Computing*
  30 (2008) 2635–2654.
- H. Niederreiter, *Random Number Generation and Quasi-Monte Carlo Methods*, SIAM, 1992.
- A. B. Owen, *Practical Quasi-Monte Carlo Integration*, 2023, <https://artowen.su.domains/mc/practicalqmc.pdf>.
