# The fast Fourier transform

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Compute the discrete Fourier transform of $n$ complex samples in $O(n\log n)$ operations |
| **Devised by** | Carl Friedrich Gauss (1805, unpublished); Gordon Danielson and Cornelius Lanczos (1942); James Cooley and John Tukey (1965); split radix: R. Yavne (1968), Pierre Duhamel and Henk Hollmann (1984); cache-oblivious and adaptive FFTs: Matteo Frigo and Steven Johnson (FFTW, 1998) |
| **Accuracy** | Relative error $O(\varepsilon\log n)$ with accurate twiddle factors |
| **Cost** | Radix 2: $5n\log_2n$ real flops; split radix: about $4n\log_2n$; the direct DFT: $8n^2$ |
| **API** | [`Fft`](../../api/lib_jsl_ffts/fft/struct.Fft.html), [`BestFft`](../../api/lib_jsl_ffts/best_fft/struct.BestFft.html), [`SimdFft`](../../api/lib_jsl_ffts/simd_fft/struct.SimdFft.html), [`FftEngine1D`](../../api/lib_jsl_ffts/fft_engine_trait/trait.FftEngine1D.html); reference: [`simple_cooley_tukey`](../../api/lib_jsl_ffts_reference/simple_cooley_tukey/index.html), [`optimized_radix2`](../../api/lib_jsl_ffts_reference/optimized_radix2/index.html), [`optimized_split_radix`](../../api/lib_jsl_ffts_reference/optimized_split_radix/index.html) |

</div>

## Overview

The FFT is a family of algorithms that compute the DFT by splitting it into smaller DFTs. The simplest, the radix-2
**Cooley-Tukey** algorithm, splits a transform of even length $n$ into two of length $n/2$, of the even- and odd-indexed samples,
and combines them with $n/2$ multiplications by **twiddle factors** $\omega^k = e^{-2\pi ik/n}$. Repeating the split down to length 1 gives
$\log_2n$ stages of $n/2$ "butterflies" each. **Split radix** splits more cleverly and does the fewest operations of the classical
algorithms.

On modern processors the operation count is only part of the story: memory access patterns, vector instructions and the
reuse of data in cache decide the speed. The production engine is a radix-2/4 Cooley-Tukey algorithm tuned for all three, and
runs five to ten times faster than the textbook engines, which perform about the same number of operations.

## History

Gauss devised the algorithm in 1805 to interpolate the orbits of the asteroids Pallas and Juno from observations, splitting a
transform of length 12 into transforms of lengths 3 and 4; his notes were published only posthumously, in Latin, in 1866, and
went unnoticed. Several later rediscoveries also passed without impact, including Danielson and Lanczos's doubling method (1942)
for X-ray crystallography. James Cooley of IBM and John Tukey of Princeton published the general algorithm in 1965, prompted by
the need to detect Soviet nuclear tests from seismometer data, and it spread immediately. R. Yavne (1968) found the split-radix
algorithm, rediscovered and analysed by Duhamel and Hollmann (1984). Matteo Frigo and Steven Johnson's FFTW (1998) showed that
generated, adaptively planned code could approach the hardware's peak, and set the standard for FFT libraries.

## The idea

### The radix-2 Cooley-Tukey algorithm

The DFT of $x_0, \ldots, x_{n-1}$ is

$$
X_k = \sum_{m=0}^{n-1}x_m\,\omega^{mk}, \qquad \omega = e^{-2\pi i/n}, \qquad k = 0, \ldots, n - 1 .
$$

Evaluated directly, each of the $n$ outputs is a sum of $n$ terms: $n^2$ multiplications. The FFT
reaches the same result in about $n\log_2n$ by never computing the same partial sum twice.

**Step 1: split the sum.** Assume $n$ is even, and separate the even-indexed samples
($m = 2j$) from the odd-indexed ones ($m = 2j + 1$):

$$
X_k = \sum_{j=0}^{n/2-1}x_{2j}\,\omega^{2jk} + \omega^k\sum_{j=0}^{n/2-1}x_{2j+1}\,\omega^{2jk} .
$$

**Step 2: recognise two smaller DFTs.** Both sums involve powers of $\omega^2 = e^{-2\pi i/(n/2)}$,
which is exactly the $\omega$ of a DFT of length $n/2$. So the first sum is the DFT of the
even-indexed samples, call it $E_k$, and the second is the DFT of the odd-indexed samples, $O_k$:

$$
X_k = E_k + \omega^kO_k .
$$

**Step 3: use each half twice.** $E$ and $O$ are DFTs of length $n/2$, so they are defined for
$k = 0, \ldots, n/2 - 1$ and repeat with period $n/2$. The outputs in the upper half,
$X_{k+n/2}$, therefore involve the *same* $E_k$ and $O_k$. Only the factor in front changes:
$\omega^{k+n/2} = \omega^k\omega^{n/2} = -\omega^k$, since $\omega^{n/2} = e^{-\pi i} = -1$. Altogether

$$
X_k = E_k + \omega^kO_k, \qquad X_{k+n/2} = E_k - \omega^kO_k, \qquad 0 \le k < n/2 .
$$

Each such pair of outputs costs one complex multiplication (by the **twiddle factor**
$\omega^k$), one addition and one subtraction. The pattern of data flow gives the operation its
name, a **butterfly**.

**The cost.** A transform of length $n$ has become two transforms of length $n/2$ plus $n/2$
butterflies. Apply the same idea to the halves, and to their halves, down to transforms of
length 1, which are just the samples. If $n = 2^p$ there are $p = \log_2n$ levels, and every
level does $n/2$ butterflies in total. That is $\tfrac n2\log_2n$ complex multiplications instead of
$n^2$: for $n = 2^{20}$, about $10^7$ instead of $10^{12}$.

**Bit reversal.** The recursion sorts the samples by the parity of their index, then by the
parity of half the index, and so on. The first split looks at the lowest binary digit of the
index, the next at the second lowest. When the recursion bottoms out, sample $m$ has arrived at
the position whose binary index is that of $m$ with its digits reversed. For $n = 8$ the order
is $0, 4, 2, 6, 1, 5, 3, 7$: for instance $3 = 011_2$ lands in position $110_2 = 6$. An iterative
FFT therefore avoids recursion altogether: it permutes the input into **bit-reversed** order
once, and then runs the levels from the bottom up, combining transforms of length 1 into
transforms of length 2, those into length 4, and so on, all in place.

### Split radix

In the radix-2 formulas the even half $E_k$ enters without any multiplication, while the odd
half needs a twiddle factor. This asymmetry can be exploited. **Split radix** treats the two
halves differently: it leaves the even samples as one transform $U$ of length $n/2$, as before,
but splits the odd samples once more, into those with indices $4m + 1$ and $4m + 3$, with
transforms $Z$ and $Z'$ of length $n/4$. The four quarters of the output are then

$$
X_k = U_k + (\omega^kZ_k + \omega^{3k}Z'_k), \qquad X_{k+n/2} = U_k - (\omega^kZ_k + \omega^{3k}Z'_k),
$$

$$
X_{k+n/4} = U_{k+n/4} - i(\omega^kZ_k - \omega^{3k}Z'_k), \qquad X_{k+3n/4} = U_{k+n/4} + i(\omega^kZ_k - \omega^{3k}Z'_k) .
$$

The gain comes from the factors $\pm i$ and $\pm1$ that appear when the output index is shifted
by a quarter or half of $n$. Multiplying by $i$ only swaps the real and imaginary parts and
changes a sign, so it costs nothing. The two products $\omega^kZ_k$ and $\omega^{3k}Z'_k$ thus serve
four outputs. The total is about $4n\log_2n - 6n + 8$ real operations, against $5n\log_2n$ for
radix 2. For decades this was the lowest known count, until Johnson and Frigo (2007) shaved off a
few per cent more.

### Twiddle factors and accuracy

The FFT is not only faster than the direct sum but also more accurate, for the same reason:
each output is reached through only $\log_2n$ butterflies, so rounding errors accumulate only
logarithmically. That holds provided each twiddle factor is itself accurate to $\varepsilon$.

There is a tempting shortcut that spoils it. The twiddle factors of a stage are successive
powers of one number, so many textbooks generate them by the recurrence
$\omega^{k+1} = \omega^k\cdot\omega$, one multiplication each, instead of calling sine and cosine. But
every multiplication adds a rounding error to a value that the next multiplication builds on,
so the error of $\omega^k$ grows in proportion to $k$, up to $n\varepsilon$. The reference crate's textbook
engine does exactly that, and its error reaches $10^{-11}$ at $n = 2^{20}$ (see the
[figure in the chapter introduction](index.md#accuracy)). The other engines compute every twiddle
from its exact angle and stay below $10^{-15}$.

### The production engines

`BestFft` is an iterative decimation-in-time algorithm on bit-reversed data, made fast in five ways:

1. **radix-4 passes**: pairs of radix-2 stages are fused, so each pass reads and writes the array once instead of twice and needs
   three twiddle multiplications per four outputs instead of four;
2. **SIMD**: on x86-64 with AVX2 and FMA (detected at run time) two complex values are processed per 256-bit register, with complex
   multiplication by `fmaddsub`;
3. **cache blocking**: all the stages whose butterflies fit in a block are run on one block at a time, so it stays in cache;
4. **bandwidth-saving large passes**: passes over the whole array store only $\omega^j$ and form $\omega^{2j}$ and $\omega^{3j}$ in registers;
5. **a cache-friendly permutation**: the bit reversal is done in $16 \times 16$ tiles, writing straight into the output with any scaling
   folded in.

`SimdFft` provides vectorised radix-2 kernels for ARM (NEON) and RISC-V as well as x86. The default `Fft` uses `BestFft` for powers of
two and [Bluestein's algorithm](bluestein-fractional.md) for every other length.

![FFT speed](../generated/fft/speed.svg)

The three reference engines run at 1 to 4 GFLOP/s; `BestFft` reaches 10 to 20, falling off once the data no longer fit in cache. The
direct DFT, at well under 0.1 GFLOP/s-equivalent, is hopeless beyond a few thousand points.

## Algorithm

**Iterative radix 2**: permute the input into bit-reversed order; for $s = 1, \ldots, \log_2n$, with $m = 2^s$, combine pairs of transforms of
length $m/2$ by butterflies with twiddles $e^{\mp2\pi ij/m}$; scale if requested.

**Split radix** (recursive): transform the even samples (length $n/2$) and the two odd subsequences (length $n/4$), then combine with the
four formulas above.

## Implementation

The stages of the textbook engine, with its twiddle recurrence:

```rust,ignore
{{#include ../../../crates/ffts_reference/src/simple_cooley_tukey.rs:radix2_stages}}
```

The split-radix recursion:

```rust,ignore
{{#include ../../../crates/ffts_reference/src/optimized_split_radix.rs:split_radix}}
```

A radix-4 pass of the production engine (portable version):

```rust,ignore
{{#include ../../../crates/ffts/src/best_fft.rs:radix4_pass}}
```

- `plan(size, scale, direction, ordering)` builds the twiddle tables; `execute(input)` returns a new vector. `FftScaleFactor` is `None`,
  `SqrtN` (unitary) or `N` (for the inverse); `FftOrdering::BitReversed` accepts input already in bit-reversed order.
- The reference split-radix engine replaced an earlier engine of the same name, which repeated its first stage and returned wrong
  results; its tests had computed the error without checking it.

## In practice

```rust,ignore
{{#include ../../../crates/ffts_reference/examples/fft_engines.rs}}
```

**Strengths**

- $O(n\log n)$ and $O(\varepsilon\log n)$: fast and accurate.
- Powers of two run at 10 to 20 GFLOP/s on one core (`cargo bench -p lib_jsl_ffts` compares the engines with RustFFT); any
  length is supported.

**Weaknesses**

- Non-power-of-two lengths go through Bluestein's algorithm, several times slower than a mixed-radix FFT would be for lengths
  with small prime factors.
- One-dimensional engines return a new vector (no in-place API).

## Exercises

1. Count the real flops of radix-2 and split-radix FFTs for $n = 8$ and $n = 16$.
2. Show that the in-place radix-2 algorithm reads its input in bit-reversed order.
3. Measure the error of the textbook engine for $n = 2^{10}, \ldots, 2^{20}$ and fit it to $Cn\varepsilon$. Then replace the recurrence by exact
   twiddles and measure again.
4. Use the FFT to multiply two polynomials of degree 1000, and check against direct multiplication.
5. Compute the DFT of a sampled sine whose frequency is not an integer number of cycles, and explain the spectral leakage.

## Further reading

- J. W. Cooley and J. W. Tukey, "An algorithm for the machine calculation of complex Fourier series", *Mathematics of Computation* 19
  (1965) 297–301.
- M. T. Heideman, D. H. Johnson and C. S. Burrus, "Gauss and the history of the fast Fourier transform", *IEEE ASSP Magazine* 1 (1984)
  14–21.
- P. Duhamel and H. Hollmann, "Split radix FFT algorithm", *Electronics Letters* 20 (1984) 14–16.
- M. Frigo and S. G. Johnson, "The design and implementation of FFTW3", *Proceedings of the IEEE* 93 (2005) 216–231.
- C. Van Loan, *Computational Frameworks for the Fast Fourier Transform*, SIAM, 1992.
