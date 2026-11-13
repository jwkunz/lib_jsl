# Discrete cosine and sine transforms

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Expand real data in cosines or sines: the discrete cosine and sine transforms of types I to IV, their inverses and orthonormal forms |
| **Devised by** | Nasir Ahmed, T. Natarajan and K. R. Rao (1974), the DCT-II; Zhongde Wang and others (1984), the classification into types I–IV; the JPEG committee (1992), image compression; John Princen and Alan Bradley (1986), the modified DCT of audio coding |
| **Accuracy** | $O(\varepsilon\log n)$, through complex FFTs |
| **Cost** | One complex FFT of length about $2n$ plus $O(n)$ work |
| **Read first** | [The fast Fourier transform](fft.md) |
| **API** | [`dct`](../../api/lib_jsl_ffts/dct/fn.dct.html), [`idct`](../../api/lib_jsl_ffts/dct/fn.idct.html), [`dst`](../../api/lib_jsl_ffts/dct/fn.dst.html), [`idst`](../../api/lib_jsl_ffts/dct/fn.idst.html), [`TrigTransformType`](../../api/lib_jsl_ffts/dct/enum.TrigTransformType.html), [`TrigNormalization`](../../api/lib_jsl_ffts/dct/enum.TrigNormalization.html) |

</div>

## Overview

The DFT implicitly treats data as one period of a periodic signal. For data that is not periodic, the jump between the last
sample and the first produces large high-frequency coefficients, decaying only like $1/k$. The **discrete cosine transform**
instead treats the data as half of an *even* signal, its mirror image appended, which has no jump: for smooth data its coefficients
decay like $1/k^2$ or faster, so a few of them capture most of the signal. That **energy compaction** made the DCT-II the transform of
JPEG images and MPEG video, and its relative the modified DCT the transform of MP3 and AAC audio.

The cosine and sine transforms are also the eigenvector bases of the second-difference matrix with Neumann and Dirichlet boundary
conditions, so they diagonalise Poisson's equation on rectangles and appear in [spectral methods](../integration/spectral-methods.md)
(the Chebyshev transform is a DCT-I).

## History

Nasir Ahmed proposed the cosine transform in 1972 as an approximation of the optimal Karhunen-Loève transform for correlated signals,
and published it with T. Natarajan and K. R. Rao in 1974. Zhongde Wang (1984) classified the eight variants by the symmetry of the
extension (types I to IV of cosines and sines). The JPEG image standard (1992) used the $8 \times 8$ DCT-II, and John Princen and Alan Bradley's
modified DCT (1986), a lapped DCT-IV, underlies MP3, AAC and Vorbis audio.

## The idea

### Symmetric extensions

The DFT treats its $n$ samples as one period of a periodic signal. If the last sample differs
from the first, the periodic signal has a jump at every period boundary, and a jump needs many
high frequencies to represent. The cosine and sine transforms avoid this by a different
assumption about what lies beyond the ends of the data.

Take the samples and reflect them, as in a mirror, before repeating. For the samples
$a, b, c, d$ one such extension is

$$
\ldots\ a\ b\ c\ d\ \ d\ c\ b\ a\ \ a\ b\ c\ d\ \ldots
$$

This sequence is continuous at the joins, whatever the data are. It is also even (symmetric),
and the Fourier series of an even sequence contains only cosines. Its DFT, of length $2n$, is
therefore real (after a phase factor accounting for the position of the mirror), and only $n$ of
its values are independent. Those $n$ real numbers are a **discrete cosine transform** (DCT) of
the original samples. Reflecting with a change of sign (an odd extension) gives sines instead,
and a **discrete sine transform** (DST).

There are several types, because there are choices to make: the mirror at each end can sit
exactly on a sample or halfway between two samples, and the extension at each end can be even or
odd. The example above, with both mirrors between samples and both extensions even, is the
DCT-II. With the unnormalised conventions of SciPy and FFTW, the four common types of each are:

| type | cosine transform $y_k$ | sine transform $y_k$ |
|---|---|---|
| I | $x_0 + (-1)^kx_{n-1} + 2\sum_{m=1}^{n-2}x_m\cos\frac{\pi km}{n-1}$ | $2\sum x_m\sin\frac{\pi(k+1)(m+1)}{n+1}$ |
| II | $2\sum x_m\cos\frac{\pi k(2m+1)}{2n}$ | $2\sum x_m\sin\frac{\pi(k+1)(2m+1)}{2n}$ |
| III | $x_0 + 2\sum_{m\ge1}x_m\cos\frac{\pi m(2k+1)}{2n}$ | $(-1)^kx_{n-1} + 2\sum_{m<n-1}x_m\sin\frac{\pi(m+1)(2k+1)}{2n}$ |
| IV | $2\sum x_m\cos\frac{\pi(2m+1)(2k+1)}{4n}$ | $2\sum x_m\sin\frac{\pi(2m+1)(2k+1)}{4n}$ |

In the DCT-II formula the argument $\pi k(2m + 1)/2n$ places sample $m$ at the position
$m + \tfrac12$, the midpoint of its cell, which is the "mirror between samples" of the extension.

Types II and III are inverses of each other (up to a factor $2n$), and types I and IV are their
own inverses (up to $2(n \mp 1)$ and $2n$). The **orthonormal** normalisation rescales each
transform to an orthogonal matrix, so that its inverse is its transpose and it preserves the sum
of squares of the data. "The DCT" usually means the DCT-II, and "the inverse DCT" the DCT-III.

### Energy compaction

The smoother a periodic function is, the faster its Fourier coefficients decay. A jump gives
coefficients that decay like $1/k$; a continuous function with a jump in its slope gives $1/k^2$.

Take a smooth function on $[0, 1]$ that is not periodic, sampled at the midpoints of $n$ cells.
Its periodic extension, which the DFT sees, has a jump, so the DFT coefficients decay like
$1/k$. Its even extension, which the DCT-II sees, is continuous with at worst a corner, so the
DCT-II coefficients decay like $1/k^2$:

![Energy compaction of the DCT](../generated/fft/dct_compaction.svg)

<p class="caption" id="fig-8-4"><strong>Figure 8.4.</strong> Energy compaction of the DCT.</p>

Most of the signal's energy is therefore packed into the first few DCT coefficients, and
discarding the rest changes the signal very little. This property is called **energy
compaction**, and it is the reason the DCT is used for compression: JPEG quantises the
high-frequency DCT coefficients of each $8 \times 8$ block coarsely or discards them.

### Computing through the FFT

Since each transform is the DFT of a symmetric extension, each can be computed with one complex
FFT of length about $2n$ applied to the extended sequence, followed by a correction for the
position of the mirror:

- the DCT-I from the even extension of length $2n - 2$;
- the DCT-II from the even extension of length $2n$, followed by multiplication by the twiddle
  factor $e^{-\pi ik/2n}$, which shifts the origin by the half sample between the mirror and the
  first sample;
- the DCT-III by the inverse of that;
- the DCT-IV by twiddled zero-padding to length $2n$;
- the DST-I from the odd extension of length $2n + 2$;
- the DST-II and DST-III from the corresponding cosine transforms, by reversing the order of
  the data and alternating signs;
- the DST-IV as the imaginary part of the same twiddled FFT as the DCT-IV.

All are $O(n\log n)$ and inherit the FFT's accuracy: compared with a direct sum with exactly
reduced angles, the DCT-II is accurate to $2 \times 10^{-15}$ at $n = 4096$.

## Algorithm

`dct(x, kind, norm)` computes any of the four types through one complex FFT.

1. **Validate** the length (type I needs at least two samples).
2. **Build the FFT input** for the type:
   - type I: the even extension of length $2n - 2$, that is $x_0, \ldots, x_{n-1}$ followed by
     $x_{n-2}, \ldots, x_1$;
   - type II: the even extension of length $2n$, that is $x_0, \ldots, x_{n-1}$ followed by
     $x_{n-1}, \ldots, x_0$;
   - type III: the samples multiplied by the twiddle factors $e^{\pi ik/2n}$, arranged so that
     the transform that follows inverts type II;
   - type IV: the samples multiplied by a twiddle factor and zero-padded to length $2n$.
3. **Transform** with a complex FFT.
4. **Extract.** Take the first $n$ outputs. For type II multiply output $k$ by
   $e^{-\pi ik/2n}$, and for type IV by its own twiddle factor; keep the real parts.
5. **Normalise.** With `norm = Ortho`, scale the outputs (the first, and for type I the last,
   by a different factor from the rest) so that the transform is an orthogonal matrix.

`idct` applies the inverse type (III for II, II for III, the same type for I and IV) with the
scaling that makes the round trip exact. The sine transforms follow the same five steps with
odd extensions, or reduce to a cosine transform by reversing the order of the input and
alternating the signs of the output. Every transform costs one FFT of length about $2n$.

## Implementation

```rust,ignore
{{#include ../../../crates/ffts/src/dct.rs:dct_raw}}
```

- `TrigNormalization::Backward` is the unnormalised forward transform with the scaling on the inverse (SciPy's default);
  `Ortho` makes both orthogonal.
- The extensions are transformed with the complex `Fft`, of any length.

## In practice

```rust,ignore
{{#include ../../../crates/ffts/examples/fft_dct.rs}}
```

**Strengths**

- All eight classical transforms, unnormalised or orthonormal, for any length, in $O(n\log n)$.
- Excellent energy compaction for smooth, non-periodic data.

**Weaknesses**

- Computed through complex FFTs of twice the length, about four times the work of a dedicated real-to-real algorithm.
- No modified (lapped) DCT and no multidimensional DCT (apply `dct` along each axis).

## Exercises

1. Show that the DCT-II of $x$ is, up to a twiddle, the DFT of its even extension of length $2n$.
2. Verify numerically that the orthonormal DCT-II matrix is orthogonal for $n = 8$.
3. Compress a sampled smooth function by keeping the $k$ largest DCT-II coefficients and plot the error against $k$; compare with the DFT.
4. Show that the vectors $\sin\frac{\pi jk}{n+1}$ are eigenvectors of the second-difference matrix with zero boundary values, and use the DST-I to
   solve $-u'' = f$ on $(0, 1)$.
5. Implement the $8 \times 8$ two-dimensional DCT-II of an image block with `dct` along rows and columns.

<details>
<summary>Hints and answers</summary>

1. The even extension $x_0, \ldots, x_{n-1}, x_{n-1}, \ldots, x_0$ of length $2n$ is symmetric about $n - \tfrac12$. Its DFT sums $x_j\cos(\pi k(j + \tfrac12)/n)$ times a phase $e^{i\pi k/(2n)}$: that phase is the twiddle, and the cosine sum is the DCT-II.
2. Build the matrix $C_{kj} = \sqrt{2/n}\,c_k\cos(\pi k(2j + 1)/(2n))$ with $c_0 = 1/\sqrt2$ and check $CC^{\mathsf T} = I$ to rounding.
3. A smooth function has rapidly decaying DCT coefficients because its even extension is also smooth, whereas the DFT's periodic extension has a jump unless the function is periodic, so DFT coefficients decay slowly. Expect the DCT error to fall much faster with $k$.
4. The second-difference matrix $T$ (tridiagonal $(-1, 2, -1)$) satisfies $T\mathbf s_k = (2 - 2\cos\frac{\pi k}{n+1})\mathbf s_k$ for $\mathbf s_k = (\sin\frac{\pi jk}{n+1})_j$. Transform $f$ with the DST-I, divide by $(2 - 2\cos\frac{\pi k}{n+1})h^2$ and transform back.
5. Apply the one-dimensional `dct` to each row, then to each column of the result.

</details>

## Further reading

- N. Ahmed, T. Natarajan and K. R. Rao, "Discrete cosine transform", *IEEE Transactions on Computers* C-23 (1974) 90–93.
- G. Strang, "The discrete cosine transform", *SIAM Review* 41 (1999) 135–147.
- K. R. Rao and P. Yip, *Discrete Cosine Transform: Algorithms, Advantages, Applications*, Academic Press, 1990.
- S. A. Martucci, "Symmetric convolution and the discrete sine and cosine transforms", *IEEE Transactions on Signal Processing* 42 (1994)
  1038–1051.
