# Real and multidimensional FFTs

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Compute the DFT of real data, returning only the non-redundant half of the spectrum; compute DFTs of two-, three- and $d$-dimensional arrays |
| **Devised by** | The two-for-one and half-length tricks: James Cooley, Peter Lewis and Peter Welch (1970); real-data FFTs: Henrik Sorensen, Douglas Jones, Michael Heideman and C. Sidney Burrus (1987); multidimensional transforms by separability: classical |
| **Accuracy** | $O(\varepsilon\log n)$, as the underlying complex FFT |
| **Cost** | Real FFT: one complex FFT of length $n/2$ plus $O(n)$ work. $d$-dimensional: $O(N\log N)$ for $N$ points in all |
| **API** | [`RealFft`](../../api/lib_jsl_ffts/real_fft/struct.RealFft.html), [`rfft`](../../api/lib_jsl_ffts/real_fft/fn.rfft.html), [`irfft`](../../api/lib_jsl_ffts/real_fft/fn.irfft.html), [`fft2`](../../api/lib_jsl_ffts/multidimensional_fft/fn.fft2.html), [`ifft2`](../../api/lib_jsl_ffts/multidimensional_fft/fn.ifft2.html), [`fft3`](../../api/lib_jsl_ffts/multidimensional_fft/fn.fft3.html), [`fftn`](../../api/lib_jsl_ffts/multidimensional_fft/fn.fftn.html), [`fft_axis`](../../api/lib_jsl_ffts/multidimensional_fft/fn.fft_axis.html), [`fftshift`](../../api/lib_jsl_ffts/multidimensional_fft/fn.fftshift.html) |

</div>

## Overview

Most measured signals are real, and the DFT of a real signal is **conjugate symmetric**: $X_{n-k} = \overline{X_k}$. Half the spectrum is
redundant, and a complex FFT of real data wastes half its work on zero imaginary parts. A **real FFT** packs the $n$ real samples into
$n/2$ complex numbers, transforms those, and untangles the result: about half the arithmetic, returning the $n/2 + 1$ independent bins.

Images, volumes and fields are multidimensional, and the multidimensional DFT is **separable**: it is the composition of
one-dimensional DFTs along each axis. A two-dimensional transform of an image is a transform of every row followed by a transform
of every column.

## History

James Cooley, Peter Lewis and Peter Welch (1970) described the standard ways to exploit real data: transforming two real signals
with one complex FFT, and transforming one real signal of length $n$ with a complex FFT of length $n/2$. Henrik Sorensen, Douglas Jones,
Michael Heideman and C. Sidney Burrus (1987) developed dedicated real-valued FFT algorithms. Multidimensional FFTs by rows and columns
are as old as the FFT itself; the row-column method's poor use of cache for large arrays later motivated transposition and
vector-radix algorithms.

## The idea

### The real FFT by half-length packing

**Half the spectrum is redundant.** If the samples $x_m$ are real, taking the complex conjugate
of the DFT sum only changes the sign of the exponent, which is the same as replacing $k$ by
$n - k$. So $X_{n-k} = \overline{X_k}$: the upper half of the spectrum is the mirror image of the
lower half, conjugated. Only the bins $k = 0, \ldots, n/2$ carry information, and a transform
that takes $n$ real numbers in and gives $n/2 + 1$ complex numbers out should be able to do about
half the work.

**The packing trick.** For even $n = 2m$, one way to achieve this is to let a complex FFT of half
the length carry two real sequences at once, one in its real part and one in its imaginary part.
Pack the even- and odd-indexed samples as

$$
z_j = x_{2j} + i\,x_{2j+1}, \qquad j = 0, \ldots, m - 1,
$$

and compute the complex DFT $Z$ of length $m$. Let $E$ and $O$ be the DFTs of the even and odd
samples separately. The DFT is linear, so $Z_k = E_k + iO_k$. That is one complex equation for two
complex unknowns, but a second comes from the symmetry just described. $E$ and $O$ are DFTs of
*real* sequences, so $\overline{E_{m-k}} = E_k$ and $\overline{O_{m-k}} = O_k$, and therefore

$$
\overline{Z_{m-k}} = \overline{E_{m-k}} - i\,\overline{O_{m-k}} = E_k - iO_k .
$$

Adding and subtracting the two equations separates the transforms:

$$
E_k = \tfrac12\big(Z_k + \overline{Z_{m-k}}\big), \qquad O_k = \tfrac1{2i}\big(Z_k - \overline{Z_{m-k}}\big) .
$$

Finally $E$ and $O$ are exactly the two half-length transforms of the
[radix-2 FFT](fft.md#the-radix-2-cooley-tukey-algorithm), so one more butterfly step combines
them into the spectrum of the original signal:

$$
X_k = E_k + e^{-2\pi ik/n}O_k, \qquad k = 0, \ldots, m .
$$

The inverse transform reverses these steps. Odd lengths cannot be packed this way and use a
complex transform of the full length.

**What it buys in practice.** The arithmetic is about halved, but the packing and untangling are
extra $O(n)$ passes over memory, which the fast vectorised FFT does not hide. On this machine:

{{#include ../generated/fft/real_fft.md}}

`RealFft` takes a quarter to a third less time than a complex FFT of the real data for lengths of a few thousand or more, but gains
nothing for short signals. The untangling loop was much slower until recently: it took two integer remainders per bin to handle the
wrap-around at $k = 0$ and $k = m$, and `RealFft` was 1.6 to 2.1 times *slower* than the complex FFT it is meant to replace. It now
handles those two bins separately and computes the bins $k$ and $m - k$ together.

### Multidimensional transforms

The DFT of a $d$-dimensional array, such as an image ($d = 2$) or a volume ($d = 3$), has one
frequency index per axis:

$$
X_{k_1\cdots k_d} = \sum_{m_1}\cdots\sum_{m_d}x_{m_1\cdots m_d}\,e^{-2\pi i(k_1m_1/n_1 + \cdots + k_dm_d/n_d)} .
$$

The exponential of a sum is a product of exponentials, one per axis. In two dimensions, for
example,

$$
X_{k_1k_2} = \sum_{m_1}e^{-2\pi ik_1m_1/n_1}\Big(\sum_{m_2}x_{m_1m_2}\,e^{-2\pi ik_2m_2/n_2}\Big) .
$$

The inner sum is a one-dimensional DFT of row $m_1$; the outer sum is then a one-dimensional DFT
down each column of the result. So no new algorithm is needed: transform every line parallel to
axis 1, then every line parallel to axis 2, and so on. The cost is $O(N\log N)$ for
$N = n_1\cdots n_d$ points.

Each output is the amplitude of a plane wave, a sinusoid with $k_1$ cycles along the first axis
and $k_2$ along the second. A real plane wave in the data transforms to a pair of peaks, at
$(k_1, k_2)$ and its mirror image. `fftshift` moves the zero frequency from the corner of the
array to its centre for display, as is conventional for images.

## Algorithm

**`RealFft::forward`**: pack, transform with a complex FFT of length $n/2$, set $X_0$ and $X_{n/2}$ from $Z_0$, and untangle the bins $k$ and $m - k$ in pairs.

**`fftn`**: for each axis, copy each line along that axis to a buffer, transform it with an `Fft` planned for that length, and write it back.

## Implementation

```rust,ignore
{{#include ../../../crates/ffts/src/real_fft.rs:real_forward}}
```

```rust,ignore
{{#include ../../../crates/ffts/src/multidimensional_fft.rs:fft_axis}}
```

- `RealFft::forward` is unscaled; `RealFft::inverse` divides by $n$, so the pair round-trips exactly.
- The multidimensional transforms work on `ndarray` arrays of complex numbers and accept any lengths along each axis.

## In practice

```rust,ignore
{{#include ../../../crates/ffts/examples/fft_real_and_2d.rs}}
```

**Strengths**

- Real transforms return only the non-redundant half spectrum and save time for long signals.
- Any number of dimensions and any lengths, through one-dimensional engines.

**Weaknesses**

- The real FFT's $O(n)$ passes limit its gain to about a quarter to a third; a dedicated real-data algorithm would do better.
- The row-column method copies every line; transposition-based or vector-radix algorithms make better use of cache for large arrays.

## Exercises

1. Prove that the DFT of a real sequence is conjugate symmetric.
2. Derive $E_k$ and $O_k$ from $Z$ and verify the untangling formula for $n = 4$.
3. Use one complex FFT to transform two real signals at once, and separate the results.
4. Show that the two-dimensional DFT of a separable array $x_{ij} = a_ib_j$ is the outer product of the one-dimensional DFTs.
5. Compute the 2D power spectrum of an image with a periodic stripe pattern and locate the stripe frequency.

## Further reading

- J. W. Cooley, P. A. W. Lewis and P. D. Welch, "The fast Fourier transform algorithm: programming considerations in the calculation of
  sine, cosine and Laplace transforms", *Journal of Sound and Vibration* 12 (1970) 315–337.
- H. V. Sorensen, D. L. Jones, M. T. Heideman and C. S. Burrus, "Real-valued fast Fourier transform algorithms", *IEEE Transactions on
  Acoustics, Speech, and Signal Processing* 35 (1987) 849–863.
- W. H. Press et al., *Numerical Recipes*, 3rd ed., Cambridge University Press, 2007, §12.3–12.5.
