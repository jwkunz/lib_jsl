# Spectra at chosen frequencies: Goertzel, chirp-z and Fourier integrals

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Evaluate a transform at a few frequencies, or on a fine grid over a narrow band, or the continuous Fourier integral of a function, without computing a full FFT |
| **Devised by** | Gerald Goertzel (1958); Lawrence Rabiner, Ronald Schafer and Charles Rader (1969), the chirp-z transform, after Leo Bluestein (1968) |
| **Accuracy** | The chirp-z transform agrees with SciPy's `czt` to $10^{-15}$; Fourier integrals converge spectrally for smooth, decaying functions (table below) |
| **Cost** | Goertzel: $N$ real multiply-adds per bin. Chirp-z: three FFTs of length $\ge N + M - 1$ for $M$ output points |
| **API** | [`goertzel`](../../api/lib_jsl_dsp/discrete/spectral/goertzel/fn.goertzel.html), [`goertzel_power`](../../api/lib_jsl_dsp/discrete/spectral/goertzel/fn.goertzel_power.html), [`goertzel_frequency`](../../api/lib_jsl_dsp/discrete/spectral/goertzel/fn.goertzel_frequency.html), [`chirpz`](../../api/lib_jsl_dsp/discrete/spectral/chirpz/fn.chirpz.html), [`fourier_integral`](../../api/lib_jsl_dsp/continuous/spectral/fourier_integrals/fn.fourier_integral.html) ([`FourierDirection`](../../api/lib_jsl_dsp/continuous/spectral/fourier_integrals/enum.FourierDirection.html)) |

</div>

## Overview

The FFT computes all $N$ DFT bins at once in $O(N\log N)$ operations. Often fewer are wanted. Tone detection, as in telephone keypads, needs a handful of
known frequencies, and the **Goertzel algorithm** computes each with a second-order recursion of one multiply-add per sample. Zooming in on a narrow band at fine
resolution needs many frequencies, but not the FFT's evenly spaced bins around the whole circle; the **chirp-z transform** evaluates the
$z$-transform on any arc or spiral of equally spaced points, again with FFTs. The same machinery computes **continuous Fourier integrals** of sampled functions at
chosen frequencies.

## History

Gerald Goertzel (1958) described the recursion for evaluating single terms of a trigonometric series, and it became the standard
detector for the dual-tone multi-frequency (DTMF) signals of touch-tone telephones. Leo Bluestein (1968) noticed that the identity
$nk = \frac12(n^2 + k^2 - (k - n)^2)$ turns a DFT of any length into a convolution with a "chirp", and Lawrence Rabiner, Ronald Schafer and Charles Rader (1969) generalised it to the chirp-z transform. The [FFT chapter](../fft/bluestein-fractional.md)
uses the same identity for prime-length transforms.

## The idea

### Goertzel's recursion

An FFT computes all $N$ frequency bins at once. When only one or two frequencies matter, as in
detecting a telephone dialling tone, most of that work is wasted. Goertzel's algorithm computes
a single bin with a two-term recursion.

**A DFT bin as a filter output.** The bin is the sum $X(k) = \sum_nx_ne^{-i\omega n}$ with
$\omega = 2\pi k/N$. Consider the one-pole recursive filter $y_n = x_n + e^{i\omega}y_{n-1}$,
starting from zero. Unrolling it, $y_{N-1} = \sum_nx_ne^{i\omega(N-1-n)}$, which is the wanted sum
multiplied by the known phase factor $e^{i\omega(N-1)}$. So running this filter over the block and
reading its final output gives the bin. The filter has a single pole at $e^{i\omega}$ on the unit
circle: it resonates at exactly the frequency of interest.

**Making it real.** That filter needs a complex multiplication per sample. Multiplying its
transfer function above and below by the conjugate factor gives an equivalent filter whose
recursive part has *real* coefficients, with the two poles $e^{\pm i\omega}$:

$$
s_n = x_n + 2\cos\omega\,s_{n-1} - s_{n-2} .
$$

This is run over the whole block with one real multiplication per sample. The complex arithmetic
is postponed to a single step at the end,
$X(k) = e^{-i\omega(N-1)}(s_{N-1} - e^{-i\omega}s_{N-2})$. If only the power is wanted, even that is
unnecessary:

$$
|X(k)|^2 = s_{N-1}^2 + s_{N-2}^2 - 2\cos\omega\,s_{N-1}s_{N-2} .
$$

Nothing in the derivation requires $k$ to be an integer, so any frequency can be targeted, not
only the bin centres of an FFT.

**When it pays.** Each bin costs one real multiply per sample; the FFT costs about $\log_2N$ per
sample for all bins. In practice the FFT's efficiency narrows the margin:

<p class="caption" id="tab-15-5"><strong>Table 15.5.</strong> Goertzel against a full FFT: time and break-even number of bins.</p>

{{#include ../generated/spectral/goertzel_time.md}}

Here one Goertzel bin costs half an FFT, so Goertzel pays off for one or two frequencies, or when samples arrive one at a time and
the result is wanted the moment the block ends.

### The chirp-z transform

The DFT evaluates the spectrum at $N$ frequencies spaced evenly round the whole unit circle. To
look closely at a narrow band, one wants many points within that band only. The
**chirp-z transform** evaluates $X(z_k) = \sum_nx_nz_k^{-n}$ at $M$ points $z_k = AW^{-k}$,
which start at $A$ and advance by the constant ratio $W^{-1}$. With $|A| = |W| = 1$ the points lie
on an arc of the unit circle, beginning at any frequency, with any spacing, and $M$ need not
equal $N$. (With other magnitudes they lie on a spiral.)

It is computed by the same device as
[Bluestein's algorithm](../fft/bluestein-fractional.md#bluesteins-algorithm), of which it is the
general form. Substituting $z_k$ gives a sum containing $W^{nk}$, and writing
$nk = \frac12(n^2 + k^2 - (k - n)^2)$ separates the exponent into parts depending on $n$, on $k$
and on $k - n$:

$$
X(z_k) = W^{k^2/2}\sum_n\big(x_nA^{-n}W^{n^2/2}\big)W^{-(k-n)^2/2} .
$$

The sum is a convolution of the premultiplied input with the chirp $W^{-m^2/2}$, computed with
FFTs in $O((N + M)\log(N + M))$ operations. Zooming on two tones 1.8 bins apart in a 256-sample
record:

![Chirp-z zoom](../generated/spectral/chirpz_zoom.svg)

<p class="caption" id="fig-15-5"><strong>Figure 15.5.</strong> Chirp-z zoom.</p>

The FFT's bins fall on either side of the two peaks; 400 chirp-z points over the band trace the response between them. Zooming adds
points, not resolution. The two peaks are resolved because they are more than one bin apart, and the chirp-z transform shows the shape
the FFT samples coarsely. (Zero-padding the FFT would do the same, at the cost of transforming the whole circle.)

### Fourier integrals

The continuous transform $F(\omega) = \int f(x)e^{-i\omega x}dx$ of a function on $[a, b]$ is approximated by the trapezoidal rule on $M$ equally spaced samples, and evaluated at $M$ equally spaced frequencies
in any range by the chirp-z transform. For smooth functions that decay at the ends of the interval, the trapezoidal rule converges
spectrally:

<p class="caption" id="tab-15-6"><strong>Table 15.6.</strong> Accuracy of a Fourier integral computed with the trapezoidal rule.</p>

{{#include ../generated/spectral/fourier_integral.md}}

With 32 samples the Gaussian's transform is exact to rounding. With 16, the sample spacing exceeds the Nyquist spacing for the
frequencies requested, and the result aliases.

## Algorithm

**Goertzel**, for one frequency $\omega$ and a block of $N$ samples.

1. Precompute $c = 2\cos\omega$. Set $s_{-1} = s_{-2} = 0$.
2. For each sample $x_n$: $s_n = x_n + c\,s_{n-1} - s_{n-2}$. Only the two latest values of $s$
   are kept.
3. After the last sample, compute the complex bin
   $X = e^{-i\omega(N-1)}(s_{N-1} - e^{-i\omega}s_{N-2})$, or the power
   $s_{N-1}^2 + s_{N-2}^2 - c\,s_{N-1}s_{N-2}$.

Cost: one multiplication and two additions per sample.

**Chirp-z transform**, for $N$ samples and $M$ output points $z_k = AW^{-k}$.

1. Premultiply: $a_n = x_nA^{-n}W^{n^2/2}$ for $n = 0, \ldots, N - 1$.
2. Form the chirp $W^{-m^2/2}$ for $m = -(N - 1), \ldots, M - 1$.
3. Convolve the two with FFTs of a power-of-two length at least $N + M - 1$.
4. Postmultiply: $X(z_k)$ is the convolution output for lag $k$ times $W^{k^2/2}$,
   $k = 0, \ldots, M - 1$.

**Fourier integral** of $f$ over $[a, b]$ at $M$ equally spaced frequencies.

1. Sample $f$ at $M$ equally spaced points and multiply by the trapezoidal weights (the
   spacing, halved at the two ends).
2. Evaluate the resulting sum $\sum_nw_nf(x_n)e^{-i\omega x_n}$ at all the requested frequencies
   at once with the chirp-z transform, taking $W$ from the product of the sample spacing and
   the frequency spacing and $A$ from the first frequency.
3. Multiply by the phase factor $e^{-i\omega a}$ that accounts for the starting point of the
   interval.

## Implementation

```rust,ignore
{{#include ../../../crates/dsp/src/discrete/spectral/goertzel.rs:goertzel}}
```

```rust,ignore
{{#include ../../../crates/dsp/src/discrete/spectral/chirpz.rs:chirpz}}
```

- `chirpz` defaults to $M = N$, $A = 1$, $W = e^{-2\pi i/M}$, which reproduces the DFT.
- `fourier_integral` takes a plain function `fn(f64) -> Complex<f64>`; `FourierDirection::Inverse` includes the factor $1/2\pi$.

## In practice

```rust,ignore
{{#include ../../../crates/dsp/examples/dsp_chosen_frequencies.rs}}
```

```text
697:    7.6   770:  102.5   852:    5.6   941:    6.7   1209:    7.7   1336:  102.5   1477:    8.1   
power in bin 20: 8169.6
chirp-z peak at 770 Hz
F(0.0) = 2.0000 (exact 2.0000)
F(2.0) = 0.9090 (exact 0.9093)
F(4.0) = -0.3778 (exact -0.3784)
F(6.0) = -0.0928 (exact -0.0931)
```

The 205-sample block is the length used by DTMF decoders, which places the eight tones close to bin centres; the digit is "5" (770 Hz and 1336 Hz). The box
function's transform converges only at second order, because the box is discontinuous at the ends of its support.

**Strengths**

- Goertzel: minimal cost and memory for a few frequencies, any frequency, and streaming operation.
- Chirp-z: any band at any spacing with FFT efficiency; the DFT of any length.

**Weaknesses**

- Neither improves resolution, which is fixed by the record length.
- The Goertzel recursion's error grows with $N$ for frequencies near 0 or Nyquist, where $2\cos\omega \approx \pm 2$.

## Exercises

1. Derive the Goertzel recursion from the filter $1/(1 - e^{i\omega}z^{-1})$ by multiplying numerator and denominator by $1 - e^{-i\omega}z^{-1}$.
2. Show that the power formula $s_{N-1}^2 + s_{N-2}^2 - 2\cos\omega\,s_{N-1}s_{N-2}$ equals $|X(k)|^2$.
3. Prove Bluestein's identity $nk = \frac12(n^2 + k^2 - (k - n)^2)$, and count the FFT lengths needed for an $M$-point chirp-z transform of $N$ samples.
4. Explain why the trapezoidal rule converges spectrally for a function whose derivatives vanish at both ends of the interval.
5. Choose a block length for DTMF detection at 8 kHz so that all eight tones are within 0.2 bins of integer bins.

<details>
<summary>Hints and answers</summary>

1. The recursion is $s_n = x_n + 2\cos\omega\,s_{n-1} - s_{n-2}$: the denominator $(1 - e^{i\omega}z^{-1})(1 - e^{-i\omega}z^{-1}) = 1 - 2\cos\omega z^{-1} + z^{-2}$ is real, and the numerator $1 - e^{-i\omega}z^{-1}$ is applied once at the end.
2. After $N$ steps $X(k) = e^{i\omega}s_{N-1} - s_{N-2}$ up to a phase; take its squared modulus: $s_{N-1}^2 - 2\cos\omega\,s_{N-1}s_{N-2} + s_{N-2}^2$.
3. $nk = \tfrac12(n^2 + k^2 - (k - n)^2)$ by expanding the square. The chirp-z transform needs one convolution of length at least $N + M - 1$, so three FFTs of that length (rounded up to a convenient size): two forward and one inverse.
4. The Euler-Maclaurin error terms of the trapezoidal rule are the odd derivatives at the end points; if all of them vanish the error is smaller than every power of the step: spectral accuracy.
5. No block length achieves 0.2 bins for all eight DTMF tones at 8 kHz up to $N = 1000$: the best is $N = 299$ (worst deviation $0.22$ bins). The classical $N = 205$ gives $0.27$ bins, which is why a threshold of about $0.3$ bins is used in practice.

</details>

## Further reading

- G. Goertzel, "An algorithm for the evaluation of finite trigonometric series", *American Mathematical Monthly* 65 (1958) 34–35.
- L. R. Rabiner, R. W. Schafer and C. M. Rader, "The chirp z-transform algorithm", *IEEE Transactions on Audio and Electroacoustics* 17 (1969) 86–92.
- A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010, sections 9.2 and 9.6.
