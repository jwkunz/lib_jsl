# FIR design: windows, frequency sampling and Parks-McClellan

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Find the taps of a linear-phase finite impulse response filter whose frequency response approximates a desired one |
| **Devised by** | Window method: James Kaiser (1966, 1974), the Kaiser window and design formulas; frequency sampling: Lawrence Rabiner, Bernard Gold and C. A. McGonegal (1970); equiripple design: Thomas Parks and James McClellan (1972), with Lawrence Rabiner (1973), after Evgeny Remez (1934) |
| **Accuracy** | `firwin`, `firwin2` and `remez` agree with SciPy to $10^{-15}$, $10^{-16}$ and $10^{-6}$ (`remez` stops on a grid, as SciPy's does) |
| **Cost** | Window and frequency sampling: $O(N)$ and $O(N\log N)$. Parks-McClellan: $O(N^2)$ per exchange for a grid of $O(N)$ points, typically under ten exchanges |
| **API** | [`firwin`](../../api/lib_jsl_dsp/discrete/filters/fir/firwin/fn.firwin.html), [`firwin2`](../../api/lib_jsl_dsp/discrete/filters/fir/firwin2/fn.firwin2.html), [`remez`](../../api/lib_jsl_dsp/discrete/filters/fir/remez/fn.remez.html) ([`RemezFilterType`](../../api/lib_jsl_dsp/discrete/filters/fir/remez/enum.RemezFilterType.html)), [`kaiser_beta`](../../api/lib_jsl_dsp/discrete/windows/kaiser/fn.kaiser_beta.html), [`kaiser_atten`](../../api/lib_jsl_dsp/discrete/windows/kaiser/fn.kaiser_atten.html), [`kaiser_estimate_numtaps`](../../api/lib_jsl_dsp/discrete/windows/kaiser/fn.kaiser_estimate_numtaps.html), [`generate_derivative_taps`](../../api/lib_jsl_dsp/discrete/filters/fir/derivative_filter/fn.generate_derivative_taps.html), [`BoxcarIntegrator`](../../api/lib_jsl_dsp/discrete/filters/fir/boxcar_integrator/struct.BoxcarIntegrator.html) |

</div>

## Overview

A finite impulse response filter is a set of taps $h[0], \ldots, h[N - 1]$, applied by [convolution](convolution.md). FIR filters are always stable, and when the taps are symmetric
or antisymmetric about their centre they have exactly **linear phase**, a pure delay of $(N - 1)/2$ samples at every frequency, which no causal IIR filter
can achieve. Their cost is length: matching a sharp IIR filter takes many more taps. Three design methods cover most needs:

- **windowing** truncates the ideal (infinite) impulse response and tapers it with a window;
- **frequency sampling** specifies the response on a dense grid and inverts the DFT;
- **Parks-McClellan** finds the taps that minimise the largest weighted error, an equiripple design that is optimal in that sense.

## History

The window method is as old as digital filtering. James Kaiser (1966) introduced the Bessel-function window that bears his name,
whose single parameter trades main-lobe width against side-lobe level, and (1974) empirical formulas for its parameter and the
filter length. Lawrence Rabiner, Bernard Gold and C. A. McGonegal (1970) developed frequency-sampling design. Thomas Parks and
James McClellan (1972), at Rice University, recognised that the optimal linear-phase design is a Chebyshev approximation problem
and solved it with Evgeny Remez's exchange algorithm (1934). Their FORTRAN program, with Lawrence Rabiner (1973), became the
standard design tool, and SciPy's `remez` is a direct descendant.

## The idea

### Linear phase and the four types

A symmetric filter of length $N$ has frequency response $H(e^{j\omega}) = e^{-j\omega(N-1)/2}A(\omega)$ with a **real** amplitude $A(\omega)$. Depending on whether $N$ is odd or even and the taps
symmetric or antisymmetric, $A$ is a sum of cosines or sines, giving the four **types**:

| Type | Taps | Length | $A(\omega)$ is forced to zero at | Suits |
|---|---|---|---|---|
| I | symmetric | odd | nowhere | any response |
| II | symmetric | even | $\omega = \pi$ | low-pass, band-pass |
| III | antisymmetric | odd | $\omega = 0$ and $\pi$ | Hilbert transformers, band-pass |
| IV | antisymmetric | even | $\omega = 0$ | differentiators, Hilbert transformers, high-pass |

Every type can be written $A(\omega) = Q(\omega)P(\omega)$ with $P$ a cosine polynomial and $Q$ equal to $1$, $\cos(\omega/2)$, $\sin\omega$ or $\sin(\omega/2)$. This is how one optimisation algorithm serves all four.

### The window method

The ideal low-pass filter with cutoff $\omega_c$ has impulse response $h_d[n] = \frac{\omega_c}{\pi}\operatorname{sinc}(\omega_c n/\pi)$, infinitely long. Truncating it to $N$ taps multiplies it by a rectangle, which
convolves the ideal response with the rectangle's transform. The result has ripple near the edge (the Gibbs phenomenon, 9% overshoot
whatever $N$) and side lobes only 21 dB down. Multiplying by a smoother **window** $w[n]$ instead trades a wider transition for lower side lobes:

![Window-method filters](../generated/filters/fir_windows.svg)

[`firwin`](../../api/lib_jsl_dsp/discrete/filters/fir/firwin/fn.firwin.html) builds the ideal response from sinc functions, one difference of two per band, applies the window, and scales the result to unit gain at the
centre of the first passband. Its window can be any of the library's window functions, which a later chapter treats in detail.

### Kaiser's formulas

The Kaiser window has a parameter $\beta$ that sets the stopband attenuation $A$ (in dB) independently of the length, and the length then sets the
transition width $\Delta\omega$. Kaiser fitted

$$
\beta = \begin{cases} 0.1102(A - 8.7), & A > 50, \\ 0.5842(A - 21)^{0.4} + 0.07886(A - 21), & 21 \le A \le 50, \end{cases} \qquad N - 1 \approx \frac{A - 7.95}{2.285\,\Delta\omega}.
$$

These make window design a recipe: choose $A$ and $\Delta\omega$, read off $\beta$ and $N$. They are accurate to about a decibel:

{{#include ../generated/filters/kaiser_design.md}}

The passband ripple of a window design equals its stopband ripple, so a 60 dB filter also has 60 dB passband accuracy, whether that is needed or not.

### Frequency sampling

[`firwin2`](../../api/lib_jsl_dsp/discrete/filters/fir/firwin2/fn.firwin2.html) takes a piecewise-linear desired gain, given at a list of frequencies (with repeated frequencies for jumps). It interpolates it onto a dense grid,
attaches the linear-phase factor, inverse-transforms by FFT, and windows the result. Any shape can be specified:

![Frequency sampling](../generated/filters/firwin2.svg)

More taps follow the specification more closely, but discontinuities still ring.

### Parks-McClellan: equiripple design

Window designs spread their error unevenly: largest near the band edges, wasted far from them. The **minimax** design instead chooses the taps
to minimise the largest weighted error over the bands of interest,

$$
\min_{h}\max_{\omega \in \text{bands}}W(\omega)\,\big|D(\omega) - A(\omega)\big|.
$$

By Chebyshev's **alternation theorem**, the optimal $P$ (a polynomial of degree $r - 1$ in $\cos\omega$) is characterised by an error that reaches its maximum magnitude $\delta$ at least
$r + 1$ times with alternating signs. Remez's **exchange algorithm** finds it:

1. Guess $r + 1$ reference frequencies.
2. Solve for the polynomial that makes the weighted error exactly $\pm\delta$, alternating, at those frequencies. A closed formula gives $\delta$, and barycentric
   Lagrange interpolation gives the polynomial.
3. Evaluate the error on a dense grid, and move the reference to its local extrema.
4. Repeat until the reference stops changing. The error is then equiripple and optimal.

![Equiripple against window](../generated/filters/fir_remez.svg)

At the same length, the equiripple filter has a sharper transition and an even stopband; the window design's side lobes fall away
where no attenuation was needed. For a fixed transition, Parks-McClellan reaches more attenuation from the same number of taps:

{{#include ../generated/filters/remez_vs_kaiser.md}}

Writing this chapter replaced the library's `remez`. The previous version, a simplified exchange, stopped well short of the optimum.
On a 51-tap low-pass filter its largest error was 12 times SciPy's (−65 dB against −87 dB). It also rejected even lengths,
differentiators and Hilbert transformers, and its tests compared against files the library had generated itself. The new
implementation follows Parks, McClellan and Rabiner for all four types and matches SciPy on six reference designs.

### Differentiators and integrators

An ideal differentiator has $A(\omega) = \omega$. `remez` with `RemezFilterType::Differentiator` designs one up to a chosen band edge, weighting the error by $1/\omega$ so that it is relative;
its desired value is a slope per cycle per sample, as in SciPy. [`generate_derivative_taps`](../../api/lib_jsl_dsp/discrete/filters/fir/derivative_filter/fn.generate_derivative_taps.html) instead smooths a central difference, for derivatives of noisy
data. It is accurate at low frequencies only, and suppresses the rest:

![Differentiators](../generated/filters/differentiators.svg)

Writing this chapter also fixed `generate_derivative_taps`. Every order above one returned first-derivative taps, and the gain
was 0.995 instead of 1. The [`BoxcarIntegrator`](../../api/lib_jsl_dsp/discrete/filters/fir/boxcar_integrator/struct.BoxcarIntegrator.html) is the opposite case, a moving sum of the last $N$ samples computed with one addition and
one subtraction per sample, whatever $N$.

## Algorithm

1. **firwin**: for each band $[f_1, f_2]$, add $f_2\operatorname{sinc}(f_2m) - f_1\operatorname{sinc}(f_1m)$ at offsets $m = n - (N - 1)/2$; multiply by the window; normalise.
2. **firwin2**: interpolate the gains onto $2^k + 1$ points, multiply by $e^{-j\pi(N-1)f}$ ($\times j$ for antisymmetric), inverse real FFT, keep $N$ taps, window.
3. **remez**: build a grid of 16 points per extremal frequency over the bands; divide $D$ by $Q$ and multiply $W$ by $Q$; run the exchange on $x = \cos\omega$; then recover the
   taps by sampling the optimal response at $N$ equally spaced frequencies and inverting the DFT.

## Implementation

```rust,ignore
{{#include ../../../crates/dsp/src/discrete/filters/fir/firwin.rs:firwin}}
```

```rust,ignore
{{#include ../../../crates/dsp/src/discrete/filters/fir/remez.rs:remez_exchange}}
```

- The weighted error's local extrema are found with signed comparisons, so an extremum at a band edge next to a sign change is not missed.
- A reference point whose error rounding puts a hair below $|\delta|$ (possible when $\delta$ is tiny, as for differentiators) is still accepted.
- `remez` frequencies are in the units of `fs`; `firwin` and `firwin2` likewise, with `fs = 2` meaning fractions of Nyquist.

## In practice

```rust,ignore
{{#include ../../../crates/dsp/examples/dsp_fir_design.rs}}
```

```text
Kaiser window: 75 taps, stopband 60.4 dB
Parks-McClellan: 41 taps, stopband 50.2 dB
Parks-McClellan: 51 taps, stopband 58.6 dB
Parks-McClellan: 61 taps, stopband 64.9 dB
firwin2 boost: gains [1.000, 1.027, 1.800] at 0, 0.5 and 0.9 × Nyquist
```

With the stopband weighted ten times the passband, Parks-McClellan meets 60 dB with about 55 taps, where the Kaiser window needs 75.

**Strengths**

- Always stable, exactly linear phase, and simple to apply with FFT methods.
- Arbitrary responses (`firwin2`) and optimal equiripple designs (`remez`), including differentiators and Hilbert transformers.

**Weaknesses**

- Many more taps than an IIR filter of similar sharpness, and a delay of half the length.
- `remez` can fail to converge for very long filters or awkward band specifications; very wide transition bands give poorly
  controlled responses there.

## Exercises

1. Show that a symmetric filter of even length has $A(\pi) = 0$.
2. Derive the window-method taps for an ideal band-pass filter from two low-pass filters.
3. Explain why the passband and stopband ripples of a window design are equal.
4. State the alternation theorem, and check it numerically on a `remez` design by counting the extrema of its weighted error.
5. Design a 31-tap Hilbert transformer with `remez` and compare its magnitude with that of a windowed ideal Hilbert impulse response $2\sin^2(\pi n/2)/(\pi n)$.

## Further reading

- T. W. Parks and J. H. McClellan, "Chebyshev approximation for nonrecursive digital filters with linear phase", *IEEE Transactions on Circuit Theory* 19 (1972) 189–194.
- J. H. McClellan, T. W. Parks and L. R. Rabiner, "A computer program for designing optimum FIR linear phase digital filters", *IEEE Transactions on Audio and Electroacoustics* 21 (1973) 506–526.
- J. F. Kaiser, "Nonrecursive digital filter design using the I₀-sinh window function", *Proceedings of the IEEE International Symposium on Circuits and Systems* (1974) 20–23.
- A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010, sections 7.5–7.8.
