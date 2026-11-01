# Trigonometric interpolation and Prony's method

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Interpolate equally spaced samples by sums of sines and cosines (periodic data), or of damped complex exponentials (decaying oscillations) |
| **Devised by** | Trigonometric interpolation: Alexis Clairaut (1754), Joseph-Louis Lagrange (1762), Carl Friedrich Gauss (1805, including the first FFT); Prony's method: Gaspard de Prony (1795) |
| **Accuracy** | Trigonometric: exponential for smooth periodic functions. Prony: exact for exact exponential sums; with noise, reliable when more exponentials are fitted than the signal contains |
| **Cost** | Trigonometric: $O(n^2)$ set-up (a direct DFT), $O(n)$ per evaluation. Prony: two least-squares problems and an eigenvalue problem of the model order |
| **API** | [`FourierInterpolator1D`](../../api/lib_jsl_interpolation/one_d/fourier_1d/struct.FourierInterpolator1D.html), [`PronyModel`](../../api/lib_jsl_interpolation/one_d/pronys_method/struct.PronyModel.html) |

</div>

## Overview

Polynomials are the natural building blocks for functions on an interval. For functions that
repeat, or that oscillate and decay, other building blocks fit far better. **Trigonometric
interpolation** passes a finite Fourier series through equally spaced samples of a periodic
function; for smooth periodic functions it converges exponentially, and it is the reason the
[discrete Fourier transform](../../api/lib_jsl_ffts/index.html) is such a powerful tool.

**Prony's method** goes one step further: it fits a sum of damped oscillations
$\sum_i c_i e^{s_i t}$ with *unknown* frequencies and damping rates, determined from the data
themselves. That makes it both an interpolator and a spectral analysis tool that can resolve
frequencies more finely than the Fourier transform of the same samples, and extrapolate the signal
beyond the data.

## History

Trigonometric interpolation was developed in the eighteenth century to predict the motions of
planets and comets, which are periodic. Clairaut (1754) and Lagrange (1762) gave formulas for the
coefficients, and in 1805 Gauss, computing the orbits of the asteroids Pallas and Juno, devised an
efficient way to evaluate them that is now recognised as the first fast Fourier transform, a century
and a half before Cooley and Tukey.

Gaspard de Prony, an engineer of the French revolutionary government, published his method in 1795
in a study of the expansion of gases, fitting measured pressures with sums of exponentials. It
remained a curiosity until the age of digital signal processing, when it became the ancestor of the
**linear prediction** and **high-resolution spectral estimation** methods (Pisarenko, MUSIC, ESPRIT,
the matrix pencil) used in radar, speech and the analysis of vibrations.

## The idea

### Trigonometric interpolation

Given $n$ samples $y_j = f(x_j)$ at $x_j = x_0 + jT/n$ over one period $T$, the discrete Fourier
transform gives coefficients

$$
\hat y_k = \frac1n \sum_{j=0}^{n-1} y_j\, e^{-2\pi i jk/n},
$$

and the **trigonometric interpolant** is the Fourier series built from them, with frequencies from
$-n/2$ to $n/2$:

$$
p(x) = \sum_{|k| < n/2} \hat y_k\, e^{2\pi i k (x - x_0)/T} \;\; \Big[+\, \hat y_{n/2} \cos\big(\pi n (x - x_0)/T\big) \text{ for even } n\Big].
$$

Pairing $\pm k$ keeps the interpolant real for real data. For even $n$ the highest frequency, the
**Nyquist** frequency, appears only once in the samples, and splitting it evenly between $\pm n/2$ gives
the cosine term above. $p$ passes through every sample, and between them it is the smoothest periodic
function that does so: it contains no frequencies the samples cannot resolve.

For a smooth periodic $f$ the Fourier coefficients decay exponentially, and so does the interpolation
error: 24 samples of $e^{\sin 2\pi x} + 0.3\cos 6\pi x$ give a maximum error of $10^{-13}$. For data that
is *not* periodic, the interpolant treats the jump between the last and first samples as a
discontinuity and develops **Gibbs oscillations** near the ends:

![Trigonometric interpolation](../generated/interpolation/fourier_interpolation.svg)

### Prony's method

Suppose the samples come from a sum of $p$ complex exponentials,
$y_k = \sum_{i=1}^p c_i z_i^k$ with $z_i = e^{s_i \Delta t}$. Each $z_i$ is a root of the polynomial
$z^p + a_1 z^{p-1} + \cdots + a_p$, and therefore the samples satisfy the **linear recurrence**

$$
y_{k+p} + a_1 y_{k+p-1} + \cdots + a_p y_k = 0 \quad \text{for every } k.
$$

Prony's method uses this in three steps, each of them linear:

1. **Linear prediction**: find $a_1, \ldots, a_p$ from the recurrence written for all available $k$, an
   overdetermined system solved by least squares.
2. **Poles**: the $z_i$ are the roots of the polynomial, computed as the eigenvalues of its companion
   matrix. Each pole gives a decay rate $-\operatorname{Re} s_i$ and a frequency $\operatorname{Im} s_i/2\pi$, with
   $s_i = \ln z_i/\Delta t$.
3. **Amplitudes**: with the $z_i$ known, $y_k = \sum c_i z_i^k$ is a linear (Vandermonde) least-squares
   problem for the $c_i$.

For an exact exponential sum, the method recovers every frequency and damping rate exactly from just
$2p$ samples, however close the frequencies, which is why it can outperform Fourier analysis.

### Prony's method and noise

The price of this resolution is sensitivity to noise. With measurement noise the recurrence no
longer holds exactly, and fitting exactly as many exponentials as the signal contains often gives
nonsense. The remedy, analysed by Ramdas Kumaresan and Donald Tufts, is to fit **more** exponentials
than expected: the extra poles absorb the noise as heavily damped components, and the true ones
emerge accurately. On data with 2% noise:

{{#include ../generated/interpolation/prony_table.md}}

![Prony's method on noisy data](../generated/interpolation/prony.svg)

With four exponentials (the true number) the fit finds a single, wrong oscillation. With eight it
recovers both true components to within a few per cent, and its extrapolation follows the true signal
long after the data end.

## Algorithm

- **Trigonometric**: compute the DFT coefficients directly; evaluate the paired series at any $x$,
  extended periodically.
- **Prony**: least squares for the prediction coefficients; companion matrix eigenvalues for the poles;
  complex least squares for the amplitudes; evaluate $\operatorname{Re}\sum_i c_i z_i^{(x - x_0)/\Delta t}$.

## Implementation

```rust,ignore
{{#include ../../../crates/interpolation/src/one_d/pronys_method.rs:prony_steps}}
```

- The trigonometric interpolant computes its coefficients with a direct $O(n^2)$ DFT, once; for long
  signals the [FFT](../../api/lib_jsl_ffts/index.html) does the same in $O(n\log n)$.
- Prony's least-squares problems and the companion eigenvalues use
  [`lib_jsl_linalg`](../../api/lib_jsl_linalg/index.html); the amplitudes are complex, and for real data
  they come in conjugate pairs, so the model's real part is the fitted signal.

## In practice

```rust,ignore
{{#include ../../../crates/interpolation/examples/interpolation_fourier_prony.rs}}
```

**Strengths**

- Trigonometric interpolation: exponential accuracy for smooth periodic data, and exact frequency
  content.
- Prony: frequencies and damping rates from short records, beyond the Fourier resolution limit, and
  meaningful extrapolation.

**Weaknesses**

- Trigonometric interpolation of non-periodic data suffers Gibbs oscillations.
- Prony's method is sensitive to noise and to the model order; overfitting the order helps, but the
  spurious components must be recognised (they are strongly damped) and discarded.

## Exercises

1. Show that the trigonometric interpolant of $\cos(2\pi m x)$ from $n > 2m$ samples is exact, and find what
   it becomes for $n = m$ (aliasing).
2. Derive the linear recurrence satisfied by $y_k = c_1 z_1^k + c_2 z_2^k$.
3. Generate two undamped sinusoids at 100 Hz and 103 Hz, sampled at 1 kHz for 0.1 s, so that the
   Fourier resolution is 10 Hz. Can the DFT separate them? Can Prony's method?
4. Repeat the noisy fit with model orders 4 to 20 and plot the fitted damping rates. Which poles
   are stable as the order increases?
5. Use a trigonometric interpolant of samples of $x(1 - x)$ (which is continuous but has a kink when
   extended periodically) and measure the convergence rate.

## Further reading

- G. R. de Prony, "Essai expérimental et analytique", *Journal de l'École Polytechnique* 1 (1795)
  24–76.
- M. T. Heideman, D. H. Johnson and C. S. Burrus, "Gauss and the history of the fast Fourier
  transform", *IEEE ASSP Magazine* 1 (1984) 14–21.
- R. Kumaresan and D. W. Tufts, "Estimating the parameters of exponentially damped sinusoids and
  pole-zero modeling in noise", *IEEE Transactions on Acoustics, Speech, and Signal Processing* 30
  (1982) 833–840.
- S. M. Kay, *Modern Spectral Estimation: Theory and Application*, Prentice-Hall, 1988.
- L. N. Trefethen, *Approximation Theory and Approximation Practice*, SIAM, 2019, on trigonometric
  interpolation and its convergence.
