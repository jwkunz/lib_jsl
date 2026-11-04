# Parametric spectra: linear prediction, Burg and MUSIC

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Estimate a spectrum by fitting a model, an autoregressive filter driven by white noise or a sum of sinusoids in noise, to resolve detail finer than the record's DFT bins |
| **Devised by** | George Udny Yule (1927) and Gilbert Walker (1931), autoregressive models; Norman Levinson (1947) and James Durbin (1960), the fast recursion; John Burg (1967), maximum entropy spectra; Ralph Schmidt (1979, 1986), MUSIC |
| **Accuracy** | Levinson-Durbin agrees with SciPy's `solve_toeplitz` to $10^{-15}$; estimation accuracy depends on the model order and the data (tables below) |
| **Cost** | Levinson-Durbin $O(p^2)$ after $O(Np)$ for the autocorrelation; Burg $O(Np)$; MUSIC $O(m^3)$ for an $m \times m$ covariance plus $O(m^2)$ per frequency |
| **API** | [`autocorrelation`](../../api/lib_jsl_dsp/discrete/spectral/linear_predictive_coding/fn.autocorrelation.html), [`levinson_durbin`](../../api/lib_jsl_dsp/discrete/spectral/linear_predictive_coding/fn.levinson_durbin.html), [`linear_predictive_coding`](../../api/lib_jsl_dsp/discrete/spectral/linear_predictive_coding/fn.linear_predictive_coding.html), [`lpc_residual`](../../api/lib_jsl_dsp/discrete/spectral/linear_predictive_coding/fn.lpc_residual.html), [`lpc_synthesize`](../../api/lib_jsl_dsp/discrete/spectral/linear_predictive_coding/fn.lpc_synthesize.html), [`burg`](../../api/lib_jsl_dsp/discrete/spectral/maximum_entropy/fn.burg.html), [`ar_psd`](../../api/lib_jsl_dsp/discrete/spectral/maximum_entropy/fn.ar_psd.html), [`maximum_entropy_spectrum`](../../api/lib_jsl_dsp/discrete/spectral/maximum_entropy/fn.maximum_entropy_spectrum.html), [`music`](../../api/lib_jsl_dsp/discrete/spectral/music/fn.music.html), [`music_real`](../../api/lib_jsl_dsp/discrete/spectral/music/fn.music_real.html), [`music_doa`](../../api/lib_jsl_dsp/discrete/spectral/music/fn.music_doa.html) |

</div>

## Overview

A periodogram assumes nothing about the signal and resolves no finer than one DFT bin, $f_s/N$. If the signal is known to have structure, a model
with a few parameters can be fitted and its spectrum computed exactly, at any resolution. Two models dominate:

- **Autoregressive** (AR): the signal is white noise passed through an all-pole filter, $x_n = -\sum_{k=1}^pa_kx_{n-k} + e_n$. Fitting the $a_k$ is **linear prediction**. The **Levinson-Durbin** recursion does it from the
  autocorrelation, and **Burg's method** does it directly from the data. The fitted spectrum $\sigma^2/|A(e^{i\omega})|^2$ is smooth with sharp peaks, and Burg called it the **maximum entropy** spectrum.
- **Sinusoids in white noise**: the signal's covariance matrix splits into a signal subspace and an orthogonal noise subspace. **MUSIC**
  finds the frequencies whose steering vectors are orthogonal to the noise subspace.

Both resolve tones far closer than a bin when the model fits. Both mislead when it doesn't.

## History

George Udny Yule (1927) fitted an autoregression to sunspot numbers, and Gilbert Walker (1931) generalised it. Their normal equations bear
both names. Norman Levinson (1947) solved Toeplitz systems in $O(p^2)$, and James Durbin (1960) adapted the method to autoregressions. Linear prediction became the core
of speech coding (Bishnu Atal and Manfred Schroeder; Fumitada Itakura and Shuzo Saito, late 1960s). John Burg (1967), working on
seismic data, proposed the maximum entropy method, which estimates the reflection coefficients from forward and backward prediction
errors and always gives a stable model. Ralph Schmidt's MUSIC (multiple signal classification, 1979, published 1986) introduced subspace methods
for frequency and direction-of-arrival estimation.

## The idea

### Linear prediction and the Yule-Walker equations

Predicting $x_n$ from its past by $\hat x_n = -\sum_{k=1}^pa_kx_{n-k}$ with the smallest mean squared error leads to the **Yule-Walker** equations $\sum_{k=1}^pa_kr_{|j-k|} = -r_j$ for $j = 1, \ldots, p$, where $r$ is the autocorrelation. The matrix is Toeplitz, and
the **Levinson-Durbin** recursion solves it in $O(p^2)$ by building the order-$m$ predictor from the order-$(m - 1)$ one:

$$
k_m = -\frac{r_m + \sum_{j=1}^{m-1}a_j^{(m-1)}r_{m-j}}{E_{m-1}}, \qquad a_j^{(m)} = a_j^{(m-1)} + k_ma_{m-j}^{(m-1)}, \qquad E_m = E_{m-1}(1 - k_m^2).
$$

The **reflection coefficients** $k_m$ have magnitude below one exactly when the autocorrelation is positive definite, which guarantees a stable synthesis filter $1/A(z)$.
They are also the coefficients of a lattice filter, a model of the vocal tract as a tube of uniform sections in speech coding. The prediction error power $E_p$
is the driving noise variance, and the AR spectrum is $\sigma^2/|A(e^{i\omega})|^2$.

### Burg's method

The autocorrelation method uses the biased estimate $r_k = \frac1N\sum x_nx_{n+k}$, which implicitly pads the data with zeros and biases the fit. Burg's method skips the autocorrelation. At
each order it chooses the reflection coefficient that minimises the sum of the forward and backward prediction error powers,

$$
k_m = \frac{-2\sum_nf_n^{(m-1)}b_{n-1}^{(m-1)}}{\sum_n\big((f_n^{(m-1)})^2 + (b_{n-1}^{(m-1)})^2\big)},
$$

then updates the errors and the predictor by Levinson's step. By the Cauchy-Schwarz inequality $|k_m| \le 1$, so the model is always stable. For sharply resonant processes,
Burg's estimates are far better than the autocorrelation method's:

{{#include ../generated/spectral/ar_estimation.md}}

The process is Marple's classic test case, $A(z) = 1 - 2.7607z^{-1} + 3.8106z^{-2} - 2.6535z^{-3} + 0.9238z^{-4}$, with poles at radius 0.98. The table shows the RMS error of the four coefficients, averaged over 50 records.
The autocorrelation method's zero-padding bias dominates even at 1024 samples.

### Resolution beyond the periodogram

Two tones 0.015 cycles per sample apart, in 64 samples, are less than one bin apart ($1/64 = 0.0156$):

![Parametric resolution](../generated/spectral/parametric_resolution.svg)

The Hann periodogram shows a single broad hump. A Burg AR(16) model resolves two peaks, and MUSIC places sharp peaks almost exactly on the true frequencies:
{{#include ../generated/spectral/music_peaks.md}}

### MUSIC

For $q$ complex exponentials in white noise, the $m \times m$ covariance matrix of the data is $R = \sum_i P_i\,\mathbf a(\omega_i)\mathbf a(\omega_i)^{\mathsf H} + \sigma^2I$, with **steering vectors** $\mathbf a(\omega) = (1, e^{i\omega}, \ldots, e^{i(m-1)\omega})$. Its $m - q$ smallest eigenvalues equal $\sigma^2$, and their
eigenvectors $E_n$ span a **noise subspace** orthogonal to every steering vector of a true frequency. The MUSIC **pseudospectrum**

$$
P(\omega) = \frac{1}{\lVert E_n^{\mathsf H}\mathbf a(\omega)\rVert^2}
$$

peaks sharply at the true frequencies. It is not a power spectrum: its height says nothing about power. A real sinusoid is two complex
exponentials, so `music_real` uses a signal subspace of dimension $2q$. The same pseudospectrum, with array steering vectors, estimates directions of arrival (`music_doa`).
The covariance eigenvalues confirm the model order: a gap after the $q$-th (or $2q$-th) separates signal from noise.

## Algorithm

1. **Levinson-Durbin**: $E_0 = r_0$; for $m = 1, \ldots, p$: compute $k_m$, update $a$ and $E$; fail if $|k_m| \ge 1$.
2. **Burg**: initialise $f = b = x$; for each order compute $k_m$ from $f$ and the lagged $b$, update both error sequences and $a$.
3. **MUSIC**: form the forward-backward averaged sample covariance from sliding windows of $m$ samples, take its eigendecomposition,
   evaluate the pseudospectrum on a grid, and refine the peaks by parabolic interpolation.

## Implementation

```rust,ignore
{{#include ../../../crates/dsp/src/discrete/spectral/linear_predictive_coding.rs:levinson}}
```

```rust,ignore
{{#include ../../../crates/dsp/src/discrete/spectral/maximum_entropy.rs:burg}}
```

```rust,ignore
{{#include ../../../crates/dsp/src/discrete/spectral/music.rs:music}}
```

- `levinson_durbin` returns $A(z) = 1 + \sum a_kz^{-k}$; SciPy's `solve_toeplitz` solves for the predictor coefficients, which have the opposite sign.
- `lpc_residual` filters by $A(z)$ (the whitening, analysis filter); `lpc_synthesize` by $1/A(z)$.

## In practice

```rust,ignore
{{#include ../../../crates/dsp/examples/dsp_parametric.rs}}
```

```text
LPC   A(z) = [1.0000, -1.5921, 0.8912], reflection [-0.8419, 0.8912]
Burg  A(z) = [1.0000, -1.5934, 0.8921], noise variance 0.08145 (true 0.08333)
prediction gain 12.3 dB
MUSIC peaks [0.1000, 0.1300, 0.3100]
largest covariance eigenvalues [8.10, 7.89, 7.87, 7.55, 7.10, 6.32, 0.00, 0.00]
```

For a moderately resonant AR(2) process and 2000 samples, both fits are close to the true $1 - 1.6z^{-1} + 0.9z^{-2}$. MUSIC finds the three tones, and the eigenvalues fall off sharply after
the sixth, the signal subspace of three real sinusoids.

**Strengths**

- Resolution beyond the DFT bin width from short records; smooth spectra with few parameters.
- Linear prediction is the basis of speech coding; MUSIC extends directly to sensor arrays.

**Weaknesses**

- The model order must be chosen: too low merges peaks, too high produces spurious ones.
- Model mismatch (coloured noise, non-sinusoidal signals) gives confidently wrong answers; MUSIC's pseudospectrum carries no power information.

## Exercises

1. Derive the Yule-Walker equations from the orthogonality of the prediction error to the past samples.
2. Show that $E_m = E_{m-1}(1 - k_m^2)$, and deduce $|k_m| < 1$ for a positive definite autocorrelation.
3. Show that Burg's reflection coefficient minimises the sum of forward and backward error powers, and that $|k_m| \le 1$.
4. Show that the steering vector of a true frequency is orthogonal to the noise subspace of $R$.
5. Fit AR models of orders 2 to 30 to the two-tone example and plot the spectra. Where do spurious peaks appear?

## Further reading

- S. L. Marple, *Digital Spectral Analysis with Applications*, Prentice Hall, 1987 (2nd ed., Dover, 2019).
- J. P. Burg, "Maximum entropy spectral analysis", PhD thesis, Stanford University, 1975.
- R. O. Schmidt, "Multiple emitter location and signal parameter estimation", *IEEE Transactions on Antennas and Propagation* 34 (1986) 276–280.
- J. Makhoul, "Linear prediction: a tutorial review", *Proceedings of the IEEE* 63 (1975) 561–580.
- P. Stoica and R. Moses, *Spectral Analysis of Signals*, Prentice Hall, 2005.
