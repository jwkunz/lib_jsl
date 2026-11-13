# The Hilbert transform and the analytic signal

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Form the complex signal whose real part is a given real signal and whose spectrum has no negative frequencies; read off the instantaneous amplitude, phase and frequency |
| **Devised by** | David Hilbert (1905), the integral transform; Dennis Gabor (1946), the analytic signal; Lawrence Marple (1999), the discrete-time construction by the DFT |
| **Accuracy** | Agrees with SciPy's `hilbert` to $10^{-16}$ |
| **Cost** | Two FFTs of the signal length |
| **Read first** | [The fast Fourier transform](../fft/fft.md) |
| **API** | [`analytic_signal`](../../api/lib_jsl_dsp/discrete/spectral/hilbert/fn.analytic_signal.html), [`hilbert_transform`](../../api/lib_jsl_dsp/discrete/spectral/hilbert/fn.hilbert_transform.html), [`instantaneous_attributes`](../../api/lib_jsl_dsp/discrete/spectral/hilbert/fn.instantaneous_attributes.html) ([`InstantaneousAttributes`](../../api/lib_jsl_dsp/discrete/spectral/hilbert/struct.InstantaneousAttributes.html)) |

</div>

## Overview

A real signal's spectrum is symmetric: every positive frequency has a negative twin carrying the same information. Discarding the negative
half and doubling the positive half gives the **analytic signal** $z = x + i\mathcal Hx$, whose imaginary part is the **Hilbert transform** of $x$, the signal with every component shifted by
90°: $\cos \to \sin$, $\sin \to -\cos$. For a narrow-band signal $a(t)\cos\phi(t)$, the analytic signal is $a(t)e^{i\phi(t)}$, so its magnitude is the **envelope** and the derivative of its phase is the **instantaneous frequency**.
Demodulation, envelope detection, single-sideband modulation and the analysis of chirps and vibrations all rest on it.

## History

David Hilbert studied the integral transform $\mathcal Hf(t) = \frac1\pi\,\mathrm{p.v.}\int f(\tau)/(t - \tau)\,d\tau$ in 1905 in connection with boundary-value problems for analytic functions, whose real and imaginary parts on the
boundary are Hilbert transforms of each other. Dennis Gabor (1946) introduced the analytic signal in his "Theory of communication", defining
the instantaneous amplitude and frequency of a real signal. Lawrence Marple (1999) gave the discrete-time analytic signal computed with the
DFT, the construction used by MATLAB and SciPy.

## The idea

### Removing negative frequencies

**The motivating example.** A real oscillation $A\cos\theta(t)$ hides its amplitude and its
phase behind each other: from the value at one instant there is no telling whether the signal is
small because $A$ is small or because the cosine is near zero. The complex signal
$Ae^{i\theta(t)}$ has no such ambiguity. Its magnitude is $A$ and its angle is $\theta$, at every
instant. The **analytic signal** is the systematic way to pass from the first to the second.

The link is Euler's formula, $\cos\theta = \tfrac12(e^{i\theta} + e^{-i\theta})$. A real cosine is the
sum of a positive-frequency exponential and a negative-frequency one, each of half the
amplitude. Deleting the negative-frequency half and doubling the positive one leaves exactly
$e^{i\theta}$. Doing this to every frequency component of a real signal $x$ gives its analytic
signal $z$.

**The Hilbert transform.** The imaginary part of $z$ is called the **Hilbert transform**
$\mathcal Hx$ of $x$, so that $z = x + i\mathcal Hx$. It turns every cosine into a sine: each
frequency component is shifted in phase by a quarter cycle, which in the frequency domain is
multiplication by $-i\,\mathrm{sgn}(f)$. The analytic signal therefore has spectrum
$X(f)(1 + \mathrm{sgn}(f))$: twice $X$ for $f > 0$, $X$ at $f = 0$, and zero for $f < 0$, as
described. In discrete time, with an $N$-point DFT, the recipe is:

$$
Z_k = \begin{cases} X_k, & k = 0 \text{ or } k = N/2, \\ 2X_k, & 0 < k < N/2, \\ 0, & k > N/2, \end{cases} \qquad z = \mathrm{IDFT}(Z).
$$

The DC and Nyquist bins are their own negatives and are kept once. The real part of $z$ is exactly $x$.

### Envelope and instantaneous frequency

![Envelope of an AM signal](../generated/spectral/hilbert_envelope.svg)

<p class="caption" id="fig-15-6"><strong>Figure 15.6.</strong> Envelope of an AM signal.</p>

The magnitude $|z|$ of the analytic signal is the **envelope** of $x$, the slowly varying
amplitude within which it oscillates. For an amplitude-modulated carrier, $|z|$ recovers the modulation, here exactly, because the modulation (3 Hz) is far below the carrier (60 Hz) and the record holds
whole periods of both. The angle $\arg z$ is the instantaneous phase, and its rate of change is the
**instantaneous frequency**, $\frac1{2\pi}\frac{d}{dt}\arg z$, computed from phase differences between samples:

![Instantaneous frequency of a chirp](../generated/spectral/hilbert_frequency.svg)

<p class="caption" id="fig-15-7"><strong>Figure 15.7.</strong> Instantaneous frequency of a chirp.</p>

The estimate follows the chirp's linear sweep except at the ends. There the DFT's implicit periodic extension joins the high-frequency
end to the low-frequency start, and the Gibbs ripple of that discontinuity disturbs the phase. Taper or pad the signal if the ends matter.

### When it makes sense

The envelope and instantaneous frequency are meaningful only for **narrow-band** signals. For a sum of two tones, the analytic signal's phase
advances at a rate that beats between them, and the instantaneous frequency can even be negative. The decomposition of a signal into amplitude and phase is not unique,
and the analytic signal is one particular, well-behaved choice; Bedrosian's theorem says when it separates them correctly.

## Algorithm

**Analytic signal** of a real signal of $N$ samples.

1. Take the FFT of the signal.
2. Modify the bins: leave bin 0 (DC) unchanged; double bins $1, \ldots, \lceil N/2\rceil - 1$
   (the positive frequencies); leave the Nyquist bin $N/2$ unchanged if $N$ is even; set the
   remaining bins (the negative frequencies) to zero.
3. Take the inverse FFT. The result $z$ is complex: its real part is the original signal and
   its imaginary part is the Hilbert transform.

**Attributes** computed from $z$:

1. the envelope is $|z_n|$;
2. the instantaneous phase is $\arg z_n$, **unwrapped**: whenever the phase jumps by more than
   $\pi$ between consecutive samples, a multiple of $2\pi$ is added to remove the jump, so that
   the phase is continuous;
3. the instantaneous frequency is the difference of the unwrapped phase between consecutive
   samples, times $f_s/2\pi$. It has one value fewer than the signal.

The cost is two FFTs.

## Implementation

```rust,ignore
{{#include ../../../crates/dsp/src/discrete/spectral/hilbert.rs:analytic_signal}}
```

- The transform is circular: it assumes the record is one period of a periodic signal.
- `instantaneous_attributes` returns frequencies one sample shorter than the signal, at the midpoints between samples.
- An FIR Hilbert transformer for streaming use can be designed with [`remez`](../filters/fir-design.md#parks-mcclellan-equiripple-design) and `RemezFilterType::Hilbert`.

## In practice

```rust,ignore
{{#include ../../../crates/dsp/examples/dsp_hilbert.rs}}
```

```text
cos → analytic signal at n = 0, 5: 1.000+0.000i, 0.000+1.000i
envelope at t = 0.5 s: 1.0000 (true 1.0000)
instantaneous frequency: mean 100.000 Hz
```

**Strengths**

- Exact, fast, and parameter-free envelope and phase extraction for narrow-band signals.

**Weaknesses**

- Circular edge effects; meaningless "instantaneous" quantities for broadband or multi-component signals.

## Exercises

1. Show that the Hilbert transform of $\cos\omega t$ is $\sin\omega t$ for $\omega > 0$, and that $\mathcal H(\mathcal Hx) = -x$ for signals without a DC component.
2. Show that the analytic signal of $a(t)\cos(\omega_0t)$ is $a(t)e^{i\omega_0t}$ when the spectrum of $a$ lies below $\omega_0$ (Bedrosian's theorem).
3. Compute the instantaneous frequency of $\cos t + 0.5\cos 3t$ and find where it is negative.
4. Explain why the analytic signal's DC and Nyquist bins must not be doubled.

<details>
<summary>Hints and answers</summary>

1. The spectrum of $\cos\omega t$ is two impulses at $\pm\omega$; the transform multiplies positive frequencies by $-j$ and negative by $+j$, giving $\sin\omega t$. Applying it twice multiplies each non-zero frequency by $(-j)^2 = -1$ (for positive) and $(j)^2 = -1$ (for negative), so $\mathcal H^2x = -x$.
2. $\mathcal H[a\cos\omega_0t] = a\,\mathcal H[\cos\omega_0t] = a\sin\omega_0t$ when the spectrum of $a$ does not overlap that of the carrier; then the analytic signal $x + j\mathcal Hx = ae^{j\omega_0t}$.
3. The analytic signal is $z = e^{it} + 0.5e^{3it}$ and the instantaneous frequency is $\operatorname{Im}(z'/z) = \dfrac{1.75 + 2\cos2t}{1.25 + \cos2t}$. It is negative where $\cos2t < -\tfrac78$, i.e. near $t = \pi/2 + k\pi$ (within about $\pm0.25$ rad): negative instantaneous frequency arises because the signal is not a single narrow-band component.
4. The DC and Nyquist bins are their own conjugates (they have no mirror image to cancel), so doubling them would change the real part; they are left unscaled, and only the positive-frequency bins are doubled.

</details>

## Further reading

- D. Gabor, "Theory of communication", *Journal of the IEE, Part III* 93 (1946) 429–457.
- S. L. Marple, "Computing the discrete-time analytic signal via FFT", *IEEE Transactions on Signal Processing* 47 (1999) 2600–2603.
- B. Boashash, "Estimating and interpreting the instantaneous frequency of a signal", *Proceedings of the IEEE* 80 (1992) 520–568.
