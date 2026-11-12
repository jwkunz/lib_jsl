# DSP: filter design, FIR and IIR filters, adaptive filters

A **filter** changes the frequency content of a signal: it keeps a band and removes the rest, removes one interfering tone,
differentiates, or matches a known waveform. Digital filters come in two kinds. **FIR** filters compute each output from a finite window
of inputs, by convolution. They are always stable and can have exactly linear phase, but need many taps. **IIR** filters also feed back
past outputs, which buys sharp responses from a handful of coefficients at the cost of nonlinear phase and numerical fragility.
**Adaptive** filters adjust their own taps as they run. This chapter covers how to design all three, and how to run them efficiently and
accurately. It is the first of the book's DSP chapters; spectral estimation, windows and multirate processing follow.

## The central questions

- **FIR or IIR?** Linear phase, guaranteed stability and easy adaptivity favour FIR; sharpness at low cost favours IIR.
- **How sharp, and at what order?** A specification (passband ripple, stopband attenuation, transition width) fixes the order of an IIR
  family or the length of an FIR filter. The order-selection functions and Kaiser's formulas compute it.
- **Which approximation?** Butterworth, Chebyshev, elliptic and Bessel trade flatness, sharpness and phase for IIR filters. Windows,
  frequency sampling and equiripple design do the same for FIR filters.
- **How to compute it?** Direct convolution for short FIR filters and FFT methods for long ones. For IIR filters, structures that
  tolerate rounding.

## Background: frequency responses and transfer functions

The DSP chapters use a small vocabulary that is worth collecting in one place. Readers who have
taken a signals course can skip this section.

**Signals and frequencies.** A digital signal is a sequence of samples $x[n]$, taken $f_s$ times
per second (the **sampling rate**). A sinusoid of frequency $f$ hertz advances by
$\omega = 2\pi f/f_s$ radians from one sample to the next; $\omega$ is the frequency in radians
per sample. The highest frequency a sampled signal can represent is the **Nyquist frequency**
$f_s/2$, or $\omega = \pi$, at which the samples alternate in sign. It is convenient to work with
the complex exponential $e^{j\omega n} = \cos\omega n + j\sin\omega n$, which carries a cosine and
a sine together. (Engineers write $j$ for $\sqrt{-1}$, and so do these chapters.)

**Linear time-invariant filters.** A digital filter computes each output sample as a weighted
sum of current and past inputs and, possibly, past outputs:

$$
y[n] = b_0x[n] + b_1x[n-1] + \cdots + b_Mx[n-M] - a_1y[n-1] - \cdots - a_Ny[n-N] .
$$

If there are no feedback terms $a_k$ the filter is FIR, and its **impulse response** (the output
when the input is a single 1 followed by zeros) is just the list of coefficients $b_k$, called
**taps**. With feedback the impulse response goes on for ever: the filter is IIR.

**The frequency response.** Feed such a filter the complex exponential $x[n] = e^{j\omega n}$. A
delay of $k$ samples multiplies it by $e^{-j\omega k}$, so every term of the sum is a multiple of
the input, and the output is the same exponential multiplied by a complex number,

$$
y[n] = H(e^{j\omega})\,e^{j\omega n}, \qquad
H(e^{j\omega}) = \frac{b_0 + b_1e^{-j\omega} + \cdots + b_Me^{-j\omega M}}{1 + a_1e^{-j\omega} + \cdots + a_Ne^{-j\omega N}} .
$$

A filter does not change the frequency of a sinusoid; it changes only its amplitude, by the
factor $|H|$ (the **gain**), and its phase, by $\arg H$. $H$ as a function of $\omega$ is the
**frequency response**. Since any signal is a sum of sinusoids, $H$ describes completely what the
filter does. A **low-pass** filter has $|H| \approx 1$ at low frequencies (its **passband**) and
$|H| \approx 0$ at high ones (its **stopband**), with a **transition band** between.

**Decibels.** Gains are quoted on a logarithmic scale, $20\log_{10}|H|$ decibels (dB). A gain of
$\tfrac1{10}$ in amplitude is $-20$ dB, $\tfrac1{100}$ is $-40$ dB, and $-3$ dB is a factor of
$1/\sqrt2$ in amplitude, or one half in power.

**The transfer function, poles and zeros.** Writing $z$ for $e^{j\omega}$, the frequency response
is the value on the unit circle $|z| = 1$ of a ratio of two polynomials in $z^{-1}$,

$$
H(z) = \frac{B(z)}{A(z)} = \frac{b_0 + b_1z^{-1} + \cdots + b_Mz^{-M}}{1 + a_1z^{-1} + \cdots + a_Nz^{-N}},
$$

the **transfer function**. Here $z^{-1}$ stands for "delay by one sample". The roots of the
numerator are the **zeros** of the filter and the roots of the denominator its **poles**; with an
overall gain factor, they determine it completely. They also give a geometric picture of the
response: as $\omega$ moves round the unit circle, the gain dips where the point $e^{j\omega}$
passes near a zero and peaks where it passes near a pole.

**Stability.** Each pole $p$ contributes a term proportional to $p^n$ to the impulse response.
If $|p| < 1$ the term decays; if $|p| > 1$ it grows without bound. A filter is **stable** exactly
when all its poles lie inside the unit circle. FIR filters have no poles (other than at the
origin) and are always stable.

**Analogue filters.** A circuit is described in the same way, with continuous time. Its response
to $e^{j\Omega t}$ is $H(j\Omega)e^{j\Omega t}$, where the transfer function $H(s)$ is a ratio of
polynomials in the complex variable $s$, and the frequency response is its value on the
imaginary axis $s = j\Omega$. A pole $p$ contributes $e^{pt}$ to the impulse response, so an
analogue filter is stable when all its poles lie in the left half-plane, $\operatorname{Re}p < 0$.

## Testing against SciPy

Each design function mirrors a SciPy function. For 45 cases, covering every IIR family and band type, order selection,
window and equiripple FIR design, notch, peak and comb filters, frequency responses, filtering and convolution, the library's
results agree with SciPy 1.17.1:

{{#include ../generated/filters/reference.md}}

The band-stop critical frequencies differ by about $10^{-7}$ because SciPy's bounded minimiser stops within $10^{-5}$ of the optimal edge, where the library's search
converges fully.

Writing this chapter fixed six problems:

- `remez` did not find the equiripple optimum (its worst error was up to 12 times SciPy's), rejected even lengths, differentiators and
  Hilbert transformers, and was tested against its own output. It is now a full Parks-McClellan implementation.
- `buttord`, `cheb1ord`, `cheb2ord` and `ellipord` returned band-stop orders one too high: two of them searched for the optimal passband
  edges in the wrong direction, and two did not search at all.
- `generate_derivative_taps` returned the first derivative for every order.
- The Kaiser window's Bessel function was accurate to only seven digits.
- `DiscreteLinearFilter` silently assumed $a_0 = 1$.
- Fourteen modules' descriptions were attached to `use` statements, where the API reference does not show them.

## In the library

The crate is `lib_jsl_dsp` ([API reference](../../api/lib_jsl_dsp/index.html)). Analogue designs are under `continuous::filter`; digital filters under `discrete::filters`
(`fir`, `iir`, `adaptive`); convolution and correlation directly under `discrete`. Frequencies follow SciPy: digital edges in the units of `fs`,
which defaults to 2 so that edges are fractions of Nyquist.

| You need | Use | Entry |
|---|---|---|
| To apply an FIR filter, in one piece or as a stream; to find a delay | `convolve`, `fft_convolve`, `OverlapAddFir`, `FftFir`, `cross_correlate`, `fft_correlation_peak` | [Convolution and correlation](convolution.md) |
| An analogue low-pass prototype or a band transformation | `buttap`, `cheb1ap`, `cheb2ap`, `ellipap`, `besselap`, `lp2*_zpk`, `freqs` | [Analogue prototypes](analog-prototypes.md) |
| A digital IIR filter meeting a specification | `buttord`/`butter`, `cheb1ord`/`cheby1`, `cheb2ord`/`cheby2`, `ellipord`/`ellip`, `bessel`, `freqz` | [Digital IIR design](iir-design.md) |
| To run an IIR filter; to remove one tone or a harmonic series | `DiscreteLinearFilter`, `BiquadFilter`, `iirnotch`, `iirpeak`, `iircomb` | [IIR structures](iir-structures.md) |
| A linear-phase FIR filter | `firwin`, `firwin2`, `remez`, `kaiser_beta`, `generate_derivative_taps` | [FIR design](fir-design.md) |
| A filter that learns | `LmsFilter`, `RlsFilter` | [Adaptive filters](adaptive.md) |
