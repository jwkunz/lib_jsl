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
