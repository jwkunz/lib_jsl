# Synchronisation: phase-locked loops, Costas loops and timing recovery

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Lock a local oscillator to a received carrier, recover the carrier of suppressed-carrier modulation, and find the right instants to sample each symbol |
| **Devised by** | Henri de Bellescize (1932), the phase-locked loop; John Costas (1956), the Costas loop; Kurt Mueller and Markus Müller (1976), timing recovery; C. W. Farrow (1988), the variable fractional-delay filter |
| **Accuracy** | A type-2 loop tracks a phase and a constant frequency offset with zero steady-state error; the Lagrange interpolator's error falls with its order (table below) |
| **Cost** | A few operations per sample for the loops; $(N + 1)^2$ multiply-adds per output for an order-$N$ Farrow interpolator |
| **API** | [`PhaseLockedLoop`](../../api/lib_jsl_dsp/discrete/synchronization/phase_locked_loop/struct.PhaseLockedLoop.html), [`LoopFilter`](../../api/lib_jsl_dsp/discrete/synchronization/phase_locked_loop/struct.LoopFilter.html), [`CostasLoop`](../../api/lib_jsl_dsp/discrete/synchronization/costas_loop/struct.CostasLoop.html) ([`CostasModulation`](../../api/lib_jsl_dsp/discrete/synchronization/costas_loop/enum.CostasModulation.html)), [`MuellerTimingRecovery`](../../api/lib_jsl_dsp/discrete/digital_communications/mueller_timing_detector/struct.MuellerTimingRecovery.html), [`mueller_timing_detector`](../../api/lib_jsl_dsp/discrete/digital_communications/mueller_timing_detector/fn.mueller_timing_detector.html), [`FarrowFilter`](../../api/lib_jsl_dsp/discrete/digital_communications/farrow_filter/struct.FarrowFilter.html), [`farrow_filter`](../../api/lib_jsl_dsp/discrete/digital_communications/farrow_filter/fn.farrow_filter.html) |

</div>

## Overview

A receiver's oscillator never runs at exactly the transmitter's frequency, and its sampling clock never matches the transmitter's symbol clock. Both
must be recovered from the signal itself. A **phase-locked loop** (PLL) compares the incoming phase with its own oscillator and steers it with the filtered
error until they agree. A **Costas loop** does the same for modulated signals whose carrier is suppressed, using a phase detector that ignores the data. **Timing recovery** steers
the sampling instant: a timing-error detector such as **Mueller and Müller's** measures how early or late the samples are, and an interpolator, here a **Farrow filter**, computes the signal between the
samples actually taken.

## History

Henri de Bellescize (1932) patented the phase-locked loop for synchronous radio reception. PLLs spread through television (to lock horizontal
sweep), space communications (Jet Propulsion Laboratory, 1950s), and every modern clock and frequency synthesiser. John Costas (1956), at General Electric,
invented the loop that bears his name for suppressed-carrier communication. Kurt Mueller and Markus Müller (1976) derived a timing-error detector needing only one sample per symbol,
and C. W. Farrow (1988) introduced the filter structure that interpolates at a continuously variable fractional delay. Floyd Gardner's *Phaselock Techniques* and Michael Rice's
*Digital Communications* are the standard engineering references.

## The idea

### The loop

Every loop here has three parts: a **phase detector** producing an error $e[n]$, a **loop filter**, and a **numerically controlled oscillator**. The proportional-integral loop filter
updates a frequency estimate and the phase,

$$
\hat\omega \leftarrow \hat\omega + K_ie, \qquad \hat\phi \leftarrow \hat\phi + K_pe + \hat\omega,
$$

which makes a **type-2** loop. It tracks a constant frequency offset with no steady-state phase error. The gains follow from the normalised noise bandwidth $B_nT$ and the
damping $\zeta$: with $\theta = B_nT/(\zeta + 1/4\zeta)$, $K_p = 4\zeta\theta/(1 + 2\zeta\theta + \theta^2)$ and $K_i = 4\theta^2/(1 + 2\zeta\theta + \theta^2)$. A wider bandwidth acquires faster and tracks better, but lets in more noise:

![PLL acquisition](../generated/comms/pll_lock.svg)

The PLL's detector is $\arg(xe^{-i\hat\phi})$, exact for any phase error, so the loop pulls in from any initial phase.

### Costas loops

For BPSK the received phase is $\phi + \pi d$ with unknown data $d$, and a phase detector must not care which way the symbol points. The Costas detector $\mathrm{sgn}(I)\,Q$ multiplies by the decision and
strips the data, leaving $\sin\theta$ for a phase error $\theta$. It locks with a 180° ambiguity. For QPSK, $\mathrm{sgn}(I)Q - \mathrm{sgn}(Q)I$ removes the four symbols and locks with a 90° ambiguity, which differential coding or a known
preamble resolves:

![Costas loop on QPSK](../generated/comms/costas.svg)

{{#include ../generated/comms/costas_facts.md}} The estimate fluctuates with the noise in proportion to the loop bandwidth.

### Timing recovery

At $L$ samples per symbol, the best sampling instant of each symbol must be found and tracked. The **Mueller-Müller** detector compares consecutive symbol-rate samples $y$ with their decisions $d$,

$$
e = \mathrm{Re}\big(\bar d_{k-1}y_k - \bar d_ky_{k-1}\big),
$$

which is zero on average when the samples fall at the pulse peaks and changes sign with the timing error. A loop filter turns it into the next sampling position, generally between
two samples, where an interpolator evaluates the signal:

![Timing recovery](../generated/comms/timing_recovery.svg)

{{#include ../generated/comms/timing_facts.md}}

### Fractional delay: the Farrow filter

Evaluating a signal at $n + \mu$ with $0 \le \mu < 1$ means fitting a polynomial through neighbouring samples. Lagrange interpolation of order $N$ through $N + 1$ samples is a set of $N + 1$ fixed FIR filters whose outputs
$v_m$ combine as $y(\mu) = v_0 + \mu(v_1 + \mu(v_2 + \cdots))$. That is the **Farrow structure**: only the cheap Horner step depends on $\mu$, so the delay can change every sample. Higher orders are exact for polynomials of higher degree and
accurate to higher frequencies:

{{#include ../generated/comms/farrow.md}}

Cubic interpolation (order 3), the usual choice for timing recovery, has an error of $2 \times 10^{-4}$ (−74 dB) at 0.05 cycles per sample, one reason receivers oversample by four or more.

## Algorithm

1. **PLL**: derotate the input by $e^{-i\hat\phi}$; $e = \arg$ of the result; update the loop filter.
2. **Costas**: derotate, normalise, apply the data-stripping detector; update the loop filter.
3. **Mueller-Müller**: interpolate at the current position; decide; compute $e$ (clipped to ±1); update the period estimate $\omega$ (limited to ±0.5% of nominal) and the position $\mu \mathrel{+}= \omega + g_\mu e$.

## Implementation

```rust,ignore
{{#include ../../../crates/dsp/src/discrete/synchronization/phase_locked_loop.rs:loop_filter}}
```

```rust,ignore
{{#include ../../../crates/dsp/src/discrete/digital_communications/mueller_timing_detector.rs:mueller}}
```

```rust,ignore
{{#include ../../../crates/dsp/src/discrete/digital_communications/farrow_filter.rs:farrow}}
```

- The Costas loops take one sample per symbol (after timing recovery); the PLL takes any complex tone.
- `MuellerTimingRecovery` interpolates with a cubic Farrow filter and implements [`StreamOperator`](../../api/lib_jsl_dsp/discrete/stream_operator/trait.StreamOperator.html), emitting one sample per recovered symbol.
- `farrow_filter(x, delay, order)` delays a whole signal by a fraction of a sample.

## In practice

```rust,ignore
{{#include ../../../crates/dsp/examples/dsp_synchronisation.rs}}
```

```text
PLL frequency 0.01000 cycles/sample (true 0.01000)
Costas: residual imaginary part 2.2e-16, phase ambiguity ±1 on the real axis: -1.000
Farrow delay 0.25: y[20] = -0.72317, sin(0.2 × 19.75) = -0.72319
Mueller-Müller: 800 symbols out, worst error after settling 0.123
```

The BPSK Costas loop locked with the 180° ambiguity: the recovered symbols are the transmitted ones negated. The Mueller-Müller example's
residual error comes from the crude linear pulse, not from the loop.

**Strengths**

- Standard, well-understood loops with gains set from bandwidth and damping; detectors that work through the data.
- Farrow interpolation makes timing recovery possible at any oversampling ratio.

**Weaknesses**

- Phase ambiguities in Costas loops; frequency offsets beyond the pull-in range need a coarse estimator first.
- The Mueller-Müller detector relies on correct decisions, so it needs a reasonable signal-to-noise ratio and carrier lock.

## Exercises

1. Linearise the type-2 loop and find its closed-loop transfer function and natural frequency in terms of $K_p$ and $K_i$.
2. Show that the BPSK Costas detector's output is $\mathrm{sgn}(\cos\theta)\sin\theta$ whatever the data, and explain the 180° ambiguity.
3. Show that the Mueller-Müller error is zero on average for a Nyquist pulse sampled at its peaks.
4. Derive the cubic Lagrange interpolation coefficients as polynomials in $\mu$.
5. Measure how long the PLL takes to lock as a function of $B_nT$ for a fixed frequency offset, and compare with the loop's theoretical acquisition time.

## Further reading

- F. M. Gardner, *Phaselock Techniques*, 3rd ed., Wiley, 2005.
- M. Rice, *Digital Communications: A Discrete-Time Approach*, Pearson, 2009, chapters 7–8 and appendix C.
- K. H. Mueller and M. Müller, "Timing recovery in digital synchronous data receivers", *IEEE Transactions on Communications* 24 (1976) 516–531.
- C. W. Farrow, "A continuously variable digital delay element", *Proceedings of the IEEE International Symposium on Circuits and Systems* (1988) 2641–2645.
