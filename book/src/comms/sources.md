# Signal sources: oscillators, CORDIC and waveforms

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Generate sinusoids, quadrature carriers, periodic waveforms and noise at exact frequencies, cheaply and reproducibly, and shift signals in frequency |
| **Devised by** | Jack Volder (1959), CORDIC; John Walther (1971), the unified CORDIC; Joseph Tierney, Charles Rader and Bernard Gold (1971), the direct digital synthesiser |
| **Accuracy** | The 64-bit phase accumulator has a frequency resolution of $f_s/2^{64}$; a 32-bit CORDIC gains about one bit per iteration (table below) |
| **Cost** | One addition per sample for the phase; one `sin_cos` or $k$ shift-and-add steps for the waveform |
| **API** | [`Nco`](../../api/lib_jsl_dsp/discrete/sources/nco/struct.Nco.html), [`NcoSource`](../../api/lib_jsl_dsp/discrete/sources/nco/struct.NcoSource.html) (with [`SineWave`](../../api/lib_jsl_dsp/discrete/sources/sinusoid/struct.SineWave.html), [`QuadratureWave`](../../api/lib_jsl_dsp/discrete/sources/quadrature/struct.QuadratureWave.html), [`SquareWave`](../../api/lib_jsl_dsp/discrete/sources/square/struct.SquareWave.html), [`TriangleWave`](../../api/lib_jsl_dsp/discrete/sources/triangle/struct.TriangleWave.html), [`RampWave`](../../api/lib_jsl_dsp/discrete/sources/ramp/struct.RampWave.html), [`PulseTrainWave`](../../api/lib_jsl_dsp/discrete/sources/pulse_train/struct.PulseTrainWave.html), [`NoiseWave`](../../api/lib_jsl_dsp/discrete/sources/noise/struct.NoiseWave.html)), [`Cordic32`](../../api/lib_jsl_dsp/discrete/sources/cordic/struct.Cordic32.html), [`Cordic64`](../../api/lib_jsl_dsp/discrete/sources/cordic/struct.Cordic64.html), [`cordic_sin_cos`](../../api/lib_jsl_dsp/discrete/sources/cordic/fn.cordic_sin_cos.html), [`FrequencyMixer`](../../api/lib_jsl_dsp/discrete/transformations/frequency_mixer/struct.FrequencyMixer.html) |

</div>

## Overview

Test signals, carriers and local oscillators all start from a **numerically controlled oscillator** (NCO): an integer phase accumulator that wraps around
once per cycle, advanced by a fixed **tuning word** each sample. A waveform function turns the phase into a sample: a sine, a quadrature pair $\cos + i\sin$, a ramp, triangle,
square or pulse train, or held noise. Integer phase makes the frequency exact and the phase free of drift. The sine itself can come from the floating-point `sin`
or from **CORDIC**, which rotates a vector by shifts and additions only, as hardware does. A **frequency mixer** multiplies a signal by a complex exponential to move its spectrum.

## History

Jack Volder (1959) devised CORDIC (coordinate rotation digital computer) at Convair for the navigation computer of the B-58 bomber, which
had to compute trigonometric functions without a multiplier. John Walther (1971) at Hewlett-Packard unified it to cover hyperbolic and linear functions, and it
became the algorithm of the HP-35 calculator. Joseph Tierney, Charles Rader and Bernard Gold (1971) described the direct digital frequency
synthesiser: a phase accumulator addressing a sine table. NCOs of this kind are in every software radio.

## The idea

### Phase accumulation

With a $B$-bit accumulator, the tuning word for frequency $f$ at rate $f_s$ is $\Delta = \mathrm{round}(2^Bf/f_s)$, and the phase after $n$ samples is $n\Delta \bmod 2^B$, read as a fraction of a turn. Integer
overflow is exactly the wrap-around of the phase. The library uses $B = 64$, so the frequency is exact to $f_s/2^{64}$ (about $10^{-14}$ Hz at 1 MHz), and the phase never accumulates
rounding error, however long the oscillator runs. `next_phase_modulated` adds a per-sample offset, giving frequency modulation.

### CORDIC

To rotate the vector $(1, 0)$ by an angle $\theta$, CORDIC performs $k$ micro-rotations by the fixed angles $\pm\arctan2^{-i}$, choosing each sign to drive the residual angle towards zero:

$$
x_{i+1} = x_i - d_iy_i2^{-i}, \qquad y_{i+1} = y_i + d_ix_i2^{-i}, \qquad z_{i+1} = z_i - d_i\arctan2^{-i}.
$$

Each step is two shifts and three additions. The micro-rotations stretch the vector by $K = \prod\sqrt{1 + 2^{-2i}} \approx 1.6468$, which is pre-compensated by starting from $1/K$.
The quadrant is taken from the top two bits of the angle first, so the iterations only cover $[0, \pi/2)$. Each iteration adds about one bit of accuracy:

{{#include ../generated/comms/cordic_accuracy.md}}

A short CORDIC's errors are periodic in the phase, so they appear in the spectrum as **spurs**, not noise:

![NCO spurs](../generated/comms/nco_spurs.svg)

`Cordic32` uses 32-bit integers and reaches about $10^{-8}$; `Cordic64` takes the oscillator's full 64-bit phase and reaches double precision with 52 or more iterations.

### Waveforms and noise

The other waveforms read the phase directly. A **ramp** is the phase itself, scaled to $[-1, 1)$. A **triangle** folds it. A **square** wave or **pulse train** compares it with a duty cycle.
**Noise** sources draw a new uniform or Gaussian value every sample (white), or each time the phase wraps (sample-and-hold, a crude
band-limited noise). All are seeded, so a test signal is reproducible.

### Frequency mixing

Multiplying a complex signal by $e^{2\pi if_0n/f_s}$ shifts its spectrum by $f_0$: down-conversion to baseband in a receiver, or up-conversion in a transmitter. Multiplying a real signal
by a real cosine instead produces both the sum and the difference frequencies, which is why receivers mix with a quadrature (complex) oscillator. [`FrequencyMixer`](../../api/lib_jsl_dsp/discrete/transformations/frequency_mixer/struct.FrequencyMixer.html)
keeps a floating-point phase, wrapped every cycle.

## Algorithm

1. **NCO**: output the current phase, then add the tuning word modulo $2^{64}$.
2. **CORDIC**: pick the quadrant from the top two bits; start from $(1/K)$ rotated by that many quarter turns; run the micro-rotations on the remaining angle with integer shifts (rounded).
3. **Waveform**: map the phase to a sample; scale by the amplitude and add the offset.

## Implementation

```rust,ignore
{{#include ../../../crates/dsp/src/discrete/sources/nco.rs:nco}}
```

```rust,ignore
{{#include ../../../crates/dsp/src/discrete/sources/cordic.rs:cordic}}
```

- The CORDIC types are generated by one macro for 32-bit and 64-bit arithmetic. The arctangent table and the gain compensations
  for every iteration count are constants.
- `NcoSource::generate(n)` returns `n` samples; every source also implements [`Waveform`](../../api/lib_jsl_dsp/discrete/sources/nco/trait.Waveform.html), so new shapes plug into the same oscillator.

## In practice

```rust,ignore
{{#include ../../../crates/dsp/examples/dsp_sources.rs}}
```

```text
phase after 12 samples: 0.250000 turns (exact 0.25)
sine: [0.0000, 0.1305, 0.2588, 0.3827]
quadrature: 1.0000+0.0000i, 0.9914+0.1305i
square, duty 0.25: [1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0]
CORDIC sin/cos(0.5) = 0.479425540, 0.877582562 (exact 0.479425539, 0.877582562)
10-iteration CORDIC: sin(0.5) = 0.47850, gain constant 1.646759
```

**Strengths**

- Exact frequencies and drift-free phase; multiplier-free sines and cosines with CORDIC.
- Reproducible, composable sources for testing every other algorithm.

**Weaknesses**

- Square, ramp and pulse waveforms are not band-limited: their harmonics alias.
- Short CORDICs and truncated phases produce spurs, not white noise.

## Exercises

1. Find the tuning word and the actual frequency for 1 kHz at 48 kHz with a 32-bit accumulator, and the frequency error.
2. Show that the CORDIC gain $K$ converges to about 1.64676 as the number of iterations grows.
3. Show that $\sum_{i \ge 0}\arctan2^{-i} > \pi/2$, so CORDIC covers a quarter turn.
4. Mix a real 1 kHz tone with a real 1.1 kHz cosine and identify the two products; repeat with a complex exponential.
5. Generate a band-limited square wave by summing odd harmonics below Nyquist, and compare its spectrum with `SquareWave`'s.

## Further reading

- J. E. Volder, "The CORDIC trigonometric computing technique", *IRE Transactions on Electronic Computers* 8 (1959) 330–334.
- J. S. Walther, "A unified algorithm for elementary functions", *Proceedings of the Spring Joint Computer Conference* (1971) 379–385.
- J. Tierney, C. M. Rader and B. Gold, "A digital frequency synthesizer", *IEEE Transactions on Audio and Electroacoustics* 19 (1971) 48–57.
- R. Andraka, "A survey of CORDIC algorithms for FPGA based computers", *Proceedings of the ACM/SIGDA FPGA Symposium* (1998) 191–200.
