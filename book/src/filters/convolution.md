# Convolution and correlation

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Apply a finite impulse response to a signal, in one piece or as a stream; measure the similarity of two signals as a function of their relative shift |
| **Devised by** | The convolution sum is classical (Euler, Laplace, Cauchy); James Cooley and John Tukey (1965), the FFT that makes fast convolution practical; Thomas Stockham (1966), sectioned high-speed convolution (overlap-add and overlap-save); Charles Knapp and G. Clifford Carter (1976), time-delay estimation by correlation |
| **Accuracy** | Direct sums are exact up to rounding; FFT methods agree with them to about $10^{-15}$ relative (tables below) |
| **Cost** | Direct: $O(NM)$ for $N$ samples and $M$ taps. FFT: $O(N\log N)$ for whole signals, $O(\log M)$ per sample for streams |
| **API** | [`convolve`](../../api/lib_jsl_dsp/discrete/convolve/fn.convolve.html) (`ConvolveMode::{Full, Same, Valid}`, `ConvolveMethod::{Direct, Fft, Auto}`), [`fft_convolve`](../../api/lib_jsl_dsp/discrete/fft_convolution/fn.fft_convolve.html), [`cross_correlate`](../../api/lib_jsl_dsp/discrete/cross_correlation/fn.cross_correlate.html), [`fft_cross_correlate`](../../api/lib_jsl_dsp/discrete/fft_cross_correlation/fn.fft_cross_correlate.html), [`fft_correlation_peak`](../../api/lib_jsl_dsp/discrete/fft_cross_correlation/fn.fft_correlation_peak.html), [`OverlapAddFir`](../../api/lib_jsl_dsp/discrete/filters/fir/overlap_and_add_fir/struct.OverlapAddFir.html), [`FftFir`](../../api/lib_jsl_dsp/discrete/filters/fir/fft_fir/struct.FftFir.html), [`StreamOperator`](../../api/lib_jsl_dsp/discrete/stream_operator/trait.StreamOperator.html) |

</div>

## Overview

A **linear time-invariant** system is completely described by its **impulse response** $h$: the output for any input $x$ is the **convolution**

$$
y[n] = (h * x)[n] = \sum_kh[k]\,x[n - k].
$$

When $h$ has finitely many taps, the system is a **finite impulse response** (FIR) filter, and computing $y$ is the most common operation in
digital signal processing. Done directly it costs one multiply-add per tap per sample. By the convolution theorem, the discrete
Fourier transform turns convolution into multiplication, so with the FFT, long filters cost a few operations per sample whatever
their length. **Cross-correlation** is convolution with one signal reversed (and conjugated). Its peak shows the shift at which two
signals best match, which is how echoes, delays and known waveforms are found.

## History

Convolution sums and integrals go back to Euler and Laplace, and the name to the twentieth century. Before the FFT, digital
filtering meant direct sums. James Cooley and John Tukey's FFT (1965) changed that almost at once: Thomas Stockham (1966)
showed that sectioning a long signal into blocks, transforming each, and joining the results by overlap-add or overlap-save
made "high-speed convolution" faster than direct filtering for filters beyond a few dozen taps. Cross-correlation for delay
estimation is older still, in radar and sonar; Charles Knapp and G. Clifford Carter's generalised cross-correlation (1976)
is the standard analysis of it.

## The idea

### Output lengths and modes

Convolving $N$ samples with $M$ taps gives $N + M - 1$ outputs, the **full** convolution, including the partial overlaps at both ends. **Same** keeps the
central $N$ samples, aligned with the input. **Valid** keeps only the $N - M + 1$ samples where the filter lies entirely inside the signal. The modes
follow NumPy and SciPy, and the library's results agree with them to rounding error (see the [chapter's reference table](index.md#testing-against-scipy)).

### Fast convolution

The DFT of a circular convolution is the product of the DFTs. To see why, take the $L$-point DFT of the circular
convolution $y_n = \sum_{m} h_m\,x_{(n-m) \bmod L}$ and exchange the order of summation:

$$
Y_k = \sum_{n}\sum_{m} h_m\,x_{(n-m) \bmod L}\,e^{-2\pi i kn/L}
    = \sum_m h_m\,e^{-2\pi i km/L}\sum_{r} x_r\,e^{-2\pi i kr/L} = H_k X_k,
$$

where $r = (n - m) \bmod L$. The step uses that $e^{-2\pi i k n/L}$ is periodic in $n$ with period $L$, so replacing
$n$ by $r + m$ only multiplies by $e^{-2\pi i km/L}$ whatever the wrap-around. This is also where the *circular*
nature enters: indices wrap modulo $L$, so products of samples near the end of a block land at the start. Padding both sequences with zeros to a length $L \ge N + M - 1$ makes the
circular convolution equal the linear one, so

$$
h * x = \mathrm{IDFT}_L\big(\mathrm{DFT}_L(h)\cdot\mathrm{DFT}_L(x)\big),
$$

at a cost of three FFTs, $O(L\log L)$, instead of $NM$ multiply-adds. For a whole signal, [`fft_convolve`](../../api/lib_jsl_dsp/discrete/fft_convolution/fn.fft_convolve.html) does exactly this. `convolve` with
`ConvolveMethod::Auto` chooses direct summation when $NM \le 4096$.

### Streams: overlap-add and overlap-save

A stream cannot be transformed whole. Both sectioned methods cut it into blocks of $B$ samples and use FFTs of length $L \ge B + M - 1$ (a power of two):

- **Overlap-add** convolves each block separately, each result being $B + M - 1$ long, and adds the $M - 1$ samples that spill past each block into the start of the next one.
- **Overlap-save** prefixes each block with the last $M - 1$ input samples, convolves circularly, and discards the first $M - 1$ outputs, which are corrupted by
  wrap-around. The remaining $B$ outputs are exact.

Both cost two FFTs per block (the filter's transform is computed once), about $O(\log L)$ operations per sample. Their outputs are identical
to the direct sum:

<p class="caption" id="tab-14-2"><strong>Table 14.2.</strong> Columns: filter taps (20 000 samples), direct (ms), whole-signal FFT (ms), overlap-add stream (ms), and more.</p>

{{#include ../generated/filters/convolution_time.md}}

Direct filtering wins for short filters. The crossover lies near a hundred taps, beyond which the streaming FFT filters cost almost nothing
more per tap. Transforming the whole 20 000-sample signal at once needs three FFTs of 32 768 points, and is slower than sectioning
until the filter is thousands of taps long.

### Cross-correlation and delay estimation

The cross-correlation $r_{xy}[\ell] = \sum_nx[n]\,\overline{y[n - \ell]}$ is large where $y$, shifted by $\ell$, resembles $x$. If $x$ is a delayed copy of $y$ plus independent noise, $r_{xy}$ peaks at the delay. The noise
adds a random background whose standard deviation grows only as $\sqrt N$, while the peak grows as $N$:

![Cross-correlation of a noisy delayed copy](../generated/filters/xcorr_delay.svg)

<p class="caption" id="fig-14-1"><strong>Figure 14.1.</strong> Cross-correlation of a noisy delayed copy.</p>

Here the copy is buried in noise with 1.5 times its amplitude, yet the peak at lag 37 stands far above the background. This is the
principle of the **matched filter** used in radar (pulse compression) and communications (synchronisation).

## Algorithm

For an input of $N$ samples and a filter of $M$ taps.

**Direct convolution.**

1. For each output index $n = 0, \ldots, N + M - 2$, sum $h[k]\,x[n - k]$ over the values of $k$
   for which both indices are in range.
2. Trim the result to the requested mode: all of it (full), the central $N$ samples (same), or
   the $N - M + 1$ samples computed from complete overlaps (valid).

Cost: about $NM$ multiply-adds.

**FFT convolution.**

1. Choose $L$, the next power of two at or above $N + M - 1$.
2. Pad both sequences with zeros to length $L$ and transform them.
3. Multiply the two transforms element by element.
4. Inverse-transform, keep the first $N + M - 1$ values, and trim to the mode.

Cost: three FFTs of length $L$.

**Overlap-save**, for a stream processed in blocks of $B$ samples, with an FFT length
$L \ge B + M - 1$. The transform of the zero-padded filter is computed once. The state is the
last $M - 1$ input samples, initially zeros.

1. Wait until $B$ new samples have arrived.
2. Form the buffer of the $M - 1$ remembered samples followed by the $B$ new ones (padded to
   $L$ if necessary) and transform it.
3. Multiply by the filter's transform and inverse-transform.
4. Discard the first $M - 1$ outputs, which are corrupted by wrap-around, and emit the next
   $B$.
5. Remember the last $M - 1$ input samples for the next block.

At the end of the stream, `flush` pads the remaining input with zeros, so as to emit the last
partial block and the $M - 1$ samples of the filter's tail. The total output is then the full
convolution.

**Overlap-add** uses the same blocks: transform each block of $B$ new samples (zero-padded to
$L$), multiply, inverse-transform to get $B + M - 1$ outputs, add the saved tail of the
previous block to the first $M - 1$ of them, emit $B$, and save the new tail.

## Implementation

```rust,ignore
{{#include ../../../crates/dsp/src/discrete/filters/fir/fft_fir.rs:overlap_save}}
```

- Both streaming filters implement [`StreamOperator`](../../api/lib_jsl_dsp/discrete/stream_operator/trait.StreamOperator.html): `process` accepts chunks of any size and returns whole blocks as they
  complete, `flush` ends the stream, and `reset` starts a new one.
- The FFT length is rounded up to a power of two. The block length is the rest of it, after the $M - 1$ samples of history.
- `fft_correlation_peak` returns the lag of the largest-magnitude correlation; a positive lag means the first signal lags the second.

## In practice

```rust,ignore
{{#include ../../../crates/dsp/examples/dsp_convolution.rs}}
```

```text
Full: [1.0, 1.0, 1.0, 1.0, 1.0, -5.0]
Same: [1.0, 1.0, 1.0, 1.0, 1.0]
Valid: [1.0, 1.0, 1.0, 1.0]
overlap-save: 1062 outputs (block 450, FFT 512), largest difference 2.7e-15
echo delay: 23 samples (correlation 4.00)
```

Convolving with $[1, -1]$ takes first differences. Feeding the stream in chunks of 137 samples, unrelated to its block size of 450,
changes nothing in the output.

**Strengths**

- Exact linear filtering at a cost per sample that, with FFTs, barely depends on the filter length.
- Correlation finds delays and known waveforms in noise far below the signal level.

**Weaknesses**

- FFT methods add a block of latency and give up a few digits to rounding for very long transforms.
- Sectioned filtering assumes a fixed filter; adaptive or time-varying filters need the direct form.

## Exercises

1. Show that zero-padding to $L \ge N + M - 1$ makes circular convolution equal linear convolution, and that $L = N + M - 2$ does not.
2. Count the real multiplications per output sample of overlap-save with block $B$ and FFT length $L = B + M - 1$, and find the $B$ that minimises it for $M = 255$.
3. Show that cross-correlating with a signal is the same as convolving with its time-reversed conjugate.
4. With white noise of variance $\sigma^2$ added to a delayed copy of a unit-power signal of length $N$, compute the ratio of the correlation peak to the background's standard deviation.
5. Implement overlap-add with FFTs of length $2B$ for a block length $B$, and check it against `convolve`.

<details>
<summary>Hints and answers</summary>

1. Linear convolution has $N + M - 1$ non-zero outputs, which fit into $L \ge N + M - 1$ without wrapping. With $L = N + M - 2$ the last output wraps onto the first and the two are summed.
2. Each block costs about $2\cdot\tfrac L2\log_2L$ complex multiplications for the two FFTs plus $L$ for the pointwise product, and produces $B = L - M + 1$ outputs. Minimise the cost per output over $L$; for $M = 255$ the best $L$ is a power of two near $2$ to $4$ times $M$ (for example 2048).
3. $r_{xy}[k] = \sum_nx[n]y^*[n - k]$ is the convolution of $x$ with $h[n] = y^*[-n]$.
4. The peak is $\sum|s_n|^2 = N$ (unit power) while the background has variance $N\sigma^2$, so the ratio of peak to background standard deviation is $\sqrt N/\sigma$.
5. Split the input into blocks of $B$, convolve each with the filter via FFT length $2B$ (taps up to $B$), and add the overlapping tails of consecutive blocks.

</details>

## Further reading

- T. G. Stockham, "High-speed convolution and correlation", *Proceedings of the AFIPS Spring Joint Computer Conference* 28 (1966) 229–233.
- A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010, chapters 2 and 8.
- C. H. Knapp and G. C. Carter, "The generalized correlation method for estimation of time delay", *IEEE Transactions on Acoustics,
  Speech, and Signal Processing* 24 (1976) 320–327.
