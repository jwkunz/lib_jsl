# Polyphase filter banks

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Split a wideband signal into many equally spaced frequency channels, each at a reduced sampling rate (a channeliser), and recombine channels into one signal |
| **Devised by** | Maurice Bellanger and J. L. Daguet (1974), the polyphase DFT filter bank; Ronald Crochiere and Lawrence Rabiner (1981), weighted overlap-add; P. P. Vaidyanathan (1987–1993), perfect-reconstruction theory; fred harris (2003), polyphase channelisers for radio |
| **Accuracy** | Analysis channels have the prototype's response (60 dB stopband by default, test below); synthesis reconstructs the input only for one-tap-per-phase prototypes (table below) |
| **Cost** | Per block of $D$ input samples: $L$ multiply-adds for a prototype of $L$ taps, and one $M$-point FFT |
| **API** | [`CriticallySampledPolyphaseAnalysisFilterBank`](../../api/lib_jsl_dsp/discrete/spectral/critically_sampled_polyphase_filter_bank/struct.CriticallySampledPolyphaseAnalysisFilterBank.html), [`CriticallySampledPolyphaseSynthesisFilterBank`](../../api/lib_jsl_dsp/discrete/spectral/critically_sampled_polyphase_filter_bank/struct.CriticallySampledPolyphaseSynthesisFilterBank.html), [`OversampledPolyphaseAnalysisFilterBank`](../../api/lib_jsl_dsp/discrete/spectral/oversampled_polyphase_filter_bank/struct.OversampledPolyphaseAnalysisFilterBank.html), [`OversampledPolyphaseSynthesisFilterBank`](../../api/lib_jsl_dsp/discrete/spectral/oversampled_polyphase_filter_bank/struct.OversampledPolyphaseSynthesisFilterBank.html) |

</div>

## Overview

A **uniform DFT filter bank** splits a signal into $M$ channels centred on $c\,f_s/M$, $c = 0, \ldots, M - 1$, each band-pass filtered by a shifted copy of one low-pass **prototype** $h$ and decimated. Built directly it needs $M$ filters
running at the full rate. The **polyphase** structure gets every channel from one shared polyphase filter and a single FFT per output block, at roughly
the cost of filtering the input once. With decimation by $M$ the bank is **critically sampled**. With decimation by $M/K$ it is **oversampled** by $K$, which relaxes the prototype's transition band
and suppresses aliasing between channels. A **synthesis** bank reverses the process. Channelisers of this kind sit in every wideband radio receiver,
spectrum monitor and radio telescope correlator.

## History

Maurice Bellanger and J. L. Daguet (1974) introduced the polyphase network with an FFT to build transmultiplexers for telephony, which
converted between frequency-division and time-division multiplexed channels. Ronald Crochiere and Lawrence Rabiner (1981) gave the weighted
overlap-add formulation. P. P. Vaidyanathan, Mark Smith, Thomas Barnwell and others (1980s) worked out the conditions for perfect
reconstruction. fred harris's work (2003) on polyphase channelisers made oversampled banks with arbitrary output rates standard in software radio.

## The idea

### The polyphase analysis bank

Channel $c$ of the bank filters by $h[n]e^{2\pi icn/M}$ and decimates. Splitting the filter sum into $M$ interleaved parts, $n = rM + j$, the output at block time $t$ is

$$
X_c[t] = \sum_{j=0}^{M-1}e^{2\pi icj/M}\underbrace{\sum_rh[rM + j]\,x[tD - rM - j]}_{v_j[t]},
$$

an $M$-point DFT of the **polyphase components** $v_j$, the input delayed by $j$ and filtered by the $j$-th phase of the prototype. One block costs $L$ multiply-adds and one FFT, however many channels.
The default prototype is a Kaiser-window low-pass with cutoff at half a channel spacing and 60 dB attenuation:

![Channel responses](../generated/spectral/filter_bank_channels.svg)

Each channel is a shifted copy of the prototype's response, with 60 dB of rejection outside its band.

Writing this chapter fixed both analysis banks. They multiplied the sample $rM + j$ places back by the tap $h[rM + M - 1 - j]$, reversing the prototype within each block of $M$
taps. With the default prototype the channels' side lobes were only 15–20 dB down. The tests used a prototype of ones, which the reversal leaves
unchanged.

### Critical sampling and oversampling

Decimating each channel by $M$ keeps the total sample rate equal to the input's, but a practical prototype's transition band then overlaps the neighbouring channels, and their
aliases fold into each channel. Decimating by $M/2$ (oversampling by 2) leaves room for the transition band, so a channel is free of aliases from its neighbours.
This is the usual choice for channelisers.

### Synthesis and reconstruction

A synthesis bank interpolates each channel, modulates it back to its centre frequency, and sums: an inverse FFT per block, polyphase filtering, and overlap-add.
For a critically sampled bank, perfect reconstruction with FIR filters requires specially designed prototypes. For an oversampled bank, a prototype whose shifted
power responses sum to a constant reconstructs nearly perfectly. The library's synthesis banks reconstruct exactly with a prototype of one tap per phase
(a rectangular window, which makes the bank a block DFT and its inverse), but **not** with longer prototypes. Measured with white noise, after the best
delay and complex gain are removed:

{{#include ../generated/spectral/filter_bank_reconstruction.md}}

With the default prototypes, the synthesis output differs from the input by 18–41% of its level, at a gain near $1/M^2$. The analysis banks are
correct and can be used on their own as channelisers. The synthesis banks should be used only with one-tap prototypes until they are reworked.

## Algorithm

1. **Analysis**: keep the last $L$ inputs; every $D$ inputs, form $v_j = \sum_rh[rM + j]\,x[\text{newest} - rM - j]$ for $j = 0, \ldots, M - 1$ (arranged in reverse order for the FFT), and transform.
2. **Synthesis**: inverse-transform each frame, weight the polyphase history by the prototype, and overlap-add blocks of the hop length.

## Implementation

```rust,ignore
{{#include ../../../crates/dsp/src/discrete/spectral/critically_sampled_polyphase_filter_bank.rs:pfb_analysis}}
```

- The number of channels must be a power of two, and a supplied prototype's length a multiple of it.
- Oversampled banks take the oversampling factor $K$ (a power of two dividing $M$); their hop is $M/K$.
- The banks take complex input and produce one complex vector of $M$ channels per block, through the [`StreamOperator`](../../api/lib_jsl_dsp/discrete/stream_operator/trait.StreamOperator.html) interface.

## In practice

```rust,ignore
{{#include ../../../crates/dsp/examples/dsp_filter_banks.rs}}
```

```text
256 frames of 8 channels (hop 4)
channel 0:   -91.3 dB
channel 1:   -74.4 dB
channel 2:     0.0 dB
channel 3:   -74.4 dB
channel 4:   -89.2 dB
channel 5:   -20.0 dB
channel 6:   -91.9 dB
channel 7:   -95.8 dB
```

The strong tone appears in channel 2 and the weak one (−20 dB, slightly off the centre of channel 5) in channel 5. The other channels see only
the prototype's stopband leakage, more than 74 dB down.

**Strengths**

- Many channels for the cost of one filter and one FFT per block; every channel has the prototype's selectivity.
- Oversampling suppresses aliasing between channels with modest prototypes.

**Weaknesses**

- The synthesis banks do not yet reconstruct with practical prototypes (see above).
- Fixed, uniform channel spacing; power-of-two channel counts only.

## Exercises

1. Derive the polyphase form of the analysis bank from the definition of channel $c$'s filter $h[n]e^{2\pi icn/M}$.
2. Show that with a one-tap-per-phase prototype the critically sampled analysis bank is a block DFT, and its synthesis bank the inverse.
3. For an oversampled bank with $K = 2$, find the largest transition band a prototype can have without aliasing between adjacent channels.
4. State the condition on the prototype for near-perfect reconstruction in an oversampled weighted overlap-add bank, and check whether a Kaiser-window low-pass with cutoff $1/(2M)$ satisfies it.

## Further reading

- M. G. Bellanger and J. L. Daguet, "TDM-FDM transmultiplexer: digital polyphase and FFT", *IEEE Transactions on Communications* 22 (1974) 1199–1205.
- R. E. Crochiere and L. R. Rabiner, *Multirate Digital Signal Processing*, Prentice Hall, 1983, chapter 7.
- P. P. Vaidyanathan, *Multirate Systems and Filter Banks*, Prentice Hall, 1993.
- f. j. harris, *Multirate Signal Processing for Communication Systems*, Prentice Hall, 2004.
