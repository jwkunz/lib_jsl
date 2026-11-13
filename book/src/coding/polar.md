# Polar codes

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Build codes that provably achieve the capacity of symmetric binary channels, with $O(N\log N)$ encoding and decoding |
| **Devised by** | Erdal Arıkan (2009); Ido Tal and Alexander Vardy (2011–15), list decoding |
| **Accuracy** | Capacity-achieving as $N \to \infty$; at $N = 1024$, rate 1/2, successive cancellation reaches a bit error rate of $10^{-4}$ near 3 dB (figure below) |
| **Cost** | $O(N\log N)$ for encoding and for successive-cancellation decoding; the code construction costs $O(N\log N)$ |
| **Read first** | [Low-density parity-check codes](ldpc.md) |
| **API** | [`PolarCode`](../../api/lib_jsl_information/forward_error_correction/polar/struct.PolarCode.html) (`new`, `encode`, `decode`, `decode_hard`, `info_positions`, `n`, `k`) |

</div>

## Overview

Shannon proved that every noisy channel has a **capacity**: a rate, in data bits per transmitted
bit, below which communication can be made as reliable as desired. For sixty years no code was
known that provably reached capacity with a practical decoder. **Polar codes** were the first,
and the idea behind them is unusual.

Instead of spreading the noise evenly over all the data bits, a polar code *redistributes* it. A
simple transformation is applied to $N = 2^m$ uses of the channel. Seen by a decoder that
recovers the input bits one at a time, in order, each input bit then has its own effective
channel, called a **bit channel**. The transformation makes these **polarise**: most become
either almost perfect or almost useless, and the fraction of nearly perfect ones tends to the
capacity of the original channel as $N$ grows.

Coding is then simple. The $k$ data bits are sent on the $k$ most reliable bit channels. The
others carry nothing: they are **frozen** to zeros that the decoder knows in advance. Encoding
is a network of XORs with the same butterfly structure as the
[FFT](../fft/fft.md#the-radix-2-cooley-tukey-algorithm), and **successive-cancellation** (SC)
decoding walks the same network recursively. With list decoding, polar codes protect the control
channels of 5G.

## History

Erdal Arıkan of Bilkent University introduced channel polarisation in 2009, building on his study of sequential decoding's cutoff rate. He proved that polar codes with SC decoding achieve the capacity of any binary-input symmetric memoryless channel. At practical lengths SC decoding was weaker than LDPC or turbo decoding. Ido Tal and
Alexander Vardy's **list decoding** (2011, published 2015), with a CRC to pick from the list, closed the gap. In 2016 3GPP adopted polar codes for the 5G control channels.

## The idea

### The polarising transform

**The channel.** Everything can be seen on the simplest noisy channel, the **binary erasure
channel**: each transmitted bit either arrives intact or, with probability $z$, is erased and
arrives as "unknown". Its capacity is $1 - z$, the fraction of bits that get through.

**One step.** Take two independent uses of the channel and two input bits $u_1, u_2$. Instead of
sending them directly, send

$$
x_1 = u_1 \oplus u_2, \qquad x_2 = u_2 .
$$

The decoder recovers $u_1$ first and then $u_2$.

- *Deciding $u_1$, with $u_2$ unknown.* Since $u_1 = x_1 \oplus x_2$, both transmitted bits must
  arrive. $u_1$ is lost if either is erased, which has probability
  $1 - (1 - z)^2 = 2z - z^2$. This is a *worse* channel than the original; call it $W^-$.
- *Deciding $u_2$, with $u_1$ now known.* There are two ways to learn $u_2$: directly, from
  $x_2$, or from $x_1$, as $u_2 = x_1 \oplus u_1$. It is lost only if both are erased, with
  probability $z^2$. This is a *better* channel, $W^+$.

With $z = 0.5$, the two bit channels have erasure probabilities 0.75 and 0.25. Nothing has been
gained or lost overall: the capacities add up to $(1 - z)^2 + (1 - z^2) = 2(1 - z)$, twice the
original. In general, writing $I(W)$ for the capacity of a channel $W$,
$I(W^-) + I(W^+) = 2I(W)$. The step has only moved reliability from one bit to the other.

**Repeating.** Apply the same step to two copies of $W^-$ and to two copies of $W^+$, giving four
bit channels, two of them more extreme than before; then again, and so on, $m$ times. At each
level every channel splits into a worse and a better one. A channel that is already nearly
perfect or nearly useless hardly changes, while those in the middle keep being pushed apart, so
after many levels almost all have drifted to one extreme or the other.

In matrix form, one step is $x = uF$ with $F = \begin{pmatrix}1 & 0\\1 & 1\end{pmatrix}$ over
GF(2), and $m$ steps give $x = u\,F^{\otimes m}$, the $m$-fold Kronecker power, computed by $m$
stages of XOR butterflies in $\tfrac N2\log_2N$ operations. The library uses the natural order,
without Arıkan's bit-reversal permutation, so the first split decides the most significant bit
of a channel's index.

### Choosing the frozen set

To build a code we must know which bit channels are the good ones. For the erasure channel the
two formulas above give the answer exactly: starting from the erasure probability $z$ of the
channel, each level replaces every value $z$ by the pair $2z - z^2$ (for $W^-$) and $z^2$ (for
$W^+$). After $m$ levels there are $N$ values, one per bit channel.

For other channels, such as the Gaussian channel, there is no such exact formula. The reliability
of a channel is then measured by its **Bhattacharyya parameter**, a number between 0 (perfect)
and 1 (useless) that bounds its error probability, and for which the same recursion gives upper
bounds. The library designs a code by running the recursion from a chosen *design* erasure
probability and freezing the $N - k$ channels with the largest values. A lower design value
suits a higher signal-to-noise ratio.

![Polarization](../generated/coding/polarization.svg)

<p class="caption" id="fig-19-7"><strong>Figure 19.7.</strong> Polarization.</p>

{{#include ../generated/coding/polar_facts.md}} The rest of the channels are still in between at this length; polarisation is slow, which is why short polar codes need better decoders than SC.

### Successive cancellation

The decoder mirrors the analysis: it decides the bits in index order, each using the channel
outputs together with the decisions already made. It works with
[log-likelihood ratios](../comms/modulation.md#soft-demapping) (LLRs), positive for "probably 0",
and applies the two cases of the single step recursively. For a block of LLRs split into a first
half $a$ and a second half $b$ (the LLRs of the $x_1$ and $x_2$ parts):

1. **The worse channel first.** The first half of the inputs was XORed into the first half of
   the outputs, so its LLRs are those of the XOR of the two halves,

   $$
   f(a, b) = \operatorname{sgn}a\,\operatorname{sgn}b\,\min(|a|, |b|) .
   $$

   The sign is that of an XOR, and the confidence is that of the weaker of the two inputs. (This
   is the min-sum form of the check-node rule of [LDPC decoding](ldpc.md#belief-propagation).)
   Decode the first half recursively from these.
2. **Then the better channel.** Re-encode the decisions just made to get the bits $\hat v$ that
   were XORed onto the first half. Each bit of the second half now has two independent
   observations, $b$ directly and $a$ with the known bit removed, and independent LLRs add:

   $$
   g = b + (-1)^{\hat v}a .
   $$

   Decode the second half recursively from these.
3. Return the re-encoded bits of both halves to the level above.

At the bottom of the recursion a single bit is decided: a frozen bit is set to zero whatever its
LLR, and an information bit takes the sign of its LLR.

The weakness of SC decoding is that a decision, once made, is never revisited. A wrong early bit
corrupts everything that follows. **List decoders** keep the $L$ most likely partial paths
instead of one, and choose among them at the end, usually with the help of a checksum.

![Polar BER](../generated/coding/polar_ber.svg)

<p class="caption" id="fig-19-8"><strong>Figure 19.8.</strong> Polar BER.</p>

The design matters: a code designed for a BEC(0.32), closer to the operating point, beats one designed for a BEC(0.5) at higher signal-to-noise ratios.

## Algorithm

For a code of length $N = 2^m$ with $k$ information bits.

**Construction.**

1. Start with the single value $z$, the design erasure probability.
2. Repeat $m$ times: replace every value $z$ in the list by the pair $2z - z^2$ and $z^2$, in
   the order that matches the library's indexing of the bit channels. The list now has $N$
   values.
3. The $k$ positions with the smallest values are the information positions; the rest are
   frozen.

**Encoding.**

1. Put the information bits in the information positions of a block of $N$ bits, and zeros in
   the frozen positions.
2. Apply the $m$ butterfly stages, one for each span $\ell = 1, 2, 4, \ldots, N/2$ (their order
   does not affect the result). In the stage with span $\ell$, the
   block is divided into groups of $2\ell$ bits, and in each group every bit of the first half
   is replaced by its XOR with the bit $\ell$ places further on: $x_i \leftarrow x_i \oplus x_{i+\ell}$.

The cost is $\tfrac N2\log_2N$ XORs.

**Successive-cancellation decoding**, from $N$ channel LLRs. `decode(llrs, positions)` returns
the re-encoded bits of the block:

1. if the block has one bit: if its position is frozen, the bit is 0; otherwise it is 0 if
   the LLR is positive and 1 if not. Record it and return it;
2. otherwise split the LLRs into halves $a$ and $b$. Compute $f(a, b)$ and call `decode` on it
   for the first half of the positions, obtaining the bits $\hat v$;
3. compute $g = b + (-1)^{\hat v}a$, element by element, and call `decode` on it for the second
   half, obtaining the bits $\hat w$;
4. return the block $(\hat v \oplus \hat w, \hat w)$.

The decoded message is the recorded decisions at the information positions. The cost is
$O(N\log N)$.

## Implementation

```rust,ignore
{{#include ../../../crates/information/src/forward_error_correction/polar.rs:sc_decode}}
```

- Lengths up to $2^{20}$ are supported; decoding takes soft LLRs or hard bits.

## In practice

```rust,ignore
{{#include ../../../crates/information/examples/coding_polar.rs}}
```

```text
N = 8, k = 4: information on channels [3, 5, 6, 7]
codeword [1, 0, 1, 0, 0, 1, 0, 1]
one flipped bit, SC decoding: [1, 0, 1, 1]
N = 1024, k = 512, 41 erasures: recovered true
```

The (8, 4) polar code designed for a BEC(0.5) is the (8, 4) Reed-Muller code, with information on channels 3, 5, 6 and 7.

**Strengths**

- Provably capacity-achieving; explicit construction; simple $O(N\log N)$ encoder and decoder; any rate by choosing $k$.

**Weaknesses**

- SC decoding is weak at short and moderate lengths, and sequential, so slow in hardware; list decoding (not provided here) is needed for competitive performance.
- The construction depends on the channel; the BEC design is an approximation for AWGN.

## Exercises

1. Show that for a BEC($\varepsilon$), $W^-$ and $W^+$ are BECs with erasure probabilities $2\varepsilon - \varepsilon^2$ and $\varepsilon^2$, and that capacity is conserved.
2. Show that $F^{\otimes m}$ is its own inverse over GF(2).
3. Verify that the (8, 4) polar code with information channels 3, 5, 6, 7 is the first-order Reed-Muller code.
4. Count the operations of SC decoding and show that it is $O(N\log N)$.
5. Implement CRC-aided list decoding with $L = 4$ and compare its error rate with SC's.

<details>
<summary>Hints and answers</summary>

1. For a BEC($\varepsilon$), the combined bit $u_1$ (the 'minus' channel) is erased unless both outputs are received: erasure probability $1 - (1 - \varepsilon)^2 = 2\varepsilon - \varepsilon^2$; the 'plus' channel is erased only if both are: $\varepsilon^2$. The capacities $1 - 2\varepsilon + \varepsilon^2 + 1 - \varepsilon^2 = 2(1 - \varepsilon)$ add up to twice the original: conserved.
2. Over GF(2), $F = \begin{pmatrix}1&0\\1&1\end{pmatrix}$ satisfies $F^2 = I$, and the Kronecker power $F^{\otimes m}$ satisfies $(F^{\otimes m})^2 = (F^2)^{\otimes m} = I$.
3. The rows of $F^{\otimes3}$ with indices 3, 5, 6, 7 (0-based, zero-indexed rows of weight at least 4) are those of weight 4 or 8, which span the first-order Reed-Muller code of length 8, dimension 4 and minimum distance 4.
4. SC decoding evaluates $\log_2N$ levels of $N/2$ node operations each; the recursion is $T(N) = 2T(N/2) + O(N)$, which solves to $O(N\log N)$.
5. A list decoder keeps the $L = 4$ best paths and selects, among those, the one that passes the CRC; compare the frame error rates at the same $E_b/N_0$.

</details>

## Further reading

- E. Arıkan, "Channel polarization: a method for constructing capacity-achieving codes for symmetric binary-input memoryless channels", *IEEE Transactions on Information Theory* 55 (2009) 3051–3073.
- I. Tal and A. Vardy, "List decoding of polar codes", *IEEE Transactions on Information Theory* 61 (2015) 2213–2226.
- E. Şaşoğlu, "Polarization and polar codes", *Foundations and Trends in Communications and Information Theory* 8 (2012) 259–381.
