# Convolutional codes and the Viterbi algorithm

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Encode a continuous bit stream with a short shift register, and decode it with maximum likelihood using soft channel information |
| **Devised by** | Peter Elias (1955), convolutional codes; Andrew Viterbi (1967), the decoding algorithm; Jim Omura (1969) and David Forney (1973), its interpretation as dynamic programming on a trellis |
| **Accuracy** | Viterbi decoding is maximum-likelihood over the whole sequence; the K = 7 code with soft decisions reaches $10^{-5}$ at about 4 dB, 5.5 dB better than uncoded BPSK (figure below) |
| **Cost** | $O(2^{K-1})$ add-compare-select operations per information bit, and $O(L\,2^{K-1})$ memory for $L$ bits |
| **API** | [`ConvolutionalCode`](../../api/lib_jsl_information/forward_error_correction/convolutional/struct.ConvolutionalCode.html) (`new`, `nasa_k7`, `encode`, `decode_soft`, `decode_hard`, `outputs`, `tail_len`) |

</div>

## Overview

A **convolutional encoder** passes the data through a shift register of $K - 1$ memory cells. For each input bit it emits $n$ output bits, each the parity of a fixed subset of the register and the input, given by the **generator polynomials**. Since every output depends on the last $K$ inputs, an error affects a
stretch of output and can be corrected from the bits around it. The encoder is a finite-state machine with $2^{K-1}$ states. Its possible outputs over time form a **trellis**, and the **Viterbi algorithm** finds the path through the trellis closest to what was received. It works equally well on hard bits or **soft** reliabilities.

## History

Peter Elias introduced convolutional codes in 1955. Early decoders were sequential (Wozencraft 1957, Fano 1963). Andrew Viterbi's 1967 algorithm was first presented as a proof device. Jim Omura recognised it as dynamic programming, and David Forney (1973) showed it was maximum-likelihood. With the
NASA standard $K = 7$, rate-1/2 code, generators 171 and 133 octal, it flew on Voyager (concatenated with Reed-Solomon) and became the inner code of satellite TV, 802.11a/g Wi-Fi and GSM. Viterbi co-founded Qualcomm. Turbo codes (1993) and LDPC codes later surpassed it, but the trellis idea lives on in their decoders and in equalisers.

## The idea

### The encoder

For constraint length $K$, the register holds the current input and the $K - 1$ previous ones. Generator $g_j$, a $K$-bit mask whose most significant bit multiplies the newest input, gives output $j$ as the parity of the masked register. The classic $K = 3$ code with generators $(7, 5)$ octal = $(111, 101)$ has **free distance** 5:
any two distinct output sequences differ in at least 5 bits, so it corrects 2 errors within a constraint span. The $K = 7$ code has free distance 10. **Termination** appends $K - 1$ zeros to return the register to the zero state, which costs a little rate but makes the end of the trellis known.

### Maximum-likelihood decoding on the trellis

Over AWGN, with channel log-likelihood ratios $L_i = \ln P(c_i = 0)/P(c_i = 1)$, the most likely codeword minimises $\sum_{i: c_i = 1} L_i$. That cost is additive along a trellis path, so dynamic programming applies. For each state, keep only the cheapest path reaching it (the **survivor**). At each step, extend each survivor by both input bits
(**add**), compare the candidates entering each new state, and keep the cheaper (**select**). At the end, trace back from the zero state (or the cheapest state if unterminated). Hard decisions are the special case $L_i = \pm 1$, where the cost is the Hamming distance.

![Viterbi BER](../generated/coding/viterbi_ber.svg)

Soft decisions are worth about 2 dB over hard ones, a classic rule of thumb that the figure confirms. The larger constraint length gains almost 2 dB more, at 16 times the decoding work. {{#include ../generated/coding/viterbi_facts.md}}

## Algorithm

1. **Encode**: for each bit (and the tail zeros), form the register, emit the parity of each generator mask, shift.
2. **Decode**: path metrics start at 0 for state 0 and $\infty$ elsewhere; for each step and state, add the branch costs for inputs 0 and 1 and keep the smaller metric for each next state with its predecessor; trace back from state 0; drop the tail.

## Implementation

```rust,ignore
{{#include ../../../crates/information/src/forward_error_correction/convolutional.rs:viterbi}}
```

- `decode_hard` maps bits to LLRs of ±1 and calls `decode_soft`.
- The decoder keeps the full survivor history and traces back once at the end. Streaming decoders instead trace back over a fixed window of about $5K$ steps.

## In practice

```rust,ignore
{{#include ../../../crates/information/examples/coding_convolutional.rs}}
```

```text
K = 3 (7, 5): 7 bits -> 18 coded bits [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]
hard Viterbi with 2 bit errors: [1, 0, 1, 1, 0, 0, 1]
soft Viterbi with 4 weak errors: [1, 0, 1, 1, 0, 0, 1]
NASA K = 7: 2 outputs per bit, 6 tail bits
```

Four wrong but weak values are outvoted by their reliable neighbours in the soft decoder.

**Strengths**

- Simple encoders; optimal, soft-input decoding at a cost that is linear in the length.
- Natural for streams; well understood performance (union bounds from the distance spectrum).

**Weaknesses**

- Decoding cost grows as $2^K$, which limits the distance achievable; several dB from capacity.
- Errors at the output come in bursts, which an outer code (such as Reed-Solomon) must handle.

## Exercises

1. Draw the state diagram and trellis of the $K = 3$ $(7, 5)$ code, and find its free distance.
2. Show that minimising $\sum_{c_i = 1}L_i$ is equivalent to maximising the likelihood of the codeword.
3. Encode 1011 with the $(7, 5)$ code, flip one bit, and run the Viterbi algorithm by hand.
4. Puncture the $K = 7$ code to rate 2/3 by deleting every fourth coded bit, and decode by setting the missing LLRs to zero.
5. Estimate the bit error rate at high SNR from the free distance: $P_b \approx c\,Q\big(\sqrt{2d_{\text{free}}RE_b/N_0}\big)$.

## Further reading

- A. J. Viterbi, "Error bounds for convolutional codes and an asymptotically optimum decoding algorithm", *IEEE Transactions on Information Theory* 13 (1967) 260–269.
- G. D. Forney, "The Viterbi algorithm", *Proceedings of the IEEE* 61 (1973) 268–278.
- P. Elias, "Coding for noisy channels", *IRE Convention Record* 3, part 4 (1955) 37–46.
- S. Lin and D. J. Costello, *Error Control Coding*, 2nd ed., Pearson, 2004, chapter 12.
