# Entropy, divergence and mutual information

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Measure how much information a source produces, how far one distribution is from another, and how much one variable reveals about another |
| **Devised by** | Claude Shannon (1948), entropy and mutual information; Solomon Kullback and Richard Leibler (1951), the divergence; Frank Gray (1947, patented 1953), the reflected binary code |
| **Accuracy** | Exact sums; the library agrees with SciPy's `stats.entropy` to the 12 digits shown (table below) |
| **Cost** | $O(n)$ for $n$ bytes or probabilities; $O(rc)$ for an $r \times c$ joint table |
| **API** | [`entropy`](../../api/lib_jsl_information/operators/entropy/fn.entropy.html), [`entropy_from_probabilities`](../../api/lib_jsl_information/operators/entropy/fn.entropy_from_probabilities.html), [`cross_entropy`](../../api/lib_jsl_information/operators/entropy/fn.cross_entropy.html), [`kl_divergence`](../../api/lib_jsl_information/operators/entropy/fn.kl_divergence.html), [`mutual_information`](../../api/lib_jsl_information/operators/entropy/fn.mutual_information.html), [`gray_code`](../../api/lib_jsl_information/source_coders/gray_codes/fn.gray_code.html), [`gray_decode`](../../api/lib_jsl_information/source_coders/gray_codes/fn.gray_decode.html), [`gray_code_sequence`](../../api/lib_jsl_information/source_coders/gray_codes/fn.gray_code_sequence.html) |

</div>

## Overview

How much information does a message carry? Shannon's answer is the **entropy**: the average number of bits needed to describe the outcome of a random source, if the description is as efficient as possible. It
sets the limit for lossless compression, which the [next entries](entropy-coding.md) approach. The **cross entropy** and the **Kullback-Leibler divergence** measure the cost of coding with the wrong distribution, and the **mutual
information** measures how much one variable tells about another. Mutual information sets the capacity of a noisy channel, which the [error-correcting codes](../coding/index.md) of the next chapter approach. This entry also covers **Gray codes**, which order binary numbers so that neighbours differ in one bit.

## History

Claude Shannon's "A mathematical theory of communication" (1948) founded information theory. It defined entropy, showed that it is the limit of lossless compression (the **source coding theorem**), defined mutual information and channel capacity, and showed that
reliable communication is possible at any rate below capacity (the **channel coding theorem**). Ralph Hartley (1928) had measured information by the logarithm of the number of possible messages, and the name "entropy" came at John von Neumann's suggestion, from statistical mechanics. Solomon
Kullback and Richard Leibler (1951) introduced the divergence for statistical discrimination. Frank Gray at Bell Labs filed his patent on the reflected binary code in 1947, for pulse-code modulation. The same code is the solution of the Chinese rings puzzle, known for centuries.

## The idea

### Entropy

A source emitting symbol $x$ with probability $p(x)$ has entropy

$$
H(X) = -\sum_x p(x)\log_2 p(x) \quad\text{bits},
$$

with the convention $0\log 0 = 0$.

The formula is an average, and the quantity being averaged has a meaning of its own. Learning
that an event of probability $p$ has occurred conveys $-\log_2 p$ bits of information, its
**surprise**. The definition is forced by two reasonable requirements: a rarer event should be
more informative, and the information from two independent events should add (their
probabilities multiply, and the logarithm turns the product into a sum). An event of probability
$\tfrac12$ conveys 1 bit, the answer to one yes-or-no question; one of probability $\tfrac18$
conveys 3 bits; a certain event conveys nothing. The entropy is the average surprise of the
source's symbols.

It ranges from 0, for a source whose outcome is certain, to $\log_2 n$, for $n$ equally likely
outcomes, which is the most unpredictable a source with $n$ symbols can be. A fair coin has 1 bit
and a byte of random data 8. A biased coin that lands heads 90% of the time has only
$-0.9\log_20.9 - 0.1\log_20.1 = 0.47$ bits: its outcomes are mostly predictable. [`entropy`](../../api/lib_jsl_information/operators/entropy/fn.entropy.html) computes it
for the empirical distribution of the bytes of a slice; [`entropy_from_probabilities`](../../api/lib_jsl_information/operators/entropy/fn.entropy_from_probabilities.html) for a given distribution.

![Binary entropy](../generated/information/binary_entropy.svg)

<p class="caption" id="fig-18-1"><strong>Figure 18.1.</strong> Binary entropy.</p>

Entropy is not just a measure of uncertainty; it is the answer to a practical question. Shannon's
**source coding theorem** says that no lossless code can use fewer than $H$ bits per symbol on
average, and that codes exist using fewer than $H + 1$ (coding single symbols) or $H + \epsilon$
(coding long blocks). Entropy is the limit of lossless compression, and the
[entropy coders](entropy-coding.md) approach it.

There is a caveat about what "the source" is.
The byte entropy only describes symbols taken one at a time. Real data has structure between symbols, and the entropy of blocks, per symbol, falls as the blocks lengthen:

<p class="caption" id="tab-18-1"><strong>Table 18.1.</strong> Columns: source, entropy (bits per byte).</p>

{{#include ../generated/information/entropy_table.md}}

The entropy rate of English is thought to be about 1–1.5 bits per character. Dictionary coders exploit this structure; order-0 entropy coders cannot.

### Cross entropy and divergence

An ideal code gives a symbol of probability $q(x)$ a codeword of $-\log_2 q(x)$ bits. Suppose the
code was designed for a distribution $q$, but the symbols actually occur with distribution $p$.
The average cost per symbol is then the **cross entropy**

$$
H(p, q) = -\sum_x p(x)\log_2 q(x) = H(p) + D(p\,\Vert\,q),
$$

where the **Kullback-Leibler divergence** $D(p\,\Vert\,q) = \sum_x p(x)\log_2\big(p(x)/q(x)\big)$ is the extra cost. It is never negative, it is zero only when $q = p$ (Gibbs' inequality), and it is not symmetric. The cross entropy is infinite if $q$ gives zero probability to a symbol that occurs.
In machine learning, minimising cross entropy is the same as maximising likelihood.

<p class="caption" id="tab-18-2"><strong>Table 18.2.</strong> Columns: quantity (bits), library, SciPy `stats.entropy`.</p>

{{#include ../generated/information/entropy_reference.md}}

### Mutual information

For a pair of variables with joint distribution $p(x, y)$,

$$
I(X; Y) = \sum_{x,y} p(x, y)\log_2\frac{p(x, y)}{p(x)\,p(y)} = H(X) - H(X \mid Y),
$$

where $H(X \mid Y)$ is the uncertainty about $X$ that remains, on average, once $Y$ is known. The
**mutual information** is therefore the reduction in uncertainty about $X$ from knowing $Y$: the
number of bits that one variable tells us about the other. It is symmetric in the two. It is also the divergence of the joint distribution from the product of its marginals, so it is zero exactly when the variables are independent. [`mutual_information`](../../api/lib_jsl_information/operators/entropy/fn.mutual_information.html) takes the joint distribution as a
row-major table. A binary symmetric channel, which flips each bit with probability $\varepsilon$, carries $1 - H(\varepsilon)$ bits per use for a uniform input. That is its capacity:

![BSC mutual information](../generated/information/bsc_information.svg)

<p class="caption" id="fig-18-2"><strong>Figure 18.2.</strong> BSC mutual information.</p>

### Gray codes

In ordinary binary, 3 → 4 changes three bits (011 → 100). A sensor reading the bits of a rotating disc at the moment of that change can briefly report anything from 000 to 111. The **reflected binary code** $g = n \oplus (n \gg 1)$ changes exactly
one bit between consecutive values, so a misread is off by at most one step. {{#include ../generated/information/gray_facts.md}} Decoding XORs together all the right shifts of $g$, which [`gray_decode`](../../api/lib_jsl_information/source_coders/gray_codes/fn.gray_decode.html) does in six doubling steps. Gray codes also label
[constellation points](../comms/modulation.md#constellations-and-gray-labelling) so that the likeliest symbol error flips a single bit.

## Algorithm

**Entropy of a byte sequence.**

1. Count the occurrences of each of the 256 byte values.
2. For each value that occurs, with count $c$ out of $n$ bytes, add
   $\tfrac cn\log_2\tfrac nc$ to the total. Values that do not occur contribute nothing.

**Entropy of a distribution**: the same sum over the given probabilities, skipping zeros.

**Cross entropy and divergence** of $p$ from $q$: sum $-p(x)\log_2q(x)$ over the symbols with
$p(x) > 0$; the result is infinite if any of those has $q(x) = 0$. The divergence is the cross
entropy minus the entropy of $p$.

**Mutual information**, from a joint distribution given as a table.

1. Sum each row to get the marginal distribution $p_r$ of the first variable, and each column
   to get $p_c$ for the second.
2. Sum $p\log_2\big(p/(p_rp_c)\big)$ over the cells with $p > 0$.

**Gray code.** Encode: $g = n \oplus (n \gg 1)$. Decode: start with $n = g$ and apply
$n \leftarrow n \oplus (n \gg 1)$, then with shifts of 2, 4, 8, 16 and 32. Each step doubles the
number of leading bits that are correct, so six steps suffice for 64 bits.

All of these cost one pass over the data or the table.

## Implementation

```rust,ignore
{{#include ../../../crates/information/src/operators/entropy.rs:entropy}}
```

```rust,ignore
{{#include ../../../crates/information/src/source_coders/gray_codes.rs:gray}}
```

- All measures are in bits (base-2 logarithms).
- The distribution functions return `NaN` for negative or non-finite entries and for mismatched lengths, and do not check that probabilities sum to one.

## In practice

```rust,ignore
{{#include ../../../crates/information/examples/information_entropy.rs}}
```

```text
entropy of "aaaa": 0 bits per byte
entropy of "abab": 1 bits per byte
entropy of "abcd": 2 bits per byte
entropy of a fair die: 2.5850 bits (log2 6 = 2.5850)
H(p) = 1.5, cross entropy H(p, q) = 1.5850, D(p||q) = 0.0850
I(X; Y) = 0.5310 bits = 1 - H(0.1) = 0.5310
3-bit Gray code: 000 001 011 010 110 111 101 100
gray_decode(0b110) = 4
```

**Strengths**

- Exact, simple and fast; the measures that bound compression and communication.

**Weaknesses**

- Byte entropy is order-0: it ignores all dependence between symbols, so it is not a compression limit for real data.
- Plug-in estimates from samples are biased low when the alphabet is large compared with the sample.

## Exercises

1. Show that $H(X) \le \log_2 n$ for $n$ outcomes, with equality only for the uniform distribution.
2. Prove Gibbs' inequality $D(p\,\Vert\,q) \ge 0$ from $\ln x \le x - 1$.
3. Show that $I(X; Y) = H(X) + H(Y) - H(X, Y)$, and that it is symmetric.
4. Compute the capacity of the binary erasure channel, which loses each bit with probability $\varepsilon$.
5. Prove that consecutive Gray codes differ in exactly one bit, and that `gray_decode` inverts `gray_code`.

## Further reading

- C. E. Shannon, "A mathematical theory of communication", *Bell System Technical Journal* 27 (1948) 379–423 and 623–656.
- T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley, 2006, chapters 2 and 5.
- S. Kullback and R. A. Leibler, "On information and sufficiency", *Annals of Mathematical Statistics* 22 (1951) 79–86.
- F. Gray, "Pulse code communication", US patent 2 632 058, 1953.
