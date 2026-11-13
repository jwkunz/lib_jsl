# Entropy coding: Huffman, Shannon-Fano and arithmetic coding

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Compress a sequence of symbols losslessly by giving frequent symbols short codes and rare ones long codes, as close to the entropy as possible |
| **Devised by** | Claude Shannon (1948) and Robert Fano (1949), Shannon-Fano coding; David Huffman (1952), optimal prefix codes; Jorma Rissanen and Richard Pasco (1976), Ian Witten, Radford Neal and John Cleary (1987), practical arithmetic coding |
| **Accuracy** | Huffman is within 1 bit per symbol of the entropy, Shannon-Fano within 2; arithmetic coding reaches the entropy of its model to within a few bytes per message (tables below) |
| **Cost** | $O(n + k\log k)$ for $n$ bytes and $k$ distinct symbols; arithmetic coding is several times slower per symbol than table lookup |
| **API** | [`SourceCoder`](../../api/lib_jsl_information/source_coders/source_coder/trait.SourceCoder.html), [`try_huffman_compress`](../../api/lib_jsl_information/source_coders/huffman/fn.try_huffman_compress.html), [`try_huffman_decompress`](../../api/lib_jsl_information/source_coders/huffman/fn.try_huffman_decompress.html), [`try_shannon_fano_compress`](../../api/lib_jsl_information/source_coders/shannon_fano/fn.try_shannon_fano_compress.html), [`try_shannon_fano_decompress`](../../api/lib_jsl_information/source_coders/shannon_fano/fn.try_shannon_fano_decompress.html), [`try_arithmetic_compress`](../../api/lib_jsl_information/source_coders/arithmetic/fn.try_arithmetic_compress.html), [`try_arithmetic_decompress`](../../api/lib_jsl_information/source_coders/arithmetic/fn.try_arithmetic_decompress.html) |

</div>

## Overview

An **entropy coder** compresses by exploiting unequal symbol frequencies. Morse code already gave the common letter E a single dot. A **prefix code** assigns each symbol a bit string, none the prefix of another, so the concatenated codes can be decoded without separators.
**Shannon-Fano coding** builds one top-down by splitting the symbols into halves of nearly equal probability. **Huffman coding** builds one bottom-up by merging the two least likely symbols, which is provably optimal among prefix codes. But any prefix
code spends a whole number of bits per symbol. **Arithmetic coding** escapes that limit by encoding the entire message as one number in an interval that shrinks with each symbol. It reaches the entropy even for highly skewed sources. All three coders here use a static
**order-0** model: the byte frequencies of the whole input, stored in the header.

## History

Shannon's 1948 paper and Robert Fano's 1949 MIT report described the top-down splitting construction. In 1951 Fano set his information-theory students the term-paper problem of finding an optimal code. David Huffman, about to give up and sit the
exam instead, found the bottom-up construction and proved it optimal. It was published in 1952 and became the entropy-coding stage of fax, JPEG, MP3 and DEFLATE. Arithmetic coding grew from ideas of Peter Elias (early 1960s, unpublished) and was made practical with finite-precision
arithmetic by Jorma Rissanen and Richard Pasco (1976). Ian Witten, Radford Neal and John Cleary's 1987 article gave the integer implementation with underflow handling followed here. Patents slowed its adoption until the 2000s. It is now standard in H.264, HEVC and modern compressors.

## The idea

### Prefix codes and Kraft's inequality

To compress, give frequent symbols short codewords and rare symbols long ones. The codewords are
simply written one after another, so the decoder must be able to tell where each one ends. A
**prefix code** guarantees this: no codeword is the beginning of another, so as soon as the bits
read so far form a codeword, it is the right one.

For example, for four symbols with probabilities $\tfrac12, \tfrac14, \tfrac18, \tfrac18$, the code

| symbol | A | B | C | D |
|---|---|---|---|---|
| probability | 1/2 | 1/4 | 1/8 | 1/8 |
| codeword | `0` | `10` | `110` | `111` |

is a prefix code. The string `0100110` can only be read as A, B, A, C. Its average length is
$\tfrac12\cdot1 + \tfrac14\cdot2 + \tfrac18\cdot3 + \tfrac18\cdot3 = 1.75$ bits per symbol, against 2
bits for a fixed-length code, and 1.75 bits is exactly the [entropy](entropy.md#entropy) of this
source.

**How short can the codewords be?** Short codewords are a limited resource: a codeword of length
$\ell$ rules out the fraction $2^{-\ell}$ of all longer bit strings, those that begin with it.
The fractions used by all the codewords cannot exceed the whole. This is **Kraft's inequality**:
a binary prefix code with lengths $\ell_x$ exists if and only if

$$
\sum_x 2^{-\ell_x} \le 1 .
$$

Minimising the average length $L = \sum_x p(x)\ell_x$ subject to this constraint, and ignoring for
the moment that lengths must be whole numbers, gives $\ell_x = -\log_2 p(x)$: each symbol's ideal
length is its surprise, and then $L = H$. Rounding the lengths up to whole bits,
$\ell_x = \lceil -\log_2 p(x)\rceil$ (Shannon's code), still satisfies the inequality and costs
less than one extra bit per symbol: $H \le L < H + 1$.

### Shannon-Fano coding

Sort the symbols by decreasing frequency. Split the list where the two parts' total frequencies are as equal as possible, give the first part a `0` and the second a `1`, and recurse. The result is a prefix code, usually good but not always optimal:
an early split that is slightly unbalanced is never revisited.

### Huffman coding

Huffman's algorithm builds the *optimal* prefix code, from the bottom up. A prefix code is a
binary tree whose leaves are the symbols; the codeword of a symbol is the sequence of left (0) and
right (1) branches on the path from the root to its leaf.

Put every symbol in a priority queue keyed by frequency. Repeatedly remove the two least
frequent nodes and join them as the two children of a new node, whose frequency is their sum,
and put that node back in the queue. When one node remains, it is the root. In the example
above, C and D (1/8 each) are joined first into a node of weight 1/4; that node and B (1/4) are
joined into one of weight 1/2; and that is joined with A. The result is the code in the table.

Why is the greedy choice right? An exchange argument shows that the two least likely symbols can always be
siblings at the deepest level of an optimal tree, so the greedy merge is optimal. Ties are broken by the smallest symbol in each subtree, so encoder and decoder build the same tree from the stored frequencies.

### Arithmetic coding

A prefix code must spend a whole number of bits on every symbol, at least one. For a very
predictable source that is far too many: a symbol of probability 0.99 deserves 0.014 bits.
**Arithmetic coding** removes the restriction by encoding the whole message as a single number.

Divide the interval $[0, 1)$ into sub-intervals, one per symbol, with widths equal to the symbol
probabilities. To encode a symbol, replace the current interval by the sub-interval for that
symbol, and divide *that* in the same proportions for the next symbol. For two symbols A
(probability 0.75) and B (0.25), with A assigned the lower part, the message "AB" narrows the
interval first to $[0, 0.75)$ and then to its upper quarter, $[0.5625, 0.75)$.

After $n$ symbols the interval has width $\prod p(x_i)$, the probability of the message, and any
number inside it identifies the message. Specifying a number inside an interval of width $w$
takes about $-\log_2w$ bits, so the message costs about
$-\log_2\prod p(x_i) = \sum -\log_2 p(x_i)$ bits: the sum of the surprises of its symbols, with
no rounding per symbol. On average that is the entropy.

The interval soon becomes far narrower than any floating-point number can represent, so a
practical coder never holds it in full. In finite precision the interval is kept as 32-bit integers $[\text{low}, \text{high}]$. Whenever it lies wholly in the lower or upper half, the leading bit of every number in it is known: the coder emits that bit and doubles the interval (E1, E2). When it straddles the middle within the central half, the next bit is
not yet known. The coder doubles about the centre and counts a **pending** bit, emitted with the opposite value once the next E1 or E2 resolves it (E3). The decoder mirrors these steps. A symbol's sub-interval must never become empty, which requires the model's total frequency to stay below a quarter of the code range, $2^{30}$. For larger inputs the library rescales the model.

{{#include ../generated/information/entropy_coders_table.md}}

The figures are payload bits per symbol, excluding the stored frequency table. For a **dyadic** source, with probabilities $1/2, 1/4, 1/8, \ldots$, the ideal lengths are whole numbers and all three coders reach the entropy. A skewed binary source shows the prefix codes' weakness: a symbol cannot take less than one bit,
so they spend 1 bit where 0.08 suffice, while the arithmetic coder matches the entropy.

### On real text

{{#include ../generated/information/kjv_facts.md}} The order-0 coders all land near that figure; the [dictionary coders](dictionary-coding.md) do better by exploiting repeated words:

{{#include ../generated/information/kjv_table.md}}

The standard compressors combine a dictionary stage with an entropy stage (zlib: LZ77 then Huffman), or use block sorting (bzip2) or context modelling with arithmetic coding (LZMA).

## Algorithm

1. **Model**: count the byte frequencies; store them in the header (symbol and 32-bit count for each byte present).
2. **Shannon-Fano**: sort by frequency; split recursively at the most balanced point, appending a bit at each level.
3. **Huffman**: merge the two least frequent nodes until one remains; read the codes off the tree; pack the bits most significant first.
4. **Arithmetic**: narrow `[low, high]` to each symbol's cumulative range; renormalise by E1, E2 and E3 steps; finish with two bits that select the final interval.
5. **Decoding**: rebuild the same model from the header; walk the tree bit by bit (prefix codes) or find the symbol whose range contains the scaled code value (arithmetic), until the stored length is reached.

## Implementation

```rust,ignore
{{#include ../../../crates/information/src/source_coders/huffman.rs:huffman_tree}}
```

```rust,ignore
{{#include ../../../crates/information/src/source_coders/shannon_fano.rs:shannon_fano_split}}
```

```rust,ignore
{{#include ../../../crates/information/src/source_coders/arithmetic.rs:arithmetic_encode}}
```

- All six coders of this chapter implement [`SourceCoder`](../../api/lib_jsl_information/source_coders/source_coder/trait.SourceCoder.html) (`try_compress`, `try_decompress`), so generic code can swap them. The plain `*_compress` and `*_decompress` functions return an empty vector on error.
- The decoders validate the header (magic, table, lengths) and return an error for malformed input.
- A single-symbol input is stored as its header alone.

## In practice

```rust,ignore
{{#include ../../../crates/information/examples/information_source_coding.rs}}
```

```text
1840 bytes, order-0 entropy 3.917 bits per byte
      Huffman:  1061 bytes (4.61 bits per input byte)
 Shannon-Fano:  1062 bytes (4.62 bits per input byte)
   arithmetic:  1058 bytes (4.60 bits per input byte)
         LZ77:   176 bytes (0.77 bits per input byte)
         LZ78:  1708 bytes (7.43 bits per input byte)
          LZW:   984 bytes (4.28 bits per input byte)
```

On a short message the stored model is a large overhead: 27 symbols × 5 bytes, an eighth of the output. The repeated text suits the dictionary coders, LZ77 above all.

**Strengths**

- Simple, fast prefix coding (Huffman is optimal among symbol-by-symbol codes); arithmetic coding reaches the model's entropy.
- Self-contained streams with validated headers.

**Weaknesses**

- An order-0 static model ignores context, and its stored table is costly for short inputs; adaptive and context models do far better.
- Arithmetic coding is slower and harder to implement correctly.

## Exercises

1. Prove Kraft's inequality for prefix codes.
2. Construct the Huffman code for probabilities (0.4, 0.2, 0.2, 0.1, 0.1) and find its average length and the entropy.
3. Find a distribution on which Shannon-Fano coding is worse than Huffman coding.
4. Show that the Huffman code of a dyadic distribution has $\ell_x = -\log_2 p(x)$ exactly.
5. Encode "ABA" with $p(A) = 0.8$, $p(B) = 0.2$ by exact interval arithmetic, and find the shortest binary fraction inside the final interval.

## Further reading

- D. A. Huffman, "A method for the construction of minimum-redundancy codes", *Proceedings of the IRE* 40 (1952) 1098–1101.
- I. H. Witten, R. M. Neal and J. G. Cleary, "Arithmetic coding for data compression", *Communications of the ACM* 30 (1987) 520–540.
- T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley, 2006, chapter 5.
- D. Salomon and G. Motta, *Handbook of Data Compression*, 5th ed., Springer, 2010.
