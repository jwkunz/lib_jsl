# Linear feedback shift registers, m-sequences and Gold codes

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Generate long, noise-like binary sequences with a few gates: for scramblers, test patterns, spread-spectrum codes and sequences with sharp autocorrelation and low cross-correlation |
| **Devised by** | Solomon Golomb (1955–67), the theory of shift-register sequences; Robert Gold (1967), families of codes with bounded cross-correlation |
| **Accuracy** | Exact: periods, primitivity and correlation values match the theory (tables below) |
| **Cost** | One parity and one shift per output bit; the primitivity test needs $O(n^2\log 2^n)$ bit operations plus factoring $2^n - 1$ |
| **API** | [`Lfsr`](../../api/lib_jsl_information/lfsr/linear_feedback_shift_register/struct.Lfsr.html) (`new`, `from_polynomial`, `step`, `bits`, `period`), [`is_primitive_polynomial`](../../api/lib_jsl_information/lfsr/maximum_length_polynomials/fn.is_primitive_polynomial.html), [`maximum_length_polynomial`](../../api/lib_jsl_information/lfsr/maximum_length_polynomials/fn.maximum_length_polynomial.html), [`maximum_length_polynomials`](../../api/lib_jsl_information/lfsr/maximum_length_polynomials/fn.maximum_length_polynomials.html), [`all_maximum_length_polynomials`](../../api/lib_jsl_information/lfsr/maximum_length_polynomials/fn.all_maximum_length_polynomials.html), [`gold_codes`](../../api/lib_jsl_information/lfsr/gold_codes/fn.gold_codes.html), [`gold_correlation_bound`](../../api/lib_jsl_information/lfsr/gold_codes/fn.gold_correlation_bound.html) |

</div>

## Overview

A **linear feedback shift register** (LFSR) holds $n$ bits. At each step it outputs one bit, shifts, and feeds in the XOR of a fixed set of its bits, the **taps**. Its output obeys a linear recurrence over GF(2), so its behaviour is governed by a polynomial.
When the polynomial is **primitive**, the register runs through all $2^n - 1$ nonzero states before repeating. Its output is then a **maximum-length sequence** (m-sequence), which is balanced and looks random in many statistical senses. Its autocorrelation is nearly an impulse. Two m-sequences
chosen as a **preferred pair** generate a **Gold family** of $2^n + 1$ sequences whose cross-correlations are all small, so many users can share a channel.

## History

Shift-register sequences were studied in the 1950s for radar, missile guidance and cryptography. Solomon Golomb, at the Jet Propulsion Laboratory, developed their theory and published it as *Shift Register Sequences* (1967). His randomness postulates describe the balance, runs and correlation of m-sequences. JPL used them to range
spacecraft, and LFSR scramblers and test patterns (PRBS) appear in nearly every serial link standard. Robert Gold (1967) found the preferred pairs and the families named after him, and Tadao Kasami found related sets. Gold codes became the spreading codes of GPS (the C/A code, from two 10-stage registers) and of W-CDMA.

## The idea

### The recurrence and its polynomial

A **linear feedback shift register** (LFSR) is a row of $n$ one-bit cells. At each clock tick the
bits shift along by one place, the bit that falls off the end is the output, and the vacated
cell is filled with the XOR of a fixed selection of the old bits, the **taps**. It is about the
cheapest circuit that can generate a long, random-looking sequence of bits.

In symbols, the library's register holds bits $s_t, \ldots, s_{t+n-1}$, with the oldest in bit 0. It outputs $s_t$ and shifts in

$$
s_{t+n} = \sum_{k=0}^{n-1} c_k\,s_{t+k} \pmod 2,
$$

the parity of the state ANDed with the tap mask $c$. The **characteristic polynomial** is $p(x) = x^n + \sum_k c_kx^k$. [`Lfsr::from_polynomial`](../../api/lib_jsl_information/lfsr/linear_feedback_shift_register/struct.Lfsr.html#method.from_polynomial) takes it as a bit mask, so `0b1011` is $x^3 + x + 1$. **An example.** Take $n = 3$ and $p(x) = x^3 + x + 1$, so that $c_0 = c_1 = 1$ and the recurrence
is $s_{t+3} = s_{t+1} + s_t$. Starting from the bits $1, 0, 0$, the sequence continues

$$
1, 0, 0, 1, 0, 1, 1,\ 1, 0, 0, 1, 0, 1, 1,\ \ldots
$$

and repeats with period 7. During one period the three-bit state takes each of the seven
nonzero values exactly once.

**The period.** The next state depends only on the current one, and there are finitely many
states, so the sequence must eventually repeat. The all-zero state is a fixed point (the XOR of
zeros is zero), so it must be avoided, and it cannot occur in a cycle with other states. That
leaves $2^n - 1$ nonzero states, so the period is at most $2^n - 1$. A sequence that achieves
this is a **maximum-length sequence**, or **m-sequence**. Whether it is achieved depends only on
the polynomial. Algebraically, one step of the register is multiplication by $x$ modulo $p(x)$,
so the period is the **order** of $x$ modulo $p(x)$: the smallest $T$ with $x^T \equiv 1$. The
order equals $2^n - 1$ exactly when $p$ is a **primitive** polynomial.

### Testing primitivity

$p$ of degree $n$ is primitive if $x^{2^n - 1} \equiv 1 \pmod p$ and $x^{(2^n - 1)/q} \not\equiv 1$ for every prime $q$ dividing $2^n - 1$. Then the order is exactly $2^n - 1$, and $p$ is irreducible because its quotient ring has $2^n - 1$ units and so is a field. [`is_primitive_polynomial`](../../api/lib_jsl_information/lfsr/maximum_length_polynomials/fn.is_primitive_polynomial.html) computes the powers by square-and-multiply with carry-less arithmetic. It factors $2^n - 1$ by trial division, which limits it to degree 40.
There are $\varphi(2^n - 1)/n$ primitive polynomials of degree $n$. [`maximum_length_polynomial`](../../api/lib_jsl_information/lfsr/maximum_length_polynomials/fn.maximum_length_polynomial.html) returns one of minimal weight (fewest taps, the cheapest hardware), the numerically smallest among them:

<p class="caption" id="tab-18-6"><strong>Table 18.6.</strong> Primitive polynomials found against the count $\varphi(2^n - 1)/n$.</p>

{{#include ../generated/information/primitive_table.md}}

The counts found by [`all_maximum_length_polynomials`](../../api/lib_jsl_information/lfsr/maximum_length_polynomials/fn.all_maximum_length_polynomials.html) agree with the formula.

### Properties of m-sequences

Because the state passes through every nonzero value once per period, an m-sequence of period
$N = 2^n - 1$ has several properties expected of a random sequence.

- **Balance.** It has $2^{n-1}$ ones and $2^{n-1} - 1$ zeros (the example has four ones and three
  zeros). The missing zero corresponds to the excluded all-zero state.
- **Windows.** Every nonzero pattern of $n$ consecutive bits appears exactly once per period.
- **Autocorrelation.** Map the bits to $\pm1$ and compare the sequence with a cyclically shifted
  copy of itself, adding up the products. At zero shift every product is $+1$ and the sum is $N$.
  At every other shift the agreements and disagreements almost cancel, and the sum is exactly
  $-1$:

![m-sequence autocorrelation](../generated/information/msequence_autocorrelation.svg)

<p class="caption" id="fig-18-4"><strong>Figure 18.4.</strong> m-sequence autocorrelation.</p>

A single sharp peak with a flat floor is the ideal shape for measuring a delay. That makes
m-sequences ideal for ranging and channel sounding: a
[correlator](../filters/convolution.md#cross-correlation-and-delay-estimation) finds the delay
unambiguously.

### Gold codes

Systems in which many transmitters share a channel, such as satellite navigation, give each one
its own code and separate them by correlation. That needs a large family of sequences in which
every pair has low *cross*-correlation, at every shift. Two m-sequences of the same length can still cross-correlate strongly. Gold showed that if $v$ is the decimation $v_t = u_{qt}$ with $q = 2^k + 1$, where $\gcd(n, k) = 1$ for odd $n$ or $\gcd(n, k) = 2$ for $n \equiv 2 \pmod 4$, then $u$ and $v$ are a preferred pair. Their cross-correlation takes only the three values $-1$, $-t(n)$ and $t(n) - 2$, with
$t(n) = 1 + 2^{\lfloor (n+2)/2\rfloor}$. The family $\{u, v, u \oplus T^\tau v\}$ of all shifts $\tau$ keeps the same three values between any two members. [`gold_codes`](../../api/lib_jsl_information/lfsr/gold_codes/fn.gold_codes.html) uses $k = 1$ ($q = 3$) for odd $n$ and $k = 2$ ($q = 5$) for $n \equiv 2 \pmod 4$. No family exists when $4 \mid n$.

{{#include ../generated/information/gold_facts.md}}

## Algorithm

**One step of the register**, with the state held in an integer of $n$ bits:

1. the output is bit 0 of the state;
2. the feedback bit is the parity of the state ANDed with the tap mask (the XOR of the tapped
   bits);
3. shift the state right by one, and place the feedback bit in bit $n - 1$.

**Testing whether a polynomial of degree $n$ is primitive.**

1. Factor $2^n - 1$ into primes by trial division.
2. Compute $x^{2^n - 1} \bmod p(x)$ by
   [square-and-multiply](../number-theory/modular-arithmetic.md#inverses-and-powers), in
   arithmetic on polynomials with coefficients modulo 2 (shifts and XORs). If the result is
   not 1, the polynomial is not primitive.
3. For each prime factor $q$, compute $x^{(2^n - 1)/q} \bmod p(x)$. If any is 1, the order of
   $x$ is a proper divisor of $2^n - 1$, and the polynomial is not primitive.
4. Otherwise it is primitive.

**Finding a maximum-length polynomial**: try the polynomials of degree $n$ in order of
increasing number of terms, and within that in increasing numerical order, and return the
first that is primitive.

**A Gold family** for degree $n$.

1. Generate one period, $N = 2^n - 1$ bits, of the m-sequence $u$ from the minimal-weight
   primitive polynomial.
2. Form $v$ by decimation: $v_t = u_{qt \bmod N}$, with $q = 3$ for odd $n$ and $q = 5$ for
   $n \equiv 2 \pmod 4$.
3. The family is $u$, $v$, and the XOR of $u$ with each of the $N$ cyclic shifts of $v$:
   $N + 2$ sequences.

## Implementation

```rust,ignore
{{#include ../../../crates/information/src/lfsr/linear_feedback_shift_register.rs:lfsr_step}}
```

```rust,ignore
{{#include ../../../crates/information/src/lfsr/maximum_length_polynomials.rs:primitive}}
```

```rust,ignore
{{#include ../../../crates/information/src/lfsr/gold_codes.rs:gold}}
```

- `Lfsr` is an `Iterator` over its output bits, up to 64 stages wide.
- Gold codes are returned as 0/1 bits; map them to chips with $1 - 2b$.
- The table of minimal-weight polynomials (degrees 2–32) is computed once, on first use.

## In practice

```rust,ignore
{{#include ../../../crates/information/examples/information_lfsr.rs}}
```

```text
x^4 + x + 1: period Some(15), bits [1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1]
x^4 + x^3 + x^2 + x + 1: primitive false, period Some(5)
degree-16 maximum-length polynomial: 0x1002d
33 Gold codes of length 31; cross-correlation values [-9, -1, 7] (bound t(5) = 9)
```

The second polynomial divides $x^5 - 1$, so although it is irreducible its period is only 5.

**Strengths**

- Very cheap generators with exactly known period and correlation; Gold families give many nearly orthogonal codes.

**Weaknesses**

- Linear, so useless for cryptography on their own: $2n$ output bits reveal the whole register (Berlekamp-Massey).
- Primitivity testing is limited to degree 40 by trial-division factoring.

## Exercises

1. Show that the output of an LFSR from a nonzero state is periodic, and that its period divides the order of $x$ modulo $p$.
2. Prove that an m-sequence has $2^{n-1}$ ones per period.
3. Derive the two-valued autocorrelation from the shift-and-add property: $u \oplus T^\tau u$ is a shift of $u$.
4. Verify $\varphi(2^n - 1)/n$ for $n = 4, 6$ by listing the primitive polynomials.
5. Use the Berlekamp-Massey algorithm to recover the polynomial of the example's register from 8 output bits.

<details>
<summary>Hints and answers</summary>

1. The state evolves by multiplication by $x$ modulo $p(x)$ in $\mathrm{GF}(2)[x]/(p)$, so it is periodic, and the sequence repeats after $k$ steps if $x^k \equiv 1$ (for a nonzero state); the period divides the order of $x$ modulo $p$.
2. In a period of $2^n - 1$ steps every nonzero $n$-bit state occurs exactly once, and exactly $2^{n-1}$ of those states have the output bit equal to 1.
3. $u \oplus T^\tau u$ is a nonzero m-sequence satisfying the same recurrence, hence a shift $T^\sigma u$ of $u$. Over one period the agreements minus disagreements of $u$ and $T^\tau u$ are therefore $(2^{n-1} - 1) - 2^{n-1} = -1$, so the autocorrelation is $-1/(2^n - 1)$ for $\tau \ne 0$ and $1$ at $\tau = 0$.
4. The number of primitive polynomials of degree $n$ is $\varphi(2^n - 1)/n$: $\varphi(15)/4 = 8/4 = 2$ for $n = 4$ and $\varphi(63)/6 = 36/6 = 6$ for $n = 6$.
5. From $2n$ output bits, Berlekamp-Massey finds the shortest LFSR that generates them; with a register of length $n$, $2n$ bits suffice, so 8 bits recover a register of up to 4 stages.

</details>

## Further reading

- S. W. Golomb, *Shift Register Sequences*, Holden-Day, 1967; 3rd ed., World Scientific, 2017.
- R. Gold, "Optimal binary sequences for spread spectrum multiplexing", *IEEE Transactions on Information Theory* 13 (1967) 619–621.
- D. V. Sarwate and M. B. Pursley, "Crosscorrelation properties of pseudorandom and related sequences", *Proceedings of the IEEE* 68 (1980) 593–619.
- R. Lidl and H. Niederreiter, *Finite Fields*, 2nd ed., Cambridge University Press, 1997, chapters 3 and 8.
