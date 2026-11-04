# Divisibility and modular arithmetic

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Find greatest common divisors and Bézout coefficients, invert numbers and raise them to powers modulo $m$, take exact integer square roots, and solve simultaneous congruences |
| **Devised by** | Euclid (c. 300 BC), the algorithm; Étienne Bézout (1779), the identity; Sunzi (3rd–5th century), the remainder theorem; Pingala (c. 200 BC), square-and-multiply; Newton's method for square roots goes back to Heron |
| **Accuracy** | Exact integer arithmetic; agrees with SymPy on every reference case (see the [chapter overview](index.md#testing-against-sympy)) |
| **Cost** | Euclid $O(\log\min(a, b))$ divisions; modular power $O(\log e)$ multiplications; integer square root $O(\log\log n)$ Newton steps; CRT $O(k)$ merges |
| **API** | [`gcd`](../../api/lib_jsl_number_theory/greatest_common_divisor/fn.gcd.html), [`extended_gcd`](../../api/lib_jsl_number_theory/greatest_common_divisor/fn.extended_gcd.html), [`multiplicative_inverse_modulo`](../../api/lib_jsl_number_theory/multiplicative_inverse_modulo/fn.multiplicative_inverse_modulo.html), [`base_exponent_in_modulus`](../../api/lib_jsl_number_theory/base_exponent_in_modulus/fn.base_exponent_in_modulus.html), [`integer_square_root_newton`](../../api/lib_jsl_number_theory/integer_square_root/fn.integer_square_root_newton.html), [`integer_square_root_binary_search`](../../api/lib_jsl_number_theory/integer_square_root/fn.integer_square_root_binary_search.html), [`chinese_remainder`](../../api/lib_jsl_number_theory/chinese_remainder_theorem/fn.chinese_remainder.html) |

</div>

## Overview

Modular arithmetic works with remainders: $a \equiv b \pmod m$ when $m$ divides $a - b$. Its basic tools are some of the oldest algorithms known. **Euclid's algorithm** finds the greatest common divisor by repeated remainders. Its extended form also finds integers $x, y$ with $ax + by = \gcd(a, b)$, which give **modular inverses**. **Square-and-multiply** raises a number to a huge power modulo $m$ in a number of steps proportional to the exponent's bit length.
The **Chinese remainder theorem** combines congruences to different moduli into one. The integer square root completes the set; it underlies sieving and factoring. Together they are the arithmetic of public-key cryptography, error-correcting codes and hashing.

## History

Euclid's *Elements* (Book VII, c. 300 BC) gives the algorithm, the oldest still in everyday use. Indian mathematicians (Āryabhaṭa, c. 500) used its extension, the *kuṭṭaka*, to solve linear congruences, and Bézout (1779) proved the identity for polynomials. The problem of Sunzi's *Mathematical Classic* (3rd–5th century) asks for a number leaving remainders 2, 3 and 2 on division by 3, 5 and 7. Qin Jiushao (1247) gave the general method, and Gauss's
*Disquisitiones Arithmeticae* (1801) put congruences on their modern footing. Gabriel Lamé (1844) proved one of the first complexity bounds of any algorithm: Euclid needs at most about five times as many steps as the smaller number has decimal digits. In 1977 Rivest, Shamir and Adleman built RSA from exactly these tools.

## The idea

### Euclid's algorithm and Bézout coefficients

Since $\gcd(a, b) = \gcd(b, a \bmod b)$, repeated remainders shrink the pair until the remainder is zero. The extended algorithm keeps coefficients $(s, t)$ with $as + bt$ equal to the current remainder at every step, so the last nonzero remainder comes with its Bézout coefficients. The worst case is consecutive Fibonacci numbers, where every quotient is 1 (Lamé). Random inputs need about $(12\ln 2/\pi^2)\ln n$ steps (Heilbronn, 1969):

![Euclid steps](../generated/number_theory/euclid_steps.svg)

### Inverses and powers

If $\gcd(a, m) = 1$, then $ax + my = 1$ makes $x$ the inverse of $a$ modulo $m$. [`multiplicative_inverse_modulo`](../../api/lib_jsl_number_theory/multiplicative_inverse_modulo/fn.multiplicative_inverse_modulo.html) returns it in $[0, |m|)$, or an error if none exists. To compute $b^e \bmod m$, write $e$ in binary: square the base for each bit, and multiply it into the result where the bit is 1. That takes $O(\log e)$ multiplications, each reduced modulo $m$.
Products of residues below $m$ fit in 128 bits when $m < 2^{64}$. For larger moduli the library multiplies by double-and-add, so every positive `i128` modulus works.

### Integer square roots

$\lfloor\sqrt n\rfloor$ is needed exactly, which floating point cannot guarantee beyond $2^{53}$. Newton's iteration $x \leftarrow \lfloor(x + \lfloor n/x\rfloor)/2\rfloor$, started at a power of two above $\sqrt n$, decreases monotonically to $\lfloor\sqrt n\rfloor$ and stops when an iterate fails to decrease. Started below the root, the integer iteration can instead alternate between $\lfloor\sqrt n\rfloor$ and $\lfloor\sqrt n\rfloor + 1$ forever, as the library's original version did. Bisection is the slower, simpler alternative.

### The Chinese remainder theorem

If the moduli are pairwise coprime, $x \equiv r_i \pmod{m_i}$ has exactly one solution modulo $\prod m_i$. The library merges congruences one at a time. To combine $x \pmod m$ with $r \pmod{m_i}$, it solves $mt \equiv r - x \pmod{m_i}$ using $g = \gcd(m, m_i)$. The congruences are consistent only if $g$ divides $r - x$, and the merged modulus is $\mathrm{lcm}(m, m_i)$, so non-coprime moduli are handled too.

{{#include ../generated/number_theory/rsa_facts.md}}

## Algorithm

1. **Extended Euclid**: $(r_0, r_1) = (a, b)$, $(s_0, s_1) = (1, 0)$, $(t_0, t_1) = (0, 1)$; while $r_1 \ne 0$: $q = \lfloor r_0/r_1\rfloor$ and $(r, s, t)_{\text{new}} = (r, s, t)_0 - q(r, s, t)_1$; make $r_0$ non-negative at the end.
2. **Modular power**: result $= 1$; for each bit of $e$ from the lowest: if set, result $\leftarrow$ result·base; base $\leftarrow$ base².
3. **CRT**: merge pairwise as above, returning $(x, M)$ with $0 \le x < M$.

## Implementation

```rust,ignore
{{#include ../../../crates/number_theory/src/greatest_common_divisor.rs:extended_euclid}}
```

```rust,ignore
{{#include ../../../crates/number_theory/src/base_exponent_in_modulus.rs:square_and_multiply}}
```

```rust,ignore
{{#include ../../../crates/number_theory/src/integer_square_root.rs:newton_isqrt}}
```

```rust,ignore
{{#include ../../../crates/number_theory/src/chinese_remainder_theorem.rs:crt_merge}}
```

- Arguments are `i128` (gcd, inverses, powers, CRT) or `u128` (square roots); the GCD is never negative.
- Writing this chapter fixed four problems here: the Newton square root looped forever on $n = k^2 - 1$; the modular power accepted a zero modulus (its checks used bitwise NOT) and overflowed for moduli above about $10^{19}$; the GCD could be negative; and a negative modulus gave a negative inverse.

## In practice

```rust,ignore
{{#include ../../../crates/number_theory/examples/number_theory_arithmetic.rs}}
```

```text
gcd(1071, 462) = 21
240·(-9) + 46·(47) = 2
17⁻¹ mod 3120 = 2753
2^1000 mod 1000000007 = 688423210
⌊√(10^30 + 1)⌋ = 1000000000000000
Sunzi: x ≡ 23 (mod 105)
RSA with n = 3233, e = 17, d = 2753: 65 → 2790 → 65
```

**Strengths**

- Exact, fast and overflow-safe within `i128`; the CRT accepts non-coprime moduli and reports inconsistency.

**Weaknesses**

- Fixed-width integers: no arbitrary-precision moduli (see the [decimal-string integers](polynomials.md#decimal-string-integers) for unbounded but slow arithmetic).
- Not constant-time, so unsuitable for cryptography where timing leaks matter.

## Exercises

1. Prove that $\gcd(a, b) = \gcd(b, a \bmod b)$, and that the extended algorithm's invariant $as + bt = r$ holds at every step.
2. Show that Euclid's algorithm on consecutive Fibonacci numbers takes the most steps for inputs of their size, and derive Lamé's bound.
3. Compute $3^{200} \bmod 1000$ by hand with square-and-multiply.
4. Show that the integer Newton iteration started above $\sqrt n$ decreases monotonically to $\lfloor\sqrt n\rfloor$, and find a starting point below the root from which it cycles.
5. Solve $x \equiv 1 \pmod 4$, $x \equiv 3 \pmod 6$, and explain why the answer is unique modulo 12, not 24.

## Further reading

- D. E. Knuth, *The Art of Computer Programming*, vol. 2, *Seminumerical Algorithms*, 3rd ed., Addison-Wesley, 1997, sections 4.5.2–4.6.3.
- V. Shoup, *A Computational Introduction to Number Theory and Algebra*, 2nd ed., Cambridge University Press, 2009, chapters 1–4.
- R. L. Rivest, A. Shamir and L. Adleman, "A method for obtaining digital signatures and public-key cryptosystems", *Communications of the ACM* 21 (1978) 120–126.
