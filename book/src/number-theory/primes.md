# Primes: sieves, Miller–Rabin and factorisation

<div class="entry-summary">

| | |
|---|---|
| **Problem** | List the primes up to a bound or in a window, decide whether a 64-bit number is prime, and factor it completely |
| **Devised by** | Eratosthenes (c. 240 BC), the sieve; Gary Miller (1976) and Michael Rabin (1980), the strong probable-prime test; John Pollard (1975), the rho method; Richard Brent (1980), its faster cycle detection |
| **Accuracy** | Deterministic and exact for every `u64`: the twelve Miller–Rabin bases have no common strong pseudoprime below $3.2 \times 10^{23}$; agrees with SymPy on all reference cases |
| **Cost** | Sieve $O(n\log\log n)$ time and $n/16$ bytes; segmented sieve $O(\sqrt{h} + w)$ memory for a window of width $w$ below $h$; Miller–Rabin twelve modular powers; Pollard's rho about $\sqrt p$ steps for the smallest factor $p$ |
| **API** | [`sieve_of_eratosthenes`](../../api/lib_jsl_number_theory/prime_sieves/fn.sieve_of_eratosthenes.html), [`segmented_sieve`](../../api/lib_jsl_number_theory/prime_sieves/fn.segmented_sieve.html), [`prime_count`](../../api/lib_jsl_number_theory/prime_sieves/fn.prime_count.html), [`is_prime`](../../api/lib_jsl_number_theory/primality/fn.is_prime.html), [`is_strong_probable_prime`](../../api/lib_jsl_number_theory/primality/fn.is_strong_probable_prime.html), [`next_prime`](../../api/lib_jsl_number_theory/primality/fn.next_prime.html), [`pow_mod`](../../api/lib_jsl_number_theory/primality/fn.pow_mod.html), [`factorize`](../../api/lib_jsl_number_theory/integer_factorization/fn.factorize.html), [`pollard_rho`](../../api/lib_jsl_number_theory/integer_factorization/fn.pollard_rho.html), [`euler_totient`](../../api/lib_jsl_number_theory/integer_factorization/fn.euler_totient.html), [`divisors`](../../api/lib_jsl_number_theory/integer_factorization/fn.divisors.html) |

</div>

## Overview

Three questions about primes need three different methods. To list all primes up to $n$, the **sieve of Eratosthenes** crosses out multiples and keeps what survives, in nearly linear time. A **segmented** sieve does the same for a window far out, such as near $10^{18}$, without sieving everything below it. To test one large number, **Miller–Rabin** checks a property every prime has and almost no composite shares. With the right twelve bases, no 64-bit composite passes them all. To **factor**, trial division removes small primes, and **Pollard's rho**
finds a factor $p$ of what remains in about $\sqrt p$ steps, using the birthday paradox in the residues modulo $p$.

## History

Eratosthenes of Cyrene described the sieve around 240 BC. Fermat's little theorem (1640) gave the first fast compositeness test, but Carmichael numbers (Robert Carmichael, 1910) fool it for every base coprime to them. Gary Miller (1976) strengthened the test and proved it deterministic assuming the generalised Riemann hypothesis. Michael Rabin (1980) made it a randomised test with error at most $1/4$ per base. Gerhard Jaeschke and later Jonathan Sorenson and Jonathan Webster (2015) found the smallest numbers fooling the first $k$ prime bases, which makes it deterministic for bounded inputs.
John Pollard's rho method (1975) and Richard Brent's improvement (1980) factored the eighth Fermat number in 1980. They remain the method of choice for factors up to about $10^{10}$.

## The idea

### Sieves

Mark every multiple of each prime $p \le \sqrt n$, starting at $p^2$. The unmarked numbers are prime. The library stores only odd numbers, one bit each, so $10^9$ needs 62 MB. A **segmented sieve** first sieves the base primes up to $\sqrt{h}$, then crosses their multiples off just the window $[l, h]$. The prime-counting function $\pi(x)$ follows by counting. By the prime number theorem it is approximately $x/\ln x$, and better approximated by the logarithmic integral $\mathrm{Li}(x)$:

![Prime counting](../generated/number_theory/prime_counting.svg)

{{#include ../generated/number_theory/sieve_table.md}}

### Miller–Rabin

For an odd prime $n$, write $n - 1 = 2^sd$ with $d$ odd. For every base $a$, either $a^d \equiv 1$, or $a^{2^rd} \equiv -1$ for some $0 \le r < s$: the only square roots of 1 modulo a prime are $\pm1$. A composite $n$ that passes for base $a$ is a **strong pseudoprime** to that base. At most a quarter of the bases fool any composite, and composites that fool many small bases are very rare:

{{#include ../generated/number_theory/pseudoprimes.md}}

The smallest number fooling all of the first twelve primes as bases is $318\,665\,857\,834\,031\,151\,167\,461$, so testing those twelve, after trial division, decides primality for every `u64` with certainty.

### Pollard's rho

Iterate $x \leftarrow x^2 + c \pmod n$. Modulo an unknown prime factor $p$ of $n$, the sequence is pseudo-random and repeats after about $\sqrt{\pi p/2}$ steps, tracing a shape like the letter ρ. When $x_i \equiv x_j \pmod p$, $\gcd(x_i - x_j, n)$ reveals $p$. Brent's variant compares against a saved value at doubling distances, and multiplies many differences together before taking one gcd. If it hits $n$ itself, it backs up and steps one at a time; if that fails too, it changes $c$:

![Pollard's rho](../generated/number_theory/pollard_rho.svg)

[`factorize`](../../api/lib_jsl_number_theory/integer_factorization/fn.factorize.html) removes the primes below 50 by trial division, then repeatedly splits composites with Pollard's rho and certifies the pieces with Miller–Rabin. The totient $\varphi(n) = n\prod_{p \mid n}(1 - 1/p)$ and the list of divisors follow from the factorisation.

## Algorithm

1. **Sieve**: for odd $p$ from 3 while $p^2 \le n$, if $p$ is unmarked, mark $p^2, p^2 + 2p, \ldots$.
2. **Miller–Rabin**: trial division by the twelve bases; for each base, compute $a^d$ and square up to $s - 1$ times, looking for $1$ or $-1$.
3. **Factorise**: trial division; then a stack of composites, each split by Pollard–Brent rho until every piece passes the primality test.

## Implementation

```rust,ignore
{{#include ../../../crates/number_theory/src/prime_sieves.rs:eratosthenes}}
```

```rust,ignore
{{#include ../../../crates/number_theory/src/primality.rs:miller_rabin}}
```

```rust,ignore
{{#include ../../../crates/number_theory/src/integer_factorization.rs:brent_rho}}
```

- 64-bit products are formed in 128 bits ([`mul_mod`](../../api/lib_jsl_number_theory/primality/fn.mul_mod.html)).
- Writing this chapter fixed three problems here: `factorize(0)` looped forever; Pollard's $x^2 + c$ could overflow for $n$ near $2^{64}$; and the segmented sieve overflowed for windows ending near `u64::MAX`. It also corrected the documented pseudoprime bound.

## In practice

```rust,ignore
{{#include ../../../crates/number_theory/examples/number_theory_primes.rs}}
```

```text
primes below 50: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
π(10⁶) = 78498
primes in [10¹², 10¹² + 100]: [1000000000039, 1000000000061, 1000000000063, 1000000000091]
2047: base 2 says prime true, base 3 says prime false, is_prime false
largest 64-bit prime is_prime: true; next prime after 10^18: Some(1000000000000000003)
18446744073709551615 = [(3, 1), (5, 1), (17, 1), (257, 1), (641, 1), (65537, 1), (6700417, 1)]
1000000016000000063 = [(1000000007, 1), (1000000009, 1)]
φ(360) = 96, divisors of 360: [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360]
```

**Strengths**

- Deterministic primality and complete factorisation for every 64-bit integer, typically in microseconds to milliseconds.
- Compact bit-set sieves, and windows far beyond memory.

**Weaknesses**

- 64-bit only; larger numbers need the elliptic-curve method or the quadratic and number field sieves.
- Pollard's rho is slow for two factors near $2^{32}$ (about 65 000 steps); the sieve of all primes to $10^9$ takes seconds.

## Exercises

1. Show that the sieve may start crossing out multiples of $p$ at $p^2$.
2. Prove that $\pm 1$ are the only square roots of 1 modulo an odd prime, and derive the Miller–Rabin condition.
3. Show that 561 is a Carmichael number but not a strong pseudoprime to base 2.
4. Explain the expected $\sqrt{\pi p/2}$ steps of Pollard's rho with the birthday paradox.
5. Use the segmented sieve to count the primes between $10^{12}$ and $10^{12} + 10^6$, and compare with $10^6/\ln 10^{12}$.

## Further reading

- R. Crandall and C. Pomerance, *Prime Numbers: A Computational Perspective*, 2nd ed., Springer, 2005, chapters 3 and 5.
- M. O. Rabin, "Probabilistic algorithm for testing primality", *Journal of Number Theory* 12 (1980) 128–138.
- J. M. Pollard, "A Monte Carlo method for factorization", *BIT* 15 (1975) 331–334.
- R. P. Brent, "An improved Monte Carlo factorization algorithm", *BIT* 20 (1980) 176–184.
- J. Sorenson and J. Webster, "Strong pseudoprimes to twelve prime bases", *Mathematics of Computation* 86 (2017) 985–1003.
