# Number theory: modular arithmetic, primes, factorisation and polynomials

Number theory studies the integers, and its algorithms are among the oldest and the most used. Every public-key handshake runs modular exponentiation, every hash table picks a prime, and every error-correcting code computes in a finite field. This chapter covers `lib_jsl_number_theory`:
- the arithmetic of remainders: gcd, inverses, powers, square roots and the Chinese remainder theorem;
- the primes: sieves, primality testing and factorisation;
- polynomials, and integers of any size held as decimal strings.

## The central questions

- **Exact or approximate?** Number-theoretic algorithms are exact, so their failures are not rounding but overflow, infinite loops and wrong conventions. Most of the defects fixed here were of those kinds.
- **Certain or probable?** Miller–Rabin is probabilistic in general but deterministic for 64-bit inputs with the right bases.
- **How big?** Everything here works in 64 or 128 bits; beyond that, methods change (and so does their cost).

## Testing against SymPy

The references come from SymPy 1.14, by `book/generator/data/make_number_theory_refs.py`. They include the classic strong pseudoprimes and Carmichael numbers, the factorisation of $2^{64} - 1$, products of two primes near $2^{32}$, and moduli up to $10^{30}$:

<p class="caption" id="tab-22-1"><strong>Table 22.1.</strong> Number-theory routines against SymPy.</p>

{{#include ../generated/number_theory/reference.md}}

## In the library

All of these are in `lib_jsl_number_theory` ([API reference](../../api/lib_jsl_number_theory/index.html)).

| You need | Use | Entry |
|---|---|---|
| A gcd, an inverse, a modular power, an exact square root, simultaneous congruences | `gcd`, `extended_gcd`, `multiplicative_inverse_modulo`, `base_exponent_in_modulus`, `integer_square_root_newton`, `chinese_remainder` | [Modular arithmetic](modular-arithmetic.md) |
| Primes up to a bound or in a window; a primality test; a factorisation | `sieve_of_eratosthenes`, `segmented_sieve`, `is_prime`, `factorize`, `euler_totient`, `divisors` | [Primes](primes.md) |
| Polynomial evaluation and roots; integers of any size | `polynomial_eval`, `polynomial_from_roots`, `polynomial_roots`, `TextCodedInteger` | [Polynomials and decimal-string integers](polynomials.md) |
