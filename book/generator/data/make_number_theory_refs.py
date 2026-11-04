"""Reference values for the chapter on number theory, from SymPy (the book used SymPy 1.14).

    python3 book/generator/data/make_number_theory_refs.py > book/generator/data/number_theory_reference.txt

Each output line is `case|values` with comma-separated integers.
"""

import sympy

def out(name, *values):
    print(name + "|" + ",".join(str(v) for v in values))

# Primality, including strong pseudoprimes to several small bases and Carmichael numbers.
for n in [2, 561, 1105, 2047, 3215031751, 2152302898747, 3474749660383, 341550071728321,
          3825123056546413051, 18446744073709551557, 18446744073709551559, 1000000007, 999999999989]:
    out(f"isprime {n}", int(sympy.isprime(n)))

# Factorisations.
for n in [360, 600851475143, 18446744073709551615, 18446743979220271189,
          2**61 - 1, 1000000016000000063, 3825123056546413051]:
    f = sympy.factorint(n)
    out(f"factor {n}", *[x for p in sorted(f) for x in (p, f[p])])
    out(f"totient {n}", sympy.totient(n))

for k in range(1, 9):
    out(f"primepi {10**k}", sympy.primepi(10**k))

# sympy.crt takes the moduli first, then the residues.
out("crt", *sympy.ntheory.modular.crt([3, 5, 7], [2, 3, 2]))
out("crt noncoprime", *sympy.ntheory.modular.crt([6, 10, 15], [4, 8, 13], check=True))
out("inverse", sympy.mod_inverse(17, 3120), sympy.mod_inverse(12345678901234567, 10**30 + 57))
out("isqrt", sympy.integer_nthroot(10**30 + 12345, 2)[0], sympy.integer_nthroot(2**127 - 1, 2)[0])
out("powmod", pow(7, 10**18 + 9, 10**30 + 57), pow(3, 2**100, 2**89 - 1))
