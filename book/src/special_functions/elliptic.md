# Elliptic integrals and Jacobian elliptic functions

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Evaluate the complete and incomplete elliptic integrals of the first, second and third kinds, Carlson's symmetric integrals, and the Jacobian elliptic functions $\operatorname{sn}$, $\operatorname{cn}$, $\operatorname{dn}$ |
| **Devised by** | Giulio Fagnano (1718) and Leonhard Euler (1750s); John Landen (1775), the transformation; Carl Friedrich Gauss (1799), the arithmetic-geometric mean; Adrien-Marie Legendre (1825), the three kinds; Niels Henrik Abel and Carl Gustav Jacobi (1827–1829), elliptic functions; Bille Carlson (1979), the symmetric forms and their duplication algorithm |
| **Accuracy** | About $10^{-15}$ relative; $10^{-12}$ for the first and third kinds very close to the logarithmic singularity at $k = 1$ |
| **Cost** | Carlson's duplication: about 6 to 10 iterations of a few square roots; Jacobian functions: an AGM of about 5 steps |
| **API** | [`elliptic_integral`](../../api/lib_jsl_special_functions/elliptic_integrals/fn.elliptic_integral.html), [`elliptic_integral_second`](../../api/lib_jsl_special_functions/elliptic_integrals/fn.elliptic_integral_second.html), [`elliptic_integral_third`](../../api/lib_jsl_special_functions/elliptic_integrals/fn.elliptic_integral_third.html), [`elliptic_f`](../../api/lib_jsl_special_functions/elliptic_integrals/fn.elliptic_f.html), [`elliptic_e`](../../api/lib_jsl_special_functions/elliptic_integrals/fn.elliptic_e.html), [`elliptic_pi`](../../api/lib_jsl_special_functions/elliptic_integrals/fn.elliptic_pi.html), [`carlson_rf`](../../api/lib_jsl_special_functions/elliptic_integrals/fn.carlson_rf.html), [`carlson_rd`](../../api/lib_jsl_special_functions/elliptic_integrals/fn.carlson_rd.html), [`carlson_rj`](../../api/lib_jsl_special_functions/elliptic_integrals/fn.carlson_rj.html), [`carlson_rc`](../../api/lib_jsl_special_functions/elliptic_integrals/fn.carlson_rc.html), [`jacobian_elliptic`](../../api/lib_jsl_special_functions/jacobian_elliptic_functions/fn.jacobian_elliptic.html) |

</div>

## Overview

**Elliptic integrals** are integrals of rational functions involving the square root of a cubic or
quartic polynomial. They got their name from the arc length of an ellipse, and they give the period
of a pendulum swinging through a large angle, the shape of a bent elastic rod, the inductance of
coils, the field of a current loop, and the geodesics on an ellipsoid. Inverting them, as the sine
inverts $\int dt/\sqrt{1 - t^2}$, gives the **Jacobian elliptic functions** $\operatorname{sn}$, $\operatorname{cn}$, $\operatorname{dn}$, doubly periodic
generalisations of the trigonometric functions that describe the motion of a pendulum and of a
rigid body, nonlinear waves (cnoidal waves) and elliptic filters.

## History

Giulio Fagnano (1718) found that arcs of the lemniscate could be doubled by algebraic operations,
and Euler (1750s) generalised this into the addition theorem for elliptic integrals. John Landen
(1775) discovered the transformation that relates integrals of different moduli, and Gauss (1799)
found that the **arithmetic-geometric mean** computes the complete integral of the first kind:
$K(k) = \pi/(2\operatorname{agm}(1, \sqrt{1 - k^2}))$, one of the fastest algorithms in all of mathematics. Legendre spent forty
years on elliptic integrals and reduced them all to three standard kinds (1825-1828), just before
Niels Henrik Abel and Carl Gustav Jacobi revolutionised the subject by inverting them, creating
the theory of elliptic functions.

For computation, Bille Carlson (1979) replaced Legendre's forms by symmetric integrals that can all be
computed by one simple, provably convergent **duplication** algorithm. His method underlies the
elliptic integrals of most modern libraries, including this one.

## The idea

### Elliptic integrals

In Legendre's form, with modulus $k$ (parameter $m = k^2$) and amplitude $\varphi$,

$$
F(\varphi, k) = \int_0^\varphi\frac{d\theta}{\sqrt{1 - k^2\sin^2\theta}}, \quad
E(\varphi, k) = \int_0^\varphi\sqrt{1 - k^2\sin^2\theta}\,d\theta, \quad
\Pi(\varphi, n, k) = \int_0^\varphi\frac{d\theta}{(1 - n\sin^2\theta)\sqrt{1 - k^2\sin^2\theta}},
$$

and the **complete** integrals $K(k) = F(\tfrac\pi2, k)$, $E(k) = E(\tfrac\pi2, k)$, $\Pi(n, k) = \Pi(\tfrac\pi2, n, k)$. At $k = 0$ they reduce to
$\pi/2$; as $k \to 1$, $K$ diverges logarithmically, $K \approx \ln(4/\sqrt{1 - k^2})$, while $E \to 1$.

![Complete elliptic integrals](../generated/special_functions/elliptic_ke.svg)

**Carlson's symmetric forms** are

$$
R_F(x, y, z) = \tfrac12\int_0^\infty\frac{dt}{\sqrt{(t + x)(t + y)(t + z)}}, \qquad
R_D(x, y, z) = \tfrac32\int_0^\infty\frac{dt}{(t + z)\sqrt{(t + x)(t + y)(t + z)}},
$$

and similarly $R_J$ (third kind) and the degenerate $R_C(x, y) = R_F(x, y, y)$, which is elementary. Legendre's
integrals are combinations of them, for example

$$
F(\varphi, k) = \sin\varphi\,R_F(\cos^2\varphi,\ 1 - k^2\sin^2\varphi,\ 1), \qquad K(k) = R_F(0,\ 1 - k^2,\ 1) .
$$

The symmetry gives the **duplication theorem**: with $\lambda = \sqrt{xy} + \sqrt{yz} + \sqrt{zx}$,

$$
R_F(x, y, z) = R_F\Big(\frac{x + \lambda}4, \frac{y + \lambda}4, \frac{z + \lambda}4\Big).
$$

Each step brings the three arguments four times closer together. Once they nearly agree,
$R_F \approx \mu^{-1/2}$ (for their mean $\mu$) plus a short Taylor series in their relative deviations, which
completes the evaluation to full precision. The same idea computes $R_D$ and $R_J$. Arguments of $F$, $E$
and $\Pi$ beyond $\pi/2$ are reduced by periodicity: $F(\varphi + j\pi, k) = F(\varphi, k) + 2jK(k)$.

### Jacobian elliptic functions

If $u = F(\varphi, k)$, the **Jacobian elliptic functions** are

$$
\operatorname{sn}(u \mid m) = \sin\varphi, \qquad \operatorname{cn}(u \mid m) = \cos\varphi, \qquad \operatorname{dn}(u \mid m) = \sqrt{1 - m\sin^2\varphi},
$$

with $m = k^2$. They have real period $4K$ ($\operatorname{sn}$, $\operatorname{cn}$) or $2K$ ($\operatorname{dn}$), and reduce to $\sin$, $\cos$, 1 at $m = 0$ and to
$\tanh$, $\operatorname{sech}$, $\operatorname{sech}$ at $m = 1$. The motion of a pendulum released at angle $\theta_0$ is $\sin(\theta/2) = k\,\operatorname{sn}(K - \omega t \mid k^2)$ with
$k = \sin(\theta_0/2)$:

![Jacobian elliptic functions](../generated/special_functions/jacobi_elliptic.svg)

They are computed by the **descending Landen transformation**: the arithmetic-geometric mean of 1
and $\sqrt{1 - m}$ is run until its sequences agree, which takes about five steps; then the amplitude is
recovered by a backward recursion $\sin(2\varphi_{n-1} - \varphi_n) = (c_n/a_n)\sin\varphi_n$, and
$\operatorname{dn} = \sqrt{(1 - m) + m\,\operatorname{cn}^2}$.

### Accuracy

{{#include ../generated/special_functions/elliptic_accuracy.md}}

The errors of $10^{-12}$ at $k = 0.999999$ come from the logarithmic singularity: $1 - k^2$ is formed with a relative error
near $10^{-10}$, and $K$ depends on it logarithmically.

The table includes points at the quarter period $u = K$, where an earlier version computed dn as
$\operatorname{cn}/\cos(\varphi_1 - \varphi_0)$, following Abramowitz and Stegun. Both factors vanish there, and the quotient was
0.2418 instead of 0.2236 for $m = 0.95$: spikes in the plot above, at every minimum of dn. It now uses the
identity above, a sum of non-negative terms.

## Algorithm

- $R_F$, $R_D$, $R_J$, $R_C$: duplicate until the relative deviations are below about $10^{-3}$ (error $\propto$ deviation$^6$),
  then sum the Taylor series.
- $F$, $E$, $\Pi$: reduce the amplitude to $[-\tfrac\pi2, \tfrac\pi2]$, then combine Carlson integrals.
- $\operatorname{sn}$, $\operatorname{cn}$, $\operatorname{dn}$: series near $m = 0$ and $m = 1$; otherwise the AGM and the backward amplitude recursion.

## Implementation

Carlson's $R_F$ by duplication:

```rust,ignore
{{#include ../../../crates/special_functions/src/elliptic_integrals.rs:carlson_rf}}
```

The Jacobian elliptic functions:

```rust,ignore
{{#include ../../../crates/special_functions/src/jacobian_elliptic_functions.rs:jacobi}}
```

- The Legendre forms take the modulus $k$, the Jacobian functions the parameter $m = k^2$, as in Abramowitz
  and Stegun. Check the convention when comparing with other software: SciPy's `ellipk` takes $m$.
- Principal-value cases of the third kind ($1 - n\sin^2\theta$ changing sign) are not supported and give `NaN`.

## In practice

```rust,ignore
{{#include ../../../crates/special_functions/examples/special_elliptic.rs}}
```

**Strengths**

- One uniform, provably convergent algorithm for all three kinds.
- Fast: a handful of square roots per evaluation.

**Weaknesses**

- Real arguments only; no complex modulus or Weierstrass functions.
- The modulus/parameter conventions differ between sources and must be checked.

## Exercises

1. Show that the period of a pendulum of amplitude $\theta_0$ is $4\sqrt{L/g}\,K(\sin\tfrac{\theta_0}2)$, and expand it to second
   order in $\theta_0$.
2. Compute $K(k)$ by Gauss's AGM formula and compare with `elliptic_integral`. How many AGM steps are needed
   for full precision?
3. Show that the perimeter of an ellipse with semi-axes $a \ge b$ is $4aE(e)$ with $e = \sqrt{1 - b^2/a^2}$, and compare
   with Ramanujan's approximation $\pi[3(a + b) - \sqrt{(3a + b)(a + 3b)}]$.
4. Verify $\operatorname{sn}^2 + \operatorname{cn}^2 = 1$ and $\operatorname{dn}^2 + m\operatorname{sn}^2 = 1$ numerically, and that $\frac{d}{du}\operatorname{sn} = \operatorname{cn}\operatorname{dn}$.
5. Prove the duplication theorem for $R_C(x, y) = \tfrac12\int_0^\infty(t + x)^{-1/2}(t + y)^{-1}\,dt$ by a change of variable.

## Further reading

- B. C. Carlson, "Computing elliptic integrals by duplication", *Numerische Mathematik* 33 (1979)
  1–16, and "Numerical computation of real or complex elliptic integrals", *Numerical Algorithms* 10
  (1995) 13–26.
- J. M. Borwein and P. B. Borwein, *Pi and the AGM*, Wiley, 1987.
- NIST Digital Library of Mathematical Functions, chapters 19 and 22, <https://dlmf.nist.gov/19>.
- M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, chapters 16 and 17, 1964.
