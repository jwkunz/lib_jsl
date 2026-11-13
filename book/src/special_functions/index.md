# Special functions

The **special functions** are the functions beyond the elementary ones (powers, exponentials,
logarithms, trigonometric functions) that appear so often in mathematics, physics, statistics and
engineering that they have names, notations and centuries of theory. The gamma function extends
the factorial; the error function gives normal probabilities; Bessel functions describe vibrating
drums, waveguides and heat flow in cylinders; elliptic integrals give the period of a pendulum and
the perimeter of an ellipse; orthogonal polynomials underlie Gaussian quadrature and spectral
methods; the zeta function encodes the distribution of the primes.

Most special functions are defined as solutions of differential equations, integrals or series,
and computing them accurately everywhere is surprisingly hard: a series that converges quickly for
small arguments may be useless for large ones, a formula valid in one region may suffer
catastrophic cancellation in another, and functions with zeros or poles need care near them. Good
implementations stitch together several methods, each used where it is accurate.

## The central techniques

- **Power series**, accurate for small arguments, but slow or subject to cancellation for large
  ones.
- **Asymptotic expansions**, divergent series that are nevertheless excellent for large
  arguments if truncated at the smallest term.
- **Continued fractions**, which often converge where the series does not, evaluated by the
  modified Lentz algorithm.
- **Recurrence relations**, which link functions of neighbouring orders. They must be run in the
  direction in which they are stable (Miller's backward recurrence for the minimal solution).
- **Functional equations** (reflection, duplication, connection formulas), which map an argument
  into the region where another method works.
- **Rational and Chebyshev approximations**, fitted once, for the fastest evaluation of a
  function of one variable.

## Accuracy

Every entry ends its idea section with an accuracy table: the library's values at a few dozen
arguments per function, chosen to include hard regions (large and small arguments, near zeros,
poles and branch points), compared with reference values computed by the arbitrary-precision
library mpmath at 50 significant digits. The arguments and reference values are in the book
generator's `data/special_functions_reference.csv`, made by `data/make_refs.py`. Relative error
is undefined at a zero of the function, so exact zeros are left out. Across the 2,265 test points
of the chapter, the median relative error is a few units of $10^{-16}$ and the largest about $10^{-11}$,
for a few oscillatory functions far out along their oscillations, where every computed value inherits
the rounding error in the argument.

## Choosing a function

| You need | Entry |
|---|---|
| $\Gamma(x)$, $\ln\Gamma(x)$, $\psi(x)$, $B(a, b)$, $n!$, $\binom nk$ | [Gamma and related functions](gamma.md) |
| $\operatorname{erf}$, $\operatorname{erfc}$, incomplete gamma $P(a, x)$, incomplete beta $I_x(a, b)$, Dawson's integral | [Incomplete gamma and beta functions and the error function](incomplete.md) |
| $E_1$, $E_n$, $\operatorname{Ei}$, $\operatorname{Si}$, $\operatorname{Ci}$, Fresnel $S$ and $C$ | [Exponential, sine, cosine and Fresnel integrals](exponential-integrals.md) |
| $J_\nu$, $Y_\nu$, $I_\nu$, $K_\nu$, $j_n$, $y_n$, $\operatorname{Ai}$, $\operatorname{Bi}$ | [Bessel and Airy functions](bessel.md) |
| $K(k)$, $E(k)$, $\Pi(n, k)$, incomplete forms, Carlson's $R_F$, $R_D$, $R_J$, $R_C$; $\operatorname{sn}$, $\operatorname{cn}$, $\operatorname{dn}$ | [Elliptic integrals and functions](elliptic.md) |
| ${}_2F_1$, ${}_1F_1$, ${}_0F_1$ | [Hypergeometric functions](hypergeometric.md) |
| Legendre, Chebyshev, Hermite, Laguerre, Jacobi polynomials; $P_l^m$, $Y_l^m$ | [Orthogonal polynomials and spherical harmonics](orthogonal-polynomials.md) |
| $\zeta(s)$, $\zeta(s, a)$, $\operatorname{Li}_s(z)$, Fermi-Dirac $F_j(x)$ | [Zeta functions, polylogarithms and Fermi-Dirac integrals](zeta.md) |
| Lambert $W$, solutions of $x\ln x = y$ | [The Lambert W function](lambert-w.md) |
| Normal, Student's $t$, $\chi^2$, $F$, Poisson and binomial distribution functions and quantiles | [Statistical distribution functions](statistical.md) |

## In the library

The special functions crate is `lib_jsl_special_functions`
([API reference](../../api/lib_jsl_special_functions/index.html)). Functions take and return `f64`
(complex values for the spherical harmonics and the complex Lambert W), and return `NaN` for
arguments outside their domain, `±∞` at poles and on overflow. Guards are written as
`!(x > 0.0)` rather than `x <= 0.0` so that a `NaN` argument is rejected too.
