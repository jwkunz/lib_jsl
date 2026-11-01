# Bracketing methods: bisection, false position, Ridders and Brent

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Find a root of a continuous function $f$ in an interval $[a, b]$ over which $f$ changes sign |
| **Devised by** | Bisection: implicit in Bolzano's intermediate value theorem (1817). False position: ancient (Egypt, China, India; Fibonacci 1202); the Illinois modification: Dowell and Jarratt (1971). Ridders' method: C. J. F. Ridders (1979). Brent's method: Theodorus Dekker (1969) and Richard Brent (1973), after van Wijngaarden and colleagues |
| **Accuracy** | To the requested tolerance, or the limit of floating point; the root always stays bracketed |
| **Cost** | Bisection: $\log_2((b - a)/\text{tol})$ evaluations. False position (Illinois): order 1.44 per evaluation. Ridders: order 1.41 per evaluation. Brent: order up to 1.84, never far from bisection's worst case |
| **API** | [`bracket_root`](../../api/lib_jsl_root_finding/bracketing/fn.bracket_root.html), [`bracket_roots_scan`](../../api/lib_jsl_root_finding/bracketing/fn.bracket_roots_scan.html), [`bisection`](../../api/lib_jsl_root_finding/bisection/fn.bisection.html), [`false_position`](../../api/lib_jsl_root_finding/false_position_method/fn.false_position.html), [`ridders_method`](../../api/lib_jsl_root_finding/ridders_method/fn.ridders_method.html), [`brent_method`](../../api/lib_jsl_root_finding/van_wijngaarden_dekker_brent/fn.brent_method.html) |

</div>

## Overview

If a continuous function is negative at $a$ and positive at $b$, it must cross zero somewhere in
between. This **intermediate value theorem** is the basis of the most reliable methods in numerical
analysis. A **bracket** $[a, b]$ with $f(a)$ and $f(b)$ of opposite signs is a trap for a root:
evaluate $f$ at a point inside, keep the half (or part) of the interval where the sign still changes,
and repeat. The trap only shrinks, so the method cannot fail.

The methods of this entry differ in how they choose the point inside. **Bisection** takes the
midpoint, which is slow but has a guaranteed rate. **False position** and **Ridders' method** use the
function values to guess where the root is. **Brent's method** combines fast interpolation with
bisection as a safeguard, and is the method of choice for a scalar equation with a bracket.

## History

Bisection is as old as the intermediate value theorem, proved by Bernard Bolzano in 1817 (and used
implicitly long before). **False position** (*regula falsi*) is one of the oldest algorithms of all:
a single false position, guessing an answer and correcting it proportionally, appears in the Egyptian
Rhind papyrus (about 1650 BC), and the double false position used here, interpolating linearly
between two guesses, is in the Chinese *Nine Chapters on the Mathematical Art* (around 200 BC) and in
Fibonacci's *Liber Abaci* (1202). Its tendency to stall was cured by the **Illinois** modification,
analysed by M. Dowell and P. Jarratt in 1971.

C. J. F. Ridders published his exponential variant of false position in 1979, in an electronics
journal. The idea of combining a fast open method with bisection was developed by Adriaan van
Wijngaarden and colleagues at the Mathematisch Centrum in Amsterdam in the 1960s and published by
Theodorus Dekker in 1969; Richard Brent (1973) added inverse quadratic interpolation and a stricter
safeguard, and proved a bound on the worst case. Brent's ALGOL procedure `zero` became `fzero` in
MATLAB and `brentq` in SciPy; this library's version follows the `zbrent` of *Numerical Recipes*.

## The idea

### Finding a bracket

A bracketing method needs a sign change to start. Two simple strategies find one:

- **Expansion** ([`bracket_root`](../../api/lib_jsl_root_finding/bracketing/fn.bracket_root.html)):
  starting from $[a, b]$, repeatedly move the end where $|f|$ is smaller outward, by 1.6 times the
  current width, until $f$ changes sign (or 50 attempts fail).
- **Scanning** ([`bracket_roots_scan`](../../api/lib_jsl_root_finding/bracketing/fn.bracket_roots_scan.html)):
  divide $[a, b]$ into $n$ pieces and report each piece over which $f$ changes sign.

Neither can be guaranteed: a function such as $x^2 + 10^{-20}$ has no sign change to find, and a scan
misses pairs of roots closer together than its spacing, since the two sign changes cancel. Roots of
even multiplicity, where $f$ touches zero without crossing, are invisible to all bracketing methods.

### Bisection

Evaluate $f$ at the midpoint $m = (a + b)/2$ and keep the half over which the sign changes. After $k$
steps the root lies in an interval of width $(b - a)/2^k$, whatever $f$ is, so the number of
evaluations to reach a tolerance $\delta$ is $\lceil\log_2((b - a)/\delta)\rceil$: about 46 for an interval of
length 1 and $\delta = 10^{-13}$. This **linear** convergence, one binary digit per evaluation, is slow,
but it is the best possible guarantee for a method that uses only the signs of $f$.

### False position and the Illinois modification

False position uses the values, not just the signs: it draws the straight line through
$(a, f(a))$ and $(b, f(b))$ and takes its zero,

$$
c = \frac{a\,f(b) - b\,f(a)}{f(b) - f(a)},
$$

keeping the part of the interval where the sign changes. For a straight line it is exact. For a
convex function, however, one end of the bracket never moves, and the method converges only
linearly, sometimes extremely slowly. The **Illinois** modification fixes this: when the same end
has been kept twice in a row, its stored function value is halved, which pulls the next estimate
towards it. The order of convergence becomes $3^{1/3} \approx 1.44$ per evaluation.

### Ridders' method

Ridders' method evaluates $f$ at the midpoint $x_3$ of the bracket $[x_1, x_2]$ and then finds the
exponential factor $e^{Qx}$ that makes $f(x)\,e^{Qx}$ a straight line through the three points. False
position applied to that straightened function gives

$$
x_4 = x_3 + (x_3 - x_1)\,\frac{\operatorname{sign}(f_1 - f_2)\,f_3}{\sqrt{f_3^2 - f_1 f_2}},
$$

which always lies inside the bracket. With two evaluations per step and quadratic convergence per
step, its order is $\sqrt 2 \approx 1.41$ per evaluation. It is simple and robust, a good alternative to
Brent's method when a short implementation is wanted.

### Brent's method

Brent's method keeps three points: the best estimate $b$, the previous one $a$, and a contrapoint $c$
with $f(c)$ of opposite sign to $f(b)$, so that the root is bracketed by $b$ and $c$. At each step it
tries a fast step:

- **inverse quadratic interpolation**, fitting $x$ as a quadratic function of $y$ through the three
  points and evaluating it at $y = 0$, when the three function values are distinct;
- the **secant** step through $a$ and $b$ otherwise.

It accepts the fast step only if it falls inside the bracket and is less than half the size of the
step before last; otherwise it **bisects**. Near a simple root the interpolation steps are always
accepted and the order is about 1.84; on difficult functions the safeguard forces bisection often
enough that the method is never much worse than bisection. Brent proved that it needs at most about
the square of bisection's count of evaluations.

### Comparison

The figure at the top of the [chapter](index.md) compares the methods on Newton's own test equation,
$x^3 - 2x - 5 = 0$: the interpolating methods need 8 to 13 evaluations where bisection needs 46. On
harder problems the picture is more varied (evaluations to a tolerance of $10^{-13}$; the open methods,
from the [next entry](newton.md), start from the ends or the midpoint of the interval, and Newton's
count includes its evaluations of $f'$):

{{#include ../generated/root_finding/evaluations.md}}

- On smooth functions with simple roots, Brent's method is the fastest bracketing method.
- $x^{10} - 1$ is nearly flat at 0 and steep at 1.3: the open methods are thrown far away, and the
  secant method fails; plain false position takes 123 evaluations, the Illinois version 17.
- At a **triple root**, $(x - 1/3)^3$, every interpolating method degrades to linear convergence, and
  even Brent's method, whose safeguard bisects only every fourth step, takes almost three times as
  long as bisection. Bisection, which ignores the values, is unaffected.
- $\tanh(20(x - 0.3))$ is almost a step: Newton's method from 0 is thrown out to where the function is
  flat, and fails, while the bracketing methods are untroubled.

## Algorithm

All four methods share the same skeleton:

1. Check that $f(a)$ and $f(b)$ have opposite signs (a zero at an end is returned at once).
2. Repeat: choose a point $x$ inside the bracket by the method's rule, evaluate $f(x)$, and replace
   the end of the bracket with the same sign as $f(x)$.
3. Stop when the bracket (or the step) is smaller than the tolerance, or $f(x) = 0$ exactly.

## Implementation

Bisection is short enough to show whole:

```rust,ignore
{{#include ../../../crates/root_finding/src/bisection.rs:bisection}}
```

The heart of Brent's method is the choice between interpolation and bisection:

```rust,ignore
{{#include ../../../crates/root_finding/src/van_wijngaarden_dekker_brent.rs:brent_step}}
```

- The signs of $f$ are compared directly, never through the product $f(a)\,f(m)$, which underflows to
  zero when both values are tiny (below about $10^{-162}$) and once sent bisection to the wrong end
  of its bracket.
- Brent's tolerance is relative as well as absolute, $2\varepsilon|b| + \text{tol}/2$, so it cannot ask for
  more digits than floating point can give.
- Every method also stops when $f$ is exactly zero; bisection never errs once it has a bracket, while
  the others give up after a fixed number of iterations (200 for Brent).
- The functions return `Result<f64, String>`; the error explains the failure (no sign change, no
  convergence).

## In practice

```rust,ignore
{{#include ../../../crates/root_finding/examples/root_finding_bracketing.rs}}
```

**Strengths**

- Guaranteed convergence once a bracket is known, with no derivative needed.
- Brent's method is nearly as fast as the open methods on smooth functions.
- Bisection's cost is known in advance and independent of $f$.

**Weaknesses**

- A bracket is needed, and finding one is not always easy.
- Roots of even multiplicity (no sign change) cannot be found, and roots of odd multiplicity are found
  slowly by the interpolating methods.
- They apply only to one equation in one unknown.

## Exercises

1. How many bisection steps are needed to locate a root in $[0, 1]$ to the full precision of double
   precision arithmetic? Why does it not help to ask for more?
2. Show that plain false position on $f(x) = x^2 - 2$ over $[0, 2]$ keeps the end $b = 2$ fixed for ever,
   and find its rate of convergence.
3. Derive Ridders' formula: find $Q$ such that $f(x)\,e^{Qx}$ takes values on a straight line at $x_1$,
   $x_3$ and $x_2$, and apply false position to it.
4. Write down the inverse quadratic interpolation formula through three points and show that it
   is exact when $x$ is a quadratic function of $y$.
5. Use `bracket_roots_scan` and `brent_method` to find all the roots of $\sin(1/x)$ in $[0.05, 1]$. How
   fine must the scan be to find them all?

## Further reading

- R. P. Brent, *Algorithms for Minimization without Derivatives*, Prentice-Hall, 1973 (Dover reprint,
  2002), chapter 4.
- T. J. Dekker, "Finding a zero by means of successive linear interpolation", in *Constructive Aspects of
  the Fundamental Theorem of Algebra*, Wiley, 1969.
- M. Dowell and P. Jarratt, "A modified regula falsi method for computing the root of an equation",
  *BIT* 11 (1971) 168–174.
- C. J. F. Ridders, "A new algorithm for computing a single root of a real continuous function",
  *IEEE Transactions on Circuits and Systems* 26 (1979) 979–980.
- W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery, *Numerical Recipes*, 3rd ed.,
  Cambridge University Press, 2007, §9.1–9.3.
