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

Evaluate $f$ at the midpoint $m = (a + b)/2$. If $f(m)$ has the same sign as $f(a)$, the sign
change is in $[m, b]$; otherwise it is in $[a, m]$. Keep that half and repeat.

Each step halves the bracket, whatever $f$ is. After $k$ steps the root lies in an interval of
width $(b - a)/2^k$. To reach a tolerance $\delta$ we need $(b - a)/2^k \le \delta$, that is

$$
k = \left\lceil \log_2 \frac{b - a}{\delta} \right\rceil
$$

halvings: 44 for an interval of length 1 and $\delta = 10^{-13}$, since
$10^{13} \approx 2^{43.2}$. Counting the two evaluations at the ends of the starting bracket, that
is about 46 evaluations of $f$. The error is multiplied by $\tfrac12$ at every step. This is **linear**
convergence, one binary digit per evaluation. It is slow, but the count is known in advance and
does not depend on $f$, and it is the best possible guarantee for a method that uses only the
signs of $f$.

### False position and the Illinois modification

Bisection ignores the sizes of $f(a)$ and $f(b)$. If $|f(a)|$ is much smaller than $|f(b)|$, the
root is probably nearer to $a$. **False position** (in Latin, *regula falsi*) uses this
information: it draws the straight line through $(a, f(a))$ and $(b, f(b))$ and takes the point
where the line crosses zero,

$$
c = \frac{a\,f(b) - b\,f(a)}{f(b) - f(a)},
$$

then keeps the part of the interval where the sign changes, as bisection does. If $f$ is a
straight line the answer is exact in one step.

The weakness appears when $f$ is curved in one direction throughout the bracket (convex or
concave). The chord of a convex function lies above its graph, so the chord's zero always falls
on the same side of the root. One end of the bracket is replaced at every step, and the other
*never moves*. The bracket does not shrink to zero, and the estimates creep towards the root with
linear convergence, sometimes extremely slowly.

The **Illinois** modification fixes this with one line of code. When the same end has been kept
twice in a row, its stored function value is halved. This tilts the next chord so that its zero
moves towards the stuck end, and usually lands on the other side of the root, which frees that
end. The order of convergence becomes $3^{1/3} \approx 1.44$ per evaluation.

### Ridders' method

False position is exact for straight lines and struggles with curvature. Ridders' idea is to
remove the curvature first. One step, on a bracket $[x_1, x_2]$ with function values $f_1, f_2$:

1. Evaluate $f_3 = f(x_3)$ at the midpoint $x_3$.
2. Find the exponential factor $e^{Qx}$ for which the three values of $g(x) = f(x)\,e^{Qx}$ at
   $x_1, x_3, x_2$ lie on a straight line. (This is one condition on the one unknown $Q$, and it
   reduces to a quadratic equation for $e^{Q(x_3 - x_1)}$.) Multiplying by a positive factor does not move the root,
   so $g$ has the same root as $f$.
3. Apply false position to the straightened function $g$. The result simplifies to

$$
x_4 = x_3 + (x_3 - x_1)\,\frac{\operatorname{sign}(f_1 - f_2)\,f_3}{\sqrt{f_3^2 - f_1 f_2}} .
$$

4. Evaluate $f(x_4)$ and form the new bracket from $x_4$ and whichever neighbouring point has the
   opposite sign.

The formula has two reassuring properties. Since $f_1$ and $f_2$ have opposite signs,
$f_3^2 - f_1f_2$ is positive, so the square root is real and larger than $|f_3|$; consequently the fraction
is smaller than 1 in magnitude and $x_4$ always lies inside the bracket. Each step costs two evaluations and
converges quadratically, which makes the order $\sqrt 2 \approx 1.41$ per evaluation. Ridders'
method is simple and robust, a good alternative to Brent's method when a short implementation is
wanted.

### Brent's method

Brent's method aims for the best of both worlds: the speed of interpolation when the function is
well behaved and the guarantee of bisection when it is not.

**The points it keeps.** Three points are maintained: the best estimate so far, $b$; the previous
estimate, $a$; and a **contrapoint** $c$ at which $f$ has the opposite sign to $f(b)$. The root
always lies between $b$ and $c$.

**The fast step.** At each iteration the method proposes a new point by interpolation:

- **Inverse quadratic interpolation**, when the three function values are distinct. Instead of
  fitting $y$ as a quadratic in $x$ (a parabola, which may cross the axis twice or not at all),
  it fits $x$ as a quadratic function of $y$ through the three points and evaluates that at $y = 0$.
  This always gives exactly one answer.
- The **secant** step through $a$ and $b$ otherwise.

**The safeguard.** The proposed point is accepted only if it falls inside the bracket and the
step is less than half the size of the step before last. The second condition insists that the
steps keep shrinking quickly, as they would if interpolation were working. If either test fails,
the method **bisects** instead.

Near a simple root the interpolation steps are always accepted and the order of convergence is
about 1.84. On difficult functions the safeguard forces a bisection often enough that the method
is never much worse than bisection. Brent proved that it needs at most about the square of
bisection's count of evaluations, and in practice it rarely needs more than a few times that
count.

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

All four methods share the same skeleton. Input: a function $f$, an interval $[a, b]$ and a
tolerance.

1. **Check the bracket.** Evaluate $f(a)$ and $f(b)$. If either is exactly zero, return that end.
   If they have the same sign, report an error.
2. **Choose a point** $x$ inside the bracket by the method's rule (below) and evaluate $f(x)$.
3. **Shrink the bracket.** If $f(x) = 0$, return $x$. Otherwise replace the end at which $f$ has
   the same sign as $f(x)$, so that the sign change is preserved.
4. **Test.** Stop when the bracket, or the last step, is smaller than the tolerance. Otherwise go
   to 2.

The methods differ only in step 2:

- **Bisection**: the midpoint.
- **False position (Illinois)**: the zero of the chord through the two ends. If the same end has
  now survived two steps running, halve its stored function value before the next step.
- **Ridders**: evaluate $f$ at the midpoint, then apply Ridders' formula to the three points
  (two evaluations per iteration).
- **Brent**: the inverse quadratic or secant step if it passes the two safeguards, otherwise the
  midpoint. Its convergence test uses the tolerance $2\varepsilon|b| + \text{tol}/2$.

Bisection needs $\lceil\log_2((b - a)/\text{tol})\rceil$ steps whatever the function. The
others give up with an error after a fixed number of iterations (500 for false position, 100 for
Ridders, 200 for Brent).

**Finding a bracket by expansion** (`bracket_root`): while $f(a)$ and $f(b)$ have the same sign,
move whichever end has the smaller $|f|$ outwards by 1.6 times the current width, and re-evaluate
there; fail after 50 expansions.

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
