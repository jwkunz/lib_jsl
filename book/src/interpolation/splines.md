# Piecewise and spline interpolation

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Interpolate data $(x_i, y_i)$, $x_0 < x_1 < \cdots$, by a function that is simple on each interval: lines or cubics |
| **Devised by** | Splines: Isaac Schoenberg (1946), after the draughtsman's flexible spline; algorithms: Carl de Boor (1960s–70s). Cubic Hermite methods: Akima (1970), Fritsch and Carlson's monotone PCHIP (1980), Catmull and Rom (1974) |
| **Accuracy** | Linear: $O(h^2)$. Natural cubic spline: $O(h^4)$ in the interior, $O(h^2)$ near ends where $f'' \ne 0$. PCHIP and Akima: $O(h^3)$ for smooth data |
| **Cost** | Linear and Hermite methods: $O(n)$ set-up; spline: one tridiagonal solve. Each evaluation: a binary search, then $O(1)$ |
| **API** | [`LinearInterpolator`](../../api/lib_jsl_interpolation/one_d/linear_interpolator/struct.LinearInterpolator.html), [`NaturalCubicSplineInterpolator`](../../api/lib_jsl_interpolation/one_d/natural_cubic_spline_interpolator/struct.NaturalCubicSplineInterpolator.html), [`pchip`](../../api/lib_jsl_interpolation/one_d/pchip/fn.pchip.html), [`akima`](../../api/lib_jsl_interpolation/one_d/akima/fn.akima.html), [`cubic_basis`](../../api/lib_jsl_interpolation/one_d/cubic_basis/fn.cubic_basis.html), [`PiecewiseHermite`](../../api/lib_jsl_interpolation/one_d/hermite/struct.PiecewiseHermite.html) |

</div>

## Overview

[Polynomial interpolation](polynomial.md) fits one polynomial through all the data, which is
excellent at well-chosen points and dangerous at equally spaced or irregular ones. **Piecewise**
interpolation instead fits a low-degree polynomial on each interval between neighbouring data
points. The result can be evaluated anywhere quickly, is not troubled by the number of points, and
a bad data point affects only its neighbourhood.

The methods in this entry differ in how they join the pieces. **Linear** interpolation joins
straight lines and has corners. **Cubic splines** choose cubics whose first and second derivatives
are continuous: the smoothest possible curve, but a global one, which can overshoot near steep
changes. **Cubic Hermite** methods (PCHIP, Akima, Catmull-Rom) choose a slope at each data point from
nearby data and fit each cubic to the end values and slopes: they are local, and PCHIP in particular
never overshoots monotone data.

## History

Before computers, draughtsmen drew smooth curves through points with a **spline**: a thin flexible
strip of wood or metal held in place by lead weights called ducks. A bent elastic strip minimises its
bending energy, approximately $\int (y'')^2\,dx$, and between the weights its shape is a cubic.

Isaac Schoenberg named and founded the mathematical theory of splines in 1946, working on smoothing
actuarial tables at the US Army's Ballistic Research Laboratory. The aircraft and automotive
industries adopted them in the 1960s for designing surfaces, and Carl de Boor's work, culminating in
*A Practical Guide to Splines* (1978), provided the stable algorithms still in use.

Hiroshi Akima (1970) proposed a local cubic interpolant that avoids the wiggles of global splines on
data that changes abruptly. Edwin Catmull and Raphael Rom (1974) introduced their spline for computer
graphics, where it still drives animation paths. Frederick Fritsch and Ralph Carlson (1980) derived
conditions under which a piecewise cubic Hermite interpolant preserves monotonicity, leading to PCHIP,
the shape-preserving interpolant of MATLAB and SciPy.

## The idea

### Locating the interval

Every piecewise method first finds the interval $[x_j, x_{j+1}]$ containing the evaluation point,
by **bisection** in $O(\log n)$ steps. Interpolators that are evaluated at nearby points in sequence,
as when drawing a curve, remember the last interval and check it first ("hunting"), making most
evaluations $O(1)$. Bisection requires the abscissas to be in increasing order, and the library's
constructors check this.

### Linear interpolation

On $[x_j, x_{j+1}]$, $p(x) = y_j + (x - x_j)\,\dfrac{y_{j+1} - y_j}{x_{j+1} - x_j}$. The error is at most
$\tfrac{h^2}{8}\max|f''|$ with $h$ the largest interval: second order, robust, and shape-preserving,
but with corners at every data point.

### Cubic splines

A **cubic spline** is a function that is a cubic on each interval and has continuous first and second
derivatives at the interior points. With $n$ points there are $n - 1$ cubics, $4(n - 1)$ coefficients,
and the interpolation and continuity conditions give $4(n - 1) - 2$ equations; two **end conditions**
complete the system. The **natural** spline sets $S'' = 0$ at both ends, the condition satisfied by the
draughtsman's free-ended strip.

Writing the unknowns as the second derivatives $c_i$ (up to a factor) at the nodes reduces the problem
to a **tridiagonal** linear system, diagonally dominant and solved in $O(n)$:

$$
h_{i-1}c_{i-1} + 2(h_{i-1} + h_i)\,c_i + h_i c_{i+1} = 3\left(\frac{y_{i+1} - y_i}{h_i} - \frac{y_i - y_{i-1}}{h_{i-1}}\right).
$$

> **Minimum curvature.** Among all functions with a square-integrable second derivative that
> interpolate the data, the natural cubic spline minimises $\int_{x_0}^{x_{n-1}} \big(f''(x)\big)^2\,dx$.

This variational property, the mathematical form of the bent strip, explains the spline's smoothness
and also its weakness: the curvature is spread globally, so a sudden rise in the data pulls the curve
into oscillations on both sides.

The natural end condition is a compromise. If the underlying function has $f'' \ne 0$ at the ends,
the condition is wrong there, and the error near the ends is only $O(h^2)$, although it remains $O(h^4)$
in the interior. For $e^x$ on $[0, 1]$, the library's results show this clearly:

{{#include ../generated/interpolation/spline_convergence.md}}

The natural spline's maximum error falls by four per halving (second order, from the ends), while on
the middle half of the interval it falls by sixteen (fourth order). A **clamped** spline with the true
end slopes, or the **not-a-knot** condition, would restore fourth order everywhere.

### Cubic Hermite interpolation

If the slopes $m_j$ at the nodes are known, each interval has a unique cubic matching the two values
and the two slopes:

$$
p(x) = h_{00}(t)\,y_j + h_{10}(t)\,h\,m_j + h_{01}(t)\,y_{j+1} + h_{11}(t)\,h\,m_{j+1}, \qquad t = \frac{x - x_j}{h},
$$

with the **Hermite basis** $h_{00} = 2t^3 - 3t^2 + 1$, $h_{10} = t^3 - 2t^2 + t$, $h_{01} = -2t^3 + 3t^2$,
$h_{11} = t^3 - t^2$. The result has a continuous first derivative but, in general, not a second. The
methods differ only in how they choose the slopes, each from nearby data:

- **Catmull-Rom** (`cubic_basis` with tension 0): the central difference
  $m_j = (y_{j+1} - y_{j-1})/(x_{j+1} - x_{j-1})$. Tension $\tau$ scales it by $1 - \tau$.
- **Akima**: a weighted average of the two neighbouring secants, with weights from the changes in the
  secants on the far sides. Where the data change abruptly on one side, the slope follows the other
  side, which suppresses wiggles.
- **PCHIP**: zero where the data have a local extremum (the neighbouring secants differ in sign), and
  otherwise a weighted harmonic mean of the two secants. Fritsch and Carlson showed that this keeps the
  interpolant monotone wherever the data are monotone.

![Akima's test data](../generated/interpolation/akima_data.svg)

On Akima's classic data, flat and then rising steeply, the natural spline dips below the flat values
and overshoots the rise. Akima's interpolant follows the data with only a small ripple, and PCHIP is
monotone throughout. In the example below, only PCHIP stays within the range of the data: Akima and
Catmull-Rom can still overshoot slightly.

## Algorithm

- **Locate** the interval by bisection, checking the cached interval first.
- **Linear**: one line per interval.
- **Natural spline**: solve the tridiagonal system for the $c_i$ with $c_0 = c_{n-1} = 0$ (Thomas
  algorithm), then compute the cubic coefficients of each interval.
- **Hermite methods**: compute the secants $d_j = (y_{j+1} - y_j)/h_j$, the node slopes by the method's
  rule (PCHIP uses a shape-limited three-point formula at the ends; Akima extrapolates two extra
  secants at each end), then evaluate the Hermite form.

## Implementation

The natural spline set-up:

```rust,ignore
{{#include ../../../crates/interpolation/src/one_d/natural_cubic_spline_interpolator.rs:natural_spline}}
```

- All Hermite methods share [`PiecewiseHermite`](../../api/lib_jsl_interpolation/one_d/hermite/struct.PiecewiseHermite.html),
  which also accepts any slopes you supply, for example derivatives known from the physics.
- Outside the data range every method extends its end piece; extrapolation far beyond the data is
  rarely meaningful.
- The linear interpolator tolerates repeated abscissas (it takes the left value), which is
  occasionally convenient for step-like tables; the other methods require strictly increasing $x$.

## In practice

A temperature profile across a thermocline:

```rust,ignore
{{#include ../../../crates/interpolation/examples/interpolation_splines.rs}}
```

**Strengths**

- Any number of points, irregular spacing, fast evaluation.
- Splines give the smoothest curve, and fourth-order accuracy away from the ends.
- PCHIP preserves monotonicity and never overshoots; Akima resists wiggles near abrupt changes.

**Weaknesses**

- Splines overshoot near steep changes and need good end conditions for full accuracy.
- Hermite methods have discontinuous second derivatives, which matters for some applications
  (curvature, acceleration).
- None of them reach the exponential accuracy of [polynomial](polynomial.md) or
  [rational](rational.md) interpolation for smooth functions at well-chosen points.

## Exercises

1. Derive the tridiagonal system for the natural spline from the continuity of $S'$ at the interior
   points.
2. Show that the Hermite basis functions satisfy $h_{00}(0) = 1$, $h_{00}(1) = h_{00}'(0) = h_{00}'(1) = 0$, and
   the analogous conditions for the others.
3. Compute the PCHIP and Akima end slopes for equally spaced data and show they agree. (This is why
   their errors coincide in the table.)
4. Add a clamped end condition (known $f'$ at the ends) to the spline and confirm fourth-order
   convergence for $e^x$.
5. Construct monotone data for which the Catmull-Rom spline is not monotone.

## Further reading

- I. J. Schoenberg, "Contributions to the problem of approximation of equidistant data by analytic
  functions", *Quarterly of Applied Mathematics* 4 (1946) 45–99 and 112–141.
- C. de Boor, *A Practical Guide to Splines*, revised ed., Springer, 2001.
- H. Akima, "A new method of interpolation and smooth curve fitting based on local procedures",
  *Journal of the ACM* 17 (1970) 589–602.
- F. N. Fritsch and R. E. Carlson, "Monotone piecewise cubic interpolation", *SIAM Journal on
  Numerical Analysis* 17 (1980) 238–246.
- E. Catmull and R. Rom, "A class of local interpolating splines", in *Computer Aided Geometric
  Design*, Academic Press, 1974, 317–326.
