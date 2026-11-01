# Finite difference derivatives

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Approximate $f'(x)$, higher derivatives, gradients, Hessians and Jacobians from values of $f$ |
| **Devised by** | Difference quotients: Newton and Leibniz (1670s), Brook Taylor (1715); difference formulas of higher order from interpolation: Newton, Stirling and Bessel (18th century); step-size analysis in floating point: the early numerical analysts, systematised in texts such as Gill, Murray and Wright (1981) |
| **Accuracy** | Central difference: about $\varepsilon^{2/3}$ (10 digits); five-point first derivative: about $\varepsilon^{4/5}$ (13 digits); each higher derivative loses more |
| **Cost** | Two evaluations (central), four or five (five-point); a gradient costs $2n$, a Hessian about $2n^2$ |
| **API** | [`derivative_2_point`](../../api/lib_jsl_derivatives/derivatives_1d_scalar/fn.derivative_2_point.html), [`derivative_stencil`](../../api/lib_jsl_derivatives/derivatives_1d_scalar/fn.derivative_stencil.html), [`gradient_2_point`](../../api/lib_jsl_derivatives/derivatives_vectors/fn.gradient_2_point.html), [`hessian_2_point`](../../api/lib_jsl_derivatives/derivatives_vectors/fn.hessian_2_point.html), [`jacobian_2_point`](../../api/lib_jsl_derivatives/derivatives_vectors/fn.jacobian_2_point.html) |

</div>

## Overview

The derivative is the limit of a difference quotient, $f'(x) = \lim_{h \to 0} \big(f(x+h) - f(x)\big)/h$.
A **finite difference** stops short of the limit and uses a small but finite $h$. The idea could not
be simpler, but choosing $h$ is subtle: if $h$ is too large the quotient is a poor approximation of
the limit, and if $h$ is too small the subtraction $f(x+h) - f(x)$ cancels nearly all the digits of
the function values, leaving mostly rounding error. Understanding this trade-off, and choosing $h$
to balance it, is the heart of numerical differentiation.

## History

Difference quotients are the starting point of the calculus: both Newton and Leibniz reasoned with
the ratios of small increments before passing to the limit. Brook Taylor's *Methodus
incrementorum* (1715) developed the calculus of finite differences systematically, and the
interpolation formulas of Newton, James Stirling and Friedrich Bessel gave difference formulas of
any order, used for centuries to differentiate tables of values by hand.

The modern question, how to choose the step in floating-point arithmetic, came with the computer.
Its answer, that the best step balances the truncation and rounding errors and scales with a power
of the machine precision, became standard in the optimisation literature of the 1970s, where finite
difference gradients and Jacobians are essential; Philip Gill, Walter Murray and Margaret Wright's
*Practical Optimization* (1981) gives a careful treatment.

## The idea

### Truncation error

By Taylor's theorem,

$$
f(x \pm h) = f(x) \pm h f'(x) + \tfrac{h^2}{2} f''(x) \pm \tfrac{h^3}{6} f'''(x) + \cdots.
$$

The **forward difference** $\big(f(x+h) - f(x)\big)/h = f'(x) + \tfrac h2 f''(x) + \cdots$ has an error of
order $h$. Subtracting the two expansions cancels the even terms and gives the **central
difference**

$$
\frac{f(x+h) - f(x-h)}{2h} = f'(x) + \frac{h^2}{6} f'''(x) + O(h^4),
$$

second-order accurate at the same cost of two evaluations. This symmetric cancellation is the
same phenomenon that makes the [midpoint rule](../integration/newton-cotes.md#the-midpoint-rule)
better than it looks.

### Rounding error and the optimal step

A computed function value carries a relative error of about $\varepsilon \approx 2.2 \times 10^{-16}$ (at best),
so $f(x \pm h)$ is wrong by about $\varepsilon|f|$. The central difference divides the difference of two
such values by $2h$, so rounding contributes an error of about $\varepsilon|f|/h$. The total error is
roughly

$$
E(h) \approx \frac{h^2}{6}|f'''| + \frac{\varepsilon |f|}{h},
$$

which is smallest when the two terms are comparable: $h \sim \varepsilon^{1/3}$, giving an error of about
$\varepsilon^{2/3}$, ten correct digits. The same argument for a formula with truncation error $O(h^p)$ for
an $m$-th derivative, whose rounding error is $O(\varepsilon/h^m)$, gives

$$
h_{\text{opt}} \sim \varepsilon^{1/(p+m)}, \qquad E(h_{\text{opt}}) \sim \varepsilon^{p/(p+m)}.
$$

The figure shows the trade-off for derivatives of $e^x$: on the right of each V the truncation error
dominates (slopes 2 and 4), on the left the rounding error (slopes $-1$, $-2$ and $-4$).

![Error against step size](../generated/derivatives/step_size.svg)

The library's default steps sit at the bottom of each V:

{{#include ../generated/derivatives/default_steps.md}}

The steps are **relative**: the step at $x$ is $h \cdot \max(|x|, 1)$, which suits functions that vary
on the scale of $x$ (a step of $10^{-5}$ is meaningless at $x = 10^{10}$). A function that oscillates on
a scale of one at large $x$, such as $\sin x$ at $x = 10^8$, needs a smaller step, passed explicitly.
One more detail matters: $x + h$ is rounded, so the step actually taken differs from $h$. The library
uses $h \leftarrow (x + h) - x$, which is exact, so the denominator matches the points evaluated.

### Higher-order stencils

Combining more points cancels more terms of the Taylor series. With the five points $x$, $x \pm h$,
$x \pm 2h$:

$$
f'(x) \approx \frac{-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)}{12h},
\qquad
f''(x) \approx \frac{-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)}{12h^2},
$$

both fourth order, and second-order formulas for $f'''$ and $f''''$. Higher order allows a larger
optimal step and so a smaller total error: the five-point first derivative reaches about thirteen
digits. Higher derivatives are inherently harder, since their rounding errors grow like $h^{-m}$: the
fourth derivative can be computed to only about five digits.

These formulas can be derived by **Richardson extrapolation** of the central difference, exactly as
[Romberg integration](../integration/romberg.md#richardson-extrapolation) extrapolates the trapezoid
rule: since $D(h) = f' + c_2h^2 + c_4h^4 + \cdots$, the combination $\big(4D(h) - D(2h)\big)/3$ cancels the
$h^2$ term and gives the five-point formula.

### Gradients, Hessians and Jacobians

For a function of several variables, partial derivatives are central differences along each
coordinate in turn. The **gradient** of $f: \mathbb R^n \to \mathbb R$ costs $2n$ evaluations, and the
**Jacobian** of $\mathbf f: \mathbb R^n \to \mathbb R^m$ the same $2n$ evaluations of the vector function. The
**Hessian** uses second differences on the diagonal and the cross difference

$$
\frac{\partial^2 f}{\partial x_i \partial x_j} \approx
\frac{f(\mathbf x + h_i\mathbf e_i + h_j\mathbf e_j) - f(\mathbf x + h_i\mathbf e_i - h_j\mathbf e_j) - f(\mathbf x - h_i\mathbf e_i + h_j\mathbf e_j) + f(\mathbf x - h_i\mathbf e_i - h_j\mathbf e_j)}{4h_ih_j}
$$

off it, with the step $\varepsilon^{1/4}$ appropriate to a second derivative. The cost grows like $n^2$,
which is why optimisation methods such as [BFGS](../../api/lib_jsl_optimization/index.html) build up
Hessian approximations from gradients instead.

### Better than finite differences

Two techniques avoid the cancellation entirely, when they apply. If $f$ is analytic and can be
evaluated at complex arguments, the **complex-step derivative** $f'(x) \approx \operatorname{Im} f(x + ih)/h$
involves no subtraction, so $h$ can be as small as $10^{-100}$ and the result is accurate to machine
precision; more generally, [Cauchy's integral formula](../integration/contour-integration.md#cauchys-integral-formula-and-derivatives)
gives derivatives of any order. And **automatic differentiation** applies the chain rule to the
program that computes $f$, producing exact derivatives up to rounding. Both require access to more
than function values, which is why finite differences remain the universal fallback.

## Algorithm

1. Choose the relative step $h = \varepsilon^{1/(p+m)}$ (or the user's), scale it by $\max(|x|, 1)$, and
   make it exact with $h \leftarrow (x + h) - x$.
2. Evaluate $f$ at the stencil points and combine them with the formula's weights.
3. For several variables, repeat for each coordinate (and each pair, for the Hessian), restoring the
   point after each perturbation.

## Implementation

The central difference and the five-point formulas:

```rust,ignore
{{#include ../../../crates/derivatives/src/derivatives_1d_scalar.rs:central_difference}}
```

```rust,ignore
{{#include ../../../crates/derivatives/src/derivatives_1d_scalar.rs:five_point}}
```

- An earlier version of the library used a fixed absolute step of $10^{-6}$ for every formula. That
  is near-optimal for the central difference at $x \approx 1$, but for the five-point fourth derivative
  the rounding error $\varepsilon/h^4$ is then about $10^8$: the formula returned noise. The default steps
  now depend on the formula, as in the table above.
- The step for each coordinate is chosen separately, scaled by that coordinate's magnitude.

## In practice

```rust,ignore
{{#include ../../../crates/derivatives/examples/finite_difference_derivatives.rs}}
```

**Strengths**

- Needs nothing but function values: works for any function, including black-box procedures.
- Simple, cheap for low dimensions, and accurate to about ten digits with a balanced step.

**Weaknesses**

- Accuracy is fundamentally limited by cancellation, more severely for higher derivatives.
- Noisy functions (simulations, measured data) can make finite differences useless, since the noise
  plays the role of $\varepsilon$ and is far larger.
- The cost of gradients grows with the dimension, and that of Hessians with its square.

## Exercises

1. Show that the forward difference has optimal step $\sim\varepsilon^{1/2}$ and error $\sim\varepsilon^{1/2}$ (eight
   digits). Why is the central difference worth its extra evaluation?
2. Derive the five-point first-derivative formula by Richardson extrapolation of the central
   difference.
3. Suppose $f$ is computed with a relative error of $10^{-8}$ (for example, by an iterative solver).
   What step and accuracy should you expect for the central difference?
4. Implement the complex-step derivative for $f(x) = e^x/\sqrt{\sin^3 x + \cos^3 x}$ at $x = 1.5$ and
   compare it with the central difference at a range of steps.
5. Use [`hessian_2_point`](../../api/lib_jsl_derivatives/derivatives_vectors/fn.hessian_2_point.html)
   on $f(x, y) = \sin(xy)$ near $(10^3, 10^{-3})$. Why does the default relative step work poorly here,
   and what would you do?

## Further reading

- B. Taylor, *Methodus incrementorum directa et inversa*, London, 1715.
- P. E. Gill, W. Murray and M. H. Wright, *Practical Optimization*, Academic Press, 1981, section 8.6.
  Step-size selection for finite difference gradients.
- J. N. Lyness and C. B. Moler, "Numerical differentiation of analytic functions", *SIAM Journal on
  Numerical Analysis* 4 (1967) 202–210, and W. Squire and G. Trapp, "Using complex variables to
  estimate derivatives of real functions", *SIAM Review* 40 (1998) 110–112. The complex step.
- A. Griewank and A. Walther, *Evaluating Derivatives: Principles and Techniques of Algorithmic
  Differentiation*, 2nd ed., SIAM, 2008.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002. Rounding errors
  in general.
