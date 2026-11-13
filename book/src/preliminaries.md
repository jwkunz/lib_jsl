# Preliminaries

Most entries assume four small bodies of background: how computers store real numbers, how to measure the size of a
vector or matrix, a little probability, and complex numbers. Each is summarised here in enough detail to read the book,
with a pointer to where it is used. A reader who has met them can skip the page and return when a later entry leans on
one of them.

## Floating-point arithmetic and the three kinds of error

A computer stores a real number as a **floating-point** number: a sign, a fixed number of significant binary digits (the
**significand**) and an exponent, so that the value is $\pm\, m \times 2^{e}$. IEEE double precision, the type behind Rust's
`f64`, has a 53-bit significand. Two consequences shape almost every entry:

- **Only some numbers exist.** Between $1$ and $2$ the representable numbers are spaced $2^{-52} \approx 2.2 \times 10^{-16}$
  apart. This spacing is the **machine epsilon**, written $\varepsilon$ throughout the book. Near a number $x$ the spacing
  is about $\varepsilon |x|$, so a double carries about 16 significant decimal digits, whatever the magnitude. The
  decimal $0.1$ is not representable, which is why $0.1 + 0.2$ evaluates to $0.30000000000000004$ and not $0.3$.
- **Every operation rounds.** The computed result of $a \circ b$, for $\circ$ one of $+, -, \times, \div$, is the exact
  result moved to the nearest representable number. That is the **rounding model**:
  $$
  \mathrm{fl}(a \circ b) = (a \circ b)(1 + \delta), \qquad |\delta| \le u, \qquad u = \tfrac12\varepsilon \approx 1.1\times10^{-16}.
  $$
  The book's tolerances and "accuracy" rows are usually stated in multiples of $\varepsilon$; the factor of two between
  $\varepsilon$ and $u$ rarely matters.

Individual roundings are tiny. Trouble arises from what the algorithm does with them, and it helps to keep three sources
of error apart.

1. **Truncation (discretisation) error** is made by the method, even in exact arithmetic: replacing a derivative by a
   difference quotient, an integral by a sum, an infinite series by its first terms. It shrinks as the step shrinks or the
   work grows. Most "order of convergence" statements are about this error.
2. **Rounding error** is made by the arithmetic. It does not shrink as the work grows; if anything it accumulates.
3. **Conditioning** is a property of the *problem*, not the method: how strongly the exact answer reacts to small changes
   in the data. A problem with large condition number turns the unavoidable rounding of the *input* into a large error in the answer,
   whatever algorithm is used.

**Cancellation** is the most important way in which rounding becomes visible. Subtracting two nearly equal numbers
cancels their leading digits and leaves the digits that were already wrong. For example, in double precision
$1 + 10^{-8}$ is stored with an error of up to $1.1\times10^{-16}$, which is a relative error of about $10^{-8}$ in the
small number $10^{-8}$ that we are trying to recover. Computing $\big((1 + 10^{-8}) - 1\big)/10^{-8}$ gives
$0.99999999392\ldots$ instead of $1$: eight of the sixteen digits are gone in one subtraction. The optimal step in
[finite-difference derivatives](derivatives/finite-differences.md) is exactly the balance between truncation error (which
wants a small $h$) and the cancellation in $f(x+h) - f(x)$ (which wants a large one).

Errors in an algorithm's output are described in two complementary ways. The **forward error** is the distance between
the computed answer $\hat y$ and the true answer $y$. The **backward error** asks instead: for which slightly perturbed
input would $\hat y$ be the exact answer? An algorithm is **backward stable** if that perturbation is of the size of
rounding error. The two are linked by conditioning:

$$
\text{forward error} \;\lesssim\; \text{condition number} \times \text{backward error}.
$$

So a backward stable algorithm applied to a well-conditioned problem is accurate, while on an ill-conditioned problem
nothing can be done by choosing a better algorithm. [Linear systems](linear_algebra/index.md) develop this in detail.

## Norms: measuring the size of vectors and matrices

Error statements need a single number for the size of a vector or matrix. For $\mathbf x = (x_1, \ldots, x_n)$ the common choices
are

$$
\lVert\mathbf x\rVert_1 = \sum_i |x_i|, \qquad
\lVert\mathbf x\rVert_2 = \Big(\sum_i |x_i|^2\Big)^{1/2}, \qquad
\lVert\mathbf x\rVert_\infty = \max_i |x_i|.
$$

The 2-norm is the Euclidean length and is the default when no subscript is written. A **matrix norm** measures the largest
amount by which the matrix can stretch a vector. The **induced** norm is
$\lVert A\rVert = \max_{\mathbf x \ne 0} \lVert A\mathbf x\rVert / \lVert\mathbf x\rVert$, which gives

- $\lVert A\rVert_1$ = the largest absolute **column** sum,
- $\lVert A\rVert_\infty$ = the largest absolute **row** sum,
- $\lVert A\rVert_2$ = the largest singular value (the 2-norm, costly to compute exactly but the natural one in theory),

and the **Frobenius norm** $\lVert A\rVert_F = \big(\sum_{ij}|a_{ij}|^2\big)^{1/2}$, which treats the matrix as one long vector
and is cheap. For $A = \begin{pmatrix}1&2\\3&4\end{pmatrix}$ these are $6$, $7$, $5.465$ and $5.477$: the norms differ by a modest
factor, which is why the choice usually does not change a conclusion, only a constant. Induced norms satisfy
$\lVert A\mathbf x\rVert \le \lVert A\rVert\,\lVert\mathbf x\rVert$ and $\lVert AB\rVert \le \lVert A\rVert\,\lVert B\rVert$.

The **condition number** of a nonsingular matrix is $\kappa(A) = \lVert A\rVert\,\lVert A^{-1}\rVert \ge 1$. For the matrix above
$\kappa_2 \approx 14.9$. It is the factor in the rule of thumb of the previous section for solving $A\mathbf x = \mathbf b$:
a relative perturbation of size $\eta$ in $\mathbf b$ or $A$ can change the solution by a relative $\kappa(A)\,\eta$, so with
$\kappa(A) \approx 10^{k}$ one loses about $k$ of the 16 digits.

## Probability in brief

The [random numbers](random/index.md), [statistics](statistics/index.md), [machine learning](machine_learning/index.md) and
[information theory](information/index.md) chapters use the following ideas without redeveloping them.

- A **random variable** $X$ is a quantity whose value is determined by chance. It has a **distribution**: for a discrete $X$ the
  probabilities $p(x) = \mathbb P(X = x)$, for a continuous $X$ a density $f$ with $\mathbb P(a \le X \le b) = \int_a^b f$.
- The **expectation** (mean) is the probability-weighted average, $\mathbb E[X] = \sum_x x\,p(x)$ or $\int x f(x)\,dx$. It is linear:
  $\mathbb E[aX + bY] = a\,\mathbb E[X] + b\,\mathbb E[Y]$ always.
- The **variance** $\operatorname{Var}X = \mathbb E[(X - \mathbb E X)^2] = \mathbb E[X^2] - (\mathbb E X)^2$ measures spread, and the
  **standard deviation** $\sigma = \sqrt{\operatorname{Var}X}$ has the units of $X$.
- $X$ and $Y$ are **independent** if knowing one tells nothing about the other. Then $\mathbb E[XY] = \mathbb E[X]\,\mathbb E[Y]$ and
  variances add, $\operatorname{Var}(X + Y) = \operatorname{Var}X + \operatorname{Var}Y$. Without independence the second statement fails.
- An estimator $\hat\theta$ computed from data is itself random. It is **unbiased** if $\mathbb E[\hat\theta] = \theta$.

Two results explain why averaging works, and they are the foundation of Monte Carlo methods, bootstrap resampling and most of the
error estimates in the book. Let $X_1, \ldots, X_N$ be independent with the same mean $\mu$ and variance $\sigma^2$, and
$\bar X = \frac1N\sum_i X_i$ their average.

- **Law of large numbers.** $\bar X \to \mu$ as $N \to \infty$.
- **Central limit theorem.** $\bar X$ is close to normally distributed with mean $\mu$ and standard deviation $\sigma/\sqrt N$, and the
  approximation improves with $N$. Averaging $N$ values therefore divides the typical error by $\sqrt N$, and each extra digit of accuracy costs a hundred
  times as many samples.

For example, the mean of $N = 10\,000$ independent draws with $\sigma = 1$ is within $\pm 0.01$ of $\mu$ about 68% of the time and
within $\pm 0.02$ about 95% of the time, whatever the shape of the distribution being sampled.

## Complex numbers

A complex number is $z = a + ib$ with $i^2 = -1$, or in polar form $z = r e^{i\theta}$ with $r = |z|$ and $\theta = \arg z$; Euler's
formula $e^{i\theta} = \cos\theta + i\sin\theta$ links the two. Multiplying complex numbers multiplies their magnitudes and
adds their angles, which is why $e^{i\theta}$ is the natural way to describe a rotation or a phase. The **roots of unity**
$\omega_N^k = e^{2\pi i k/N}$, $k = 0, \ldots, N-1$, are the $N$ equally spaced points on the unit circle that satisfy $\omega^N = 1$;
they underlie the [fast Fourier transform](fft/index.md). A function of a complex variable is **analytic** in a region if it is
complex-differentiable there. That is a much stronger condition than real differentiability: an analytic function is
infinitely differentiable, equals its Taylor series near each point, and its behaviour on a real interval is controlled by
where its nearest singularities lie in the complex plane. That is the source of the geometric convergence rates in
[polynomial interpolation](interpolation/polynomial.md), [tanh-sinh quadrature](integration/tanh-sinh.md) and
[contour integration](integration/contour-integration.md). Signal processing and electromagnetics write $j$ for the
imaginary unit.
