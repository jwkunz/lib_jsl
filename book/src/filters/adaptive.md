# Adaptive filters: LMS, NLMS and RLS

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Adjust the taps of a filter while it runs, so that its output tracks a desired signal: identify an unknown system, cancel noise or echo, equalise a channel |
| **Devised by** | Bernard Widrow and Marcian Hoff (1960), least mean squares; Jin-ichi Nagumo and Atsuhiko Noda (1967) and Arthur Albert and Leland Gardner (1967), normalised LMS; Robin Plackett (1950) and Dominique Godard (1974), recursive least squares |
| **Accuracy** | LMS converges in the mean for small enough steps, with an excess error proportional to the step; RLS minimises the exponentially weighted squared error exactly at every step |
| **Cost** | LMS and NLMS: $O(M)$ per sample for $M$ taps. RLS: $O(M^2)$ per sample |
| **API** | [`LmsFilter`](../../api/lib_jsl_dsp/discrete/filters/adaptive/lms/struct.LmsFilter.html) (`new(taps, step_size, normalized)`, `adapt`, `adapt_block`, `filter`, `weights`), [`RlsFilter`](../../api/lib_jsl_dsp/discrete/filters/adaptive/rls/struct.RlsFilter.html) (`new(taps, forgetting, initial_inverse_correlation)`, `adapt`, `adapt_block`, `weights`) |

</div>

## Overview

A fixed filter needs the signal's statistics in advance. An **adaptive filter** learns them as it runs. At each sample it forms an output
$y_n = \mathbf w^{\mathsf H}\mathbf u_n$ from the most recent inputs $\mathbf u_n = (x_n, x_{n-1}, \ldots, x_{n-M+1})$, compares it with a desired response $d_n$, and nudges its weights to reduce the error $e_n = d_n - y_n$. Arranged
suitably, the same loop identifies an unknown system (the desired signal is the system's output), cancels noise (the desired signal is
the corrupted one, the input a reference correlated with the noise), cancels echoes on telephone lines, and equalises
communication channels. **LMS** takes one cheap stochastic-gradient step per sample. **RLS** solves the least-squares problem exactly
at every sample, converging far faster at quadratic cost.

## History

Bernard Widrow and his student Marcian Hoff (1960) invented the least-mean-squares rule for their ADALINE adaptive neuron at
Stanford; it became the most widely used adaptive algorithm. Normalising the step by the input power was proposed independently by
Jin-ichi Nagumo and Atsuhiko Noda (1967) and by Arthur Albert and Leland Gardner (1967). Recursive least squares goes back to Robin
Plackett's (1950) recursive form of Gauss's least squares, and to the Kalman filter, of which it is a special case; Dominique Godard (1974)
applied it to channel equalisation. Simon Haykin's *Adaptive Filter Theory* (1986, now in its fifth edition) is the standard text.

## The idea

### The Wiener solution

If the signals are stationary, the weights that minimise the mean squared error $J(\mathbf w) = E|d_n - \mathbf w^{\mathsf H}\mathbf u_n|^2$ satisfy the **Wiener-Hopf** equations $R\mathbf w_\star = \mathbf p$, with input correlation matrix
$R = E[\mathbf u\mathbf u^{\mathsf H}]$ and cross-correlation $\mathbf p = E[\mathbf u\,\bar d]$. Adaptive filters approach $\mathbf w_\star$ without knowing $R$ or $\mathbf p$, and follow it when the statistics drift.

### LMS: stochastic gradient descent

The gradient of $J$ is $-2(\mathbf p - R\mathbf w)$. Replacing the expectations by their instantaneous values gives the **LMS** update

$$
\mathbf w_{n+1} = \mathbf w_n + \mu\,\bar e_n\,\mathbf u_n,
$$

one multiply-add per tap. For small $\mu$ the weights converge in the mean when $0 < \mu < 2/\lambda_{\max}(R)$, and in practice when $\mu$ is well below $2/(M\,P_x)$, where $P_x$ is the input power. Convergence
proceeds along the eigenvectors of $R$, each at a rate set by $\mu\lambda_i$. With a large **eigenvalue spread** $\lambda_{\max}/\lambda_{\min}$, the slow modes dominate. Gradient noise leaves a
**misadjustment**, an excess mean squared error that grows with $\mu$. The choice of $\mu$ trades speed against final accuracy.

### NLMS: a step normalised by the input power

**Normalised LMS** divides the step by the regressor's energy:

$$
\mathbf w_{n+1} = \mathbf w_n + \frac{\mu}{\epsilon + \lVert\mathbf u_n\rVert^2}\,\bar e_n\,\mathbf u_n, \qquad 0 < \mu < 2.
$$

It is stable for any input level and converges faster when the input power varies. With $\mu = 1$ it makes the a posteriori error zero at each step.

### RLS: exact recursive least squares

**RLS** minimises $\sum_{i \le n}\lambda^{n-i}|d_i - \mathbf w^{\mathsf H}\mathbf u_i|^2$ exactly at every step, where the **forgetting factor** $\lambda \le 1$ discounts the past over about $1/(1 - \lambda)$ samples. Its normal equations update
by rank one at each step, and the matrix inversion lemma updates the inverse correlation matrix $P$ directly:

$$
\mathbf k = \frac{P\mathbf u}{\lambda + \mathbf u^{\mathsf H}P\mathbf u}, \qquad e = d - \mathbf w^{\mathsf H}\mathbf u, \qquad \mathbf w \leftarrow \mathbf w + \mathbf k\bar e, \qquad P \leftarrow \frac{P - \mathbf k\mathbf u^{\mathsf H}P}{\lambda}.
$$

RLS converges in about $2M$ samples whatever the eigenvalue spread, because $P$ approximates $R^{-1}$ and "whitens" the update, a Newton step instead of a gradient
step. The cost is $O(M^2)$ per sample, and in finite precision $P$ must be kept Hermitian, or it drifts indefinite and the filter diverges. The library
updates one triangle of $P$ and mirrors it.

### Learning curves

Identifying an 8-tap system from noisy observations (noise floor −40 dB), averaged over 50 runs:

| | |
|:---:|:---:|
| ![White input](../generated/filters/adaptive_white.svg) | ![Coloured input](../generated/filters/adaptive_coloured.svg) |

With white input all three reach the noise floor, RLS in a few dozen samples, LMS in a few hundred. With strongly coloured input
(a first-order autoregressive process with coefficient 0.9, eigenvalue spread 113 for 8 taps), LMS needs thousands of samples, while RLS is unaffected.
The price of RLS grows with the square of the length:

{{#include ../generated/filters/adaptive_cost.md}}

## Algorithm

1. **LMS/NLMS**: shift the new sample into the regressor; $y = \mathbf w^{\mathsf H}\mathbf u$, $e = d - y$; step $\mu$ (or $\mu/(\epsilon + \lVert\mathbf u\rVert^2)$); $\mathbf w \leftarrow \mathbf w + \text{step}\cdot\bar e\,\mathbf u$.
2. **RLS**: initialise $P = \delta I$ with large $\delta$ (little confidence in the zero initial weights). Each sample: $\boldsymbol\pi = P\mathbf u$, $\mathbf k = \boldsymbol\pi/(\lambda + \mathbf u^{\mathsf H}\boldsymbol\pi)$, $e$, $\mathbf w$ update, then
   $P \leftarrow (P - \mathbf k\boldsymbol\pi^{\mathsf H})/\lambda$ on one triangle, mirrored.

## Implementation

```rust,ignore
{{#include ../../../crates/dsp/src/discrete/filters/adaptive/lms.rs:lms}}
```

```rust,ignore
{{#include ../../../crates/dsp/src/discrete/filters/adaptive/rls.rs:rls}}
```

- Both filters work on complex samples; pass real signals with zero imaginary parts.
- `adapt` returns the a priori output and error; `filter` (LMS) applies the current weights without adapting.

## In practice

```rust,ignore
{{#include ../../../crates/dsp/examples/dsp_adaptive.rs}}
```

```text
n =   10: |e| NLMS 1.4e-1, RLS 3.7e-3
n =   50: |e| NLMS 1.5e-4, RLS 2.2e-4
n =  200: |e| NLMS 1.8e-14, RLS 7.8e-5
n = 1999: |e| NLMS 0.0e0, RLS 5.2e-7
NLMS weights [0.6000, -0.3000, 0.2000, 0.0500]
RLS weights  [0.6000, -0.3000, 0.2000, 0.0500]
```

Without noise, both recover the system exactly. RLS is far ahead after 10 samples, but its error then shrinks slowly: the initial
$P = 100I$ acts as a weak prior pulling the weights towards zero, and its influence fades only at the rate $\lambda^n$. NLMS with $\mu = 0.5$ converges geometrically to
machine precision.

**Strengths**

- LMS is simple, cheap and robust, and tracks slow changes; NLMS removes the dependence on the input level.
- RLS converges in a few multiples of the filter length, regardless of the input's spectrum.

**Weaknesses**

- LMS slows down with coloured input and trades speed against misadjustment through one parameter.
- RLS costs $O(M^2)$ per sample and needs care to stay numerically stable; both assume a linear FIR model.

## Exercises

1. Derive the Wiener-Hopf equations from $\nabla J = 0$.
2. Show that LMS converges in the mean when $0 < \mu < 2/\lambda_{\max}$, by writing the mean weight error as a linear recursion.
3. Show that NLMS with $\mu = 1$ makes the a posteriori error $d_n - \mathbf w_{n+1}^{\mathsf H}\mathbf u_n$ zero (for $\epsilon = 0$).
4. Derive the RLS update of $P$ from the matrix inversion lemma.
5. Set up a noise canceller: a sinusoid plus noise $v$ as the desired signal, and a filtered copy of $v$ as the input. Show that the error converges to the sinusoid.

## Further reading

- B. Widrow and M. E. Hoff, "Adaptive switching circuits", *IRE WESCON Convention Record* 4 (1960) 96–104.
- S. Haykin, *Adaptive Filter Theory*, 5th ed., Pearson, 2014.
- A. H. Sayed, *Fundamentals of Adaptive Filtering*, Wiley, 2003.
- B. Widrow, J. R. Glover et al., "Adaptive noise cancelling: principles and applications", *Proceedings of the IEEE* 63 (1975) 1692–1716.
