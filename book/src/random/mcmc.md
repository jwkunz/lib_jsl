# Markov chain Monte Carlo

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Draw samples from a probability density known only up to a constant factor, typically a Bayesian posterior, to estimate expectations, quantiles and uncertainties |
| **Devised by** | Nicholas Metropolis, Arianna and Marshall Rosenbluth, Augusta and Edward Teller (1953); W. Keith Hastings (1970); Stuart and Donald Geman (1984), Gibbs sampling; Gareth Roberts, Andrew Gelman and Walter Gilks (1997), optimal scaling; Heikki Haario, Eero Saksman and Johanna Tamminen (2001), adaptive proposals |
| **Accuracy** | Errors of averages fall like $\sqrt{\tau/N}$, where $\tau$ is the integrated autocorrelation time of the chain |
| **Cost** | One evaluation of the log-density per step |
| **API** | [`metropolis_hastings`](../../api/lib_jsl_random/mcmc/fn.metropolis_hastings.html), [`MetropolisOptions`](../../api/lib_jsl_random/mcmc/struct.MetropolisOptions.html), [`MarkovChain`](../../api/lib_jsl_random/mcmc/struct.MarkovChain.html) |

</div>

## Overview

In Bayesian statistics the object of interest is the posterior density $p(\theta \mid \text{data}) \propto p(\text{data} \mid \theta)\,p(\theta)$, which
is easy to evaluate up to its normalising constant, but whose normalisation is an integral over all the parameters:
intractable beyond a few dimensions. The same problem arises in statistical physics, where the Boltzmann distribution
$e^{-E(x)/kT}$ has an unknown partition function. **Markov chain Monte Carlo** (MCMC) sidesteps the constant: it constructs a
random walk whose long-run distribution is the target, using only ratios of densities, and estimates expectations by
averages along the walk.

The price is correlation. Successive states of the chain are dependent, so $N$ steps carry the information of fewer
independent samples, and the art of MCMC is to make the chain mix quickly.

## History

Nicholas Metropolis, Arianna Rosenbluth, Marshall Rosenbluth, Augusta Teller and Edward Teller (1953) introduced the
algorithm at Los Alamos to compute the equation of state of a system of interacting particles on the MANIAC computer;
Arianna Rosenbluth wrote the program. W. Keith Hastings (1970) generalised it to asymmetric proposals. Stuart and Donald
Geman (1984) introduced Gibbs sampling for image restoration, and Alan Gelfand and Adrian Smith (1990) showed statisticians
how to compute Bayesian posteriors with it, which transformed applied statistics. Gareth Roberts, Andrew Gelman and Walter
Gilks (1997) proved that random-walk proposals mix best at an acceptance rate of about 0.234 in high dimensions, and Haario,
Saksman and Tamminen (2001) began the theory of adaptive MCMC. Hamiltonian Monte Carlo (Duane et al., 1987) and its tuned
variant NUTS (Hoffman and Gelman, 2014) power modern systems such as Stan.

## The idea

### The Metropolis-Hastings algorithm

From the current state $x$, propose $x'$ from a proposal density $q(x' \mid x)$ and accept it with probability

$$
\alpha(x, x') = \min\Big(1, \frac{p(x')\,q(x \mid x')}{p(x)\,q(x' \mid x)}\Big);
$$

otherwise stay at $x$ (and count $x$ again). The unknown normalising constant of $p$ cancels in the ratio. The acceptance
probability makes the chain satisfy **detailed balance**, $p(x)\,P(x \to x') = p(x')\,P(x' \to x)$, so $p$ is its stationary
distribution, and under mild conditions the distribution of the state converges to $p$ from any start.

The library implements the **random-walk** version: $x' = x + \boldsymbol\sigma \odot \mathbf z$ with $\mathbf z$ standard normal, a symmetric
proposal for which $q$ cancels and $\alpha = \min(1, p(x')/p(x))$. Working with log-densities avoids overflow, and a log-density of $-\infty$
marks points outside the support, which are always rejected.

### Burn-in, thinning and the proposal scale

The chain starts wherever it is put, possibly in a region of negligible probability, so the first states are discarded as
**burn-in**. Keeping only every $k$-th state (**thinning**) saves storage but does not improve the estimates.

The step size decides how fast the chain explores. Too small, and almost every proposal is accepted but the chain crawls;
too large, and almost every proposal is rejected and the chain sticks. On a standard normal target from $x = 4$:

![Traces of three chains](../generated/random/mcmc_trace.svg)

![Autocorrelation](../generated/random/mcmc_acf.svg)

### Effective sample size

The variance of an MCMC average is $\sigma^2\tau/N$ rather than $\sigma^2/N$, where the **integrated autocorrelation time**
$\tau = 1 + 2\sum_{k\ge1}\rho(k)$ sums the autocorrelations of the chain; $N/\tau$ is the **effective sample size**:

{{#include ../generated/random/mcmc_table.md}}

The well-tuned chain is a hundred times more efficient than the timid one. In one dimension the optimal random-walk
acceptance rate is about 0.44, and in many dimensions about 0.234; the library adapts the scale towards these rates during
burn-in (multiplying it every 100 steps by $e^{2(r - r^*)}$ for observed rate $r$), as the last row shows, and keeps it fixed afterwards
so that the chain is a proper Markov chain.

### Difficult targets

Random-walk proposals explore curved or strongly correlated densities slowly, because the step must be small enough for the
narrowest direction. A banana-shaped density, $x \sim N(0, 4)$ and $y \mid x \sim N(x^2/4, 1)$, is still sampled correctly, with thinning:

![Banana-shaped density](../generated/random/mcmc_banana.svg)

Multimodal densities are harder still: a random walk may never cross between modes. Gradient-based methods (Hamiltonian
Monte Carlo), tempering and problem-specific proposals address these cases.

## Algorithm

1. Start at $x_0$ with $\log p(x_0)$ finite.
2. For each step: propose $x' = x + \boldsymbol\sigma \odot \mathbf z$; accept if $\log p(x') \ge \log p(x)$ or $\ln u < \log p(x') - \log p(x)$; otherwise keep $x$.
3. During burn-in, optionally rescale $\boldsymbol\sigma$ every 100 steps towards the target acceptance rate.
4. After burn-in, record every `thin`-th state.

## Implementation

```rust,ignore
{{#include ../../../crates/random/src/mcmc.rs:metropolis}}
```

- `MarkovChain` holds the samples, the post-burn-in acceptance rate and the final proposal scale, with `mean` and
  `autocorrelation` helpers.
- Check convergence: run several chains from different starts and compare them, and look at traces and autocorrelations.

## In practice

```rust,ignore
{{#include ../../../crates/random/examples/random_mcmc.rs}}
```

**Strengths**

- Needs only an unnormalised log-density; simple and general.
- Adaptive tuning of the step during burn-in.

**Weaknesses**

- Correlated samples: the effective sample size can be far below the number of steps.
- Random-walk proposals scale poorly with dimension and curvature, and cannot cross between separated modes.
- Convergence is diagnosed, never guaranteed.

## Exercises

1. Prove that the Metropolis-Hastings transition satisfies detailed balance with respect to $p$.
2. For a random walk on $N(0, 1)$ with step $\sigma$, estimate the acceptance rate and $\tau$ for $\sigma = 0.5, 1, 2.4, 5$, and find the $\sigma$ that minimises $\tau$.
3. Sample a mixture of two normals at $\pm3$ and at $\pm6$ with unit variance. How often does the chain switch modes?
4. Implement a Gibbs sampler for a bivariate normal with correlation 0.99 and compare its autocorrelation time with random-walk Metropolis.
5. Fit the dose-response model of the example by maximum likelihood and compare with the posterior means.

## Further reading

- N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller and E. Teller, "Equation of state calculations by fast computing
  machines", *Journal of Chemical Physics* 21 (1953) 1087–1092.
- W. K. Hastings, "Monte Carlo sampling methods using Markov chains and their applications", *Biometrika* 57 (1970) 97–109.
- G. O. Roberts, A. Gelman and W. R. Gilks, "Weak convergence and optimal scaling of random walk Metropolis algorithms", *Annals of
  Applied Probability* 7 (1997) 110–120.
- S. Brooks, A. Gelman, G. L. Jones and X.-L. Meng (eds.), *Handbook of Markov Chain Monte Carlo*, CRC Press, 2011.
- A. Gelman et al., *Bayesian Data Analysis*, 3rd ed., CRC Press, 2013, chapters 11–12 (the bioassay example).
