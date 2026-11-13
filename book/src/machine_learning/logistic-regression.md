# Logistic regression

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Predict the probability of each class as a function of the features, with a linear decision boundary |
| **Devised by** | Pierre-François Verhulst (1838), the logistic function; Joseph Berkson (1944), the logit model; David Cox (1958), logistic regression; Daniel McFadden (1973), the multinomial model; Dong Liu and Jorge Nocedal (1989), L-BFGS |
| **Accuracy** | Gradient of the penalised log-loss below the tolerance ($10^{-7}$ by default) |
| **Cost** | $O(nKp)$ per gradient evaluation for $K$ classes; typically 10–100 L-BFGS iterations |
| **API** | [`LogisticRegression`](../../api/lib_jsl_machine_learning/supervised/classification/logistic_regression/struct.LogisticRegression.html) (`fit`, `predict`, `predict_proba`, `coefficients`, `intercepts`) |

</div>

## Overview

Logistic regression models the **log-odds** of a class as a linear function of the features. For two classes,
$P(y = 1 \mid \mathbf x) = \sigma(\mathbf w^{\mathsf T}\mathbf x + b)$ with the logistic function $\sigma(z) = 1/(1 + e^{-z})$; for $K$ classes the **softmax** generalises it,

$$
P(y = c \mid \mathbf x) = \frac{e^{\mathbf w_c^{\mathsf T}\mathbf x + b_c}}{\sum_k e^{\mathbf w_k^{\mathsf T}\mathbf x + b_k}}.
$$

The weights are fitted by maximum likelihood. The result is a linear classifier whose probabilities are calibrated when the model fits (without regularisation, the fitted probabilities
average to the observed fraction of positives; with strong regularisation or a poorly matched boundary they need not, and a
reliability plot is the way to check) and whose coefficients are interpretable: each unit of feature $j$ multiplies the odds by $e^{w_j}$. It is the standard model for binary outcomes
in medicine and the social sciences, and the output layer of most neural network classifiers.

## History

Pierre-François Verhulst (1838) introduced the logistic curve to model population growth. Joseph Berkson (1944) proposed
the "logit" as an alternative to the probit model of bioassay, and David Cox (1958) developed logistic regression for binary
sequences. Daniel McFadden (1973) derived the multinomial logit from a theory of choice, work recognised by the 2000 Nobel
prize in economics. Iteratively reweighted least squares (Newton's method) was the classical fitting algorithm; for large
problems quasi-Newton methods such as L-BFGS (Liu and Nocedal, 1989) took over.

## The idea

### The likelihood

**From scores to probabilities.** The model computes a linear **score** for each class,
$z_c = \mathbf w_c^{\mathsf T}\mathbf x + b_c$, and the softmax turns the scores into probabilities:
exponentiating makes them positive, and dividing by the sum makes them add up to 1. A class with
a higher score gets a higher probability. For two classes this reduces to the logistic function
of the difference of the scores, which rises smoothly from 0 to 1.

**Fitting by maximum likelihood.** Given training data, how should the weights be chosen? The
model assigns a probability to each possible label of each training point. The **likelihood** of
the weights is the probability the model assigns to the labels that were actually observed, the
product over the training points of $p_{i,y_i}$, where $p_{ic} = P(y_i = c \mid \mathbf x_i)$ and
$y_i$ is the true class of point $i$. Maximum likelihood chooses the weights that make the
observed labels as probable as possible.

Products of many probabilities are tiny and awkward, so one works with the logarithm, which
turns the product into a sum, and changes the sign to obtain a quantity to minimise. Averaged
over the data this is the **cross-entropy**, or **log-loss**:

$$
L = -\frac1n\sum_i\ln p_{i,y_i} .
$$

Each point contributes $-\ln$ of the probability given to its true class: nothing if the model
was certain and right, a little if it was fairly confident and right, and a great deal if it was
confident and wrong.

**The gradient.** Differentiating $L$ through the softmax gives a remarkably simple result. For
the weight of feature $j$ in class $c$,

$$
\frac{\partial L}{\partial w_{cj}} = \frac1n\sum_i\big(p_{ic} - [y_i = c]\big)\,x_{ij},
$$

where $[y_i = c]$ is 1 if point $i$ belongs to class $c$ and 0 otherwise. The bracket is the
**prediction error**: the probability the model gave to class $c$, minus what it should have
been. The gradient is the average of the prediction errors, weighted by the feature values. It
vanishes when the errors are uncorrelated with every feature.

The loss is convex, so any minimum is global and there are no false minima to be trapped in. But
unlike least squares, setting the gradient to zero gives equations that cannot be solved in
closed form, because the probabilities depend nonlinearly on the weights. An iterative solver is
needed.

### Separation and the penalty

Maximum likelihood has a failure mode that is easy to overlook. Suppose a hyperplane separates
the classes perfectly in the training data. Then scaling the weights up makes the model more
confident about every training point, all of which it classifies correctly, and the likelihood
increases. Doubling the weights again increases it further. The likelihood has no maximum: it
approaches its upper limit only as the weights grow without bound, and the fitted probabilities
all tend to 0 or 1. This **separation** happens easily with many features and few samples.

The remedy is the one used in [ridge regression](linear-regression.md#ridge-regression): add an
$\ell_2$ penalty $\frac{\lambda}{2}\lVert W\rVert^2$ on the weights (intercepts excluded) to the
loss. The penalised minimum is unique and finite, and the less extreme probabilities it produces
are better on new data:

<p class="caption" id="tab-11-5"><strong>Table 11.5.</strong> Columns: $\lambda$, L-BFGS iterations, $\lVert W\rVert$, training log-loss, and more.</p>

{{#include ../generated/machine_learning/logistic_penalty.md}}

Without a penalty the training loss falls to zero, the weights reach $\lVert W\rVert = 21$ (and are still growing when the solver stops,
because the loss no longer changes measurably) and the test log-loss is 5.2, confidently wrong. A moderate penalty barely
changes the accuracy but improves the test log-loss ninefold.

### Probability contours

![Logistic regression contours](../generated/machine_learning/logistic_contours.svg)

<p class="caption" id="fig-11-4"><strong>Figure 11.4.</strong> Logistic regression contours.</p>

For two classes the probability depends on $\mathbf x$ only through the score
$\mathbf w^{\mathsf T}\mathbf x + b$, which is constant along lines perpendicular to $\mathbf w$. The
**decision boundary** $P = 0.5$, where the score is zero, is therefore a straight line (a
hyperplane in higher dimensions), and the other probability contours are lines parallel to it.
Their spacing reflects the overlap of the classes: closely spaced contours mean a sharp
transition, and so well-separated classes.

### L-BFGS

The library minimises the penalised loss with **L-BFGS**: a [quasi-Newton method](../optimization/newton-quasi-newton.md#l-bfgs)
that approximates the inverse Hessian from the last 10 pairs of steps and gradient changes, with a backtracking line search.
It needs only the loss and gradient, and converges in a few dozen iterations on well-scaled data. Standardise the features
first: badly scaled features slow it down.

## Algorithm

Input: the training features (which should be standardised), the labels and the penalty
$\lambda$.

1. **Initialise** all weights and intercepts to zero, which gives every class the same
   probability.
2. **Evaluate the loss and its gradient** at the current weights:
   - compute the scores $z_{ic} = \mathbf w_c^{\mathsf T}\mathbf x_i + b_c$ for every point and class;
   - turn each row of scores into probabilities by the softmax. For numerical safety, subtract
     the row's largest score from every score in the row first: this does not change the
     probabilities, and it keeps the exponentials from overflowing;
   - the loss is the mean of $-\ln p_{i,y_i}$ plus the penalty $\tfrac\lambda2\lVert W\rVert^2$;
   - the gradient is the mean of the prediction errors times the features, plus $\lambda W$.
3. **Take an L-BFGS step**: compute the search direction from the stored pairs, and backtrack
   along it from a unit step until the loss has decreased sufficiently (the Armijo condition).
4. **Test.** Stop when the largest component of the gradient is below the tolerance, or when
   the loss has stopped changing; otherwise go to 2.

Each evaluation of the loss and gradient costs $O(npK)$ for $n$ points, $p$ features and $K$
classes.

## Implementation

```rust,ignore
{{#include ../../../crates/machine_learning/src/supervised/classification/logistic_regression.rs:logistic_objective}}
```

- The model is over-parameterised (adding a constant to every class's weights changes nothing); the penalty selects the
  solution with $\sum_c\mathbf w_c = 0$. For two classes, the usual log-odds coefficients are the differences of the two rows.

## In practice

```rust,ignore
{{#include ../../../crates/machine_learning/examples/ml_logistic_regression.rs}}
```

```text
log-odds = -6.39 +0.92 study +0.41 sleep  (22 iterations)
study 2 h, sleep 8 h: P(pass) = 0.22
study 5 h, sleep 7 h: P(pass) = 0.74
study 8 h, sleep 5 h: P(pass) = 0.95
```

The fitted log-odds recover the true model ($-6 + 0.8\,\text{study} + 0.5\,\text{sleep}$) to within sampling error from 200 students.

**Strengths**

- Probabilities, not just labels; interpretable coefficients; a convex problem with a unique penalised solution.
- Fast and reliable; the natural baseline for classification.

**Weaknesses**

- Linear decision boundaries (unless features are expanded); fails on problems such as the concentric circles.
- No standard errors or p-values for the coefficients.
- Only an $\ell_2$ penalty; no sparse ($\ell_1$) logistic regression.

## Exercises

1. Show that $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ and derive the gradient of the binary cross-entropy.
2. Show that the Hessian of the cross-entropy is positive semidefinite, so the loss is convex.
3. Construct a separable data set and show that the unpenalised loss has no minimum.
4. Show that the softmax model with $K = 2$ is equivalent to binary logistic regression with $\mathbf w = \mathbf w_1 - \mathbf w_0$.
5. Implement Newton's method (iteratively reweighted least squares) for binary logistic regression and compare its iteration count with L-BFGS.

## Further reading

- D. R. Cox, "The regression analysis of binary sequences", *Journal of the Royal Statistical Society B* 20 (1958) 215–242.
- D. W. Hosmer, S. Lemeshow and R. X. Sturdivant, *Applied Logistic Regression*, 3rd ed., Wiley, 2013.
- D. C. Liu and J. Nocedal, "On the limited memory BFGS method for large scale optimization", *Mathematical Programming* 45 (1989) 503–528.
- T. Hastie, R. Tibshirani and J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009, section 4.4.
