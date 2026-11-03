# Naive Bayes

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Classify by Bayes' theorem, modelling each class's features as independent given the class |
| **Devised by** | Thomas Bayes (1763) and Pierre-Simon Laplace (1774), inverse probability; Melvin Maron (1961), text classification; Pedro Domingos and Michael Pazzani (1997), why the independence assumption does little harm; Andrew Ng and Michael Jordan (2002), generative against discriminative learning |
| **Accuracy** | Optimal when the features really are conditionally independent; otherwise often close (figures below) |
| **Cost** | Training $O(np)$, a single pass; prediction $O(Kp)$ per point |
| **API** | [`GaussianNaiveBayes`](../../api/lib_jsl_machine_learning/supervised/classification/naive_bayes/struct.GaussianNaiveBayes.html) (`predict_proba`, `joint_log_likelihood`), [`MultinomialNaiveBayes`](../../api/lib_jsl_machine_learning/supervised/classification/naive_bayes/struct.MultinomialNaiveBayes.html) |

</div>

## Overview

Bayes' theorem turns a model of how each class generates data into a classifier: $P(c \mid \mathbf x) \propto P(c)\,P(\mathbf x \mid c)$. Estimating the joint
density $P(\mathbf x \mid c)$ in many dimensions is hopeless, so **naive Bayes** assumes the features are independent within each class,
$P(\mathbf x \mid c) = \prod_jP(x_j \mid c)$, and estimates $p$ one-dimensional distributions per class instead. The assumption is almost never true, yet the
classifier works well, especially for text, where the features are word counts in thousands of dimensions. It trains in a
single pass over the data and is a **generative** model: it describes the data, not just the boundary.

## History

Thomas Bayes's essay (published posthumously in 1763) and Pierre-Simon Laplace (1774) founded inverse probability.
Melvin Maron (1961) used the independence model to classify documents automatically, and it became the standard spam
filter after Paul Graham's "A plan for spam" (2002). Pedro Domingos and Michael Pazzani (1997) explained its success: the
classifier only needs the *largest* posterior to be right, not the probabilities, and that survives strong dependence. Andrew
Ng and Michael Jordan (2002) compared it with logistic regression, its discriminative counterpart, and showed that naive
Bayes approaches its (higher) asymptotic error much faster, in a number of samples logarithmic rather than linear in the
dimension.

## The idea

### Gaussian naive Bayes

For continuous features, each class $c$ and feature $j$ gets a normal distribution with the class mean $\mu_{cj}$ and variance $\sigma^2_{cj}$ estimated
from the training data, and the prior $P(c)$ is the class frequency. Working in logarithms,

$$
\log P(c \mid \mathbf x) = \log P(c) - \frac12\sum_j\Big(\log(2\pi\sigma_{cj}^2) + \frac{(x_j - \mu_{cj})^2}{\sigma_{cj}^2}\Big) + \text{const}.
$$

The boundaries between classes are quadratic, since each class has its own variances. A small fraction of the largest
feature variance (`var_smoothing`, $10^{-9}$) is added to every variance so that a constant feature cannot cause a division by zero.

### When independence fails

With two classes sharing a correlated covariance, the optimal boundary is a line along the correlation. Naive Bayes, ignoring
the correlation, tilts its boundary; logistic regression, which models the boundary directly, finds the right one:

![Naive Bayes on correlated features](../generated/machine_learning/naive_bayes_boundary.svg)

{{#include ../generated/machine_learning/naive_bayes_accuracy.md}}

The cost of the wrong assumption here is 1.4 percentage points of accuracy.

### Multinomial naive Bayes for counts

For word counts $x_j$ in a document, the class-conditional model is a multinomial over the vocabulary: $\log P(\mathbf x \mid c) = \sum_jx_j\log\theta_{cj} + \text{const}$,
where $\theta_{cj}$ is the probability of word $j$ in class $c$. **Laplace smoothing** estimates $\theta_{cj} = (N_{cj} + \alpha)/(N_c + \alpha p)$ from the word totals, so that a word
never seen in a class does not give it probability zero. The classifier is linear in the counts.

### Generative against discriminative

Naive Bayes estimates each parameter from simple averages, so it reaches its best performance with few examples; logistic
regression needs more data to pin down its weights, but is not limited by a wrong model. On documents of 12 words from two
overlapping topics:

![Learning curves](../generated/machine_learning/naive_bayes_learning.svg)

Naive Bayes is ahead at every size up to a thousand documents, the regime Ng and Jordan describe; here the documents really
are generated word by word independently, so its model is correct and its advantage persists.

## Algorithm

1. **Fit**: class counts give the log priors; per class, the feature means and variances (Gaussian) or smoothed word
   frequencies (multinomial).
2. **Predict**: for each class, the log prior plus the sum of the per-feature log-likelihoods; return the largest, or normalise the
   exponentials (after subtracting the maximum) for probabilities.

## Implementation

```rust,ignore
{{#include ../../../crates/machine_learning/src/supervised/classification/naive_bayes.rs:gaussian_nb}}
```

- Every label in $0, \ldots, K - 1$ must occur in the training data.
- With `alpha = 0` an unseen word rules a class out, but only when it occurs in the document; absent words contribute
  nothing. (Before this chapter was written, they produced $0 \times (-\infty) = \text{NaN}$ and dropped the class from every vote.)

## In practice

```rust,ignore
{{#include ../../../crates/machine_learning/examples/ml_naive_bayes.rs}}
```

```text
free lunch meeting             -> not spam
click to claim money, winner   -> spam
project report                 -> not spam
vocabulary: free, winner, money, meeting, report, lunch, click, project
```

**Strengths**

- Trains in one pass, predicts in $O(Kp)$; handles thousands of features and little data.
- Probabilistic and generative; works for text.

**Weaknesses**

- The independence assumption distorts probabilities (they are typically too confident) and can misplace the boundary.
- Gaussian features only, or counts; no Bernoulli or categorical variants.

## Exercises

1. Show that Gaussian naive Bayes with variances shared across classes has linear decision boundaries.
2. Show that multinomial naive Bayes is a linear classifier in the word counts, and write down its weights.
3. Duplicate a feature $m$ times and show how naive Bayes's posterior changes. Why does the predicted class often not?
4. Derive the Laplace-smoothed estimate as the posterior mean under a Dirichlet prior.

## Further reading

- P. Domingos and M. Pazzani, "On the optimality of the simple Bayesian classifier under zero-one loss", *Machine Learning* 29 (1997) 103–130.
- A. Y. Ng and M. I. Jordan, "On discriminative vs. generative classifiers: a comparison of logistic regression and naive Bayes",
  *Advances in Neural Information Processing Systems* 14 (2002).
- C. D. Manning, P. Raghavan and H. Schütze, *Introduction to Information Retrieval*, Cambridge University Press, 2008, chapter 13.
