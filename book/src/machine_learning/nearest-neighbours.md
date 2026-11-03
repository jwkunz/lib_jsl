# k-nearest neighbours

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Classify a point by the labels of the training points closest to it |
| **Devised by** | Evelyn Fix and Joseph Hodges (1951); Thomas Cover and Peter Hart (1967), the error bound; Sahibsingh Dudani (1976), distance weighting |
| **Accuracy** | As the sample grows, the 1-nearest-neighbour error is at most twice the optimal (Bayes) error; with $k \to \infty$ and $k/n \to 0$, $k$-NN is consistent |
| **Cost** | No training. Prediction $O(np)$ per query by brute force |
| **API** | [`KNearestNeighbors`](../../api/lib_jsl_machine_learning/supervised/classification/k_nearest_neighbors/struct.KNearestNeighbors.html) (`new(k)`, `distance_weighted`, `predict_proba`, `neighbours`) |

</div>

## Overview

The nearest-neighbour rule is the simplest classifier that can learn any boundary: to classify a point, find the $k$ training
points closest to it and take a majority vote. There is no training, only storage, and no assumption about the shape of the
classes. The number of neighbours $k$ controls the smoothness of the boundary, from jagged ($k = 1$, every training point decides its
own neighbourhood) to nearly linear (large $k$). The price is paid at prediction time, and in high dimensions, where all points
are nearly equally far apart.

## History

Evelyn Fix and Joseph Hodges (1951), in a technical report for the US Air Force School of Aviation Medicine, introduced
nonparametric discrimination by nearest neighbours and proved its consistency. Thomas Cover and Peter Hart (1967) showed that
with unlimited data the 1-NN error rate is at most twice the Bayes rate, the smallest achievable: half the information for
classification lies in the nearest neighbour. Sahibsingh Dudani (1976) proposed weighting the votes by distance. Data
structures such as k-d trees (Jon Bentley, 1975) and, in high dimensions, approximate methods such as locality-sensitive
hashing (Piotr Indyk and Rajeev Motwani, 1998) make the search fast.

## The idea

### Voting

For a query $\mathbf x$, find the set $N_k(\mathbf x)$ of the $k$ training points nearest in Euclidean distance, and estimate
$P(c \mid \mathbf x) \approx \frac1k\#\{i \in N_k(\mathbf x) : y_i = c\}$. With **distance weighting**, each neighbour's vote counts $1/d_i$, so closer points matter more; a training point at
distance zero decides alone.

### Choosing k

| | | |
|:---:|:---:|:---:|
| ![k = 1](../generated/machine_learning/knn_1.svg) | ![k = 15](../generated/machine_learning/knn_15.svg) | ![k = 101](../generated/machine_learning/knn_101.svg) |

With $k = 1$ the boundary encloses every isolated point; with $k = 15$ it follows the moons; with $k = 101$ it approaches a straight line.
The training error is zero for $k = 1$ and grows with $k$; the test error is lowest in between:

![Error against k](../generated/machine_learning/knn_error.svg)

### The curse of dimensionality

In $p$ dimensions, a neighbourhood containing a fraction $f$ of uniformly spread data has edge length $f^{1/p}$: with $p = 10$, capturing 1% of the
data needs a cube spanning 63% of each axis. Neighbours are no longer local, distances concentrate, and nearest-neighbour
methods degrade unless the data lie near a low-dimensional manifold. Irrelevant features add noise to every distance, and
features on large scales dominate it: standardise, and select features.

## Algorithm

1. **Fit**: store the training points and labels.
2. **Predict**: compute the squared distances to all training points; select the $k$ smallest with a linear-time partial sort;
   vote (or weight by $1/d$); return the class with the largest share.

## Implementation

```rust,ignore
{{#include ../../../crates/machine_learning/src/supervised/classification/k_nearest_neighbors.rs:knn}}
```

- Ties among equidistant neighbours are broken by training index, and ties in the vote by the lower class label, so results
  are deterministic.
- The search is brute force. For many queries in two or three dimensions, the k-d tree of the `lib_jsl_geometry` crate is faster.

## In practice

```rust,ignore
{{#include ../../../crates/machine_learning/examples/ml_nearest_neighbours.rs}}
```

```text
k =   1: test accuracy 0.877, distance-weighted 0.877
k =   5: test accuracy 0.896, distance-weighted 0.891
k =  15: test accuracy 0.901, distance-weighted 0.902
k =  45: test accuracy 0.899, distance-weighted 0.901
k = 135: test accuracy 0.798, distance-weighted 0.883
neighbours of (0.5, 0.25): #29 (class 0, 0.05), #116 (class 1, 0.09), #122 (class 1, 0.10), #5 (class 0, 0.15), #51 (class 0, 0.18)
```

Distance weighting makes the choice of $k$ less critical: at $k = 135$, two thirds of the 200 training points, plain voting is swamped
by distant points while the weighted vote still works.

**Strengths**

- No training; any boundary shape; naturally multi-class; easy to explain ("these were the most similar cases").

**Weaknesses**

- Prediction costs $O(np)$ per query and the whole training set must be kept.
- Sensitive to feature scaling and irrelevant features; degrades in high dimensions.

## Exercises

1. Show that the leave-one-out error of 1-NN can be computed in $O(n^2p)$, and use it to choose $k$.
2. In $p$ dimensions with $n$ uniform points in the unit cube, estimate the median distance from the origin to the nearest point, and
   how it grows with $p$ for fixed $n$.
3. Sketch the proof of the Cover-Hart bound for two classes: as $n \to \infty$, $P(\text{error}) \to 2\eta(1 - \eta)$ at a point with $P(y = 1) = \eta$.
4. Implement a k-d tree search for two-dimensional data and compare its speed with brute force for $n = 10^5$.

## Further reading

- T. M. Cover and P. E. Hart, "Nearest neighbor pattern classification", *IEEE Transactions on Information Theory* 13 (1967) 21–27.
- E. Fix and J. L. Hodges, "Discriminatory analysis. Nonparametric discrimination: consistency properties", USAF School of Aviation
  Medicine report 4, 1951; reprinted in *International Statistical Review* 57 (1989) 238–247.
- T. Hastie, R. Tibshirani and J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009, chapter 13.
