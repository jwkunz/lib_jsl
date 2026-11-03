# Neural networks, ensembles, model selection and sequence models

This chapter completes the machine-learning part of the book with four topics that build on the
[regression and classification](../machine_learning/index.md) and [clustering](../unsupervised/index.md) chapters:

- **Neural networks** learn their own features by composing layers of simple nonlinear units, trained by backpropagation.
- **Gradient boosting** builds a strong predictor from many small trees, each fitted to the errors of the ones before.
- **Model selection** chooses among models and settings honestly, by cross-validation and grid search.
- **Hidden Markov models** handle sequences, in which the label of each step depends on its neighbours, by dynamic programming over hidden states.

## The central questions

- **How much capacity?** Hidden units, layers, stages, depth and epochs all add flexibility; weight decay, shrinkage and
  subsampling take it away. The right amount depends on the data and must be measured, not guessed.
- **How to measure it?** On data the model has not seen. Cross-validation reuses the training data for this, but its estimates are
  noisy, and the best of many cross-validated scores is optimistic.
- **Is the problem a sequence?** When observations depend on hidden states that persist over time, independent classification
  of each step throws away information that an HMM uses.

## Comparing classifiers

Test accuracy of the tree ensembles and the neural network against a single tree and a kernel machine, with default settings:

{{#include ../generated/learning/comparison.md}}

With default settings and a few hundred points, the methods are close: the RBF support vector machine is the most accurate and
among the fastest, the MLP is competitive but slowest, and both ensembles improve on a single tree. On large tabular data sets
boosted trees usually lead, and on images, sound and text neural networks do.

## Testing the library

Backpropagation is checked against finite differences for every activation and output layer, Viterbi against brute-force
enumeration of all paths, and the forward algorithm against an explicit sum over paths. Boosting's training loss and Baum-Welch's
likelihood are checked to be monotone, and Baum-Welch to recover a generating model.

Writing this chapter fixed four problems:

- An MLP with one input feature and two hidden layers could not be trained: a weight gradient came back column-major and the
  optimiser panicked reading it as a row-major slice.
- MLP predictions broadcast inputs with the wrong number of features instead of rejecting them.
- Gradient boosting predictions panicked on rows with too few features.
- The Baum-Welch likelihood history was documented as describing the model after each iteration; it describes the model before.

## In the library

All four are in `lib_jsl_machine_learning` ([API reference](../../api/lib_jsl_machine_learning/index.html)). The networks and boosted
models implement `Regressor` or `Classifier`, so the cross-validation functions accept them like any other model.

| You need | Use | Entry |
|---|---|---|
| A flexible nonlinear model trained by gradient descent | `MlpClassifier`, `MlpRegressor` | [Multilayer perceptrons](neural-networks.md) |
| The strongest general method for tabular data | `GradientBoostingRegressor`, `GradientBoostingClassifier` | [Gradient boosting](gradient-boosting.md) |
| Honest error estimates and hyperparameter tuning | `k_fold`, `stratified_k_fold`, `cross_validate_*`, `grid_search` | [Cross-validation and grid search](model-selection.md) |
| Decoding or learning a hidden state sequence | `HiddenMarkovModel`, `viterbi_path` | [Hidden Markov models](hidden-markov-models.md) |

## Demonstration programs

Four larger example programs in `crates/machine_learning/examples/` run many of the crate's methods side by side and print
tables and terminal plots:

| Program | Shows |
|---|---|
| `regression_demo` | Linear, polynomial, penalised and kernel regression on a noisy sine and a sparse linear problem |
| `classification_demo` | Seven classifiers on moons, circles and Gaussian classes, a hidden Markov model and a phylogenetic tree |
| `clustering_demo` | Centroid, density, connectivity and mixture clustering on blobs and moons |
| `embedding_demo` | PCA, kernel PCA, t-SNE and UMAP of clusters and the Swiss roll |

Run one with `cargo run --release -p lib_jsl_machine_learning --example clustering_demo`.
