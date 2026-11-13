# Multilayer perceptrons and backpropagation

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Fit a flexible nonlinear function, built from layers of weighted sums and simple nonlinearities, to regression targets or class labels |
| **Devised by** | Warren McCulloch and Walter Pitts (1943), the artificial neuron; Frank Rosenblatt (1958), the perceptron; Seppo Linnainmaa (1970), reverse-mode differentiation; Paul Werbos (1974) and David Rumelhart, Geoffrey Hinton and Ronald Williams (1986), backpropagation; George Cybenko (1989) and Kurt Hornik (1991), universal approximation; Diederik Kingma and Jimmy Ba (2014), Adam |
| **Accuracy** | Backpropagation gives exact gradients, checked against finite differences in the tests; training finds a local minimum of a nonconvex loss |
| **Cost** | Forward and backward passes each $O(B\sum_lh_{l-1}h_l)$ per mini-batch of $B$ rows, for layer widths $h_l$ |
| **API** | [`MlpClassifier`](../../api/lib_jsl_machine_learning/neural_network/struct.MlpClassifier.html) (`predict_proba`, `training_loss`), [`MlpRegressor`](../../api/lib_jsl_machine_learning/neural_network/struct.MlpRegressor.html), [`MlpParameters`](../../api/lib_jsl_machine_learning/neural_network/struct.MlpParameters.html) (`hidden`, `activation`, `learning_rate`, `epochs`, `batch_size`, `weight_decay`, `seed`), [`Activation`](../../api/lib_jsl_machine_learning/neural_network/enum.Activation.html) |

</div>

## Overview

A **multilayer perceptron** (MLP) composes simple layers: each multiplies its input by a weight matrix, adds a bias, and applies a
fixed nonlinear **activation** elementwise. The last layer gives a number for regression or class probabilities for classification.
With enough hidden units, such networks can approximate any continuous function. The weights are learned by gradient descent
on a loss, with the gradient computed by **backpropagation**, the chain rule applied layer by layer from the output back to
the input. The MLP is the simplest neural network and the building block of deep learning. On small tabular problems it competes
with, rather than beats, kernel machines and boosted trees.

## History

Warren McCulloch and Walter Pitts (1943) modelled neurons as threshold units, and Frank Rosenblatt's perceptron (1958) learned
the weights of one such layer. Marvin Minsky and Seymour Papert (1969) showed that a single layer cannot represent XOR, and
interest faded. Reverse-mode automatic differentiation was described by Seppo Linnainmaa (1970) and applied to networks by
Paul Werbos (1974). David Rumelhart, Geoffrey Hinton and Ronald Williams (1986) demonstrated that backpropagation trains hidden
layers that learn useful features, which revived the field. George Cybenko (1989) and Kurt Hornik (1991) proved universal
approximation. Better initialisation (Xavier Glorot and Yoshua Bengio, 2010; Kaiming He et al., 2015), ReLU activations
(Vinod Nair and Hinton, 2010), and adaptive optimisers such as Adam (Diederik Kingma and Jimmy Ba, 2014) made deep networks
trainable.

## The idea

### Layers and activations

A multilayer perceptron is a chain of **layers**. Each layer takes a vector, multiplies it by a
matrix of **weights**, adds a vector of **biases**, and applies a simple nonlinear function to
each component. With the input $\mathbf a^{(0)} = \mathbf x$, layer $l$ computes

$$
\mathbf z^{(l)} = W^{(l)}\mathbf a^{(l-1)} + \mathbf b^{(l)}, \qquad \mathbf a^{(l)} = f(\mathbf z^{(l)}) .
$$

Each component of $\mathbf a^{(l)}$ is called a **unit**; it computes a weighted sum of the previous
layer's outputs and passes it through the **activation function** $f$. The layers between the
input and the output are **hidden** layers. Common activations are **ReLU**, $\max(0, z)$,
**tanh**, and the **sigmoid** $1/(1 + e^{-z})$.

The last layer depends on the task. For regression its output is left linear and compared with
the target by the squared error. For classification it is passed through a **softmax**,
$p_c = e^{z_c}/\sum_ke^{z_k}$, which turns the outputs into probabilities, and the loss is the
cross-entropy $-\log p_y$, exactly as in
[logistic regression](../machine_learning/logistic-regression.md#the-likelihood). A network with
no hidden layer *is* linear or logistic regression.

**Why the nonlinearity is essential.** A composition of linear maps is a linear map: without
$f$, any number of layers would collapse into a single matrix, and the network could represent
nothing more than a linear model. With the nonlinearity, each hidden unit contributes a function
that varies along one direction of the input space (a ridge), and the next layer adds these up.
Enough of them can build curved decision boundaries of any shape:

| | | |
|:---:|:---:|:---:|
| ![2 hidden units](../generated/learning/mlp_hidden_2.svg) | ![8 hidden units](../generated/learning/mlp_hidden_8.svg) | ![64 × 64 hidden units](../generated/learning/mlp_hidden_64_64.svg) |

Two hidden units give little more than a straight line. Eight already follow the moons. Two layers of 64 units, trained for 500
epochs, start to bend around individual points.

### Backpropagation

Training adjusts the weights to reduce the loss $L$ by gradient descent, which needs the
derivative of the loss with respect to every weight in every layer. Computing each derivative
separately would be hopelessly slow. **Backpropagation** computes all of them together, by
applying the chain rule in a well-chosen order.

**The quantity to propagate.** For each layer define

$$
\boldsymbol\delta^{(l)} = \frac{\partial L}{\partial\mathbf z^{(l)}},
$$

the sensitivity of the loss to the weighted sums of that layer. Everything else follows from
these vectors.

**Start at the output.** For both standard pairings of output and loss (a linear output with
squared error, and a softmax with cross-entropy), the derivative works out to the same simple
form, the prediction error:

$$
\boldsymbol\delta^{(L)} = \hat{\mathbf y} - \mathbf y .
$$

**Step backwards.** Suppose $\boldsymbol\delta^{(l)}$ is known. Layer $l - 1$ influences the loss only
through layer $l$, by two links: $\mathbf z^{(l)} = W^{(l)}\mathbf a^{(l-1)} + \mathbf b^{(l)}$, whose
derivative with respect to $\mathbf a^{(l-1)}$ is $W^{(l)}$, and
$\mathbf a^{(l-1)} = f(\mathbf z^{(l-1)})$, whose derivative is $f'(\mathbf z^{(l-1)})$ component by
component. The chain rule multiplies them:

$$
\boldsymbol\delta^{(l-1)} = \big(W^{(l)\mathsf T}\boldsymbol\delta^{(l)}\big) \odot f'(\mathbf z^{(l-1)}),
$$

where $\odot$ is the componentwise product. The transpose appears because the sensitivities
travel in the opposite direction to the signals: each unit collects the sensitivities of all the
units it feeds, weighted by the connections.

**Read off the gradients.** Since $\mathbf z^{(l)}$ depends on the weights of its own layer through
$W^{(l)}\mathbf a^{(l-1)} + \mathbf b^{(l)}$,

$$
\frac{\partial L}{\partial W^{(l)}} = \boldsymbol\delta^{(l)}\mathbf a^{(l-1)\mathsf T}, \qquad \frac{\partial L}{\partial\mathbf b^{(l)}} = \boldsymbol\delta^{(l)} .
$$

The gradient for a weight is the sensitivity of the unit it feeds times the activation of the
unit it comes from.

So training alternates a **forward pass**, which computes and stores the activations layer by
layer, with a **backward pass**, which computes the $\boldsymbol\delta$'s from the last layer to the
first. The backward pass costs a small multiple of the forward pass, whatever the number of
weights. This is the point of reverse-mode differentiation, and the reason networks with millions
of parameters can be trained. The derivatives of the activations are written in terms of the
activation values themselves ($f' = 1 - a^2$ for tanh, $a(1 - a)$ for the sigmoid), so the stored
activations are all the backward pass needs.

### Adam and mini-batches

**Mini-batches.** The loss is an average over all the training points, and so is its gradient.
Computing the full gradient for every step is wasteful when a rough estimate would do. Each step
therefore uses the gradient of a small random subset of the data, a **mini-batch** (32 rows by
default). The data are shuffled and divided into mini-batches, and one full pass through them is
an **epoch**. The gradient is noisy but cheap, and many noisy steps make faster progress than a
few exact ones. This is **stochastic gradient descent**.

**Adam.** Plain gradient descent uses the same step size for every weight, and
[choosing it is delicate](../optimization/gradient-methods.md#gradient-descent). **Adam**
adapts the step for each weight separately. It keeps two running averages for each weight: one
of the gradient, $\mathbf m$, and one of its square, $\mathbf v$. They are exponentially decaying
averages (with rates 0.9 and 0.999), which forget old gradients gradually. Both start at zero
and so are biased towards zero during the first steps; a correction factor removes the bias,
giving $\hat m$ and $\hat v$. Each weight is then moved by

$$
-\eta\,\frac{\hat m}{\sqrt{\hat v} + \epsilon} .
$$

The numerator is a smoothed gradient, which damps the noise of the mini-batches. The denominator
is the typical recent size of that weight's gradient, so the ratio is of order 1 for every
weight, and each moves by about $\eta$ per step regardless of the scale of its gradient. This
makes the **learning rate** $\eta$ far less sensitive than in plain gradient descent, though it
still matters:

![Training loss for several learning rates](../generated/learning/mlp_learning_rate.svg)

<p class="caption" id="fig-13-1"><strong>Figure 13.1.</strong> Training loss for several learning rates.</p>

At $\eta = 0.0003$ training is slow; at $0.003$ and $0.03$ it converges; at $0.3$ the steps are too large and the loss bounces around a value several times higher.

### Initialisation and scaling

Weights start random and small, so that units differ and signals neither vanish nor explode through the layers: Glorot's
variance $2/(n_{\text{in}} + n_{\text{out}})$ for tanh and sigmoid, He's $2/n_{\text{in}}$ for ReLU. The library standardises the inputs, and for regression the targets, before training,
because gradient-based training is sensitive to scales.

### Activations and smoothness

![MLP regression of a sine](../generated/learning/mlp_sine.svg)

<p class="caption" id="fig-13-2"><strong>Figure 13.2.</strong> MLP regression of a sine.</p>

A ReLU network is piecewise linear, with kinks wherever a unit switches on. A tanh network is smooth. Both fit the sine
where there are data. Beyond the data, on the right, both continue along straight lines, since every unit is saturated or linear there.
What a network does outside the data is therefore decided by its activation function, not by the data: ReLU
units continue linearly and saturating units flatten out, and neither continues the sine. On the two
classification benchmarks:

<p class="caption" id="tab-13-2"><strong>Table 13.2.</strong> Columns: activation (32 × 32, 200 epochs), moons: test accuracy, final training loss, training time, and more.</p>

{{#include ../generated/learning/mlp_activations.md}}

The sigmoid converges slowest per epoch: its final training loss is the highest in both columns (tanh happens to take the
most wall-clock time, because evaluating it costs more than a ReLU comparison). Its derivative is at most 1/4, so gradients shrink at every layer, the **vanishing gradient** problem that ReLU avoids.

### Overfitting and weight decay

A network with thousands of weights can memorise 120 points. **Weight decay** adds $\frac\lambda2\sum w^2$ to the loss, shrinking the weights towards
zero as ridge regression does:

<p class="caption" id="tab-13-3"><strong>Table 13.3.</strong> Columns: weight decay λ (64 × 64, 120 training points), training accuracy, test accuracy.</p>

{{#include ../generated/learning/mlp_weight_decay.md}}

Without decay the network reaches 100% training accuracy and 91% on test data. A little decay trades training accuracy for
test accuracy; too much flattens the network into a constant. Early stopping, dropout and more data are the other remedies.

## Algorithm

**Training.**

1. **Prepare.** Standardise the inputs, and for regression the targets. Initialise the weights
   at random (Glorot's variance for tanh and sigmoid, He's for ReLU) and the biases to zero.
2. **For each epoch**: shuffle the rows of the training data, and divide them into
   mini-batches. For each mini-batch:
   - **forward pass**: compute and store $\mathbf z^{(l)}$ and $\mathbf a^{(l)}$ for every layer, for
     all the rows of the batch at once (as matrices with one row per sample);
   - **loss**: the mean squared error or cross-entropy over the batch;
   - **backward pass**: compute $\boldsymbol\delta^{(L)}$ from the output and the targets, then
     $\boldsymbol\delta^{(l)}$ for each earlier layer, and from them the gradients of the weights and
     biases, averaged over the batch; add $\lambda W$ to each weight gradient for weight decay;
   - **Adam step**: update the two running averages for every weight and bias, correct their
     bias, and move the parameter.
3. **Monitor.** Record the mean training loss of the epoch. If it is infinite or NaN (the sign
   of a learning rate that is far too large), stop with an error.

One epoch costs a small multiple of (number of samples) × (number of weights).

**Prediction.** Standardise the input with the stored means and standard deviations, run the
forward pass, and return the class of largest probability, or for regression the output
transformed back to the original units of the target.

## Implementation

```rust,ignore
{{#include ../../../crates/machine_learning/src/neural_network.rs:backprop}}
```

```rust,ignore
{{#include ../../../crates/machine_learning/src/neural_network.rs:adam}}
```

- The weight gradient is assigned into a row-major array. The Adam step reads gradients and weights as flat slices, so a
  column-major matrix product (which can occur with a single input feature) would be read in the wrong order.
- Predictions check the number of features, so a one-column input is not silently broadcast against two training features.
- Training is deterministic for a given `seed`, which sets both the initial weights and the shuffling.

## In practice

```rust,ignore
{{#include ../../../crates/machine_learning/examples/ml_neural_network.rs}}
```

```text
cross-entropy: epoch 1 0.407, epoch 50 0.059, epoch 200 0.043
test accuracy 0.964
P(class 1) at (0.5, 0.25): 0.159; at (-1, 0.5): 0.000
sine: MSE against sin x 0.00377
```

**Strengths**

- Universal approximators that learn their own features; one method for regression and multi-class classification.
- Training scales linearly with the data, by mini-batches.

**Weaknesses**

- Nonconvex training with many settings (architecture, learning rate, epochs, decay) that interact; results vary with the seed.
- On small tabular data, usually no better than kernel machines or boosted trees, and slower to train.
- No extrapolation, and probabilities that can be overconfident far from the data.

## Exercises

1. Show that a network with linear activations, of any depth, is a linear function of its input.
2. Derive $\boldsymbol\delta^{(L)} = \hat{\mathbf y} - \mathbf y$ for softmax with cross-entropy.
3. Construct a network with two hidden ReLU units that computes XOR exactly.
4. Show that Adam's bias correction makes $\hat m$ an unbiased estimate of the mean gradient when the gradients are stationary.
5. Train the same network with ten different seeds and report the spread of the test accuracy.

<details>
<summary>Hints and answers</summary>

1. A composition of affine maps is affine: $W_2(W_1\mathbf x + \mathbf b_1) + \mathbf b_2 = (W_2W_1)\mathbf x + (W_2\mathbf b_1 + \mathbf b_2)$.
2. For softmax outputs $\hat y_k = e^{z_k}/\sum e^{z_j}$ and cross-entropy $-\sum y_k\ln\hat y_k$, $\partial L/\partial z_k = \hat y_k - y_k$ since $\sum y_k = 1$.
3. Take $h_1 = \operatorname{ReLU}(x_1 + x_2)$, $h_2 = \operatorname{ReLU}(x_1 + x_2 - 1)$ and output $h_1 - 2h_2$: the inputs $(0,0), (1,0), (0,1), (1,1)$ give $0, 1, 1, 0$.
4. With $m_t = (1 - \beta_1)\sum_k\beta_1^{t-k}g_k$ and stationary $\mathbb E g_k = \mu$, $\mathbb Em_t = (1 - \beta_1^t)\mu$; dividing by $1 - \beta_1^t$ removes the bias.
5. Report the mean and the standard deviation (or range) of the test accuracy over the ten seeds; the spread shows how much of any difference between two configurations is just initialisation noise.

</details>

## Further reading

- D. E. Rumelhart, G. E. Hinton and R. J. Williams, "Learning representations by back-propagating errors", *Nature* 323 (1986) 533–536.
- D. P. Kingma and J. Ba, "Adam: a method for stochastic optimization", arXiv:1412.6980, 2014.
- I. Goodfellow, Y. Bengio and A. Courville, *Deep Learning*, MIT Press, 2016, chapters 6–8.
- X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks", *Proceedings of AISTATS* (2010) 249–256.
