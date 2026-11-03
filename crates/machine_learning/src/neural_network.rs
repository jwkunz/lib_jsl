//! Multilayer perceptrons trained by backpropagation.
//!
//! A fully connected feed-forward network `x → f(W₁x + b₁) → … → W_L h + b_L`, with a linear
//! output and squared-error loss for regression or a softmax output and cross-entropy loss for
//! classification. Gradients come from backpropagation; weights are updated by mini-batch Adam
//! (Kingma and Ba) with L2 weight decay. Inputs (and regression targets) are standardised
//! internally, and weights start from He (ReLU) or Glorot (tanh, sigmoid) initialisation.

use crate::common::{Classifier, MlRng, Regressor, argmax, check_xy, class_count};
use lib_jsl_core::{ErrorsJSL, R1D, R2D, VR1D, VR2D};
use ndarray::Axis;

/// The hidden-layer nonlinearity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Relu,
    Tanh,
    Sigmoid,
}

impl Activation {
    fn apply(self, z: f64) -> f64 {
        match self {
            Activation::Relu => z.max(0.0),
            Activation::Tanh => z.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-z).exp()),
        }
    }

    /// The derivative expressed through the activation value `a = f(z)`.
    fn derivative(self, a: f64) -> f64 {
        match self {
            Activation::Relu => f64::from(u8::from(a > 0.0)),
            Activation::Tanh => 1.0 - a * a,
            Activation::Sigmoid => a * (1.0 - a),
        }
    }
}

/// Network shape and training settings.
#[derive(Debug, Clone)]
pub struct MlpParameters {
    /// Sizes of the hidden layers.
    pub hidden: Vec<usize>,
    pub activation: Activation,
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
    /// L2 penalty `λ/2 Σ w²` on the weights (not the biases).
    pub weight_decay: f64,
    pub seed: u64,
}

impl Default for MlpParameters {
    fn default() -> Self {
        MlpParameters { hidden: vec![32, 32], activation: Activation::Tanh, learning_rate: 0.01, epochs: 300, batch_size: 32, weight_decay: 1e-4, seed: 0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Output {
    Linear,
    Softmax,
}

/// The network: weights `w[l]` of shape `(outputs, inputs)` and biases `b[l]`.
#[derive(Debug, Clone)]
struct Network {
    w: Vec<R2D>,
    b: Vec<R1D>,
    activation: Activation,
    output: Output,
}

impl Network {
    fn new(sizes: &[usize], activation: Activation, output: Output, rng: &mut MlRng) -> Self {
        let mut w = Vec::new();
        let mut b = Vec::new();
        for pair in sizes.windows(2) {
            let (fan_in, fan_out) = (pair[0], pair[1]);
            let scale = match activation {
                Activation::Relu => (2.0 / fan_in as f64).sqrt(),
                _ => (2.0 / (fan_in + fan_out) as f64).sqrt(),
            };
            w.push(R2D::from_shape_fn((fan_out, fan_in), |_| rng.gaussian(0.0, scale)));
            b.push(R1D::zeros(fan_out));
        }
        Network { w, b, activation, output }
    }

    /// Activations of every layer for a batch (`activations[0]` is the input).
    fn forward(&self, x: &R2D) -> Vec<R2D> {
        let mut activations = vec![x.clone()];
        let last = self.w.len() - 1;
        for (l, (w, b)) in self.w.iter().zip(&self.b).enumerate() {
            let mut z = activations[l].dot(&w.t()) + b;
            if l < last {
                z.mapv_inplace(|v| self.activation.apply(v));
            } else if self.output == Output::Softmax {
                for mut row in z.rows_mut() {
                    let m = row.fold(f64::NEG_INFINITY, |a, &v| a.max(v));
                    row.mapv_inplace(|v| (v - m).exp());
                    let s = row.sum();
                    row.mapv_inplace(|v| v / s);
                }
            }
            activations.push(z);
        }
        activations
    }

    /// The mean loss over the batch (plus the weight penalty) and its gradients.
    fn loss_and_gradient(&self, x: &R2D, targets: &R2D, weight_decay: f64) -> (f64, Vec<R2D>, Vec<R1D>) {
        let a = self.forward(x);
        let batch = x.nrows() as f64;
        let output = a.last().expect("at least one layer");
        let data_loss = match self.output {
            Output::Linear => 0.5 * (output - targets).mapv(|v| v * v).sum() / batch,
            Output::Softmax => -(output.mapv(|p| p.max(1e-300).ln()) * targets).sum() / batch,
        };
        let penalty = 0.5 * weight_decay * self.w.iter().map(|w| w.mapv(|v| v * v).sum()).sum::<f64>();
        // For both pairings (linear + squared error, softmax + cross-entropy) δ_L = (ŷ − y)/B.
        let mut delta = (output - targets) / batch;
        let layers = self.w.len();
        let mut gw = vec![R2D::zeros((0, 0)); layers];
        let mut gb = vec![R1D::zeros(0); layers];
        for l in (0..layers).rev() {
            // Assigned into a row-major array: the product can come back column-major (for example
            // when an activation matrix has a single column), and the Adam step reads the gradient
            // as a flat slice in the same order as the weights.
            let mut g = R2D::zeros(self.w[l].dim());
            g.assign(&delta.t().dot(&a[l]));
            g.scaled_add(weight_decay, &self.w[l]);
            gw[l] = g;
            gb[l] = delta.sum_axis(Axis(0));
            if l > 0 {
                let back = delta.dot(&self.w[l]);
                delta = back * a[l].mapv(|v| self.activation.derivative(v));
            }
        }
        (data_loss + penalty, gw, gb)
    }
}

/// Adam optimiser state for one parameter tensor.
struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
}

impl Adam {
    fn step(&mut self, parameters: &mut [f64], gradient: &[f64], rate: f64, t: i32) {
        let (b1, b2, eps) = (0.9f64, 0.999f64, 1e-8);
        let (c1, c2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
        for i in 0..parameters.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gradient[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * gradient[i] * gradient[i];
            parameters[i] -= rate * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + eps);
        }
    }
}

/// Feature standardisation learned from the training data.
#[derive(Debug, Clone)]
struct Standardizer {
    mean: R1D,
    scale: R1D,
}

impl Standardizer {
    fn fit(x: VR2D) -> Self {
        let mean = x.mean_axis(Axis(0)).expect("non-empty");
        let scale = x.std_axis(Axis(0), 0.0).mapv(|s| if s > 1e-12 { s } else { 1.0 });
        Standardizer { mean, scale }
    }

    fn apply(&self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        if x.ncols() != self.mean.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), self.mean.len())));
        }
        Ok((&x - &self.mean) / &self.scale)
    }
}

/// Trains a network by mini-batch Adam; returns it with the per-epoch mean training loss.
fn train(parameters: &MlpParameters, x: &R2D, targets: &R2D, output: Output) -> Result<(Network, Vec<f64>), ErrorsJSL> {
    if parameters.hidden.contains(&0) || parameters.batch_size == 0 || parameters.epochs == 0 || parameters.learning_rate.is_nan() || parameters.learning_rate <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("Need non-zero layer sizes, batch size and epochs, and a positive learning rate."));
    }
    let mut rng = MlRng::new(parameters.seed);
    let sizes: Vec<usize> = std::iter::once(x.ncols()).chain(parameters.hidden.iter().copied()).chain(std::iter::once(targets.ncols())).collect();
    let mut net = Network::new(&sizes, parameters.activation, output, &mut rng);
    let mut adam_w: Vec<Adam> = net.w.iter().map(|w| Adam { m: vec![0.0; w.len()], v: vec![0.0; w.len()] }).collect();
    let mut adam_b: Vec<Adam> = net.b.iter().map(|b| Adam { m: vec![0.0; b.len()], v: vec![0.0; b.len()] }).collect();
    let n = x.nrows();
    let mut order: Vec<usize> = (0..n).collect();
    let mut history = Vec::with_capacity(parameters.epochs);
    let mut t = 0;
    for _ in 0..parameters.epochs {
        rng.shuffle(&mut order);
        let mut epoch_loss = 0.0;
        for chunk in order.chunks(parameters.batch_size) {
            t += 1;
            let (xb, yb) = (x.select(Axis(0), chunk), targets.select(Axis(0), chunk));
            let (loss, gw, gb) = net.loss_and_gradient(&xb, &yb, parameters.weight_decay);
            epoch_loss += loss * chunk.len() as f64;
            for l in 0..net.w.len() {
                adam_w[l].step(net.w[l].as_slice_mut().expect("contiguous"), gw[l].as_slice().expect("contiguous"), parameters.learning_rate, t);
                adam_b[l].step(net.b[l].as_slice_mut().expect("contiguous"), gb[l].as_slice().expect("contiguous"), parameters.learning_rate, t);
            }
        }
        let mean_loss = epoch_loss / n as f64;
        if !mean_loss.is_finite() {
            return Err(ErrorsJSL::RuntimeError("Training diverged; lower the learning rate."));
        }
        history.push(mean_loss);
    }
    Ok((net, history))
}

/// A multilayer perceptron for regression (linear output, squared-error loss).
#[derive(Debug, Clone)]
pub struct MlpRegressor {
    pub parameters: MlpParameters,
    network: Option<Network>,
    inputs: Option<Standardizer>,
    target_mean: f64,
    target_scale: f64,
    /// Mean training loss per epoch (on standardised targets).
    pub training_loss: Vec<f64>,
}

impl MlpRegressor {
    pub fn new(parameters: MlpParameters) -> Self {
        MlpRegressor { parameters, network: None, inputs: None, target_mean: 0.0, target_scale: 1.0, training_loss: Vec::new() }
    }
}

impl Regressor for MlpRegressor {
    fn fit(&mut self, x: VR2D, y: VR1D) -> Result<(), ErrorsJSL> {
        check_xy(x, y.len())?;
        let standardizer = Standardizer::fit(x);
        self.target_mean = y.mean().expect("non-empty");
        self.target_scale = y.std(0.0).max(1e-12);
        let targets = y.mapv(|v| (v - self.target_mean) / self.target_scale).insert_axis(Axis(1));
        let (net, history) = train(&self.parameters, &standardizer.apply(x)?, &targets, Output::Linear)?;
        self.network = Some(net);
        self.inputs = Some(standardizer);
        self.training_loss = history;
        Ok(())
    }

    fn predict(&self, x: VR2D) -> Result<R1D, ErrorsJSL> {
        let (net, standardizer) = self.network.as_ref().zip(self.inputs.as_ref()).ok_or(ErrorsJSL::RuntimeError("The model has not been fitted."))?;
        let output = net.forward(&standardizer.apply(x)?).pop().expect("output layer");
        Ok(output.column(0).mapv(|v| v * self.target_scale + self.target_mean))
    }
}

/// A multilayer perceptron for classification (softmax output, cross-entropy loss).
#[derive(Debug, Clone)]
pub struct MlpClassifier {
    pub parameters: MlpParameters,
    network: Option<Network>,
    inputs: Option<Standardizer>,
    /// Mean training cross-entropy per epoch.
    pub training_loss: Vec<f64>,
}

impl MlpClassifier {
    pub fn new(parameters: MlpParameters) -> Self {
        MlpClassifier { parameters, network: None, inputs: None, training_loss: Vec::new() }
    }

    /// Class probabilities, one row per sample.
    pub fn predict_proba(&self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        let (net, standardizer) = self.network.as_ref().zip(self.inputs.as_ref()).ok_or(ErrorsJSL::RuntimeError("The model has not been fitted."))?;
        Ok(net.forward(&standardizer.apply(x)?).pop().expect("output layer"))
    }
}

impl Classifier for MlpClassifier {
    fn fit(&mut self, x: VR2D, labels: &[usize]) -> Result<(), ErrorsJSL> {
        check_xy(x, labels.len())?;
        let k = class_count(labels).max(2);
        let targets = R2D::from_shape_fn((labels.len(), k), |(i, c)| f64::from(u8::from(labels[i] == c)));
        let standardizer = Standardizer::fit(x);
        let (net, history) = train(&self.parameters, &standardizer.apply(x)?, &targets, Output::Softmax)?;
        self.network = Some(net);
        self.inputs = Some(standardizer);
        self.training_loss = history;
        Ok(())
    }

    fn predict(&self, x: VR2D) -> Result<Vec<usize>, ErrorsJSL> {
        let p = self.predict_proba(x)?;
        Ok(p.rows().into_iter().map(|r| argmax(&r.to_vec()).unwrap_or(0)).collect())
    }
}

/// The one-hot encoding of labels (exposed for building custom targets).
pub fn one_hot(labels: &[usize], classes: usize) -> R2D {
    R2D::from_shape_fn((labels.len(), classes), |(i, c)| f64::from(u8::from(labels[i] == c)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{make_moons, make_sine};
    use crate::metrics::{accuracy, mean_squared_error};
    use crate::preprocessing::train_test_split;

    #[test]
    fn backpropagation_matches_finite_differences() {
        let mut rng = MlRng::new(3);
        for (activation, output) in [(Activation::Tanh, Output::Linear), (Activation::Sigmoid, Output::Softmax), (Activation::Relu, Output::Softmax)] {
            let net = Network::new(&[3, 5, 4, 2], activation, output, &mut rng);
            let x = R2D::from_shape_fn((6, 3), |_| rng.gaussian(0.0, 1.0));
            let y = if output == Output::Linear { R2D::from_shape_fn((6, 2), |_| rng.gaussian(0.0, 1.0)) } else { one_hot(&[0, 1, 1, 0, 1, 0], 2) };
            let (_, gw, gb) = net.loss_and_gradient(&x, &y, 0.01);
            let h = 1e-6;
            for l in 0..3 {
                for idx in [(0, 0), (1, 2)] {
                    let mut plus = net.clone();
                    plus.w[l][idx] += h;
                    let mut minus = net.clone();
                    minus.w[l][idx] -= h;
                    let numeric = (plus.loss_and_gradient(&x, &y, 0.01).0 - minus.loss_and_gradient(&x, &y, 0.01).0) / (2.0 * h);
                    assert!((numeric - gw[l][idx]).abs() < 1e-6, "{activation:?} layer {l} {idx:?}: {numeric} vs {}", gw[l][idx]);
                }
                let mut plus = net.clone();
                plus.b[l][0] += h;
                let mut minus = net.clone();
                minus.b[l][0] -= h;
                let numeric = (plus.loss_and_gradient(&x, &y, 0.01).0 - minus.loss_and_gradient(&x, &y, 0.01).0) / (2.0 * h);
                assert!((numeric - gb[l][0]).abs() < 1e-6, "bias layer {l}");
            }
        }
    }

    #[test]
    fn learns_xor_and_two_moons() {
        let x = R2D::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
        let labels = [0, 1, 1, 0];
        let mut xor = MlpClassifier::new(MlpParameters { hidden: vec![8], epochs: 500, batch_size: 4, learning_rate: 0.05, ..Default::default() });
        xor.fit(x.view(), &labels).unwrap();
        assert_eq!(xor.predict(x.view()).unwrap(), labels);
        let (xm, lm) = make_moons(400, 0.15, 7);
        let (train_idx, test_idx) = train_test_split(400, 0.25, 8).unwrap();
        let lt: Vec<usize> = train_idx.iter().map(|&i| lm[i]).collect();
        let ls: Vec<usize> = test_idx.iter().map(|&i| lm[i]).collect();
        let mut moons = MlpClassifier::new(MlpParameters { epochs: 150, ..Default::default() });
        moons.fit(xm.select(Axis(0), &train_idx).view(), &lt).unwrap();
        let accuracy_value = accuracy(&ls, &moons.predict(xm.select(Axis(0), &test_idx).view()).unwrap()).unwrap();
        assert!(accuracy_value > 0.95, "{accuracy_value}");
        assert!(moons.training_loss.last().unwrap() < &(0.3 * moons.training_loss[0]));
    }

    #[test]
    fn regresses_a_sine_and_validates() {
        let (x, y) = make_sine(300, 0.05, 9);
        let mut mlp = MlpRegressor::new(MlpParameters { hidden: vec![32], epochs: 400, activation: Activation::Tanh, ..Default::default() });
        mlp.fit(x.view(), y.view()).unwrap();
        let grid = R2D::from_shape_fn((50, 1), |(i, _)| 0.2 + 5.8 * i as f64 / 49.0);
        let truth = grid.column(0).mapv(f64::sin);
        let mse = mean_squared_error(truth.view(), mlp.predict(grid.view()).unwrap().view()).unwrap();
        assert!(mse < 0.01, "{mse}");
        // One input feature through two hidden layers (a product that comes out column-major).
        let mut deep = MlpRegressor::new(MlpParameters { hidden: vec![16, 16], epochs: 300, batch_size: 16, ..Default::default() });
        deep.fit(x.view(), y.view()).unwrap();
        let deep_mse = mean_squared_error(truth.view(), deep.predict(grid.view()).unwrap().view()).unwrap();
        assert!(deep_mse < 0.02, "{deep_mse}");
        let mut bad = MlpRegressor::new(MlpParameters { hidden: vec![0], ..Default::default() });
        assert!(bad.fit(x.view(), y.view()).is_err());
        assert!(MlpClassifier::new(MlpParameters::default()).predict(x.view()).is_err());
        // A different number of features is an error, not a silent broadcast.
        assert!(mlp.predict(R2D::zeros((3, 2)).view()).is_err());
        let (xm, lm) = make_moons(20, 0.1, 1);
        let mut c = MlpClassifier::new(MlpParameters { epochs: 2, ..Default::default() });
        c.fit(xm.view(), &lm).unwrap();
        assert!(c.predict(R2D::zeros((2, 1)).view()).is_err());
    }
}
