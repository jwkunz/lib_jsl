//! Logistic (softmax) regression.

use crate::common::{Classifier, argmax, check_features, check_xy, class_count};
use lib_jsl_core::{ErrorsJSL, R2D, VR2D};

/// Multinomial logistic regression: class probabilities `softmax(W x + b)`, trained by minimising
/// the mean negative log-likelihood plus an L2 penalty `λ/2 ‖W‖²` (the intercepts are not
/// penalised) with L-BFGS. With two classes this is ordinary logistic regression. Standardise the
/// features first for fast convergence.
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    l2: f64,
    max_iterations: usize,
    tolerance: f64,
    /// `(classes, features + 1)`; the last column holds the intercepts.
    weights: R2D,
    iterations: usize,
}

impl LogisticRegression {
    /// `l2` is the penalty strength `λ >= 0`.
    pub fn new(l2: f64) -> Self {
        LogisticRegression { l2, max_iterations: 500, tolerance: 1e-7, weights: R2D::zeros((0, 0)), iterations: 0 }
    }

    pub fn with_limits(mut self, max_iterations: usize, tolerance: f64) -> Self {
        self.max_iterations = max_iterations;
        self.tolerance = tolerance;
        self
    }

    /// The learned weights, `(classes, features)` without the intercept column.
    pub fn coefficients(&self) -> R2D {
        self.weights.slice(ndarray::s![.., ..self.weights.ncols().saturating_sub(1)]).to_owned()
    }

    pub fn intercepts(&self) -> Vec<f64> {
        self.weights.column(self.weights.ncols() - 1).to_vec()
    }

    /// L-BFGS iterations used by the last fit.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Class probabilities for each row, `(rows, classes)`.
    pub fn predict_proba(&self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        if self.weights.nrows() == 0 {
            return Err(ErrorsJSL::Misconfiguration("The model has not been fitted."));
        }
        check_features(x)?;
        if x.ncols() + 1 != self.weights.ncols() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), self.weights.ncols() - 1)));
        }
        Ok(softmax_rows(&self.weights, x))
    }
}

fn softmax_rows(w: &R2D, x: VR2D) -> R2D {
    let p = x.ncols();
    let mut out = R2D::from_shape_fn((x.nrows(), w.nrows()), |(i, c)| w[[c, p]] + (0..p).map(|j| w[[c, j]] * x[[i, j]]).sum::<f64>());
    for mut row in out.rows_mut() {
        let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        row.mapv_inplace(|v| (v - m).exp());
        let s = row.sum();
        row.mapv_inplace(|v| v / s);
    }
    out
}

/// Penalised mean negative log-likelihood and its gradient with respect to the flattened weights.
fn objective(theta: &[f64], x: VR2D, labels: &[usize], classes: usize, l2: f64) -> (f64, Vec<f64>) {
    let (n, p) = x.dim();
    let w = R2D::from_shape_vec((classes, p + 1), theta.to_vec()).expect("shape");
    let proba = softmax_rows(&w, x);
    let mut loss = 0.0;
    let mut grad = R2D::zeros((classes, p + 1));
    for i in 0..n {
        loss -= proba[[i, labels[i]]].max(1e-300).ln();
        for c in 0..classes {
            let err = proba[[i, c]] - f64::from(c == labels[i]);
            for j in 0..p {
                grad[[c, j]] += err * x[[i, j]];
            }
            grad[[c, p]] += err;
        }
    }
    loss /= n as f64;
    grad /= n as f64;
    for c in 0..classes {
        for j in 0..p {
            loss += 0.5 * l2 * w[[c, j]] * w[[c, j]];
            grad[[c, j]] += l2 * w[[c, j]];
        }
    }
    (loss, grad.into_raw_vec_and_offset().0)
}

/// Minimises `f` from `x0` with L-BFGS (memory 10) and a backtracking Armijo line search.
/// Returns the solution and the iteration count.
pub(crate) fn lbfgs(mut f: impl FnMut(&[f64]) -> (f64, Vec<f64>), x0: Vec<f64>, max_iterations: usize, tolerance: f64) -> (Vec<f64>, usize) {
    const MEMORY: usize = 10;
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let mut x = x0;
    let (mut fx, mut g) = f(&x);
    let mut history: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::new(); // (s, y, 1/(y·s))
    for iter in 1..=max_iterations {
        if g.iter().fold(0.0f64, |m, v| m.max(v.abs())) < tolerance {
            return (x, iter - 1);
        }
        // Two-loop recursion for the search direction -H g.
        let mut q = g.clone();
        let mut alphas = Vec::with_capacity(history.len());
        for (s, y, rho) in history.iter().rev() {
            let a = rho * dot(s, &q);
            q.iter_mut().zip(y).for_each(|(qi, yi)| *qi -= a * yi);
            alphas.push(a);
        }
        if let Some((s, y, _)) = history.last() {
            let gamma = dot(s, y) / dot(y, y);
            q.iter_mut().for_each(|v| *v *= gamma);
        }
        for ((s, y, rho), a) in history.iter().zip(alphas.iter().rev()) {
            let b = rho * dot(y, &q);
            q.iter_mut().zip(s).for_each(|(qi, si)| *qi += (a - b) * si);
        }
        let mut dir: Vec<f64> = q.iter().map(|v| -v).collect();
        let mut slope = dot(&dir, &g);
        if slope >= 0.0 {
            dir = g.iter().map(|v| -v).collect(); // fall back to steepest descent
            slope = -dot(&g, &g);
            history.clear();
        }
        let mut step = if history.is_empty() { 1.0 / dot(&g, &g).sqrt().max(1.0) } else { 1.0 };
        let mut accepted = None;
        for _ in 0..40 {
            let trial: Vec<f64> = x.iter().zip(&dir).map(|(a, d)| a + step * d).collect();
            let (ft, gt) = f(&trial);
            if ft <= fx + 1e-4 * step * slope {
                accepted = Some((trial, ft, gt));
                break;
            }
            step *= 0.5;
        }
        let Some((nx, nf, ng)) = accepted else { return (x, iter) };
        let s: Vec<f64> = nx.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = ng.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 {
            if history.len() == MEMORY {
                history.remove(0);
            }
            history.push((s, y, 1.0 / sy));
        }
        let converged = (fx - nf).abs() < 1e-15 * fx.abs().max(1.0);
        (x, fx, g) = (nx, nf, ng);
        if converged {
            return (x, iter);
        }
    }
    (x, max_iterations)
}

impl Classifier for LogisticRegression {
    fn fit(&mut self, x: VR2D, labels: &[usize]) -> Result<(), ErrorsJSL> {
        check_xy(x, labels.len())?;
        let classes = class_count(labels);
        if classes < 2 {
            return Err(ErrorsJSL::InvalidInputRange("At least two classes are required."));
        }
        if self.l2.is_nan() || self.l2 < 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("The penalty must be non-negative."));
        }
        let p = x.ncols();
        let (theta, iterations) = lbfgs(|t| objective(t, x, labels, classes, self.l2), vec![0.0; classes * (p + 1)], self.max_iterations, self.tolerance);
        self.weights = R2D::from_shape_vec((classes, p + 1), theta).expect("shape");
        self.iterations = iterations;
        Ok(())
    }

    fn predict(&self, x: VR2D) -> Result<Vec<usize>, ErrorsJSL> {
        Ok(self.predict_proba(x)?.rows().into_iter().map(|r| argmax(&r.to_vec()).unwrap_or(0)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Transformer;
    use crate::datasets::{make_blobs, make_classification, make_moons};
    use crate::metrics::accuracy;
    use crate::preprocessing::StandardScaler;
    use ndarray::arr2;

    #[test]
    fn separable_blobs_are_classified_perfectly_with_calibrated_probabilities() {
        let (x, l) = make_blobs(40, &[vec![0.0, 0.0], vec![5.0, 5.0], vec![-5.0, 5.0]], 0.8, 3);
        let mut m = LogisticRegression::new(1e-3);
        m.fit(x.view(), &l).unwrap();
        assert_eq!(accuracy(&l, &m.predict(x.view()).unwrap()).unwrap(), 1.0);
        let p = m.predict_proba(x.view()).unwrap();
        assert!(p.rows().into_iter().all(|r| (r.sum() - 1.0).abs() < 1e-12));
        assert!(p[[0, 0]] > 0.95);
    }

    #[test]
    fn linear_model_on_overlapping_classes_and_nonlinear_failure() {
        let (x, l) = make_classification(150, 5);
        let mut sc = StandardScaler::new();
        let z = sc.fit_transform(x.view()).unwrap();
        let mut m = LogisticRegression::new(1e-2);
        m.fit(z.view(), &l).unwrap();
        assert!(accuracy(&l, &m.predict(z.view()).unwrap()).unwrap() > 0.8);
        // Moons are not linearly separable: accuracy stays well below a perfect fit.
        let (mx, ml) = make_moons(150, 0.1, 2);
        let mut lin = LogisticRegression::new(1e-3);
        lin.fit(mx.view(), &ml).unwrap();
        let acc = accuracy(&ml, &lin.predict(mx.view()).unwrap()).unwrap();
        assert!((0.75..0.95).contains(&acc), "{acc}");
    }

    #[test]
    fn regularisation_shrinks_weights_and_binary_matches_known_solution() {
        let (x, l) = make_blobs(50, &[vec![-1.0, 0.0], vec![1.0, 0.0]], 1.0, 4);
        let norm = |lam: f64| {
            let mut m = LogisticRegression::new(lam);
            m.fit(x.view(), &l).unwrap();
            m.coefficients().iter().map(|v| v * v).sum::<f64>().sqrt()
        };
        assert!(norm(10.0) < norm(0.1) && norm(0.1) < norm(0.0001) * 1.0001);
        // Perfectly separable data with no penalty: the weights grow without bound, but the fit
        // still terminates (iteration cap) and classifies correctly.
        let x1 = arr2(&[[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]]);
        let mut m = LogisticRegression::new(0.0).with_limits(2000, 1e-9);
        m.fit(x1.view(), &[0, 0, 0, 1, 1, 1]).unwrap();
        assert_eq!(m.predict(x1.view()).unwrap(), vec![0, 0, 0, 1, 1, 1]);
        assert!(m.intercepts().iter().all(|b| b.abs() < 1e-3)); // symmetric data -> boundary at 0
    }

    #[test]
    fn errors() {
        let x = arr2(&[[0.0], [1.0]]);
        assert!(LogisticRegression::new(0.1).fit(x.view(), &[0, 0]).is_err()); // one class
        assert!(LogisticRegression::new(-1.0).fit(x.view(), &[0, 1]).is_err());
        assert!(LogisticRegression::new(0.1).predict(x.view()).is_err());
        let mut m = LogisticRegression::new(0.1);
        m.fit(x.view(), &[0, 1]).unwrap();
        assert!(m.predict(arr2(&[[1.0, 2.0]]).view()).is_err());
        assert_eq!(m.intercepts().len(), 2);
    }
}
