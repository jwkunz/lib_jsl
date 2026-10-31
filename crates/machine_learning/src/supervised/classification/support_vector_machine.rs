//! Support vector machine classification with kernels.

use crate::common::{Classifier, check_features, check_xy, class_count};
use crate::kernels::Kernel;
use lib_jsl_core::{ErrorsJSL, R2D, VR2D};

/// One trained binary machine: support vectors with signed dual coefficients `α_i y_i` and bias.
#[derive(Debug, Clone)]
struct BinaryMachine {
    class_pos: usize,
    class_neg: usize,
    support: R2D,
    coefficients: Vec<f64>,
    rho: f64,
}

/// A soft-margin kernel support vector machine. The binary problem
/// `min ½ αᵀQα - eᵀα, 0 <= α <= C, yᵀα = 0` is solved by sequential minimal optimisation with the
/// second-order working-set selection of LIBSVM. More than two classes are handled one-versus-one
/// (a machine per pair of classes, majority vote).
#[derive(Debug, Clone)]
pub struct SupportVectorMachine {
    kernel: Kernel,
    c: f64,
    tolerance: f64,
    max_iterations: usize,
    machines: Vec<BinaryMachine>,
    classes: usize,
    features: usize,
}

impl SupportVectorMachine {
    pub fn new(kernel: Kernel, c: f64) -> Self {
        SupportVectorMachine { kernel, c, tolerance: 1e-3, max_iterations: 200_000, machines: Vec::new(), classes: 0, features: 0 }
    }

    pub fn with_limits(mut self, max_iterations: usize, tolerance: f64) -> Self {
        self.max_iterations = max_iterations;
        self.tolerance = tolerance;
        self
    }

    /// Total support vectors across all binary machines.
    pub fn support_vector_count(&self) -> usize {
        self.machines.iter().map(|m| m.coefficients.len()).sum()
    }

    /// For a two-class model, the signed distance-like score `f(x)`: positive means the second
    /// class (label 1). Errors if the model has more than two classes.
    pub fn decision_function(&self, x: VR2D) -> Result<Vec<f64>, ErrorsJSL> {
        if self.machines.len() != 1 {
            return Err(ErrorsJSL::Misconfiguration("decision_function needs a fitted two-class model."));
        }
        self.check_input(x)?;
        let m = &self.machines[0];
        // The machine's positive class is the lower label, so negate to make label 1 positive.
        Ok(self.score(m, x).into_iter().map(|v| -v).collect())
    }

    fn check_input(&self, x: VR2D) -> Result<(), ErrorsJSL> {
        if self.machines.is_empty() {
            return Err(ErrorsJSL::Misconfiguration("The model has not been fitted."));
        }
        check_features(x)?;
        if x.ncols() != self.features {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), self.features)));
        }
        Ok(())
    }

    fn score(&self, m: &BinaryMachine, x: VR2D) -> Vec<f64> {
        if m.coefficients.is_empty() {
            return vec![-m.rho; x.nrows()];
        }
        self.kernel.cross(x, m.support.view()).dot(&ndarray::arr1(&m.coefficients)).iter().map(|v| v - m.rho).collect()
    }

    #[allow(clippy::needless_range_loop)]
    /// Trains the machine separating `pos` (+1) from `neg` (-1).
    fn train_binary(&self, x: VR2D, labels: &[usize], pos: usize, neg: usize) -> BinaryMachine {
        let idx: Vec<usize> = (0..x.nrows()).filter(|&i| labels[i] == pos || labels[i] == neg).collect();
        let sub = R2D::from_shape_fn((idx.len(), x.ncols()), |(a, j)| x[[idx[a], j]]);
        let y: Vec<f64> = idx.iter().map(|&i| if labels[i] == pos { 1.0 } else { -1.0 }).collect();
        let k = self.kernel.gram(sub.view());
        let l = idx.len();
        let (c, eps) = (self.c, self.tolerance);
        let mut alpha = vec![0.0; l];
        let mut grad = vec![-1.0; l]; // G = Qα - e
        let q = |i: usize, j: usize| y[i] * y[j] * k[[i, j]];
        let in_up = |t: usize, a: &[f64]| (y[t] > 0.0 && a[t] < c) || (y[t] < 0.0 && a[t] > 0.0);
        let in_low = |t: usize, a: &[f64]| (y[t] > 0.0 && a[t] > 0.0) || (y[t] < 0.0 && a[t] < c);
        for _ in 0..self.max_iterations {
            // Select i: the maximal violator in I_up of -y G.
            let mut gmax = f64::NEG_INFINITY;
            let mut i = None;
            for t in 0..l {
                if in_up(t, &alpha) && -y[t] * grad[t] > gmax {
                    gmax = -y[t] * grad[t];
                    i = Some(t);
                }
            }
            let Some(i) = i else { break };
            // Select j by the second-order rule among I_low violators.
            let (mut gmin_obj, mut j, mut gmax2) = (f64::INFINITY, None, f64::NEG_INFINITY);
            for t in 0..l {
                if in_low(t, &alpha) {
                    let g = y[t] * grad[t];
                    gmax2 = gmax2.max(g);
                    let b = gmax + g;
                    if b > 0.0 {
                        let a = (k[[i, i]] + k[[t, t]] - 2.0 * k[[i, t]]).max(1e-12);
                        let obj = -(b * b) / a;
                        if obj < gmin_obj {
                            gmin_obj = obj;
                            j = Some(t);
                        }
                    }
                }
            }
            let Some(j) = j else { break };
            if gmax + gmax2 < eps {
                break;
            }
            // Update α_i, α_j keeping yᵀα constant and 0 <= α <= C.
            let (old_i, old_j) = (alpha[i], alpha[j]);
            let quad = (k[[i, i]] + k[[j, j]] - 2.0 * k[[i, j]]).max(1e-12);
            if y[i] != y[j] {
                let delta = (-grad[i] - grad[j]) / quad;
                let diff = alpha[i] - alpha[j];
                alpha[i] += delta;
                alpha[j] += delta;
                if diff > 0.0 {
                    if alpha[j] < 0.0 { alpha[j] = 0.0; alpha[i] = diff; }
                } else if alpha[i] < 0.0 { alpha[i] = 0.0; alpha[j] = -diff; }
                if diff > 0.0 {
                    if alpha[i] > c { alpha[i] = c; alpha[j] = c - diff; }
                } else if alpha[j] > c { alpha[j] = c; alpha[i] = c + diff; }
            } else {
                let delta = (grad[i] - grad[j]) / quad;
                let sum = alpha[i] + alpha[j];
                alpha[i] -= delta;
                alpha[j] += delta;
                if sum > c {
                    if alpha[i] > c { alpha[i] = c; alpha[j] = sum - c; }
                } else if alpha[j] < 0.0 { alpha[j] = 0.0; alpha[i] = sum; }
                if sum > c {
                    if alpha[j] > c { alpha[j] = c; alpha[i] = sum - c; }
                } else if alpha[i] < 0.0 { alpha[i] = 0.0; alpha[j] = sum; }
            }
            let (di, dj) = (alpha[i] - old_i, alpha[j] - old_j);
            for t in 0..l {
                grad[t] += q(t, i) * di + q(t, j) * dj;
            }
        }
        // Bias: average of y G over free support vectors, else the midpoint of the feasible range.
        let free: Vec<f64> = (0..l).filter(|&t| alpha[t] > 0.0 && alpha[t] < c).map(|t| y[t] * grad[t]).collect();
        let rho = if free.is_empty() {
            let upper = (0..l).filter(|&t| in_up(t, &alpha)).map(|t| y[t] * grad[t]).fold(f64::INFINITY, f64::min);
            let lower = (0..l).filter(|&t| in_low(t, &alpha)).map(|t| y[t] * grad[t]).fold(f64::NEG_INFINITY, f64::max);
            if upper.is_finite() && lower.is_finite() { 0.5 * (upper + lower) } else { 0.0 }
        } else {
            free.iter().sum::<f64>() / free.len() as f64
        };
        let keep: Vec<usize> = (0..l).filter(|&t| alpha[t] > 1e-12).collect();
        BinaryMachine {
            class_pos: pos,
            class_neg: neg,
            support: R2D::from_shape_fn((keep.len(), x.ncols()), |(a, j)| sub[[keep[a], j]]),
            coefficients: keep.iter().map(|&t| alpha[t] * y[t]).collect(),
            rho,
        }
    }
}

impl Classifier for SupportVectorMachine {
    fn fit(&mut self, x: VR2D, labels: &[usize]) -> Result<(), ErrorsJSL> {
        check_xy(x, labels.len())?;
        if self.c.is_nan() || self.c <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("C must be positive."));
        }
        let classes = class_count(labels);
        if classes < 2 || (0..classes).any(|c| !labels.contains(&c)) {
            return Err(ErrorsJSL::InvalidInputRange("Need at least two classes, each with samples."));
        }
        self.machines.clear();
        for a in 0..classes {
            for b in a + 1..classes {
                let m = self.train_binary(x, labels, a, b);
                self.machines.push(m);
            }
        }
        self.classes = classes;
        self.features = x.ncols();
        Ok(())
    }

    fn predict(&self, x: VR2D) -> Result<Vec<usize>, ErrorsJSL> {
        self.check_input(x)?;
        let mut votes = vec![vec![0usize; self.classes]; x.nrows()];
        for m in &self.machines {
            for (i, s) in self.score(m, x).into_iter().enumerate() {
                votes[i][if s >= 0.0 { m.class_pos } else { m.class_neg }] += 1;
            }
        }
        Ok(votes.into_iter().map(|v| v.iter().enumerate().max_by_key(|&(c, n)| (*n, std::cmp::Reverse(c))).map_or(0, |(c, _)| c)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{make_blobs, make_circles, make_classification, make_moons};
    use crate::metrics::accuracy;
    use crate::preprocessing::{select_labels, select_rows, train_test_split};
    use ndarray::arr2;

    #[test]
    fn linear_svm_on_separable_data_has_few_support_vectors() {
        let (x, l) = make_blobs(40, &[vec![-3.0, 0.0], vec![3.0, 0.0]], 0.7, 2);
        let mut m = SupportVectorMachine::new(Kernel::Linear, 10.0);
        m.fit(x.view(), &l).unwrap();
        assert_eq!(accuracy(&l, &m.predict(x.view()).unwrap()).unwrap(), 1.0);
        assert!(m.support_vector_count() < 12, "{}", m.support_vector_count());
        // The decision function is positive for label 1 and negative for label 0.
        let d = m.decision_function(arr2(&[[-3.0, 0.0], [3.0, 0.0]]).view()).unwrap();
        assert!(d[0] < 0.0 && d[1] > 0.0);
    }

    #[test]
    fn rbf_kernel_solves_the_nonlinear_problems_a_linear_model_cannot() {
        let (x, l) = make_circles(100, 0.05, 0.4, 1);
        let mut linear = SupportVectorMachine::new(Kernel::Linear, 1.0);
        linear.fit(x.view(), &l).unwrap();
        let mut rbf = SupportVectorMachine::new(Kernel::Rbf { gamma: 2.0 }, 5.0);
        rbf.fit(x.view(), &l).unwrap();
        let (a_lin, a_rbf) = (accuracy(&l, &linear.predict(x.view()).unwrap()).unwrap(), accuracy(&l, &rbf.predict(x.view()).unwrap()).unwrap());
        assert!(a_rbf > 0.97 && a_lin < 0.7, "{a_lin} {a_rbf}");
        let (mx, ml) = make_moons(120, 0.12, 3);
        let (tr, te) = train_test_split(mx.nrows(), 0.3, 4).unwrap();
        let mut m = SupportVectorMachine::new(Kernel::Rbf { gamma: 2.0 }, 5.0);
        m.fit(select_rows(mx.view(), &tr).view(), &select_labels(&ml, &tr)).unwrap();
        assert!(accuracy(&select_labels(&ml, &te), &m.predict(select_rows(mx.view(), &te).view()).unwrap()).unwrap() > 0.9);
    }

    #[test]
    fn multiclass_one_versus_one() {
        let (x, l) = make_classification(60, 9);
        let (tr, te) = train_test_split(x.nrows(), 0.3, 1).unwrap();
        let mut m = SupportVectorMachine::new(Kernel::Rbf { gamma: 0.1 }, 3.0);
        m.fit(select_rows(x.view(), &tr).view(), &select_labels(&l, &tr)).unwrap();
        assert!(accuracy(&select_labels(&l, &te), &m.predict(select_rows(x.view(), &te).view()).unwrap()).unwrap() > 0.8);
        assert!(m.decision_function(x.view()).is_err()); // three classes
    }

    #[test]
    fn errors() {
        let x = arr2(&[[0.0], [1.0]]);
        assert!(SupportVectorMachine::new(Kernel::Linear, 0.0).fit(x.view(), &[0, 1]).is_err());
        assert!(SupportVectorMachine::new(Kernel::Linear, 1.0).fit(x.view(), &[0, 0]).is_err());
        assert!(SupportVectorMachine::new(Kernel::Linear, 1.0).predict(x.view()).is_err());
        let mut m = SupportVectorMachine::new(Kernel::Linear, 1.0);
        m.fit(x.view(), &[0, 1]).unwrap();
        assert!(m.predict(arr2(&[[1.0, 2.0]]).view()).is_err());
    }
}
