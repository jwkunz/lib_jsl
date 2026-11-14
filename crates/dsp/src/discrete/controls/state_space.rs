//! Discrete-time state-space models: simulation, poles, stability, controllability and observability.
//!
//! See the book entry [Control: state-space models, PID controllers and the Kalman filter](https://jwkunz.github.io/lib_jsl/book/comms/controls.html#state-space-models).

use ndarray::s;
use lib_jsl_linalg::{Eig, SVD};

use crate::{
    dsp::discrete::stream_operator::{StreamOperator, StreamOperatorManagement},
    prelude::{C1D, C2D, ErrorsJSL},
};

/// When a [`StateSpace`] model reads its output: from the state before or after the update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputConvention {
    /// The textbook and SciPy convention, `y[k] = C x[k] + D u[k]`: the output is read from the
    /// current state, before it is updated. Use [`StateSpace::new_standard`].
    Standard,
    /// `y[k] = C x[k+1] + D u[k]`: the state is updated first and the output read from the new
    /// state, as in the C++ implementation this type mirrors. This is what [`StateSpace::new`]
    /// builds, and it equals the standard model `(A, B, C A, C B + D)`.
    Updated,
}

/// Discrete-time state-space model using complex-valued matrices and vectors.
///
/// The state evolves as `x[k+1] = A x[k] + B u[k]`. The output follows one of two conventions
/// (see [`OutputConvention`]):
///
/// - [`new_standard`](Self::new_standard): `y[k] = C x[k] + D u[k]`, the textbook and SciPy model;
/// - [`new`](Self::new): `y[k] = C x[k+1] + D u[k]`, the state updated before the output is read,
///   which matches the C++ implementation that this type mirrors.
///
/// All matrices use [`C2D`] and all vectors use [`C1D`], so real-valued models
/// can be represented as purely real complex matrices while still allowing
/// complex-valued systems when needed.
pub struct StateSpace {
    /// State-transition matrix.
    pub a: C2D,
    /// Input-to-state matrix.
    pub b: C2D,
    /// State-to-output matrix.
    pub c: C2D,
    /// Input-to-output feedthrough matrix.
    pub d: C2D,
    /// Current state vector.
    pub x: C1D,
    initial_x: C1D,
    convention: OutputConvention,
}

impl StateSpace {
    /// Construct a model that reads its output after updating the state,
    /// `y[k] = C x[k+1] + D u[k]` ([`OutputConvention::Updated`]). For the textbook and SciPy
    /// convention use [`new_standard`](Self::new_standard).
    ///
    /// The dimensions must satisfy:
    /// - `A` is `n x n`
    /// - `B` is `n x p`
    /// - `C` is `m x n`
    /// - `D` is `m x p`
    /// - `x` has length `n`
    pub fn new(a: C2D, b: C2D, c: C2D, d: C2D, x: C1D) -> Result<Self, ErrorsJSL> {
        Self::with_convention(a, b, c, d, x, OutputConvention::Updated)
    }

    // ANCHOR: new_standard
    /// Construct a model in the textbook and SciPy convention,
    /// `x[k+1] = A x[k] + B u[k]`, `y[k] = C x[k] + D u[k]`: each [`step`](Self::step) reads the
    /// output from the current state and then updates it. A model given to
    /// `scipy.signal.dlsim` as `(A, B, C, D)` is entered here unchanged.
    pub fn new_standard(a: C2D, b: C2D, c: C2D, d: C2D, x: C1D) -> Result<Self, ErrorsJSL> {
        Self::with_convention(a, b, c, d, x, OutputConvention::Standard)
    }
    // ANCHOR_END: new_standard

    /// Construct a model with an explicit [`OutputConvention`]; the dimensions are as for
    /// [`new`](Self::new).
    pub fn with_convention(a: C2D, b: C2D, c: C2D, d: C2D, x: C1D, convention: OutputConvention) -> Result<Self, ErrorsJSL> {
        if a.nrows() == 0 || a.nrows() != a.ncols() {
            return Err(ErrorsJSL::InvalidInputRange("A must be a non-empty square matrix"));
        }
        if a.nrows() != b.nrows() {
            return Err(ErrorsJSL::InvalidInputRange("A and B are not compatible size"));
        }
        if a.nrows() != x.len() {
            return Err(ErrorsJSL::InvalidInputRange("A and x are not compatible size"));
        }
        if c.nrows() != d.nrows() {
            return Err(ErrorsJSL::InvalidInputRange("C and D are not compatible size"));
        }
        if c.ncols() != a.ncols() {
            return Err(ErrorsJSL::InvalidInputRange("A and C are not compatible size"));
        }
        if b.ncols() != d.ncols() {
            return Err(ErrorsJSL::InvalidInputRange("B and D are not compatible size"));
        }

        Ok(Self {
            a,
            b,
            c,
            d,
            x: x.clone(),
            initial_x: x,
            convention,
        })
    }

    /// The output convention of this model.
    pub fn convention(&self) -> OutputConvention {
        self.convention
    }

    /// The same system in the standard convention: for an [`Updated`](OutputConvention::Updated)
    /// model, `(A, B, C A, C B + D)`, which produces identical outputs; a
    /// [`Standard`](OutputConvention::Standard) model is returned unchanged. The current and
    /// initial states are kept.
    pub fn to_standard(&self) -> StateSpace {
        let (c, d) = match self.convention {
            OutputConvention::Standard => (self.c.clone(), self.d.clone()),
            OutputConvention::Updated => (self.c.dot(&self.a), self.c.dot(&self.b) + &self.d),
        };
        StateSpace {
            a: self.a.clone(),
            b: self.b.clone(),
            c,
            d,
            x: self.x.clone(),
            initial_x: self.initial_x.clone(),
            convention: OutputConvention::Standard,
        }
    }

    // ANCHOR: step
    /// Advance the system by one discrete input sample and return the output.
    ///
    /// In the standard convention, `y <- C x + D u` and then `x <- A x + B u`; in the updated
    /// convention (the C++ ordering), `x <- A x + B u` first and then `y <- C x + D u`.
    pub fn step(&mut self, u: C1D) -> Result<C1D, ErrorsJSL> {
        if u.len() != self.b.ncols() {
            return Err(ErrorsJSL::IncompatibleArraySizes((u.len(), self.b.ncols())));
        }

        match self.convention {
            OutputConvention::Standard => {
                let y = self.c.dot(&self.x) + self.d.dot(&u);
                self.x = self.a.dot(&self.x) + self.b.dot(&u);
                Ok(y)
            }
            OutputConvention::Updated => {
                self.x = self.a.dot(&self.x) + self.b.dot(&u);
                Ok(self.c.dot(&self.x) + self.d.dot(&u))
            }
        }
    }
    // ANCHOR_END: step

    /// Check whether the discrete-time model is observable.
    ///
    /// The observability matrix is formed by vertically stacking
    /// `C, C A, C A^2, ..., C A^(n-1)`.
    pub fn is_observable(&self) -> Result<bool, ErrorsJSL> {
        let state_dim = self.a.nrows();
        let output_dim = self.c.nrows();
        let mut big_matrix = C2D::zeros((output_dim * state_dim, state_dim));
        let mut accumulation = self.c.clone();

        for block in 0..state_dim {
            let row_start = block * output_dim;
            let row_end = row_start + output_dim;
            big_matrix
                .slice_mut(s![row_start..row_end, ..])
                .assign(&accumulation);
            accumulation = accumulation.dot(&self.a);
        }

        Ok(Self::matrix_rank(&big_matrix)? == state_dim)
    }

    /// Check whether the discrete-time model is controllable.
    ///
    /// The controllability matrix is formed by horizontally stacking
    /// `B, A B, A^2 B, ..., A^(n-1) B`.
    pub fn is_controllable(&self) -> Result<bool, ErrorsJSL> {
        let state_dim = self.a.nrows();
        let input_dim = self.b.ncols();
        let mut big_matrix = C2D::zeros((state_dim, input_dim * state_dim));
        let mut accumulation = self.b.clone();

        for block in 0..state_dim {
            let col_start = block * input_dim;
            let col_end = col_start + input_dim;
            big_matrix
                .slice_mut(s![.., col_start..col_end])
                .assign(&accumulation);
            accumulation = self.a.dot(&accumulation);
        }

        Ok(Self::matrix_rank(&big_matrix)? == state_dim)
    }

    /// Return the poles of the discrete-time system, which are the eigenvalues
    /// of the state-transition matrix `A`.
    pub fn get_poles(&self) -> Result<C1D, ErrorsJSL> {
        let (eigenvalues, _) = self
            .a
            .clone()
            .eig()
            .map_err(|_| ErrorsJSL::RuntimeError("failed to compute state-space poles"))?;
        Ok(eigenvalues)
    }

    /// Check whether the discrete-time state matrix is stable.
    ///
    /// For a discrete-time model this means every pole lies strictly inside the
    /// unit circle.
    pub fn is_stable(&self) -> Result<bool, ErrorsJSL> {
        let poles = self.get_poles()?;
        Ok(poles.iter().all(|pole| pole.norm() < 1.0))
    }

    fn matrix_rank(matrix: &C2D) -> Result<usize, ErrorsJSL> {
        let (_, singular_values, _) = matrix
            .clone()
            .svd(false, false)
            .map_err(|_| ErrorsJSL::RuntimeError("failed to compute matrix rank"))?;
        let max_sigma = singular_values.iter().copied().fold(0.0_f64, f64::max);
        let tolerance = if max_sigma == 0.0 {
            0.0
        } else {
            max_sigma * (matrix.nrows().max(matrix.ncols()) as f64) * f64::EPSILON
        };
        Ok(singular_values
            .iter()
            .filter(|&&sigma| sigma > tolerance)
            .count())
    }
}

impl StreamOperatorManagement for StateSpace {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.x = self.initial_x.clone();
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl StreamOperator<C1D, C1D> for StateSpace {
    fn process(&mut self, data_in: &[C1D]) -> Result<Option<Vec<C1D>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }

        let mut output = Vec::with_capacity(data_in.len());
        for u in data_in {
            output.push(self.step(u.clone())?);
        }
        Ok(Some(output))
    }

    fn flush(&mut self) -> Result<Option<Vec<C1D>>, ErrorsJSL> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use num_complex::Complex;

    fn c(re: f64) -> Complex<f64> {
        Complex::new(re, 0.0)
    }

    fn vec1(values: &[f64]) -> C1D {
        Array1::from_iter(values.iter().copied().map(c))
    }

    fn mat(rows: usize, cols: usize, values: &[f64]) -> C2D {
        C2D::from_shape_vec((rows, cols), values.iter().copied().map(c).collect()).unwrap()
    }

    #[test]
    fn test_state_space_step_matches_reference_update_order() {
        let mut dut = StateSpace::new(
            mat(1, 1, &[0.5]),
            mat(1, 1, &[1.0]),
            mat(1, 1, &[1.0]),
            mat(1, 1, &[0.0]),
            vec1(&[0.0]),
        )
        .unwrap();

        let y0 = dut.step(vec1(&[2.0])).unwrap();
        let y1 = dut.step(vec1(&[0.0])).unwrap();

        assert!((y0[0] - c(2.0)).norm() < 1e-12);
        assert!((y1[0] - c(1.0)).norm() < 1e-12);
        assert!((dut.x[0] - c(1.0)).norm() < 1e-12);
    }

    #[test]
    fn test_state_space_process_and_reset() {
        let mut dut = StateSpace::new(
            mat(1, 1, &[0.5]),
            mat(1, 1, &[1.0]),
            mat(1, 1, &[1.0]),
            mat(1, 1, &[0.0]),
            vec1(&[0.0]),
        )
        .unwrap();

        let output = dut
            .process(&[vec1(&[1.0]), vec1(&[1.0]), vec1(&[1.0])])
            .unwrap()
            .unwrap();

        assert!((output[0][0] - c(1.0)).norm() < 1e-12);
        assert!((output[1][0] - c(1.5)).norm() < 1e-12);
        assert!((output[2][0] - c(1.75)).norm() < 1e-12);

        dut.reset().unwrap();
        assert!((dut.x[0] - c(0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_state_space_observable_controllable_and_stable() {
        let dut = StateSpace::new(
            mat(2, 2, &[0.5, 0.0, 0.0, 0.25]),
            mat(2, 1, &[1.0, 1.0]),
            mat(1, 2, &[1.0, 1.0]),
            mat(1, 1, &[0.0]),
            vec1(&[0.0, 0.0]),
        )
        .unwrap();

        assert!(dut.is_observable().unwrap());
        assert!(dut.is_controllable().unwrap());
        assert!(dut.is_stable().unwrap());

        let poles = dut.get_poles().unwrap();
        let mut poles = poles.to_vec();
        poles.sort_by(|a, b| a.re.total_cmp(&b.re));
        assert!((poles[0] - c(0.25)).norm() < 1e-12);
        assert!((poles[1] - c(0.5)).norm() < 1e-12);
    }

    #[test]
    fn test_state_space_unobservable_and_uncontrollable() {
        let dut = StateSpace::new(
            mat(2, 2, &[0.5, 0.0, 0.0, 0.25]),
            mat(2, 1, &[1.0, 0.0]),
            mat(1, 2, &[1.0, 0.0]),
            mat(1, 1, &[0.0]),
            vec1(&[0.0, 0.0]),
        )
        .unwrap();

        assert!(!dut.is_observable().unwrap());
        assert!(!dut.is_controllable().unwrap());
    }

    #[test]
    fn test_standard_convention_reads_the_current_state() {
        let mut dut = StateSpace::new_standard(
            mat(1, 1, &[0.5]),
            mat(1, 1, &[1.0]),
            mat(1, 1, &[1.0]),
            mat(1, 1, &[0.0]),
            vec1(&[0.0]),
        )
        .unwrap();
        assert_eq!(dut.convention(), OutputConvention::Standard);
        // y[0] = x[0] = 0, then x[1] = 2, y[1] = 2, x[2] = 1, y[2] = 1.
        let y: Vec<_> = [2.0, 0.0, 0.0].iter().map(|&u| dut.step(vec1(&[u])).unwrap()[0]).collect();
        assert!((y[0] - c(0.0)).norm() < 1e-12);
        assert!((y[1] - c(2.0)).norm() < 1e-12);
        assert!((y[2] - c(1.0)).norm() < 1e-12);
    }

    #[test]
    fn test_to_standard_gives_identical_outputs() {
        let a = mat(2, 2, &[0.9, 0.2, -0.1, 0.8]);
        let b = mat(2, 1, &[1.0, 0.5]);
        let cm = mat(1, 2, &[1.0, -1.0]);
        let d = mat(1, 1, &[0.3]);
        let mut updated = StateSpace::new(a, b, cm, d, vec1(&[0.5, -0.2])).unwrap();
        let mut standard = updated.to_standard();
        assert_eq!(standard.convention(), OutputConvention::Standard);
        for k in 0..20 {
            let u = vec1(&[(0.3 * k as f64).sin()]);
            let (y1, y2) = (updated.step(u.clone()).unwrap(), standard.step(u).unwrap());
            assert!((y1[0] - y2[0]).norm() < 1e-12);
        }
    }

    #[test]
    fn test_state_space_rejects_incompatible_sizes() {
        let err = StateSpace::new(
            mat(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            mat(1, 1, &[1.0]),
            mat(1, 2, &[1.0, 0.0]),
            mat(1, 1, &[0.0]),
            vec1(&[0.0, 0.0]),
        );
        assert!(err.is_err());
    }
}
