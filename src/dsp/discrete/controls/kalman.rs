use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;
use num::Complex;

use crate::{
    dsp::discrete::stream_operator::{StreamOperator, StreamOperatorManagement},
    prelude::{C1D, C2D, ErrorsJSL},
};

/// Streaming discrete-time Kalman filter with complex-valued matrices.
///
/// This type implements the classic linear Kalman filter for a discrete-time,
/// possibly complex-valued state-space system. It is intended for streaming
/// estimation workloads where each call to [`process`](StreamOperator::process)
/// consumes one or more measurement vectors and returns the updated state
/// estimate after each observation.
///
/// The filter operates on the standard state-space model:
/// `x[k] = F x[k-1] + B u[k] + w[k]`
/// `z[k] = H x[k] + v[k]`
///
/// where:
/// - `x[k]` is the hidden state vector
/// - `z[k]` is the measurement vector
/// - `F` is the state-transition matrix
/// - `H` is the observation matrix
/// - `B` is the optional control-input matrix
/// - `u[k]` is the optional control vector
/// - `w[k]` and `v[k]` are the process and measurement noise terms
///
/// All internal matrices use [`C2D`] and all state/measurement vectors use
/// [`C1D`]. The covariance update uses conjugate-transpose operations so the
/// filter works naturally with complex-valued systems.
///
/// Dimension conventions:
/// - `F` is `n x n`
/// - `H` is `m x n`
/// - `Q` is `n x n`
/// - `R` is `m x m`
/// - `P` is `n x n`
/// - `x` is length `n`
/// - `B`, when present, is `n x p`
/// - `u`, when present, is length `p`
///
/// Each streamed measurement triggers one full predict/update cycle:
/// 1. Predict the next state and covariance.
/// 2. Form the innovation from the new measurement.
/// 3. Compute the Kalman gain.
/// 4. Update the state estimate and covariance.
///
/// This filter is stateful. Reusing the same instance across multiple calls to
/// [`process`](StreamOperator::process) continues estimation from the previous
/// internal state until [`reset`](StreamOperatorManagement::reset) is called.
#[derive(Clone, Debug, PartialEq)]
pub struct KalmanResult {
    /// The updated a-posteriori state estimate after incorporating the current
    /// measurement.
    pub next_state: C1D,
    /// The innovation or residual vector `z[k] - H x_pred[k]` formed from the
    /// incoming measurement and the predicted observation.
    pub residual_error: C1D,
}

pub struct KalmanFilter {
    state_transition: C2D,
    observation_model: C2D,
    observation_model_h: C2D,
    process_noise: C2D,
    measurement_noise: C2D,
    estimate_covariance: C2D,
    control_model: Option<C2D>,
    control_input: Option<C1D>,
    state_estimate: C1D,
    initial_state_estimate: C1D,
    initial_estimate_covariance: C2D,
}

impl KalmanFilter {
    /// Creates a new streaming Kalman filter.
    ///
    /// The constructor validates the dimensional compatibility of all provided
    /// matrices and vectors before any processing begins.
    ///
    /// Arguments:
    /// - `state_transition`: State-transition matrix `F`
    /// - `observation_model`: Observation matrix `H`
    /// - `process_noise`: Process-noise covariance `Q`
    /// - `measurement_noise`: Measurement-noise covariance `R`
    /// - `estimate_covariance`: Initial estimate covariance `P`
    /// - `initial_state_estimate`: Initial state estimate `x`
    /// - `control_model`: Optional control matrix `B`
    /// - `control_input`: Optional control vector `u`
    ///
    /// Returns an error when any dimensions are incompatible, when the state
    /// matrix is not square, or when a control input is supplied without a
    /// matching control model.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        state_transition: C2D,
        observation_model: C2D,
        process_noise: C2D,
        measurement_noise: C2D,
        estimate_covariance: C2D,
        initial_state_estimate: C1D,
        control_model: Option<C2D>,
        control_input: Option<C1D>,
    ) -> Result<Self, ErrorsJSL> {
        let state_dim = state_transition.nrows();
        if state_dim == 0 || state_transition.ncols() != state_dim {
            return Err(ErrorsJSL::InvalidInputRange(
                "state_transition must be a non-empty square matrix",
            ));
        }
        if observation_model.ncols() != state_dim {
            return Err(ErrorsJSL::IncompatibleArraySizes((
                observation_model.ncols(),
                state_dim,
            )));
        }
        if process_noise.dim() != (state_dim, state_dim) {
            return Err(ErrorsJSL::IncompatibleArraySizes((
                process_noise.len(),
                state_dim * state_dim,
            )));
        }
        if estimate_covariance.dim() != (state_dim, state_dim) {
            return Err(ErrorsJSL::IncompatibleArraySizes((
                estimate_covariance.len(),
                state_dim * state_dim,
            )));
        }
        if initial_state_estimate.len() != state_dim {
            return Err(ErrorsJSL::IncompatibleArraySizes((
                initial_state_estimate.len(),
                state_dim,
            )));
        }

        let measurement_dim = observation_model.nrows();
        if measurement_noise.dim() != (measurement_dim, measurement_dim) {
            return Err(ErrorsJSL::IncompatibleArraySizes((
                measurement_noise.len(),
                measurement_dim * measurement_dim,
            )));
        }

        if let Some(b) = &control_model
            && b.nrows() != state_dim
        {
            return Err(ErrorsJSL::IncompatibleArraySizes((b.nrows(), state_dim)));
        }

        if let (Some(b), Some(u)) = (&control_model, &control_input)
            && b.ncols() != u.len()
        {
            return Err(ErrorsJSL::IncompatibleArraySizes((b.ncols(), u.len())));
        }

        if control_model.is_none() && control_input.is_some() {
            return Err(ErrorsJSL::Misconfiguration(
                "control_input requires control_model",
            ));
        }

        let observation_model_h = Self::conjugate_transpose(&observation_model);

        Ok(Self {
            state_transition,
            observation_model,
            observation_model_h,
            process_noise,
            measurement_noise,
            estimate_covariance: estimate_covariance.clone(),
            control_model,
            control_input,
            state_estimate: initial_state_estimate.clone(),
            initial_state_estimate,
            initial_estimate_covariance: estimate_covariance,
        })
    }

    /// Returns the current a-posteriori state estimate.
    ///
    /// This is the most recent state after the last successful measurement
    /// update. Before any measurements are processed, this is equal to the
    /// `initial_state_estimate` provided at construction time.
    pub fn state_estimate(&self) -> &C1D {
        &self.state_estimate
    }

    /// Returns the current estimate covariance matrix.
    ///
    /// This matrix represents the filter's current uncertainty about the state
    /// estimate. Smaller diagonal values generally indicate greater confidence
    /// in the corresponding state components.
    pub fn estimate_covariance(&self) -> &C2D {
        &self.estimate_covariance
    }

    /// Updates the observation matrix `H` used during the measurement update.
    ///
    /// The matrix must preserve the existing state dimension and measurement
    /// dimension. The cached conjugate-transpose `H^H` is refreshed whenever
    /// this matrix is changed.
    pub fn set_observation_model(&mut self, observation_model: C2D) -> Result<(), ErrorsJSL> {
        let expected_state_dim = self.state_estimate.len();
        let expected_measurement_dim = self.measurement_noise.nrows();

        if observation_model.ncols() != expected_state_dim {
            return Err(ErrorsJSL::IncompatibleArraySizes((
                observation_model.ncols(),
                expected_state_dim,
            )));
        }
        if observation_model.nrows() != expected_measurement_dim {
            return Err(ErrorsJSL::IncompatibleArraySizes((
                observation_model.nrows(),
                expected_measurement_dim,
            )));
        }

        self.observation_model_h = Self::conjugate_transpose(&observation_model);
        self.observation_model = observation_model;
        Ok(())
    }

    /// Updates the control-input vector used during the prediction step.
    ///
    /// When both a control model `B` and a control vector `u` are present, the
    /// predictor applies the term `B u` before the measurement update.
    ///
    /// Passing `None` disables the control input while leaving the control
    /// matrix configured.
    pub fn set_control_input(&mut self, control_input: Option<C1D>) -> Result<(), ErrorsJSL> {
        if let (Some(b), Some(u)) = (&self.control_model, &control_input)
            && b.ncols() != u.len()
        {
            return Err(ErrorsJSL::IncompatibleArraySizes((b.ncols(), u.len())));
        }
        if self.control_model.is_none() && control_input.is_some() {
            return Err(ErrorsJSL::Misconfiguration(
                "control_input requires control_model",
            ));
        }
        self.control_input = control_input;
        Ok(())
    }

    /// Updates the optional control matrix used during prediction.
    ///
    /// The control matrix must have one row per state component. If a control
    /// vector is already configured, its length must match the new matrix's
    /// column count.
    ///
    /// Passing `None` removes the control model, but only when no control input
    /// is currently configured.
    pub fn set_control_model(&mut self, control_model: Option<C2D>) -> Result<(), ErrorsJSL> {
        if let Some(b) = &control_model {
            if b.nrows() != self.state_estimate.len() {
                return Err(ErrorsJSL::IncompatibleArraySizes((
                    b.nrows(),
                    self.state_estimate.len(),
                )));
            }
            if let Some(u) = &self.control_input
                && b.ncols() != u.len()
            {
                return Err(ErrorsJSL::IncompatibleArraySizes((b.ncols(), u.len())));
            }
        } else if self.control_input.is_some() {
            return Err(ErrorsJSL::Misconfiguration(
                "control_model cannot be removed while control_input is set",
            ));
        }

        self.control_model = control_model;
        Ok(())
    }

    fn conjugate_transpose(matrix: &C2D) -> C2D {
        matrix.t().mapv(|x| x.conj()).to_owned()
    }

    fn identity(size: usize) -> C2D {
        Array2::from_diag(&Array1::from_elem(size, Complex::new(1.0, 0.0)))
    }

    fn predict(&mut self) {
        self.state_estimate = self.state_transition.dot(&self.state_estimate);

        if let (Some(b), Some(u)) = (&self.control_model, &self.control_input) {
            self.state_estimate += &b.dot(u);
        }

        let f_h = Self::conjugate_transpose(&self.state_transition);
        self.estimate_covariance = self
            .state_transition
            .dot(&self.estimate_covariance)
            .dot(&f_h)
            + &self.process_noise;
    }

    fn update(&mut self, measurement: &C1D) -> Result<KalmanResult, ErrorsJSL> {
        let measurement_dim = self.observation_model.nrows();
        if measurement.len() != measurement_dim {
            return Err(ErrorsJSL::IncompatibleArraySizes((
                measurement.len(),
                measurement_dim,
            )));
        }

        let innovation = measurement - &self.observation_model.dot(&self.state_estimate);
        let innovation_covariance = self
            .observation_model
            .dot(&self.estimate_covariance)
            .dot(&self.observation_model_h)
            + &self.measurement_noise;
        let gain_rhs = self.estimate_covariance.dot(&self.observation_model_h);
        let gain_rhs_t = gain_rhs.t().to_owned();
        let mut kalman_gain_t = C2D::zeros(gain_rhs_t.raw_dim());
        for (column_idx, rhs_column) in gain_rhs_t.columns().into_iter().enumerate() {
            let solved_column = innovation_covariance
                .solve_t_into(rhs_column.to_owned())
                .map_err(|_| ErrorsJSL::RuntimeError("failed to solve innovation covariance"))?;
            kalman_gain_t.column_mut(column_idx).assign(&solved_column);
        }
        let kalman_gain = kalman_gain_t.t().to_owned();

        self.state_estimate = &self.state_estimate + &kalman_gain.dot(&innovation);

        let identity = Self::identity(self.state_estimate.len());
        self.estimate_covariance =
            (identity - kalman_gain.dot(&self.observation_model)).dot(&self.estimate_covariance);

        Ok(KalmanResult {
            next_state: self.state_estimate.clone(),
            residual_error: innovation,
        })
    }

    fn step(&mut self, measurement: &C1D) -> Result<KalmanResult, ErrorsJSL> {
        self.predict();
        self.update(measurement)
    }
}

impl StreamOperatorManagement for KalmanFilter {
    /// Restores the filter to its initial state estimate and covariance.
    ///
    /// This is useful when starting a fresh estimation run over a new signal or
    /// simulation without reallocating the filter object.
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.state_estimate = self.initial_state_estimate.clone();
        self.estimate_covariance = self.initial_estimate_covariance.clone();
        Ok(())
    }

    /// Finalizes the filter.
    ///
    /// The Kalman filter does not buffer delayed output, so finalization is a
    /// no-op and simply returns `Ok(())`.
    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl StreamOperator<C1D, KalmanResult> for KalmanFilter {
    /// Processes a slice of measurement vectors and returns the updated state
    /// estimate and residual after each measurement.
    ///
    /// Input:
    /// - Each element of `data_in` is one measurement vector `z[k]`
    ///
    /// Output:
    /// - The returned vector contains one [`KalmanResult`] per input
    ///   measurement, in the same order as the measurements were processed
    ///
    /// For an empty input slice, this method returns `Ok(None)` to match the
    /// stream-operator conventions used elsewhere in the crate.
    fn process(&mut self, data_in: &[C1D]) -> Result<Option<Vec<KalmanResult>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }

        let mut out = Vec::with_capacity(data_in.len());
        for measurement in data_in {
            out.push(self.step(measurement)?);
        }
        Ok(Some(out))
    }

    /// Flushes any buffered output from the filter.
    ///
    /// The Kalman filter has no delayed-output buffering, so flushing produces
    /// no additional samples and returns `Ok(None)`.
    fn flush(&mut self) -> Result<Option<Vec<KalmanResult>>, ErrorsJSL> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::discrete::stream_operator::{StreamOperator, StreamOperatorManagement};

    fn c(re: f64) -> Complex<f64> {
        Complex::new(re, 0.0)
    }

    #[test]
    fn test_kalman_filter_tracks_constant_scalar_signal() {
        let f = Array2::from_elem((1, 1), c(1.0));
        let h = Array2::from_elem((1, 1), c(1.0));
        let q = Array2::from_elem((1, 1), c(1e-3));
        let r = Array2::from_elem((1, 1), c(1e-2));
        let p0 = Array2::from_elem((1, 1), c(1.0));
        let x0 = Array1::from_vec(vec![c(0.0)]);

        let mut dut = KalmanFilter::new(f, h, q, r, p0, x0, None, None).unwrap();
        let measurements = vec![
            Array1::from_vec(vec![c(1.2)]),
            Array1::from_vec(vec![c(0.8)]),
            Array1::from_vec(vec![c(1.1)]),
            Array1::from_vec(vec![c(0.9)]),
            Array1::from_vec(vec![c(1.0)]),
        ];

        let estimates = dut.process(&measurements).unwrap().unwrap();
        let final_estimate = estimates.last().unwrap().next_state[0].re;
        let final_residual = estimates.last().unwrap().residual_error[0].re;

        assert!((final_estimate - 1.0).abs() < 0.08);
        assert!(final_residual.abs() < 0.15);
        assert!(dut.estimate_covariance()[[0, 0]].re < 0.05);
    }

    #[test]
    fn test_kalman_filter_reset_restores_initial_conditions() {
        let f = Array2::from_elem((1, 1), c(1.0));
        let h = Array2::from_elem((1, 1), c(1.0));
        let q = Array2::from_elem((1, 1), c(1e-3));
        let r = Array2::from_elem((1, 1), c(1e-2));
        let p0 = Array2::from_elem((1, 1), c(1.0));
        let x0 = Array1::from_vec(vec![c(0.25)]);

        let mut dut = KalmanFilter::new(f, h, q, r, p0.clone(), x0.clone(), None, None).unwrap();
        let measurement = Array1::from_vec(vec![c(1.0)]);
        let _ = dut.process(&[measurement]).unwrap().unwrap();

        dut.reset().unwrap();

        assert_eq!(dut.state_estimate(), &x0);
        assert_eq!(dut.estimate_covariance(), &p0);
    }

    #[test]
    fn test_kalman_filter_rejects_bad_measurement_dimension() {
        let f = Array2::from_elem((2, 2), c(1.0));
        let h = Array2::from_shape_vec((1, 2), vec![c(1.0), c(0.0)]).unwrap();
        let q = Array2::from_diag(&Array1::from_vec(vec![c(1e-3), c(1e-3)]));
        let r = Array2::from_elem((1, 1), c(1e-2));
        let p0 = Array2::from_diag(&Array1::from_vec(vec![c(1.0), c(1.0)]));
        let x0 = Array1::from_vec(vec![c(0.0), c(0.0)]);

        let mut dut = KalmanFilter::new(f, h, q, r, p0, x0, None, None).unwrap();
        let bad_measurement = Array1::from_vec(vec![c(1.0), c(2.0)]);

        let err = dut.process(&[bad_measurement]).unwrap_err();
        assert!(matches!(err, ErrorsJSL::IncompatibleArraySizes((2, 1))));
    }
}
