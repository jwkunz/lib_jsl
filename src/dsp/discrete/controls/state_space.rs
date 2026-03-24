use ndarray::s;
use ndarray_linalg::{Eig, SVD};

use crate::{
    dsp::discrete::stream_operator::{StreamOperator, StreamOperatorManagement},
    prelude::{C1D, C2D, ErrorsJSL},
};

/// Discrete-time state-space model using complex-valued matrices and vectors.
///
/// The internal representation follows the standard linear model
///
/// `x[k+1] = A x[k] + B u[k]`
/// `y[k]   = C x[k+1] + D u[k]`
///
/// which matches the step ordering used by the C++ implementation that this
/// type mirrors. In other words, each call to [`step`](Self::step) first
/// updates the internal state and then computes the output from the updated
/// state and current input.
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
}

impl StateSpace {
    /// Construct a new discrete-time state-space model.
    ///
    /// The dimensions must satisfy:
    /// - `A` is `n x n`
    /// - `B` is `n x p`
    /// - `C` is `m x n`
    /// - `D` is `m x p`
    /// - `x` has length `n`
    pub fn new(a: C2D, b: C2D, c: C2D, d: C2D, x: C1D) -> Result<Self, ErrorsJSL> {
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
        })
    }

    /// Advance the system by one discrete input sample and return the output.
    ///
    /// This uses the same update ordering as the C++ implementation:
    /// first `x <- A x + B u`, then `y <- C x + D u`.
    pub fn step(&mut self, u: C1D) -> Result<C1D, ErrorsJSL> {
        if u.len() != self.b.ncols() {
            return Err(ErrorsJSL::IncompatibleArraySizes((u.len(), self.b.ncols())));
        }

        self.x = self.a.dot(&self.x) + self.b.dot(&u);
        Ok(self.c.dot(&self.x) + self.d.dot(&u))
    }

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
    use num::Complex;

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
