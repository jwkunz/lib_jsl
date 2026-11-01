//! Padé approximants built from Taylor series coefficients.

use lib_jsl_core::*;
use ndarray::{Array1, Array2};
use lib_jsl_linalg::Solve;

/// The `[L/M]` Padé approximant `P(x)/Q(x)` with `deg P <= L`, `deg Q <= M`, `Q(0) = 1`, whose
/// Taylor series matches the given one through order `L + M`.
#[derive(Debug, Clone)]
pub struct PadeApproximant {
    numerator: Vec<f64>,
    denominator: Vec<f64>,
}

impl PadeApproximant {
    /// Builds the `[l/m]` approximant from Taylor coefficients `c[0..=l+m]` (lowest order first).
    pub fn new(taylor: &[f64], l: usize, m: usize) -> Result<Self, ErrorsJSL> {
        if taylor.len() < l + m + 1 {
            return Err(ErrorsJSL::IncompatibleArraySizes((taylor.len(), l + m + 1)));
        }
        let c = |k: isize| if k < 0 { 0.0 } else { taylor[k as usize] };
        let mut q = vec![1.0];
        if m > 0 {
            // Σ_{j=1..m} q_j c_{L+i-j} = -c_{L+i}, i = 1..m.
            let a = Array2::from_shape_fn((m, m), |(i, j)| c((l + i + 1) as isize - (j + 1) as isize));
            let b = Array1::from_shape_fn(m, |i| -c((l + i + 1) as isize));
            let sol = a.solve(&b).map_err(|_| ErrorsJSL::RuntimeError("singular Padé system"))?;
            q.extend(sol.iter());
        }
        let p = (0..=l)
            .map(|i| (0..=i.min(m)).map(|j| q[j] * c((i - j) as isize)).sum())
            .collect();
        Ok(PadeApproximant { numerator: p, denominator: q })
    }

    /// Numerator coefficients, lowest order first.
    pub fn numerator(&self) -> &[f64] {
        &self.numerator
    }

    /// Denominator coefficients, lowest order first (`denominator()[0] == 1`).
    pub fn denominator(&self) -> &[f64] {
        &self.denominator
    }

    /// Evaluates `P(x) / Q(x)` by Horner's rule.
    pub fn evaluate(&self, x: f64) -> f64 {
        let horner = |c: &[f64]| c.iter().rev().fold(0.0, |acc, &k| acc * x + k);
        horner(&self.numerator) / horner(&self.denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_pade_2_2_known_coefficients() {
        // exp(x) series: 1, 1, 1/2, 1/6, 1/24 -> [2/2] = (1 + x/2 + x²/12)/(1 - x/2 + x²/12)
        let t = [1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0];
        let p = PadeApproximant::new(&t, 2, 2).unwrap();
        for (a, b) in p.numerator().iter().zip([1.0, 0.5, 1.0 / 12.0]) {
            assert!((a - b).abs() < 1e-12);
        }
        for (a, b) in p.denominator().iter().zip([1.0, -0.5, 1.0 / 12.0]) {
            assert!((a - b).abs() < 1e-12);
        }
        assert!((p.evaluate(1.0) - 19.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn beats_the_taylor_polynomial_beyond_its_radius() {
        // ln(1+x) at x = 3 is outside the Taylor radius; the [4/4] approximant still works.
        let t: Vec<f64> = (0..9).map(|k| if k == 0 { 0.0 } else { (-1.0f64).powi(k + 1) / k as f64 }).collect();
        let p = PadeApproximant::new(&t, 4, 4).unwrap();
        assert!((p.evaluate(3.0) - 4.0f64.ln()).abs() < 5e-3);
    }

    #[test]
    fn pure_polynomial_when_m_is_zero() {
        let p = PadeApproximant::new(&[1.0, 2.0, 3.0], 2, 0).unwrap();
        assert_eq!(p.evaluate(2.0), 17.0);
        assert!(PadeApproximant::new(&[1.0], 2, 2).is_err());
    }
}
