//! Farrow filter: a fractional-delay (arbitrary-position) interpolator.
//!
//! Interpolating a sampled signal at a point between samples means evaluating a polynomial through
//! the surrounding samples. The Farrow structure factors this as a bank of *fixed* FIR sub-filters
//! whose outputs are combined by Horner's rule in the fractional position `μ`:
//!
//! `y(μ) = v₀ + μ (v₁ + μ (v₂ + …))`, with `v_m = Σ_k c[m][k] x[k]`.
//!
//! Only the cheap Horner step depends on `μ`, so the interpolation point can change on every
//! sample (as in timing recovery) at no extra cost. Here the sub-filters implement Lagrange
//! interpolation of odd order `N` (an even number `N + 1` of points), which reproduces polynomials
//! up to degree `N` exactly. `N = 1` is linear interpolation and `N = 3` the usual cubic.
//!
//! # Position convention
//! With the window holding `x[n-N] … x[n]` (oldest first), the output at `μ ∈ [0, 1)` is the signal
//! value `(N + 1)/2 - μ` samples before the newest sample: `y = x(n - (N+1)/2 + μ)`.

use crate::{
    discrete::stream_operator::{StreamOperator, StreamOperatorManagement},
    prelude::{ErrorsJSL, IsAnalytic},
};

/// A Lagrange Farrow interpolator of odd order.
#[derive(Clone, Debug)]
pub struct FarrowFilter<T: IsAnalytic> {
    order: usize,
    /// `coefficients[m][k]`: coefficient of `μ^m` for window tap `k`.
    coefficients: Vec<Vec<f64>>,
    history: Vec<T>,
    mu: f64,
}

fn lagrange_polynomials(order: usize) -> Vec<Vec<f64>> {
    let nodes: Vec<f64> = (0..=order).map(|k| k as f64 - (order as f64 - 1.0) / 2.0).collect();
    // coefficients[m][k]
    let mut coeffs = vec![vec![0.0; order + 1]; order + 1];
    for k in 0..=order {
        // Expand prod_{j != k} (mu - t_j) / (t_k - t_j) into powers of mu.
        let mut poly = vec![1.0];
        let mut denom = 1.0;
        for (j, &tj) in nodes.iter().enumerate() {
            if j == k {
                continue;
            }
            let mut next = vec![0.0; poly.len() + 1];
            for (m, &c) in poly.iter().enumerate() {
                next[m + 1] += c;
                next[m] -= c * tj;
            }
            poly = next;
            denom *= nodes[k] - tj;
        }
        for (m, c) in poly.into_iter().enumerate() {
            coeffs[m][k] = c / denom;
        }
    }
    coeffs
}

impl<T: IsAnalytic> FarrowFilter<T> {
    /// A filter of odd `order` in `1..=9` (so `order + 1` points), initially at `μ = 0`.
    pub fn new(order: usize) -> Result<Self, ErrorsJSL> {
        if order.is_multiple_of(2) || order > 9 {
            return Err(ErrorsJSL::InvalidInputRange("Farrow order must be odd and at most 9"));
        }
        Ok(FarrowFilter { order, coefficients: lagrange_polynomials(order), history: vec![T::zero(); order + 1], mu: 0.0 })
    }

    pub fn order(&self) -> usize {
        self.order
    }

    /// Number of window samples the interpolator uses (`order + 1`).
    pub fn window_len(&self) -> usize {
        self.order + 1
    }

    /// The delay, in samples behind the newest sample, of the output at fractional position `mu`.
    pub fn delay_at(&self, mu: f64) -> f64 {
        (self.order as f64 + 1.0) / 2.0 - mu
    }

    /// Sets the fractional position used by the streaming interface (`0 <= mu <= 1`).
    pub fn set_mu(&mut self, mu: f64) -> Result<(), ErrorsJSL> {
        if !(0.0..=1.0).contains(&mu) {
            return Err(ErrorsJSL::InvalidInputRange("mu must lie in [0, 1]"));
        }
        self.mu = mu;
        Ok(())
    }

    /// Interpolates a window of exactly `order + 1` samples (oldest first) at position `mu`.
    pub fn interpolate_window(&self, window: &[T], mu: f64) -> Result<T, ErrorsJSL> {
        if window.len() != self.order + 1 {
            return Err(ErrorsJSL::IncompatibleArraySizes((window.len(), self.order + 1)));
        }
        // Sub-filter outputs v_m, then Horner in mu.
        let v = |m: usize| -> T {
            window.iter().zip(&self.coefficients[m]).fold(T::zero(), |acc, (&x, &c)| acc + x.f_scale(c))
        };
        let mut y = v(self.order);
        for m in (0..self.order).rev() {
            y = y.f_scale(mu) + v(m);
        }
        Ok(y)
    }

    /// Shifts a new sample into the window.
    pub fn push(&mut self, x: T) {
        self.history.rotate_left(1);
        *self.history.last_mut().unwrap() = x;
    }

    /// The interpolated output for the current window at position `mu`.
    pub fn output(&self, mu: f64) -> T {
        self.interpolate_window(&self.history, mu).expect("history length is order + 1")
    }
}

impl<T: IsAnalytic> StreamOperatorManagement for FarrowFilter<T> {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.history.iter_mut().for_each(|v| *v = T::zero());
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

/// Streaming operation at the position set by [`FarrowFilter::set_mu`]: each input sample yields
/// one output, a copy of the signal delayed by `delay_at(mu)` samples.
impl<T: IsAnalytic> StreamOperator<T, T> for FarrowFilter<T> {
    fn process(&mut self, data_in: &[T]) -> Result<Option<Vec<T>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        let mu = self.mu;
        Ok(Some(data_in.iter().map(|&x| {
            self.push(x);
            self.output(mu)
        }).collect()))
    }

    fn flush(&mut self) -> Result<Option<Vec<T>>, ErrorsJSL> {
        Ok(None)
    }
}

/// Delays `x` by an arbitrary, possibly fractional, number of samples: `y[n] = x(n - delay)`, with
/// zeros assumed outside the record. Works for any `delay` (negative advances the signal).
pub fn farrow_filter<T: IsAnalytic>(x: &[T], delay: f64, order: usize) -> Result<Vec<T>, ErrorsJSL> {
    if !delay.is_finite() {
        return Err(ErrorsJSL::InvalidInputRange("Delay must be finite"));
    }
    let f = FarrowFilter::<T>::new(order)?;
    let n = x.len() as isize;
    let before = ((order as isize) - 1) / 2; // window starts this many samples before the position
    let out = (0..n)
        .map(|i| {
            let p = i as f64 - delay;
            let base = p.floor();
            let mu = p - base;
            let first = base as isize - before;
            let window: Vec<T> = (0..=order as isize)
                .map(|k| {
                    let idx = first + k;
                    if (0..n).contains(&idx) { x[idx as usize] } else { T::zero() }
                })
                .collect();
            f.interpolate_window(&window, mu).expect("window length is order + 1")
        })
        .collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use std::f64::consts::TAU;

    #[test]
    fn mu_zero_and_one_land_on_the_neighbouring_samples() {
        for order in [1usize, 3, 5, 7] {
            let f = FarrowFilter::<f64>::new(order).unwrap();
            let w: Vec<f64> = (0..=order).map(|k| ((k * 37) % 11) as f64 - 4.0).collect();
            let c = (order - 1) / 2; // window index of the node at t = 0
            assert!((f.interpolate_window(&w, 0.0).unwrap() - w[c]).abs() < 1e-12, "order {order}, mu 0");
            assert!((f.interpolate_window(&w, 1.0).unwrap() - w[c + 1]).abs() < 1e-12, "order {order}, mu 1");
        }
    }

    #[test]
    fn linear_order_is_linear_interpolation() {
        let f = FarrowFilter::<f64>::new(1).unwrap();
        assert!((f.interpolate_window(&[2.0, 6.0], 0.25).unwrap() - 3.0).abs() < 1e-14);
    }

    #[test]
    fn polynomials_up_to_the_order_are_reproduced_exactly() {
        let poly = |t: f64| 1.0 - 2.0 * t + 0.5 * t * t + 0.25 * t * t * t;
        let f = FarrowFilter::<f64>::new(3).unwrap();
        let window: Vec<f64> = (-1..=2).map(|t| poly(t as f64)).collect();
        for mu in [0.0, 0.2, 0.5, 0.77, 1.0] {
            assert!((f.interpolate_window(&window, mu).unwrap() - poly(mu)).abs() < 1e-12, "mu {mu}");
        }
        // A quartic is not reproduced by the cubic filter...
        let quartic = |t: f64| t.powi(4);
        let w4: Vec<f64> = (-1..=2).map(|t| quartic(t as f64)).collect();
        assert!((f.interpolate_window(&w4, 0.5).unwrap() - quartic(0.5)).abs() > 1e-3);
        // ...but the order-5 filter reproduces it.
        let f5 = FarrowFilter::<f64>::new(5).unwrap();
        let w5: Vec<f64> = (-2..=3).map(|t| quartic(t as f64)).collect();
        assert!((f5.interpolate_window(&w5, 0.37).unwrap() - quartic(0.37)).abs() < 1e-11);
    }

    #[test]
    fn higher_orders_interpolate_a_sinusoid_more_accurately() {
        let f0 = 0.08;
        let x: Vec<f64> = (0..200).map(|n| (TAU * f0 * n as f64).sin()).collect();
        let delay = 7.3;
        let mut errors = Vec::new();
        for order in [1usize, 3, 5, 7] {
            let y = farrow_filter(&x, delay, order).unwrap();
            let e = (30..170).map(|n| (y[n] - (TAU * f0 * (n as f64 - delay)).sin()).abs()).fold(0.0, f64::max);
            errors.push(e);
        }
        assert!(errors.windows(2).all(|w| w[1] < w[0] * 0.3), "{errors:?}");
        assert!(errors[1] < 2e-3 && errors[3] < 1e-5, "{errors:?}");
    }

    #[test]
    fn integer_delays_are_exact_shifts_and_negative_delays_advance() {
        let x: Vec<f64> = (0..20).map(|n| (n * n) as f64 % 7.0).collect();
        let y = farrow_filter(&x, 3.0, 3).unwrap();
        for n in 3..20 {
            assert!((y[n] - x[n - 3]).abs() < 1e-12);
        }
        assert!(y[..3].iter().all(|v| v.abs() < 1e-12), "zeros before the record");
        let adv = farrow_filter(&x, -2.0, 3).unwrap();
        for n in 0..16 {
            assert!((adv[n] - x[n + 2]).abs() < 1e-12);
        }
    }

    #[test]
    fn streaming_matches_the_batch_delay() {
        let x: Vec<f64> = (0..100).map(|n| (0.2 * n as f64).cos()).collect();
        let mut f = FarrowFilter::<f64>::new(3).unwrap();
        f.set_mu(0.4).unwrap();
        let y = f.process(&x).unwrap().unwrap();
        let delay = f.delay_at(0.4);
        assert!((delay - 1.6).abs() < 1e-14);
        let batch = farrow_filter(&x, delay, 3).unwrap();
        for n in 10..100 {
            assert!((y[n] - batch[n]).abs() < 1e-12, "n = {n}");
        }
        StreamOperatorManagement::reset(&mut f).unwrap();
        assert!(f.process(&[]).unwrap().is_none() && f.flush().unwrap().is_none());
        assert!(f.set_mu(1.5).is_err());
    }

    #[allow(clippy::needless_range_loop)]
    #[test]
    fn complex_samples() {
        let x: Vec<Complex<f64>> = (0..100).map(|n| Complex::from_polar(1.0, TAU * 0.05 * n as f64)).collect();
        let y = farrow_filter(&x, 2.5, 5).unwrap();
        for n in 20..80 {
            let want = Complex::from_polar(1.0, TAU * 0.05 * (n as f64 - 2.5));
            assert!((y[n] - want).norm() < 1e-4, "n = {n}");
        }
    }

    #[test]
    fn invalid_arguments() {
        assert!(FarrowFilter::<f64>::new(2).is_err() && FarrowFilter::<f64>::new(11).is_err() && FarrowFilter::<f64>::new(0).is_err());
        assert!(farrow_filter(&[1.0, 2.0], f64::NAN, 3).is_err());
        let f = FarrowFilter::<f64>::new(3).unwrap();
        assert!(f.interpolate_window(&[1.0, 2.0], 0.5).is_err());
        assert_eq!((f.order(), f.window_len()), (3, 4));
    }
}
