//! One-dimensional sample axes with built-in quadrature weights.
//!
//! An [`Axis`] holds strictly increasing sample coordinates and the weights `wᵢ` of a quadrature
//! rule `∫ f(u) du ≈ Σ wᵢ f(uᵢ)` over the axis's extent. Jacobians are *not* included here: the
//! domain that combines axes multiplies them in. Periodic axes (azimuth) cover one period and wrap.

use lib_jsl_core::ErrorsJSL;
use std::f64::consts::PI;

/// How an axis's quadrature weights were constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadratureRule {
    /// Trapezoid rule on the samples (endpoints included, any spacing).
    Trapezoid,
    /// Composite Simpson rule on uniform samples (odd count).
    Simpson,
    /// Rectangle rule on a periodic grid (spectrally accurate for smooth periodic integrands).
    Periodic,
    /// Gauss-Legendre nodes and weights (exact for polynomials of degree `2n - 1`).
    GaussLegendre,
    /// Gauss-Legendre in `cos θ` for a polar axis: nodes `θᵢ = acos(xᵢ)` with weights
    /// `wᵢ / sin θᵢ`, so that multiplying by the `sin θ` Jacobian recovers the exact rule.
    GaussLegendrePolar,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Axis {
    values: Vec<f64>,
    weights: Vec<f64>,
    period: Option<f64>,
    rule: QuadratureRule,
}

/// Gauss-Legendre nodes and weights on `[-1, 1]`, by Newton iteration on the Legendre recurrence.
pub fn gauss_legendre_nodes(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 1.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let p = if n == 0 { 1.0 } else if n == 1 { z } else { p1 };
            let prev = if n == 1 { 1.0 } else { p0 };
            dp = n as f64 * (z * p - prev) / (z * z - 1.0);
            let dz = p / dp;
            z -= dz;
            if dz.abs() < 1e-16 {
                break;
            }
        }
        let wi = 2.0 / ((1.0 - z * z) * dp * dp);
        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = wi;
        w[n - 1 - i] = wi;
    }
    (x, w)
}

impl Axis {
    fn check_increasing(values: &[f64]) -> Result<(), ErrorsJSL> {
        if values.is_empty() || values.iter().any(|v| !v.is_finite()) || values.windows(2).any(|w| w[1] <= w[0]) {
            return Err(ErrorsJSL::InvalidInputRange("Axis samples must be finite and strictly increasing."));
        }
        Ok(())
    }

    /// An axis on arbitrary increasing samples with trapezoid weights over `[first, last]`.
    pub fn from_values(values: Vec<f64>) -> Result<Self, ErrorsJSL> {
        Self::check_increasing(&values)?;
        let n = values.len();
        let weights = (0..n)
            .map(|i| {
                let left = if i > 0 { values[i] - values[i - 1] } else { 0.0 };
                let right = if i + 1 < n { values[i + 1] - values[i] } else { 0.0 };
                0.5 * (left + right)
            })
            .collect();
        Ok(Axis { values, weights, period: None, rule: QuadratureRule::Trapezoid })
    }

    /// `n >= 2` uniform samples from `start` to `end` inclusive, trapezoid weights.
    pub fn uniform(start: f64, end: f64, n: usize) -> Result<Self, ErrorsJSL> {
        if n < 2 || end.partial_cmp(&start) != Some(std::cmp::Ordering::Greater) {
            return Err(ErrorsJSL::InvalidInputRange("A uniform axis needs n >= 2 and end > start."));
        }
        Self::from_values((0..n).map(|i| start + (end - start) * i as f64 / (n - 1) as f64).collect())
    }

    /// Uniform samples with composite Simpson weights (`n` odd, at least 3).
    pub fn simpson(start: f64, end: f64, n: usize) -> Result<Self, ErrorsJSL> {
        if n < 3 || n.is_multiple_of(2) {
            return Err(ErrorsJSL::InvalidInputRange("Simpson's rule needs an odd number (>= 3) of samples."));
        }
        let mut axis = Self::uniform(start, end, n)?;
        let h = (end - start) / (n - 1) as f64;
        axis.weights = (0..n).map(|i| h / 3.0 * if i == 0 || i == n - 1 { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 }).collect();
        axis.rule = QuadratureRule::Simpson;
        Ok(axis)
    }

    /// `n` uniform samples covering one period `[offset, offset + period)` (the end point is the
    /// start point again, so it is not repeated).
    pub fn periodic(offset: f64, period: f64, n: usize) -> Result<Self, ErrorsJSL> {
        if n == 0 || period.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(ErrorsJSL::InvalidInputRange("A periodic axis needs n >= 1 and a positive period."));
        }
        let h = period / n as f64;
        Ok(Axis { values: (0..n).map(|i| offset + h * i as f64).collect(), weights: vec![h; n], period: Some(period), rule: QuadratureRule::Periodic })
    }

    /// Non-uniform samples within one period (trapezoid weights including the wrap-around gap).
    pub fn periodic_from_values(values: Vec<f64>, period: f64) -> Result<Self, ErrorsJSL> {
        Self::check_increasing(&values)?;
        if values[values.len() - 1] - values[0] >= period {
            return Err(ErrorsJSL::InvalidInputRange("Periodic samples must lie within one period."));
        }
        let n = values.len();
        let gap = |i: usize| if i + 1 < n { values[i + 1] - values[i] } else { values[0] + period - values[n - 1] };
        let weights = (0..n).map(|i| 0.5 * (gap(i) + gap((i + n - 1) % n))).collect();
        Ok(Axis { values, weights, period: Some(period), rule: QuadratureRule::Periodic })
    }

    /// `n` Gauss-Legendre nodes on `[start, end]`.
    pub fn gauss_legendre(start: f64, end: f64, n: usize) -> Result<Self, ErrorsJSL> {
        if n == 0 || end.partial_cmp(&start) != Some(std::cmp::Ordering::Greater) {
            return Err(ErrorsJSL::InvalidInputRange("Gauss-Legendre needs n >= 1 and end > start."));
        }
        let (x, w) = gauss_legendre_nodes(n);
        let (mid, half) = (0.5 * (start + end), 0.5 * (end - start));
        Ok(Axis { values: x.iter().map(|v| mid + half * v).collect(), weights: w.iter().map(|v| v * half).collect(), period: None, rule: QuadratureRule::GaussLegendre })
    }

    /// A polar (θ) axis on `(0, π)` with Gauss-Legendre nodes in `cos θ`; see
    /// [`QuadratureRule::GaussLegendrePolar`]. Exact on the sphere for polynomials in `cos θ`
    /// of degree up to `2n - 1`, and it never samples the poles.
    pub fn gauss_legendre_polar(n: usize) -> Result<Self, ErrorsJSL> {
        if n == 0 {
            return Err(ErrorsJSL::InvalidInputRange("Need at least one node."));
        }
        let (x, w) = gauss_legendre_nodes(n);
        // θ increases as cos θ decreases: reverse so the axis is increasing.
        let values: Vec<f64> = x.iter().rev().map(|v| v.acos()).collect();
        let weights: Vec<f64> = w.iter().rev().zip(&values).map(|(wi, t)| wi / t.sin()).collect();
        Ok(Axis { values, weights, period: None, rule: QuadratureRule::GaussLegendrePolar })
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn period(&self) -> Option<f64> {
        self.period
    }

    pub fn is_periodic(&self) -> bool {
        self.period.is_some()
    }

    pub fn rule(&self) -> QuadratureRule {
        self.rule
    }

    pub fn first(&self) -> f64 {
        self.values[0]
    }

    pub fn last(&self) -> f64 {
        self.values[self.values.len() - 1]
    }

    /// Whether every gap between samples (including the wrap gap on periodic axes) is equal.
    pub fn is_uniform(&self) -> bool {
        let n = self.len();
        if n < 2 {
            return true;
        }
        let h = self.values[1] - self.values[0];
        let ok = self.values.windows(2).all(|w| ((w[1] - w[0]) - h).abs() <= 1e-9 * h.abs().max(1e-300));
        ok && self.period.is_none_or(|p| ((self.values[0] + p - self.values[n - 1]) - h).abs() <= 1e-9 * h.abs())
    }

    /// Maps a possibly out-of-range index to a valid one: wrapping on periodic axes, `None`
    /// beyond the ends otherwise.
    pub fn wrap_index(&self, i: isize) -> Option<usize> {
        let n = self.len() as isize;
        if self.is_periodic() {
            Some(i.rem_euclid(n) as usize)
        } else if (0..n).contains(&i) {
            Some(i as usize)
        } else {
            None
        }
    }

    /// Reduces a coordinate into the axis's period (for periodic axes), starting at the first sample.
    pub fn reduce(&self, u: f64) -> f64 {
        match self.period {
            Some(p) => self.values[0] + (u - self.values[0]).rem_euclid(p),
            None => u,
        }
    }

    /// The cell containing `u`: index `i` and fraction `t` in `[0, 1]` such that
    /// `u = values[i] + t (values[i+1] - values[i])` (the "next" sample wraps on periodic axes).
    /// `None` outside a non-periodic axis's range (inclusive of the end points), or for a
    /// single-sample non-periodic axis unless `u` equals the sample.
    pub fn locate(&self, u: f64) -> Option<(usize, f64)> {
        let n = self.len();
        if !u.is_finite() {
            return None;
        }
        if let Some(p) = self.period {
            let u = self.reduce(u);
            let i = self.values.partition_point(|&v| v <= u).saturating_sub(1);
            let next = if i + 1 < n { self.values[i + 1] } else { self.values[0] + p };
            let width = next - self.values[i];
            return Some((i, if width > 0.0 { (u - self.values[i]) / width } else { 0.0 }));
        }
        // Accept points a rounding error outside the ends (e.g. after a rotation) and clamp them.
        let tol = 1e-9 * (self.values[n - 1] - self.values[0]).abs().max(self.values[0].abs()).max(self.values[n - 1].abs()).max(1e-300);
        if u < self.values[0] - tol || u > self.values[n - 1] + tol {
            return None;
        }
        let u = u.clamp(self.values[0], self.values[n - 1]);
        if n == 1 {
            return Some((0, 0.0));
        }
        let i = self.values.partition_point(|&v| v <= u).saturating_sub(1).min(n - 2);
        Some((i, (u - self.values[i]) / (self.values[i + 1] - self.values[i])))
    }

    /// The index of the sample closest to `u` (wrapping on periodic axes; clamped otherwise).
    pub fn nearest(&self, u: f64) -> usize {
        let n = self.len();
        if self.is_periodic() {
            let (i, t) = self.locate(u).unwrap_or((0, 0.0));
            return if t > 0.5 { (i + 1) % n } else { i };
        }
        let i = self.values.partition_point(|&v| v < u);
        if i == 0 {
            0
        } else if i == n || u - self.values[i - 1] <= self.values[i] - u {
            i - 1
        } else {
            i
        }
    }

    /// The sum of the weights: the length the rule integrates over.
    pub fn measure(&self) -> f64 {
        self.weights.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integrate(axis: &Axis, f: impl Fn(f64) -> f64) -> f64 {
        axis.values().iter().zip(axis.weights()).map(|(&u, &w)| w * f(u)).sum()
    }

    #[test]
    fn rules_integrate_what_they_should_exactly() {
        let t = Axis::uniform(0.0, 2.0, 11).unwrap();
        assert!((t.measure() - 2.0).abs() < 1e-14 && (integrate(&t, |x| 3.0 * x + 1.0) - 8.0).abs() < 1e-12);
        let nu = Axis::from_values(vec![0.0, 0.1, 0.5, 1.7, 3.0]).unwrap();
        assert!((integrate(&nu, |x| 2.0 * x - 1.0) - (9.0 - 3.0)).abs() < 1e-12); // linear functions exact
        let s = Axis::simpson(0.0, 1.0, 9).unwrap();
        assert!((integrate(&s, |x| x.powi(3) - x * x) - (0.25 - 1.0 / 3.0)).abs() < 1e-14); // cubics exact
        let g = Axis::gauss_legendre(-1.0, 3.0, 5).unwrap();
        assert!((integrate(&g, |x| x.powi(9)) - (3f64.powi(10) - 1.0) / 10.0).abs() < 1e-9); // degree 9 = 2n - 1
        let (x1, w1) = gauss_legendre_nodes(1);
        assert_eq!((x1, w1), (vec![0.0], vec![2.0]));
        let p = Axis::periodic(0.0, std::f64::consts::TAU, 16).unwrap();
        assert!((integrate(&p, |x| (3.0 * x).cos().powi(2)) - PI).abs() < 1e-12); // periodic: spectral
        assert!((p.measure() - std::f64::consts::TAU).abs() < 1e-12);
    }

    #[test]
    fn polar_gauss_legendre_integrates_on_the_sphere() {
        let a = Axis::gauss_legendre_polar(6).unwrap();
        assert!(a.values().windows(2).all(|w| w[1] > w[0]) && a.first() > 0.0 && a.last() < PI);
        // ∫ f(θ) sin θ dθ with the Jacobian supplied by the caller.
        let with_sin = |f: &dyn Fn(f64) -> f64| a.values().iter().zip(a.weights()).map(|(&t, &w)| w * t.sin() * f(t)).sum::<f64>();
        assert!((with_sin(&|_| 1.0) - 2.0).abs() < 1e-13);
        assert!((with_sin(&|t| t.cos().powi(2)) - 2.0 / 3.0).abs() < 1e-13);
        assert!((with_sin(&|t| t.cos().powi(11))).abs() < 1e-13);
    }

    #[test]
    fn periodic_axes_wrap_and_locate() {
        let p = Axis::periodic(0.0, 360.0, 8).unwrap();
        assert!(p.is_uniform() && p.is_periodic());
        assert_eq!((p.wrap_index(-1), p.wrap_index(8), p.wrap_index(3)), (Some(7), Some(0), Some(3)));
        assert_eq!(p.locate(350.0), Some((7, 35.0 / 45.0)));
        assert_eq!(p.locate(-10.0), p.locate(350.0));
        assert_eq!(p.nearest(359.0), 0);
        assert_eq!(p.reduce(725.0), 5.0);
        let np = Axis::periodic_from_values(vec![0.0, 1.0, 3.0], 4.0).unwrap();
        assert_eq!(np.weights(), &[1.0, 1.5, 1.5]);
        assert!(!np.is_uniform());
        assert!(Axis::periodic_from_values(vec![0.0, 4.0], 4.0).is_err());
    }

    #[test]
    fn bounded_axes_locate_and_validate() {
        let a = Axis::from_values(vec![0.0, 1.0, 4.0]).unwrap();
        assert_eq!((a.wrap_index(-1), a.wrap_index(3), a.wrap_index(2)), (None, None, Some(2)));
        assert_eq!(a.locate(2.5), Some((1, 0.5)));
        assert_eq!(a.locate(4.0), Some((1, 1.0)));
        assert_eq!((a.locate(-0.1), a.locate(4.1), a.locate(f64::NAN)), (None, None, None));
        assert_eq!((a.nearest(-5.0), a.nearest(2.4), a.nearest(2.6), a.nearest(9.0)), (0, 1, 2, 2));
        assert!(Axis::from_values(vec![1.0, 1.0]).is_err() && Axis::from_values(vec![]).is_err());
        assert!(Axis::uniform(1.0, 1.0, 3).is_err() && Axis::simpson(0.0, 1.0, 4).is_err());
        assert!(Axis::periodic(0.0, 0.0, 4).is_err() && Axis::gauss_legendre(0.0, 1.0, 0).is_err());
        let one = Axis::from_values(vec![2.0]).unwrap();
        assert_eq!((one.locate(2.0), one.locate(2.1), one.measure()), (Some((0, 0.0)), None, 0.0));
    }
}
