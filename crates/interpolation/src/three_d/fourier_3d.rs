//! Trigonometric interpolation of triply periodic gridded data.

use super::Interpolation3D;
use lib_jsl_core::*;
use ndarray::{Array3, Axis};
use num::Complex;
use std::f64::consts::PI;

/// The band-limited triply periodic trigonometric interpolant through
/// `v[[i, j, k]] = f(x0 + i px/nx, y0 + j py/ny, z0 + k pz/nz)`. Coefficients come from a
/// separable direct DFT (O(N·(nx+ny+nz)) for N samples); each evaluation costs O(N). Nyquist bins
/// of even-length axes keep their `+n/2` frequency, which leaves the interpolant real.
#[derive(Debug, Clone)]
pub struct FourierInterpolator3D {
    origin: [f64; 3],
    period: [f64; 3],
    coefficients: Array3<Complex<f64>>,
}

fn frequency(k: usize, n: usize) -> f64 {
    if k <= n / 2 { k as f64 } else { k as f64 - n as f64 }
}

fn dft_axis(data: &Array3<Complex<f64>>, axis: usize) -> Array3<Complex<f64>> {
    let n = data.len_of(Axis(axis));
    let mut out = data.clone();
    for (lane_in, mut lane_out) in data.lanes(Axis(axis)).into_iter().zip(out.lanes_mut(Axis(axis))) {
        for k in 0..n {
            lane_out[k] = lane_in.iter().enumerate().map(|(j, &v)| v * Complex::from_polar(1.0, -2.0 * PI * ((k * j) % n) as f64 / n as f64)).sum();
        }
    }
    out
}

impl FourierInterpolator3D {
    /// `origin` is the position of sample `[0, 0, 0]`, `period` the periods along each axis.
    pub fn new(v: &Array3<f64>, origin: [f64; 3], period: [f64; 3]) -> Result<Self, ErrorsJSL> {
        if v.is_empty() || period.iter().any(|&p| p <= 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("need samples and positive periods"));
        }
        let mut c = v.mapv(|r| Complex::new(r, 0.0));
        for axis in 0..3 {
            c = dft_axis(&c, axis);
        }
        let scale = 1.0 / v.len() as f64;
        c.mapv_inplace(|z| z * scale);
        Ok(FourierInterpolator3D { origin, period, coefficients: c })
    }
}

impl Interpolation3D for FourierInterpolator3D {
    fn interpolate_at(&self, x: f64, y: f64, z: f64) -> Result<f64, ErrorsJSL> {
        let (nx, ny, nz) = self.coefficients.dim();
        let n = [nx, ny, nz];
        let p = [x, y, z];
        let phasors: Vec<Vec<Complex<f64>>> = (0..3)
            .map(|d| {
                let theta = 2.0 * PI * (p[d] - self.origin[d]) / self.period[d];
                (0..n[d]).map(|k| Complex::from_polar(1.0, frequency(k, n[d]) * theta)).collect()
            })
            .collect();
        let mut sum = Complex::new(0.0, 0.0);
        for i in 0..nx {
            for j in 0..ny {
                let pij = phasors[0][i] * phasors[1][j];
                let line: Complex<f64> = (0..nz).map(|k| self.coefficients[[i, j, k]] * phasors[2][k]).sum();
                sum += pij * line;
            }
        }
        Ok(sum.re)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_bandlimited_function_and_samples() {
        let (n, period) = ([6usize, 5, 4], [2.0, 3.0, 1.5]);
        let origin = [0.1, -0.2, 0.3];
        let f = |a: f64, b: f64, c: f64| {
            1.0 + (2.0 * PI * (a - origin[0]) / period[0]).cos() * (2.0 * PI * (b - origin[1]) / period[1]).sin()
                + 0.3 * (2.0 * PI * (c - origin[2]) / period[2] + 2.0 * PI * (a - origin[0]) / period[0]).cos()
        };
        let v = Array3::from_shape_fn((n[0], n[1], n[2]), |(i, j, k)| {
            f(origin[0] + i as f64 * period[0] / n[0] as f64, origin[1] + j as f64 * period[1] / n[1] as f64, origin[2] + k as f64 * period[2] / n[2] as f64)
        });
        let fi = FourierInterpolator3D::new(&v, origin, period).unwrap();
        for &(a, b, c) in &[(0.4, 0.5, 0.9), (1.7, -1.1, 0.0), (-0.8, 2.2, 1.2)] {
            assert!((fi.interpolate_at(a, b, c).unwrap() - f(a, b, c)).abs() < 1e-11, "{a},{b},{c}");
        }
        let g = Array3::from_shape_fn((4, 4, 3), |(i, j, k)| ((i * 5 + j * 3 + k) % 6) as f64);
        let t = FourierInterpolator3D::new(&g, [0.0; 3], [4.0, 4.0, 3.0]).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..3 {
                    assert!((t.interpolate_at(i as f64, j as f64, k as f64).unwrap() - g[[i, j, k]]).abs() < 1e-11);
                }
            }
        }
        assert!(FourierInterpolator3D::new(&g, [0.0; 3], [1.0, 0.0, 1.0]).is_err());
    }
}
