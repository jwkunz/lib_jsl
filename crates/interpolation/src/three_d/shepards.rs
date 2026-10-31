//! Shepard's inverse-distance weighting for scattered 3D data.

use super::Interpolation3D;
use lib_jsl_core::*;

/// `s(p) = Σ w_i f_i / Σ w_i` with `w_i = |p - p_i|^-power`; exact at the data points, a convex
/// combination of the data elsewhere (never over- or undershoots), flat near each point.
#[derive(Debug, Clone)]
pub struct Shepards {
    points: Vec<[f64; 3]>,
    values: Vec<f64>,
    power: f64,
}

impl Shepards {
    /// `power` is the distance exponent (2 is the classic choice; larger values localise).
    pub fn new(x: VR1D, y: VR1D, z: VR1D, f: VR1D, power: f64) -> Result<Self, ErrorsJSL> {
        let n = x.len();
        if n == 0 || y.len() != n || z.len() != n || f.len() != n {
            return Err(ErrorsJSL::IncompatibleArraySizes((n, f.len())));
        }
        if power <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("power must be positive"));
        }
        Ok(Shepards { points: (0..n).map(|i| [x[i], y[i], z[i]]).collect(), values: f.to_vec(), power })
    }
}

impl Interpolation3D for Shepards {
    fn interpolate_at(&self, x: f64, y: f64, z: f64) -> Result<f64, ErrorsJSL> {
        let (mut num, mut den) = (0.0, 0.0);
        for (p, &f) in self.points.iter().zip(&self.values) {
            let d2 = (x - p[0]).powi(2) + (y - p[1]).powi(2) + (z - p[2]).powi(2);
            if d2 == 0.0 {
                return Ok(f);
            }
            let w = d2.powf(-0.5 * self.power);
            num += w * f;
            den += w;
        }
        Ok(num / den)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_at_points_and_bounded_elsewhere() {
        let x = R1D::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        let y = R1D::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let z = R1D::from_vec(vec![0.0, 1.0, 1.0, 0.0]);
        let f = R1D::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let s = Shepards::new(x.view(), y.view(), z.view(), f.view(), 2.0).unwrap();
        for i in 0..4 {
            assert_eq!(s.interpolate_at(x[i], y[i], z[i]).unwrap(), f[i]);
        }
        let v = s.interpolate_at(0.3, 0.6, 0.2).unwrap();
        assert!((1.0..=4.0).contains(&v));
        // Symmetric centre of the tetrahedron-like point set weights all equally.
        assert!((s.interpolate_at(0.5, 0.5, 0.5).unwrap() - 2.5).abs() < 1e-12);
        assert!(Shepards::new(x.view(), y.view(), z.view(), f.view(), 0.0).is_err());
    }
}
