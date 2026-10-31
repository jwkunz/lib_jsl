//! Sampling fields at arbitrary points and resampling them onto other domains.
//!
//! Interpolation is separable over a domain's axes with [`InterpolationMethod::Nearest`],
//! [`InterpolationMethod::Linear`] (two-point) or [`InterpolationMethod::Cubic`] (four-point
//! Lagrange, exact for cubics) stencils:
//!
//! - periodic (azimuth) axes wrap around;
//! - bounded axes shift the stencil inside the grid near the ends (falling back to fewer points
//!   when an axis is too short);
//! - polar axes on spherical domains continue *across the pole* onto the reflected samples
//!   `(θ', φ + π)` when the grid allows it (see [`Domain::across_pole`]), so queries between the
//!   pole and the first Gauss-Legendre node, or stencils straddling a pole, are handled smoothly.
//!
//! Vector fields are interpolated in **Cartesian components**, which are smooth everywhere
//! (spherical θ̂/φ̂ components flip sign across the poles), and the result is expressed in the
//! field's basis at the query point.

use crate::conventions::C64;
use crate::coordinates::systems::Position;
use crate::fields::domain::{Axis, Domain, DomainKind};
use crate::fields::field_container::{HarmonicField, Rank, TimeField};
use lib_jsl_core::ErrorsJSL;
use ndarray::{Array2, Array3};
use std::f64::consts::{PI, TAU};
use std::sync::Arc;

/// The interpolation stencil.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    Nearest,
    Linear,
    Cubic,
}

/// What resampling does with target points outside the source domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extrapolation {
    /// Fail with an error.
    Error,
    /// Use zero there.
    Zero,
}

/// How a stencil index along one axis maps to a sample.
#[derive(Debug, Clone, Copy)]
struct Tap {
    /// The index along the axis, possibly outside `0..len` (polar continuation).
    index: isize,
    weight: f64,
}

fn lagrange_weights(nodes: &[f64], u: f64) -> Vec<f64> {
    (0..nodes.len())
        .map(|a| (0..nodes.len()).filter(|&b| b != a).map(|b| (u - nodes[b]) / (nodes[a] - nodes[b])).product())
        .collect()
}

/// Turns candidate `(index, coordinate)` pairs and a cell into taps for the method.
fn taps_for(method: InterpolationMethod, coords: &dyn Fn(isize) -> f64, cell: isize, lo: isize, hi: isize, u: f64) -> Vec<Tap> {
    // `lo..=hi` is the range of indices that may be used.
    let available = (hi - lo + 1).max(0) as usize;
    let want = match method {
        InterpolationMethod::Nearest => 1,
        InterpolationMethod::Linear => 2,
        InterpolationMethod::Cubic => 4,
    }
    .min(available.max(1));
    if want == 1 {
        let (a, b) = (cell.clamp(lo, hi), (cell + 1).clamp(lo, hi));
        let pick = if (u - coords(a)).abs() <= (coords(b) - u).abs() { a } else { b };
        return vec![Tap { index: pick, weight: 1.0 }];
    }
    let start = if want == 2 { cell } else { cell - 1 }.clamp(lo, hi + 1 - want as isize);
    let idx: Vec<isize> = (start..start + want as isize).collect();
    let nodes: Vec<f64> = idx.iter().map(|&i| coords(i)).collect();
    lagrange_weights(&nodes, u).into_iter().zip(idx).map(|(w, index)| Tap { index, weight: w }).collect()
}

/// Virtual coordinate of index `k` on a polar axis continued across the poles.
fn polar_coordinate(axis: &Axis, k: isize) -> f64 {
    let n = axis.len() as isize;
    let v = axis.values();
    let north = axis.first().abs() < 1e-12;
    let south = (axis.last() - PI).abs() < 1e-12;
    if k < 0 {
        let m = if north { -k } else { -k - 1 };
        -v[m.clamp(0, n - 1) as usize]
    } else if k >= n {
        let over = k - (n - 1);
        let m = if south { n - 1 - over } else { n - over };
        TAU - v[m.clamp(0, n - 1) as usize]
    } else {
        v[k as usize]
    }
}

impl Domain {
    /// The coordinates of `p` along this domain's axes. Surface domains require the point to lie
    /// on the surface (to a relative tolerance of 1e-6); the far-field sphere only uses the
    /// direction.
    pub fn axis_coordinates(&self, p: &Position) -> Result<Vec<f64>, ErrorsJSL> {
        let q = p.in_system(self.system()).coords;
        let off = |actual: f64, expected: f64| (actual - expected).abs() > 1e-6 * expected.abs().max(1.0);
        match self.kind() {
            DomainKind::Plane { z0 } => {
                if off(q[2], z0) {
                    return Err(ErrorsJSL::InvalidInputRange("The point is not on the plane."));
                }
                Ok(vec![q[0], q[1]])
            }
            DomainKind::Sphere { radius } => {
                if off(q[0], radius) {
                    return Err(ErrorsJSL::InvalidInputRange("The point is not on the sphere."));
                }
                Ok(vec![q[1], q[2]])
            }
            DomainKind::FarFieldSphere => Ok(vec![q[1], q[2]]),
            DomainKind::Cylinder { radius } => {
                if off(q[0], radius) {
                    return Err(ErrorsJSL::InvalidInputRange("The point is not on the cylinder."));
                }
                Ok(vec![q[1], q[2]])
            }
            DomainKind::Box | DomainKind::SphericalShell | DomainKind::CylindricalVolume => Ok(q.to_vec()),
        }
    }

    fn polar_axis_index(&self) -> Option<usize> {
        match self.kind() {
            DomainKind::Sphere { .. } | DomainKind::FarFieldSphere => Some(0),
            DomainKind::SphericalShell => Some(1),
            _ => None,
        }
    }

    /// Whether the polar axis may be continued across its poles (needs φ + π on the grid).
    fn pole_continuation(&self) -> bool {
        self.polar_axis_index().is_some_and(|pa| {
            let n = self.axis(pa).len() as isize;
            self.across_pole(-1, 0).is_some() && self.across_pole(n, 0).is_some()
        })
    }

    /// The interpolation stencil at `p`: `(flat sample index, weight)` pairs with weights summing
    /// to one. Errors if `p` is outside the domain.
    pub fn stencil(&self, p: &Position, method: InterpolationMethod) -> Result<Vec<(usize, f64)>, ErrorsJSL> {
        let u = self.axis_coordinates(p)?;
        let polar = self.polar_axis_index();
        let continue_pole = self.pole_continuation();
        let mut per_axis: Vec<Vec<Tap>> = Vec::with_capacity(u.len());
        for (a, (axis, &ua)) in self.axes().iter().zip(&u).enumerate() {
            let n = axis.len() as isize;
            let taps = if let Some(period) = axis.period() {
                let ur = axis.reduce(ua);
                let (i, _) = axis.locate(ur).ok_or(ErrorsJSL::InvalidInputRange("The coordinate is not finite."))?;
                let coords = move |k: isize| axis.values()[k.rem_euclid(n) as usize] + period * k.div_euclid(n) as f64;
                let lo = if n >= 4 { -4 } else { 0 };
                let hi = if n >= 4 { n + 3 } else { n - 1 };
                taps_for(method, &coords, i as isize, lo, hi, ur)
            } else if Some(a) == polar && continue_pole {
                if !(0.0..=PI).contains(&ua) {
                    return Err(ErrorsJSL::InvalidInputRange("The polar angle is outside [0, π]."));
                }
                let coords = |k: isize| polar_coordinate(axis, k);
                // Find the cell in the extended coordinate list (indices -2..n+1).
                let cell = (-2..n + 1).rev().find(|&k| coords(k) <= ua).unwrap_or(-2);
                taps_for(method, &coords, cell, -2, n + 1, ua)
            } else {
                let (i, _) = axis.locate(ua).ok_or(ErrorsJSL::InvalidInputRange("The point is outside the domain."))?;
                let coords = |k: isize| axis.values()[k as usize];
                taps_for(method, &coords, i as isize, 0, n - 1, ua)
            };
            per_axis.push(taps);
        }
        // Tensor product, mapping out-of-range indices (wrap / across the pole) to samples.
        let mut out = Vec::new();
        let dims = per_axis.len();
        let mut counters = vec![0usize; dims];
        loop {
            let mut index: Vec<isize> = (0..dims).map(|d| per_axis[d][counters[d]].index).collect();
            let weight: f64 = (0..dims).map(|d| per_axis[d][counters[d]].weight).product();
            for (d, axis) in self.axes().iter().enumerate() {
                if axis.is_periodic() {
                    index[d] = index[d].rem_euclid(axis.len() as isize);
                }
            }
            if let Some(pa) = polar {
                let (t, f) = (index[pa], index[pa + 1]);
                let (t2, f2) = self.across_pole(t, f as usize).ok_or(ErrorsJSL::InvalidInputRange("The stencil cannot cross this pole."))?;
                index[pa] = t2 as isize;
                index[pa + 1] = f2 as isize;
            }
            let multi: Vec<usize> = index.iter().map(|&i| i as usize).collect();
            out.push((self.flatten(&multi), weight));
            // Advance the odometer.
            let mut d = dims;
            loop {
                if d == 0 {
                    return Ok(out);
                }
                d -= 1;
                counters[d] += 1;
                if counters[d] < per_axis[d].len() {
                    break;
                }
                counters[d] = 0;
            }
        }
    }
}

impl HarmonicField {
    /// The field at `p`: scalar fields give one value, vector fields three components in the
    /// field's basis at `p`.
    pub fn sample_at(&self, p: &Position, method: InterpolationMethod) -> Result<Vec<C64>, ErrorsJSL> {
        let stencil = self.domain().stencil(p, method)?;
        match self.rank() {
            Rank::Scalar => Ok(vec![stencil.iter().map(|&(i, w)| self.scalar(i) * w).sum()]),
            Rank::Vector(basis) => {
                let mut cart = [C64::new(0.0, 0.0); 3];
                for &(i, w) in &stencil {
                    let c = self.cartesian(i);
                    for k in 0..3 {
                        cart[k] += c[k] * w;
                    }
                }
                Ok(basis.from_cartesian(cart, p, self.convention()).to_vec())
            }
        }
    }

    /// The field resampled onto `domain` (same basis, quantity, frequency and convention).
    pub fn resample_to(&self, domain: Arc<Domain>, method: InterpolationMethod, outside: Extrapolation) -> Result<HarmonicField, ErrorsJSL> {
        let c = self.rank().components();
        let mut values = Array2::<C64>::zeros((domain.len(), c));
        for i in 0..domain.len() {
            let p = domain.position_flat(i);
            match self.sample_at(&p, method) {
                Ok(v) => (0..c).for_each(|k| values[[i, k]] = v[k]),
                Err(e) if outside == Extrapolation::Error => return Err(e),
                Err(_) => {}
            }
        }
        HarmonicField::new(domain, self.quantity(), self.rank(), self.frequency(), self.convention(), values)
    }
}

impl TimeField {
    /// The field at `p` for every time step, components in the field's basis at `p`.
    pub fn sample_at(&self, p: &Position, method: InterpolationMethod) -> Result<Vec<Vec<f64>>, ErrorsJSL> {
        let stencil = self.domain().stencil(p, method)?;
        let steps = self.time_axis().count;
        let mut out = Vec::with_capacity(steps);
        for t in 0..steps {
            match self.rank() {
                Rank::Scalar => out.push(vec![stencil.iter().map(|&(i, w)| self.values()[[t, i, 0]] * w).sum()]),
                Rank::Vector(basis) => {
                    let mut cart = [0.0; 3];
                    for &(i, w) in &stencil {
                        let v = [self.values()[[t, i, 0]], self.values()[[t, i, 1]], self.values()[[t, i, 2]]];
                        let c = basis.convert_real(v, crate::coordinates::PolarizationBasis::CARTESIAN, &self.domain().position_flat(i))?;
                        for k in 0..3 {
                            cart[k] += c[k] * w;
                        }
                    }
                    out.push(crate::coordinates::PolarizationBasis::CARTESIAN.convert_real(cart, basis, p)?.to_vec());
                }
            }
        }
        Ok(out)
    }

    /// The field resampled onto `domain`.
    pub fn resample_to(&self, domain: Arc<Domain>, method: InterpolationMethod, outside: Extrapolation) -> Result<TimeField, ErrorsJSL> {
        let (steps, c) = (self.time_axis().count, self.rank().components());
        let mut values = Array3::<f64>::zeros((steps, domain.len(), c));
        for i in 0..domain.len() {
            match self.sample_at(&domain.position_flat(i), method) {
                Ok(v) => {
                    for t in 0..steps {
                        for k in 0..c {
                            values[[t, i, k]] = v[t][k];
                        }
                    }
                }
                Err(e) if outside == Extrapolation::Error => return Err(e),
                Err(_) => {}
            }
        }
        TimeField::new(domain, self.quantity(), self.rank(), self.time_axis(), values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conventions::TimeConvention;
    use crate::coordinates::PolarizationBasis;
    use crate::fields::field_container::{FieldQuantity, TimeAxis};

    fn c(re: f64) -> C64 {
        C64::new(re, 0.0)
    }

    #[test]
    fn plane_interpolation_orders() {
        let d = Arc::new(Domain::plane(Axis::from_values(vec![0.0, 0.4, 1.0, 1.5, 2.5]).unwrap(), Axis::uniform(-1.0, 1.0, 6).unwrap(), 0.5).unwrap());
        let f = |x: f64, y: f64| 1.0 + 2.0 * x - y + 0.5 * x * y;
        let g = |x: f64, y: f64| x.powi(3) - 2.0 * x * y * y + y.powi(3);
        let lin = HarmonicField::scalar_from_fn(d.clone(), FieldQuantity::Generic, 1.0, TimeConvention::PlusJ, |p| c(f(p.coords[0], p.coords[1]))).unwrap();
        let cub = HarmonicField::scalar_from_fn(d.clone(), FieldQuantity::Generic, 1.0, TimeConvention::PlusJ, |p| c(g(p.coords[0], p.coords[1]))).unwrap();
        for (x, y) in [(0.1, 0.3), (1.2, -0.9), (2.4, 0.95), (0.0, -1.0), (2.5, 1.0)] {
            let p = Position::cartesian(x, y, 0.5);
            assert!((lin.sample_at(&p, InterpolationMethod::Linear).unwrap()[0].re - f(x, y)).abs() < 1e-12);
            assert!((cub.sample_at(&p, InterpolationMethod::Cubic).unwrap()[0].re - g(x, y)).abs() < 1e-11, "({x},{y})");
        }
        let near = lin.sample_at(&Position::cartesian(0.39, -0.95, 0.5), InterpolationMethod::Nearest).unwrap()[0].re;
        assert!((near - f(0.4, -1.0)).abs() < 1e-12);
        assert!(lin.sample_at(&Position::cartesian(3.0, 0.0, 0.5), InterpolationMethod::Linear).is_err()); // outside
        assert!(lin.sample_at(&Position::cartesian(1.0, 0.0, 0.7), InterpolationMethod::Linear).is_err()); // off the plane
        let stencil = d.stencil(&Position::cartesian(1.1, 0.1, 0.5), InterpolationMethod::Cubic).unwrap();
        assert_eq!(stencil.len(), 16);
        assert!((stencil.iter().map(|s| s.1).sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn vector_fields_interpolate_smoothly_through_the_poles() {
        // A constant ẑ field stored in the spherical basis: (cos θ, -sin θ, 0). Cartesian
        // interpolation reproduces it exactly anywhere, including at and near the poles.
        for d in [Domain::far_field_uniform(13, 24).unwrap(), Domain::far_field_gauss(8, 16).unwrap()] {
            let d = Arc::new(d);
            let z = HarmonicField::vector_from_fn(d.clone(), FieldQuantity::ElectricField, PolarizationBasis::SPHERICAL, 1.0, TimeConvention::PlusJ, |p| {
                let (s, ct) = p.coords[1].sin_cos();
                [c(ct), c(-s), c(0.0)]
            })
            .unwrap();
            for method in [InterpolationMethod::Linear, InterpolationMethod::Cubic] {
                for (t, ph) in [(0.0, 1.0), (0.01, 4.0), (1.3, 6.2), (PI - 0.005, 0.3), (PI, 2.0)] {
                    let v = z.sample_at(&Position::spherical(1.0, t, ph), method).unwrap();
                    let (s, ct) = f64::sin_cos(t);
                    assert!((v[0] - c(ct)).norm() < 1e-12 && (v[1] - c(-s)).norm() < 1e-12 && v[2].norm() < 1e-12, "{method:?} θ={t} φ={ph}: {v:?}");
                }
            }
        }
    }

    #[test]
    fn smooth_scalars_on_the_sphere_converge_with_order_and_wrap_in_azimuth() {
        let d = Arc::new(Domain::far_field_gauss(16, 32).unwrap());
        let f = |t: f64, p: f64| (t.sin() * p.cos()).powi(2) + t.cos().powi(3) + (t.sin() * p.sin());
        let field = HarmonicField::scalar_from_fn(d.clone(), FieldQuantity::Generic, 1.0, TimeConvention::PlusJ, |pos| c(f(pos.coords[1], pos.coords[2]))).unwrap();
        let err = |m| {
            let mut worst: f64 = 0.0;
            for k in 0..60 {
                let (t, p) = (0.02 + 3.1 * (k as f64 * 0.618).fract(), TAU * (k as f64 * 0.754).fract());
                worst = worst.max((field.sample_at(&Position::spherical(1.0, t, p), m).unwrap()[0].re - f(t, p)).abs());
            }
            worst
        };
        let (lin, cub) = (err(InterpolationMethod::Linear), err(InterpolationMethod::Cubic));
        assert!(cub < lin / 5.0 && cub < 5e-3, "linear {lin}, cubic {cub}");
        // φ = 2π - ε and φ = -ε are the same point.
        let a = field.sample_at(&Position::spherical(1.0, 1.0, TAU - 1e-3), InterpolationMethod::Cubic).unwrap()[0];
        let b = field.sample_at(&Position::spherical(1.0, 1.0, -1e-3), InterpolationMethod::Cubic).unwrap()[0];
        assert!((a - b).norm() < 1e-12);
        // Resampling to another grid preserves the integral closely.
        let uniform = Arc::new(Domain::far_field_uniform(91, 64).unwrap());
        let moved = field.resample_to(uniform, InterpolationMethod::Cubic, Extrapolation::Error).unwrap();
        let (i1, i2) = (field.integrate().unwrap().re, moved.integrate().unwrap().re);
        assert!((i1 - i2).abs() < 1e-3 * i1.abs(), "{i1} {i2}");
    }

    #[test]
    fn volumes_extrapolation_and_time_fields() {
        let d = Arc::new(Domain::cartesian_box(Axis::uniform(0.0, 1.0, 4).unwrap(), Axis::uniform(0.0, 1.0, 4).unwrap(), Axis::uniform(0.0, 2.0, 5).unwrap()).unwrap());
        let t = TimeAxis::new(0.0, 0.5, 3).unwrap();
        let tf = TimeField::from_fn(d.clone(), FieldQuantity::ElectricField, Rank::Vector(PolarizationBasis::CARTESIAN), t, |p, time| vec![p.coords[0] + time, p.coords[1] * p.coords[2], 1.0]).unwrap();
        let v = tf.sample_at(&Position::cartesian(0.3, 0.6, 1.1), InterpolationMethod::Linear).unwrap();
        assert!((v[2][0] - 1.3).abs() < 1e-12 && (v[1][1] - 0.66).abs() < 1e-12 && (v[0][2] - 1.0).abs() < 1e-12);
        // The same field read in the cylindrical basis at a point on the y axis: ρ̂ = ŷ, φ̂ = -x̂.
        let cyl = TimeField::from_fn(d.clone(), FieldQuantity::ElectricField, Rank::Vector(PolarizationBasis::CYLINDRICAL), t, |_, _| vec![1.0, 0.0, 0.0]).unwrap();
        let s = cyl.sample_at(&Position::cylindrical(0.5, PI / 2.0, 1.0), InterpolationMethod::Linear).unwrap();
        assert!((s[0][0] - 1.0).abs() < 1e-9 && s[0][1].abs() < 1e-9);
        // Resampling onto a larger box: Error refuses, Zero fills.
        let big = Arc::new(Domain::cartesian_box(Axis::uniform(0.0, 2.0, 3).unwrap(), Axis::uniform(0.0, 1.0, 2).unwrap(), Axis::uniform(0.0, 2.0, 2).unwrap()).unwrap());
        assert!(tf.resample_to(big.clone(), InterpolationMethod::Linear, Extrapolation::Error).is_err());
        let filled = tf.resample_to(big, InterpolationMethod::Linear, Extrapolation::Zero).unwrap();
        assert_eq!(filled.at(0, filled.domain().flatten(&[2, 0, 0])), vec![0.0, 0.0, 0.0]);
        // Short axes fall back to lower-order stencils instead of failing.
        let tiny = Arc::new(Domain::cartesian_box(Axis::uniform(0.0, 1.0, 2).unwrap(), Axis::uniform(0.0, 1.0, 2).unwrap(), Axis::uniform(0.0, 1.0, 2).unwrap()).unwrap());
        assert_eq!(tiny.stencil(&Position::cartesian(0.5, 0.5, 0.5), InterpolationMethod::Cubic).unwrap().len(), 8);
    }
}
