//! Differential operators on surface domains: surface gradient, divergence and curl, and the
//! Laplace-Beltrami operator.
//!
//! On a surface parametrised by its two axes `(u₁, u₂)` with scale factors `(h₁, h₂)` and unit
//! tangents `û₁ × û₂ = n̂` (plane: `x̂, ŷ, ẑ`; sphere: `θ̂, φ̂, r̂`; cylinder: `φ̂, ẑ, ρ̂`):
//!
//! - `∇ₛ f = (1/h₁) ∂f/∂u₁ û₁ + (1/h₂) ∂f/∂u₂ û₂`
//! - `∇ₛ·F = 1/(h₁h₂) [∂(h₂F₁)/∂u₁ + ∂(h₁F₂)/∂u₂]`
//! - `n̂·∇×F = 1/(h₁h₂) [∂(h₂F₂)/∂u₁ - ∂(h₁F₁)/∂u₂]` (the surface curl, a scalar)
//! - `∇ₛ×(f n̂) = ∇ₛf × n̂` (the vector surface curl of a scalar)
//! - `Δₛ f = ∇ₛ·∇ₛ f` (Laplace-Beltrami)
//!
//! Vector operators act on the *tangential* part of a field (any normal component is ignored),
//! with derivatives from the same Lagrange stencils as the volume operators. The far-field sphere
//! is treated as the unit sphere of directions (angular operators, `h = (1, sin θ)`). Samples where
//! a scale factor vanishes (the poles of a uniform sphere grid) are returned as NaN; Gauss-Legendre
//! polar grids avoid them. Vector results are in the linear basis of the domain's system.
//!
//! See the book entry [Fields on grids](https://jwkunz.github.io/lib_jsl/book/em-arrays/fields.html).

use crate::conventions::C64;
use crate::coordinates::polarization::PolarizationBasis;
use crate::fields::domain::{Domain, DomainKind};
use crate::fields::field_container::{FieldQuantity, HarmonicField, Rank, TimeField};
use crate::fields::interpolation::polar_coordinate;
use crate::fields::vector_calculus::{FieldScalar, axis_stencil, derivative_weights};
use lib_jsl_core::ErrorsJSL;
use ndarray::{Array2, Array3};

/// Indices of the tangential components `(û₁, û₂)` and of the normal one in the domain's
/// linear basis.
fn component_layout(domain: &Domain) -> Result<([usize; 2], usize), ErrorsJSL> {
    match domain.kind() {
        DomainKind::Plane { .. } => Ok(([0, 1], 2)),
        DomainKind::Sphere { .. } | DomainKind::FarFieldSphere | DomainKind::Cylinder { .. } => Ok(([1, 2], 0)),
        _ => Err(ErrorsJSL::InvalidInputRange("Surface operators need a plane, sphere or cylinder domain.")),
    }
}

/// The scale factors `(h₁, h₂)` of the surface axes at a sample.
fn surface_scales(domain: &Domain, idx: &[usize]) -> [f64; 2] {
    let p = domain.position(idx).coords;
    match domain.kind() {
        DomainKind::Sphere { radius } => [radius, radius * p[1].sin()],
        DomainKind::FarFieldSphere => [1.0, p[1].sin()],
        DomainKind::Cylinder { radius } => [radius, 1.0],
        _ => [1.0, 1.0],
    }
}

/// The θ stencil on a sphere, continued across the poles onto the reflected samples
/// `(θ', φ + π)` when the grid allows it. The quantities differentiated along θ here (`f`,
/// `h₂F₁`, `h₂F₂`) are even under that reflection (the sign flips of `sin θ` and of `θ̂`, `φ̂`
/// cancel), so the reflected samples are used as they are.
fn polar_stencil(domain: &Domain, idx: &[usize]) -> Option<Vec<(usize, f64)>> {
    let axis = domain.axis(0);
    let n = axis.len() as isize;
    if n < 5 || domain.across_pole(-1, 0).is_none() || domain.across_pole(n, 0).is_none() {
        return None;
    }
    let i = idx[0] as isize;
    let window: Vec<isize> = (i - 2..=i + 2).collect();
    let nodes: Vec<f64> = window.iter().map(|&k| polar_coordinate(axis, k)).collect();
    let weights = derivative_weights(&nodes, axis.values()[idx[0]]);
    window
        .iter()
        .zip(weights)
        .map(|(&k, w)| domain.across_pole(k, idx[1]).map(|(t, p)| (domain.flatten(&[t, p]), w)))
        .collect()
}

fn derivative<T: FieldScalar>(domain: &Domain, idx: &[usize], axis: usize, value: &dyn Fn(usize) -> T) -> T {
    let spherical = matches!(domain.kind(), DomainKind::Sphere { .. } | DomainKind::FarFieldSphere);
    let stencil = if spherical && axis == 0 { polar_stencil(domain, idx) } else { None }.unwrap_or_else(|| axis_stencil(domain, idx, axis));
    stencil.into_iter().fold(T::zero(), |acc, (k, w)| acc + value(k) * w)
}

fn gradient_core<T: FieldScalar>(domain: &Domain, f: &[T]) -> Vec<[T; 2]> {
    (0..domain.len())
        .map(|flat| {
            let idx = domain.unflatten(flat);
            let h = surface_scales(domain, &idx);
            if h.iter().any(|v| v.abs() < 1e-300) {
                return [T::nan(); 2];
            }
            [derivative(domain, &idx, 0, &|k| f[k]) * (1.0 / h[0]), derivative(domain, &idx, 1, &|k| f[k]) * (1.0 / h[1])]
        })
        .collect()
}

/// `∇ₛ·F` (`curl = false`) or `n̂·∇×F` (`curl = true`) of tangential components.
fn divergence_or_curl_core<T: FieldScalar>(domain: &Domain, t: &[[T; 2]], curl: bool) -> Vec<T> {
    let scales: Vec<[f64; 2]> = (0..domain.len()).map(|k| surface_scales(domain, &domain.unflatten(k))).collect();
    (0..domain.len())
        .map(|flat| {
            let idx = domain.unflatten(flat);
            let [h1, h2] = scales[flat];
            if h1.abs() < 1e-300 || h2.abs() < 1e-300 {
                return T::nan();
            }
            let inv = 1.0 / (h1 * h2);
            if curl {
                (derivative(domain, &idx, 0, &|k| t[k][1] * scales[k][1]) - derivative(domain, &idx, 1, &|k| t[k][0] * scales[k][0])) * inv
            } else {
                (derivative(domain, &idx, 0, &|k| t[k][0] * scales[k][1]) + derivative(domain, &idx, 1, &|k| t[k][1] * scales[k][0])) * inv
            }
        })
        .collect()
}

fn native_basis(domain: &Domain) -> PolarizationBasis {
    PolarizationBasis::Linear(domain.system())
}

/// Tangential components of a harmonic vector field in the domain's native basis.
fn harmonic_tangential(field: &HarmonicField) -> Result<Vec<[C64; 2]>, ErrorsJSL> {
    let ([a, b], _) = component_layout(field.domain())?;
    let native = field.to_basis(native_basis(field.domain()))?;
    Ok((0..native.len()).map(|i| [native.values()[[i, a]], native.values()[[i, b]]]).collect())
}

fn harmonic_from_tangential(template: &HarmonicField, t: Vec<[C64; 2]>) -> Result<HarmonicField, ErrorsJSL> {
    let domain = template.domain().clone();
    let ([a, b], _) = component_layout(&domain)?;
    let mut values = Array2::<C64>::zeros((domain.len(), 3));
    for (i, v) in t.iter().enumerate() {
        values[[i, a]] = v[0];
        values[[i, b]] = v[1];
    }
    HarmonicField::new(domain.clone(), FieldQuantity::Generic, Rank::Vector(native_basis(&domain)), template.frequency(), template.convention(), values)
}

fn harmonic_scalar(template: &HarmonicField, v: Vec<C64>) -> Result<HarmonicField, ErrorsJSL> {
    let domain = template.domain().clone();
    let values = Array2::from_shape_fn((domain.len(), 1), |(i, _)| v[i]);
    HarmonicField::new(domain, FieldQuantity::Generic, Rank::Scalar, template.frequency(), template.convention(), values)
}

impl HarmonicField {
    /// The surface gradient `∇ₛ f` of a scalar field on a surface.
    pub fn surface_gradient(&self) -> Result<HarmonicField, ErrorsJSL> {
        component_layout(self.domain())?;
        if !self.is_scalar() {
            return Err(ErrorsJSL::InvalidInputRange("The surface gradient needs a scalar field."));
        }
        let f: Vec<C64> = (0..self.len()).map(|i| self.scalar(i)).collect();
        harmonic_from_tangential(self, gradient_core(self.domain(), &f))
    }

    /// The surface divergence `∇ₛ·F` of the tangential part of a vector field.
    pub fn surface_divergence(&self) -> Result<HarmonicField, ErrorsJSL> {
        if self.is_scalar() {
            return Err(ErrorsJSL::InvalidInputRange("The surface divergence needs a vector field."));
        }
        let t = harmonic_tangential(self)?;
        harmonic_scalar(self, divergence_or_curl_core(self.domain(), &t, false))
    }

    /// The surface curl `n̂·∇×F` of the tangential part of a vector field (a scalar field).
    pub fn surface_curl(&self) -> Result<HarmonicField, ErrorsJSL> {
        if self.is_scalar() {
            return Err(ErrorsJSL::InvalidInputRange("The surface curl needs a vector field."));
        }
        let t = harmonic_tangential(self)?;
        harmonic_scalar(self, divergence_or_curl_core(self.domain(), &t, true))
    }

    /// The vector surface curl `∇ₛ×(f n̂) = ∇ₛf × n̂` of a scalar field (tangential).
    pub fn surface_vector_curl(&self) -> Result<HarmonicField, ErrorsJSL> {
        let g = self.surface_gradient()?;
        let t = harmonic_tangential(&g)?;
        harmonic_from_tangential(self, t.into_iter().map(|[g1, g2]| [g2, -g1]).collect())
    }

    /// The Laplace-Beltrami operator `Δₛ f = ∇ₛ·∇ₛ f` of a scalar field.
    pub fn laplace_beltrami(&self) -> Result<HarmonicField, ErrorsJSL> {
        self.surface_gradient()?.surface_divergence()
    }
}

impl TimeField {
    fn per_step<F: Fn(&Domain, usize) -> Result<Vec<[f64; 3]>, ErrorsJSL>>(&self, rank: Rank, f: F) -> Result<TimeField, ErrorsJSL> {
        let domain = self.domain().clone();
        let (steps, n) = (self.time_axis().count, domain.len());
        let c = rank.components();
        let mut values = Array3::<f64>::zeros((steps, n, c));
        for t in 0..steps {
            let out = f(&domain, t)?;
            for i in 0..n {
                for k in 0..c {
                    values[[t, i, k]] = out[i][k];
                }
            }
        }
        TimeField::new(domain, FieldQuantity::Generic, rank, self.time_axis(), values)
    }

    fn tangential_step(&self, t: usize) -> Result<Vec<[f64; 2]>, ErrorsJSL> {
        let ([a, b], _) = component_layout(self.domain())?;
        let Rank::Vector(basis) = self.rank() else {
            return Err(ErrorsJSL::InvalidInputRange("This operator needs a vector field."));
        };
        let target = native_basis(self.domain());
        (0..self.domain().len())
            .map(|i| {
                let v = basis.convert_real([self.values()[[t, i, 0]], self.values()[[t, i, 1]], self.values()[[t, i, 2]]], target, &self.domain().position_flat(i))?;
                Ok([v[a], v[b]])
            })
            .collect()
    }

    /// `∇ₛ f` of a scalar time field at every step.
    pub fn surface_gradient(&self) -> Result<TimeField, ErrorsJSL> {
        let ([a, b], _) = component_layout(self.domain())?;
        if self.rank() != Rank::Scalar {
            return Err(ErrorsJSL::InvalidInputRange("The surface gradient needs a scalar field."));
        }
        self.per_step(Rank::Vector(native_basis(self.domain())), |d, t| {
            let f: Vec<f64> = (0..d.len()).map(|i| self.values()[[t, i, 0]]).collect();
            Ok(gradient_core(d, &f)
                .into_iter()
                .map(|g| {
                    let mut v = [0.0; 3];
                    v[a] = g[0];
                    v[b] = g[1];
                    v
                })
                .collect())
        })
    }

    /// `∇ₛ·F` of the tangential part of a vector time field.
    pub fn surface_divergence(&self) -> Result<TimeField, ErrorsJSL> {
        component_layout(self.domain())?;
        self.per_step(Rank::Scalar, |d, t| Ok(divergence_or_curl_core(d, &self.tangential_step(t)?, false).into_iter().map(|v| [v, 0.0, 0.0]).collect()))
    }

    /// `n̂·∇×F` of the tangential part of a vector time field.
    pub fn surface_curl(&self) -> Result<TimeField, ErrorsJSL> {
        component_layout(self.domain())?;
        self.per_step(Rank::Scalar, |d, t| Ok(divergence_or_curl_core(d, &self.tangential_step(t)?, true).into_iter().map(|v| [v, 0.0, 0.0]).collect()))
    }

    /// `Δₛ f` of a scalar time field.
    pub fn laplace_beltrami(&self) -> Result<TimeField, ErrorsJSL> {
        self.surface_gradient()?.surface_divergence()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conventions::TimeConvention;
    use crate::fields::domain::Axis;
    use crate::fields::field_container::TimeAxis;
    use std::f64::consts::TAU;
    use std::sync::Arc;

    fn c(re: f64) -> C64 {
        C64::new(re, 0.0)
    }

    fn sphere(radius: f64) -> Arc<Domain> {
        Arc::new(Domain::sphere(radius, Axis::gauss_legendre_polar(48).unwrap(), Axis::periodic(0.0, TAU, 96).unwrap()).unwrap())
    }

    fn scalar(d: Arc<Domain>, f: impl Fn(&crate::coordinates::systems::Position) -> f64) -> HarmonicField {
        HarmonicField::scalar_from_fn(d, FieldQuantity::Generic, 1.0, TimeConvention::PlusJ, |p| c(f(p))).unwrap()
    }

    #[test]
    fn spherical_harmonics_are_laplace_beltrami_eigenfunctions() {
        let r = 2.0;
        let d = sphere(r);
        for (l, f) in [(1, Box::new(|p: &crate::coordinates::systems::Position| p.coords[1].sin() * p.coords[2].cos()) as Box<dyn Fn(&_) -> f64>), (2, Box::new(|p: &crate::coordinates::systems::Position| 3.0 * p.coords[1].cos().powi(2) - 1.0))] {
            let field = scalar(d.clone(), &f);
            let lap = field.laplace_beltrami().unwrap();
            let k = -(l * (l + 1)) as f64 / (r * r);
            let worst = (0..d.len()).map(|i| (lap.scalar(i) - field.scalar(i) * k).norm()).fold(0.0, f64::max);
            assert!(worst < 2e-3, "l = {l}: {worst}");
        }
    }

    #[test]
    fn gradient_divergence_theorem_and_curl_on_the_sphere() {
        let r = 1.5;
        let d = sphere(r);
        // z = R cos θ: ∇ₛz = -sin θ θ̂ (the tangential part of ẑ).
        let g = scalar(d.clone(), |p| r * p.coords[1].cos()).surface_gradient().unwrap();
        assert_eq!(g.basis(), Some(PolarizationBasis::SPHERICAL));
        for i in (0..d.len()).step_by(31) {
            let t = d.position_flat(i).coords[1];
            assert!((g.vector(i)[1] - c(-t.sin())).norm() < 1e-5 && g.vector(i)[2].norm() < 1e-9 && g.vector(i)[0].norm() == 0.0);
        }
        // The surface divergence integrates to zero over a closed surface.
        let f = HarmonicField::vector_from_fn(d.clone(), FieldQuantity::Generic, PolarizationBasis::SPHERICAL, 1.0, TimeConvention::PlusJ, |p| {
            let (t, ph) = (p.coords[1], p.coords[2]);
            [c(5.0), c(t.sin() * ph.cos() + t.sin().powi(2)), c((2.0 * t).sin() * ph.sin())] // the radial part is ignored
        })
        .unwrap();
        assert!(f.surface_divergence().unwrap().integrate().unwrap().norm() < 1e-6);
        // Surface curl of a gradient vanishes; ∇ₛ·(∇ₛf × n̂) vanishes.
        let h = scalar(d.clone(), |p| p.coords[1].cos().powi(3) + p.coords[1].sin() * p.coords[2].sin());
        assert!(h.surface_gradient().unwrap().surface_curl().unwrap().values().iter().all(|v| v.norm() < 1e-3));
        assert!(h.surface_vector_curl().unwrap().surface_divergence().unwrap().values().iter().all(|v| v.norm() < 1e-3));
        // A rigid rotation about z, F = R sin θ φ̂, has surface curl 2 cos θ.
        let rot = HarmonicField::vector_from_fn(d.clone(), FieldQuantity::Generic, PolarizationBasis::SPHERICAL, 1.0, TimeConvention::PlusJ, |p| [c(0.0), c(0.0), c(r * p.coords[1].sin())]).unwrap();
        let curl = rot.surface_curl().unwrap();
        assert!((0..d.len()).all(|i| (curl.scalar(i) - c(2.0 * d.position_flat(i).coords[1].cos())).norm() < 1e-4));
    }

    #[test]
    fn plane_cylinder_and_far_field_domains() {
        let plane = Arc::new(Domain::plane(Axis::uniform(-1.0, 1.0, 9).unwrap(), Axis::from_values(vec![0.0, 0.2, 0.5, 0.9, 1.0, 1.4]).unwrap(), 0.3).unwrap());
        let f = scalar(plane.clone(), |p| p.coords[0] * p.coords[0] * p.coords[1]);
        assert!((0..plane.len()).all(|i| {
            let [x, y, _] = plane.position_flat(i).coords;
            (f.laplace_beltrami().unwrap().scalar(i) - c(2.0 * y)).norm() < 1e-9 && (f.surface_gradient().unwrap().vector(i)[0] - c(2.0 * x * y)).norm() < 1e-9
        }));
        let swirl = HarmonicField::vector_from_fn(plane.clone(), FieldQuantity::Generic, PolarizationBasis::CARTESIAN, 1.0, TimeConvention::PlusJ, |p| [c(-p.coords[1]), c(p.coords[0]), c(7.0)]).unwrap();
        assert!(swirl.surface_curl().unwrap().values().iter().all(|v| (v - c(2.0)).norm() < 1e-9));
        assert!(swirl.surface_divergence().unwrap().values().iter().all(|v| v.norm() < 1e-9));
        // Cylinder of radius 0.5: ∇ₛ z = ẑ, and F = z φ̂ has surface curl -1 (n̂ = ρ̂).
        let cyl = Arc::new(Domain::cylinder(0.5, Axis::periodic(0.0, TAU, 32).unwrap(), Axis::uniform(0.0, 2.0, 9).unwrap()).unwrap());
        let gz = scalar(cyl.clone(), |p| p.coords[2]).surface_gradient().unwrap();
        assert!((0..cyl.len()).all(|i| (gz.vector(i)[2] - c(1.0)).norm() < 1e-9 && gz.vector(i)[1].norm() < 1e-9));
        let zphi = HarmonicField::vector_from_fn(cyl.clone(), FieldQuantity::Generic, PolarizationBasis::CYLINDRICAL, 1.0, TimeConvention::PlusJ, |p| [c(0.0), c(p.coords[2]), c(0.0)]).unwrap();
        assert!(zphi.surface_curl().unwrap().values().iter().all(|v| (v - c(-1.0)).norm() < 1e-9));
        // Far-field sphere: the angular Laplacian of Y₂⁰ is -6 Y₂⁰.
        let ff = Arc::new(Domain::far_field_gauss(48, 96).unwrap());
        let y20 = scalar(ff.clone(), |p| 3.0 * p.coords[1].cos().powi(2) - 1.0);
        let lap = y20.laplace_beltrami().unwrap();
        assert!((0..ff.len()).all(|i| (lap.scalar(i) + y20.scalar(i) * 6.0).norm() < 2e-3));
    }

    #[test]
    fn poles_volumes_and_time_fields() {
        // A uniform grid samples the poles, where h_φ = 0: those results are NaN.
        let uni = Arc::new(Domain::far_field_uniform(19, 24).unwrap());
        let g = scalar(uni.clone(), |p| p.coords[1].cos()).surface_gradient().unwrap();
        assert!(g.vector(0)[1].re.is_nan() && g.vector(uni.flatten(&[9, 3]))[1].re.is_finite());
        let vol = Arc::new(Domain::cartesian_box(Axis::uniform(0.0, 1.0, 3).unwrap(), Axis::uniform(0.0, 1.0, 3).unwrap(), Axis::uniform(0.0, 1.0, 3).unwrap()).unwrap());
        assert!(scalar(vol, |_| 1.0).surface_gradient().is_err());
        let d = sphere(1.0);
        assert!(scalar(d.clone(), |_| 1.0).surface_divergence().is_err());
        // Time fields agree with the harmonic operators at each step.
        let t = TimeAxis::new(0.0, 1.0, 2).unwrap();
        let tf = TimeField::from_fn(d.clone(), FieldQuantity::Generic, Rank::Scalar, t, |p, time| vec![(1.0 + time) * (3.0 * p.coords[1].cos().powi(2) - 1.0)]).unwrap();
        let lap = tf.laplace_beltrami().unwrap();
        let i = d.flatten(&[7, 5]);
        let y = 3.0 * d.position_flat(i).coords[1].cos().powi(2) - 1.0;
        assert!((lap.at(1, i)[0] + 6.0 * 2.0 * y).abs() < 2e-3);
        let tv = tf.surface_gradient().unwrap();
        assert!(tv.surface_curl().unwrap().at(0, i)[0].abs() < 1e-3);
    }

    #[test]
    fn operators_converge_at_fourth_order() {
        let err = |nt: usize| {
            let d = Arc::new(Domain::far_field_gauss(nt, 2 * nt).unwrap());
            let y = scalar(d.clone(), |p| 3.0 * p.coords[1].cos().powi(2) - 1.0);
            let lap = y.laplace_beltrami().unwrap();
            (0..d.len()).map(|i| (lap.scalar(i) + y.scalar(i) * 6.0).norm()).fold(0.0, f64::max)
        };
        let (coarse, fine) = (err(24), err(48));
        assert!(coarse / fine > 10.0, "{coarse} -> {fine}");
    }
}
