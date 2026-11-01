//! Differential operators on volume fields: gradient, divergence, curl, Laplacians, and time
//! derivatives.
//!
//! Spatial derivatives are taken along each native axis with Lagrange stencils (five points,
//! fourth order, when an axis has at least five samples, otherwise three; any spacing; wrapping
//! on periodic axes; shifted inside the axis near the ends of bounded ones), and
//! combined into Cartesian derivatives through the local frame and scale factors,
//! `∂/∂x_j = Σ_a (û_a)_j / h_a ∂/∂u_a`, which is the curvilinear formula applied to the smooth
//! Cartesian components. Results on samples where a scale factor vanishes (r = 0, ρ = 0, or a pole
//! inside a spherical shell) are undefined and returned as NaN. Gradients and curls are returned
//! in the linear basis of the domain's coordinate system; use `to_basis` for another.
//!
//! These operators need volume domains with at least two samples per axis; surface fields have
//! no full 3D derivatives.

use crate::conventions::C64;
use crate::coordinates::polarization::PolarizationBasis;
use crate::fields::domain::Domain;
use crate::fields::field_container::{FieldQuantity, HarmonicField, Rank, TimeField};
use lib_jsl_core::ErrorsJSL;
use ndarray::{Array2, Array3};
use num_traits::Zero;
use std::ops::{Add, Mul, Sub};

/// Values the operators work on: real (time domain) or complex (harmonic).
pub trait FieldScalar: Copy + Zero + Add<Output = Self> + Sub<Output = Self> + Mul<f64, Output = Self> {
    fn nan() -> Self;
}
impl FieldScalar for f64 {
    fn nan() -> Self {
        f64::NAN
    }
}
impl FieldScalar for C64 {
    fn nan() -> Self {
        C64::new(f64::NAN, f64::NAN)
    }
}

/// Weights `w` such that `f'(x) ≈ Σ w_a f(nodes_a)` (derivative of the Lagrange interpolant).
pub(crate) fn derivative_weights(nodes: &[f64], x: f64) -> Vec<f64> {
    let n = nodes.len();
    (0..n)
        .map(|a| {
            let denom: f64 = (0..n).filter(|&b| b != a).map(|b| nodes[a] - nodes[b]).product();
            let numer: f64 = (0..n).filter(|&m| m != a).map(|m| (0..n).filter(|&b| b != a && b != m).map(|b| x - nodes[b]).product::<f64>()).sum();
            numer / denom
        })
        .collect()
}

fn require_volume(domain: &Domain) -> Result<(), ErrorsJSL> {
    if domain.is_surface() {
        return Err(ErrorsJSL::InvalidInputRange("Spatial derivatives need a volume domain."));
    }
    if domain.axes().iter().any(|a| a.len() < 2) {
        return Err(ErrorsJSL::InvalidInputRange("Every axis needs at least two samples for derivatives."));
    }
    Ok(())
}

/// The derivative stencil along axis `a` at the multi-index `idx`: `(flat neighbour, weight)`
/// pairs with `∂f/∂u_a ≈ Σ w f(neighbour)`. Five points (fourth order) when the axis has at least
/// five samples, else three (or two); centred where possible, wrapping on periodic axes and
/// shifted inside bounded ones near their ends.
pub(crate) fn axis_stencil(domain: &Domain, idx: &[usize], a: usize) -> Vec<(usize, f64)> {
    let axis = domain.axis(a);
    let len = axis.len() as isize;
    let i = idx[a] as isize;
    let width: isize = if len >= 5 { 5 } else { len.min(3) };
    let half = width / 2;
    let start = if axis.is_periodic() { i - half } else { (i - half).clamp(0, len - width) };
    let window: Vec<isize> = (start..start + width).collect();
    let coord = |k: isize| match axis.period() {
        Some(p) => axis.values()[k.rem_euclid(len) as usize] + p * k.div_euclid(len) as f64,
        None => axis.values()[k as usize],
    };
    let nodes: Vec<f64> = window.iter().map(|&k| coord(k)).collect();
    let weights = derivative_weights(&nodes, axis.values()[idx[a]]);
    window
        .iter()
        .zip(weights)
        .map(|(&k, w)| {
            let mut neighbour = idx.to_vec();
            neighbour[a] = axis.wrap_index(k).expect("stencil index inside the axis");
            (domain.flatten(&neighbour), w)
        })
        .collect()
}

/// For each sample, the Jacobian `J[k][j] = ∂F_k/∂x_j` of Cartesian components `F_k`.
fn cartesian_jacobians<T: FieldScalar>(domain: &Domain, cart: &[[T; 3]]) -> Vec<[[T; 3]; 3]> {
    let n = domain.len();
    let mut out = vec![[[T::zero(); 3]; 3]; n];
    for (flat, jac) in out.iter_mut().enumerate() {
        let idx = domain.unflatten(flat);
        let pos = domain.position(&idx);
        let h = pos.scale_factors(domain.system());
        if h.iter().any(|&v| v.abs() < 1e-300) {
            *jac = [[T::nan(); 3]; 3];
            continue;
        }
        let frame = pos.frame(domain.system());
        for a in 0..domain.axes().len() {
            let mut d = [T::zero(); 3];
            for (neighbour, w) in axis_stencil(domain, &idx, a) {
                let v = cart[neighbour];
                for c in 0..3 {
                    d[c] = d[c] + v[c] * w;
                }
            }
            for c in 0..3 {
                for j in 0..3 {
                    jac[c][j] = jac[c][j] + d[c] * (frame[a][j] / h[a]);
                }
            }
        }
    }
    out
}

fn harmonic_cartesian(field: &HarmonicField) -> Vec<[C64; 3]> {
    (0..field.len()).map(|i| field.cartesian(i)).collect()
}

fn vector_field_from_cartesian(template: &HarmonicField, cart: Vec<[C64; 3]>) -> Result<HarmonicField, ErrorsJSL> {
    let domain = template.domain().clone();
    let basis = PolarizationBasis::Linear(domain.system());
    let mut values = Array2::<C64>::zeros((domain.len(), 3));
    for (i, v) in cart.iter().enumerate() {
        let b = basis.from_cartesian(*v, &domain.position_flat(i), template.convention());
        for k in 0..3 {
            values[[i, k]] = b[k];
        }
    }
    HarmonicField::new(domain, FieldQuantity::Generic, Rank::Vector(basis), template.frequency(), template.convention(), values)
}

fn scalar_field(template: &HarmonicField, values: Vec<C64>) -> Result<HarmonicField, ErrorsJSL> {
    let domain = template.domain().clone();
    let arr = Array2::from_shape_fn((domain.len(), 1), |(i, _)| values[i]);
    HarmonicField::new(domain, FieldQuantity::Generic, Rank::Scalar, template.frequency(), template.convention(), arr)
}

impl HarmonicField {
    /// `∇f` of a scalar field (quantity `Generic`, linear basis of the domain's system).
    pub fn gradient(&self) -> Result<HarmonicField, ErrorsJSL> {
        require_volume(self.domain())?;
        if !self.is_scalar() {
            return Err(ErrorsJSL::InvalidInputRange("The gradient needs a scalar field."));
        }
        let cart: Vec<[C64; 3]> = (0..self.len()).map(|i| [self.scalar(i), C64::zero(), C64::zero()]).collect();
        let j = cartesian_jacobians(self.domain(), &cart);
        vector_field_from_cartesian(self, j.iter().map(|m| m[0]).collect())
    }

    /// `∇·F` of a vector field.
    pub fn divergence(&self) -> Result<HarmonicField, ErrorsJSL> {
        require_volume(self.domain())?;
        if self.is_scalar() {
            return Err(ErrorsJSL::InvalidInputRange("The divergence needs a vector field."));
        }
        let j = cartesian_jacobians(self.domain(), &harmonic_cartesian(self));
        scalar_field(self, j.iter().map(|m| m[0][0] + m[1][1] + m[2][2]).collect())
    }

    /// `∇×F` of a vector field.
    pub fn curl(&self) -> Result<HarmonicField, ErrorsJSL> {
        require_volume(self.domain())?;
        if self.is_scalar() {
            return Err(ErrorsJSL::InvalidInputRange("The curl needs a vector field."));
        }
        let j = cartesian_jacobians(self.domain(), &harmonic_cartesian(self));
        vector_field_from_cartesian(self, j.iter().map(|m| [m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1]]).collect())
    }

    /// `∇²f` of a scalar field (as `∇·∇f`).
    pub fn laplacian(&self) -> Result<HarmonicField, ErrorsJSL> {
        self.gradient()?.divergence()
    }

    /// The vector Laplacian `∇²F`, the scalar Laplacian of each Cartesian component.
    pub fn vector_laplacian(&self) -> Result<HarmonicField, ErrorsJSL> {
        require_volume(self.domain())?;
        if self.is_scalar() {
            return Err(ErrorsJSL::InvalidInputRange("The vector Laplacian needs a vector field."));
        }
        let cart = harmonic_cartesian(self);
        let mut out = vec![[C64::zero(); 3]; self.len()];
        for k in 0..3 {
            let comp: Vec<[C64; 3]> = cart.iter().map(|v| [v[k], C64::zero(), C64::zero()]).collect();
            let grad: Vec<[C64; 3]> = cartesian_jacobians(self.domain(), &comp).iter().map(|m| m[0]).collect();
            let second = cartesian_jacobians(self.domain(), &grad);
            for (o, m) in out.iter_mut().zip(&second) {
                o[k] = m[0][0] + m[1][1] + m[2][2];
            }
        }
        vector_field_from_cartesian(self, out)
    }

    /// `∂F/∂t`, which for a phasor is multiplication by `+jω` (or `-iω`).
    pub fn time_derivative(&self) -> HarmonicField {
        self.scaled(self.convention().time_derivative(self.omega()))
    }
}

impl TimeField {
    fn cartesian_step(&self, t: usize) -> Result<Vec<[f64; 3]>, ErrorsJSL> {
        (0..self.domain().len())
            .map(|i| match self.rank() {
                Rank::Scalar => Ok([self.values()[[t, i, 0]], 0.0, 0.0]),
                Rank::Vector(b) => b.convert_real([self.values()[[t, i, 0]], self.values()[[t, i, 1]], self.values()[[t, i, 2]]], PolarizationBasis::CARTESIAN, &self.domain().position_flat(i)),
            })
            .collect()
    }

    fn assemble_steps(&self, rank: Rank, per_step: Vec<Vec<[f64; 3]>>) -> Result<TimeField, ErrorsJSL> {
        let domain = self.domain().clone();
        let (steps, n, c) = (per_step.len(), domain.len(), rank.components());
        let mut values = Array3::<f64>::zeros((steps, n, c));
        for (t, cart) in per_step.iter().enumerate() {
            for i in 0..n {
                let v = match rank {
                    Rank::Scalar => [cart[i][0], 0.0, 0.0],
                    Rank::Vector(b) => PolarizationBasis::CARTESIAN.convert_real(cart[i], b, &domain.position_flat(i))?,
                };
                for k in 0..c {
                    values[[t, i, k]] = v[k];
                }
            }
        }
        TimeField::new(domain, FieldQuantity::Generic, rank, self.time_axis(), values)
    }

    fn vector_rank(&self) -> Rank {
        Rank::Vector(PolarizationBasis::Linear(self.domain().system()))
    }

    /// `∇f` of a scalar time field, at every time step.
    pub fn gradient(&self) -> Result<TimeField, ErrorsJSL> {
        require_volume(self.domain())?;
        if self.rank() != Rank::Scalar {
            return Err(ErrorsJSL::InvalidInputRange("The gradient needs a scalar field."));
        }
        let steps = (0..self.time_axis().count).map(|t| Ok(cartesian_jacobians(self.domain(), &self.cartesian_step(t)?).iter().map(|m| m[0]).collect())).collect::<Result<Vec<_>, ErrorsJSL>>()?;
        self.assemble_steps(self.vector_rank(), steps)
    }

    /// `∇·F` of a vector time field.
    pub fn divergence(&self) -> Result<TimeField, ErrorsJSL> {
        require_volume(self.domain())?;
        if self.rank() == Rank::Scalar {
            return Err(ErrorsJSL::InvalidInputRange("The divergence needs a vector field."));
        }
        let steps = (0..self.time_axis().count).map(|t| Ok(cartesian_jacobians(self.domain(), &self.cartesian_step(t)?).iter().map(|m| [m[0][0] + m[1][1] + m[2][2], 0.0, 0.0]).collect())).collect::<Result<Vec<_>, ErrorsJSL>>()?;
        self.assemble_steps(Rank::Scalar, steps)
    }

    /// `∇×F` of a vector time field.
    pub fn curl(&self) -> Result<TimeField, ErrorsJSL> {
        require_volume(self.domain())?;
        if self.rank() == Rank::Scalar {
            return Err(ErrorsJSL::InvalidInputRange("The curl needs a vector field."));
        }
        let steps = (0..self.time_axis().count)
            .map(|t| Ok(cartesian_jacobians(self.domain(), &self.cartesian_step(t)?).iter().map(|m| [m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1]]).collect()))
            .collect::<Result<Vec<_>, ErrorsJSL>>()?;
        self.assemble_steps(self.vector_rank(), steps)
    }

    /// `∂F/∂t` by second-order finite differences in time (one-sided at the ends).
    pub fn time_derivative(&self) -> Result<TimeField, ErrorsJSL> {
        let (steps, n, c) = self.values().dim();
        if steps < 3 {
            return Err(ErrorsJSL::InvalidInputRange("A time derivative needs at least three time steps."));
        }
        let dt = self.time_axis().step;
        let v = self.values();
        let mut out = Array3::<f64>::zeros((steps, n, c));
        for i in 0..n {
            for k in 0..c {
                out[[0, i, k]] = (-3.0 * v[[0, i, k]] + 4.0 * v[[1, i, k]] - v[[2, i, k]]) / (2.0 * dt);
                out[[steps - 1, i, k]] = (3.0 * v[[steps - 1, i, k]] - 4.0 * v[[steps - 2, i, k]] + v[[steps - 3, i, k]]) / (2.0 * dt);
                for t in 1..steps - 1 {
                    out[[t, i, k]] = (v[[t + 1, i, k]] - v[[t - 1, i, k]]) / (2.0 * dt);
                }
            }
        }
        TimeField::new(self.domain().clone(), FieldQuantity::Generic, self.rank(), self.time_axis(), out)
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

    fn cube() -> Arc<Domain> {
        Arc::new(Domain::cartesian_box(Axis::uniform(-1.0, 1.0, 7).unwrap(), Axis::from_values(vec![0.0, 0.3, 0.5, 1.0, 1.2]).unwrap(), Axis::uniform(0.0, 2.0, 5).unwrap()).unwrap())
    }

    fn harmonic_scalar(d: Arc<Domain>, f: impl Fn([f64; 3]) -> f64) -> HarmonicField {
        HarmonicField::scalar_from_fn(d, FieldQuantity::ScalarPotential, 1.0, TimeConvention::PlusJ, |p| c(f(p.to_cartesian()))).unwrap()
    }

    fn harmonic_vector(d: Arc<Domain>, f: impl Fn([f64; 3]) -> [f64; 3]) -> HarmonicField {
        HarmonicField::vector_from_fn(d, FieldQuantity::Generic, PolarizationBasis::CARTESIAN, 1.0, TimeConvention::PlusJ, |p| f(p.to_cartesian()).map(c)).unwrap()
    }

    #[test]
    fn exact_for_quadratics_on_a_nonuniform_box() {
        let d = cube();
        let f = harmonic_scalar(d.clone(), |[x, y, z]| x * x * y + 3.0 * z - y * z);
        let g = f.gradient().unwrap();
        for i in 0..d.len() {
            let [x, y, z] = d.position_flat(i).to_cartesian();
            let v = g.vector(i);
            assert!((v[0] - c(2.0 * x * y)).norm() < 1e-10 && (v[1] - c(x * x - z)).norm() < 1e-10 && (v[2] - c(3.0 - y)).norm() < 1e-10);
        }
        // ∇²(x² + y² + z²) = 6, and the identities hold exactly for polynomials of this degree.
        let r2 = harmonic_scalar(d.clone(), |[x, y, z]| x * x + y * y + z * z);
        assert!(r2.laplacian().unwrap().values().iter().all(|v| (v - c(6.0)).norm() < 1e-9));
        assert!(r2.gradient().unwrap().curl().unwrap().values().iter().all(|v| v.norm() < 1e-9));
        let quad = harmonic_vector(d.clone(), |[x, y, z]| [y * z, x * x, x * y + z * z]);
        assert!(quad.curl().unwrap().divergence().unwrap().values().iter().all(|v| v.norm() < 1e-9));
        let vl = quad.vector_laplacian().unwrap();
        assert!((0..d.len()).all(|i| (vl.vector(i)[0]).norm() < 1e-8 && (vl.vector(i)[1] - c(2.0)).norm() < 1e-8 && (vl.vector(i)[2] - c(2.0)).norm() < 1e-8));
    }

    #[test]
    fn curvilinear_domains_match_the_analytic_operators() {
        // Spherical shell: ∇·r = 3 and ∇(r²) = 2r r̂ (radial component 2r in the native basis).
        let shell = Arc::new(Domain::spherical_shell(Axis::uniform(1.0, 2.0, 6).unwrap(), Axis::gauss_legendre_polar(10).unwrap(), Axis::periodic(0.0, TAU, 48).unwrap()).unwrap());
        let pos = harmonic_vector(shell.clone(), |p| p);
        let div = pos.divergence().unwrap();
        assert!(div.values().iter().all(|v| (v - c(3.0)).norm() < 2e-3), "max err {}", div.values().iter().map(|v| (v - c(3.0)).norm()).fold(0.0, f64::max));
        let g = harmonic_scalar(shell.clone(), |[x, y, z]| x * x + y * y + z * z).gradient().unwrap();
        assert_eq!(g.basis(), Some(PolarizationBasis::SPHERICAL));
        for i in (0..shell.len()).step_by(37) {
            let r = shell.position_flat(i).coords[0];
            let v = g.vector(i);
            assert!((v[0] - c(2.0 * r)).norm() < 2e-3 * r && v[1].norm() < 2e-3 && v[2].norm() < 2e-3);
        }
        // Cylindrical volume: the rotation field (-y, x, 0) has curl 2ẑ and no divergence.
        let cyl = Arc::new(Domain::cylindrical_volume(Axis::uniform(0.5, 2.0, 5).unwrap(), Axis::periodic(0.0, TAU, 64).unwrap(), Axis::uniform(0.0, 1.0, 3).unwrap()).unwrap());
        let swirl = harmonic_vector(cyl.clone(), |[x, y, _]| [-y, x, 0.0]);
        let curl = swirl.curl().unwrap();
        assert!((0..cyl.len()).all(|i| (curl.vector(i)[2] - c(2.0)).norm() < 1e-5 && curl.vector(i)[0].norm() < 1e-5));
        assert!(swirl.divergence().unwrap().values().iter().all(|v| v.norm() < 1e-5));
    }

    #[test]
    fn singular_samples_surfaces_and_time_derivatives() {
        // A ball including r = 0: the centre samples are flagged as NaN rather than wrong numbers.
        let ball = Arc::new(Domain::spherical_shell(Axis::uniform(0.0, 1.0, 4).unwrap(), Axis::gauss_legendre_polar(4).unwrap(), Axis::periodic(0.0, TAU, 8).unwrap()).unwrap());
        let div = harmonic_vector(ball.clone(), |p| p).divergence().unwrap();
        assert!(div.scalar(0).re.is_nan() && div.scalar(ball.flatten(&[2, 1, 3])).re.is_finite());
        let surf = Arc::new(Domain::far_field_gauss(4, 8).unwrap());
        assert!(harmonic_scalar(surf, |_| 1.0).gradient().is_err());
        let d = cube();
        assert!(harmonic_scalar(d.clone(), |_| 1.0).divergence().is_err() && harmonic_vector(d.clone(), |p| p).gradient().is_err());
        // Harmonic: ∂/∂t = jω.
        let h = harmonic_scalar(d.clone(), |_| 2.0);
        assert!((h.time_derivative().scalar(0) - C64::new(0.0, 2.0 * TAU)).norm() < 1e-12);
        // Time domain: d/dt sin(ωt) = ω cos(ωt), second order.
        let omega = 3.0;
        let t = TimeAxis::new(0.0, 1e-3, 400).unwrap();
        let tf = TimeField::from_fn(d.clone(), FieldQuantity::Generic, Rank::Scalar, t, |_, time| vec![(omega * time).sin()]).unwrap();
        let dt = tf.time_derivative().unwrap();
        for k in [0, 7, 200, 399] {
            assert!((dt.at(k, 0)[0] - omega * (omega * t.time(k)).cos()).abs() < 1e-4, "step {k}");
        }
        assert!(TimeField::from_fn(d.clone(), FieldQuantity::Generic, Rank::Scalar, TimeAxis::new(0.0, 1.0, 2).unwrap(), |_, _| vec![0.0]).unwrap().time_derivative().is_err());
        // Time-domain spatial operators agree with the harmonic ones at each step.
        let tv = TimeField::from_fn(d.clone(), FieldQuantity::Generic, Rank::Vector(PolarizationBasis::CARTESIAN), TimeAxis::new(0.0, 1.0, 2).unwrap(), |p, time| {
            let [x, y, _] = p.to_cartesian();
            vec![-y * (1.0 + time), x, 0.0]
        })
        .unwrap();
        let tc = tv.curl().unwrap();
        assert!((tc.at(1, 10)[2] - 3.0).abs() < 1e-9 && tv.divergence().unwrap().at(0, 3)[0].abs() < 1e-9);
        let tg = TimeField::from_fn(d.clone(), FieldQuantity::Generic, Rank::Scalar, TimeAxis::new(0.0, 1.0, 1).unwrap(), |p, _| vec![p.to_cartesian()[2] * 2.0]).unwrap().gradient().unwrap();
        assert!((tg.at(0, 4)[2] - 2.0).abs() < 1e-12);
    }
}
