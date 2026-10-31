//! Structured surface and volume domains.
//!
//! A [`Domain`] is a tensor-product grid of [`Axis`] samples in a native coordinate system. It
//! knows the position of every sample and the quadrature weight of every sample **with the
//! Jacobian included**, so `Σ w f` approximates `∫ f dA` (surfaces), `∫ f dV` (volumes) or
//! `∫ f dΩ` (the far-field sphere). Integrals elsewhere in the crate always use these weights and
//! never form a Jacobian themselves.
//!
//! | Kind | Axes (in order) | Element |
//! |------|-----------------|---------|
//! | plane | `x`, `y` at height `z₀` | `dx dy` |
//! | sphere | `θ`, `φ` at radius `R` | `R² sin θ dθ dφ` |
//! | far-field sphere | `θ`, `φ` (directions) | `sin θ dθ dφ` (solid angle) |
//! | cylinder | `φ`, `z` at radius `ρ₀` | `ρ₀ dφ dz` |
//! | box | `x`, `y`, `z` | `dx dy dz` |
//! | spherical shell | `r`, `θ`, `φ` | `r² sin θ dr dθ dφ` |
//! | cylindrical volume | `ρ`, `φ`, `z` | `ρ dρ dφ dz` |
//!
//! Azimuth axes must be periodic with period `2π`; polar axes must lie within `[0, π]`; radii
//! must be non-negative. Samples are stored row-major (the last axis varies fastest).

use super::axis::Axis;
use crate::coordinates::systems::{CoordinateSystem, Position};
use lib_jsl_core::ErrorsJSL;
use std::f64::consts::{PI, TAU};

/// The geometry a [`Domain`] describes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DomainKind {
    /// The plane `z = z0`, axes `(x, y)`.
    Plane { z0: f64 },
    /// The sphere of radius `radius`, axes `(θ, φ)`.
    Sphere { radius: f64 },
    /// Directions on the sphere at infinity (far fields), axes `(θ, φ)`; positions are unit vectors.
    FarFieldSphere,
    /// The cylinder `ρ = radius`, axes `(φ, z)`.
    Cylinder { radius: f64 },
    /// A Cartesian box, axes `(x, y, z)`.
    Box,
    /// A spherical shell or ball, axes `(r, θ, φ)`.
    SphericalShell,
    /// A cylindrical volume, axes `(ρ, φ, z)`.
    CylindricalVolume,
}

impl DomainKind {
    /// The coordinate system the axes are expressed in.
    pub fn system(self) -> CoordinateSystem {
        match self {
            DomainKind::Plane { .. } | DomainKind::Box => CoordinateSystem::Cartesian,
            DomainKind::Sphere { .. } | DomainKind::FarFieldSphere | DomainKind::SphericalShell => CoordinateSystem::Spherical,
            DomainKind::Cylinder { .. } | DomainKind::CylindricalVolume => CoordinateSystem::Cylindrical,
        }
    }

    pub fn is_surface(self) -> bool {
        matches!(self, DomainKind::Plane { .. } | DomainKind::Sphere { .. } | DomainKind::FarFieldSphere | DomainKind::Cylinder { .. })
    }

    /// The number of axes: 2 for surfaces, 3 for volumes.
    pub fn dimension(self) -> usize {
        if self.is_surface() { 2 } else { 3 }
    }
}

/// A tensor-product grid with quadrature weights; see the module documentation.
#[derive(Debug, Clone, PartialEq)]
pub struct Domain {
    kind: DomainKind,
    axes: Vec<Axis>,
    weights: Vec<f64>,
}

fn check_polar(a: &Axis) -> Result<(), ErrorsJSL> {
    if a.is_periodic() || a.first() < -1e-12 || a.last() > PI + 1e-12 {
        return Err(ErrorsJSL::InvalidInputRange("A polar axis must be non-periodic within [0, π]."));
    }
    Ok(())
}

fn check_azimuth(a: &Axis) -> Result<(), ErrorsJSL> {
    if a.period().is_none_or(|p| (p - TAU).abs() > 1e-9) {
        return Err(ErrorsJSL::InvalidInputRange("An azimuth axis must be periodic with period 2π."));
    }
    Ok(())
}

fn check_plain(a: &Axis) -> Result<(), ErrorsJSL> {
    if a.is_periodic() {
        return Err(ErrorsJSL::InvalidInputRange("This axis must not be periodic."));
    }
    Ok(())
}

fn check_radial(a: &Axis) -> Result<(), ErrorsJSL> {
    check_plain(a)?;
    if a.first() < 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("A radial axis must be non-negative."));
    }
    Ok(())
}

impl Domain {
    fn build(kind: DomainKind, axes: Vec<Axis>) -> Self {
        let mut d = Domain { kind, axes, weights: Vec::new() };
        d.weights = (0..d.len()).map(|f| d.compute_weight(&d.unflatten(f))).collect();
        d
    }

    pub fn plane(x: Axis, y: Axis, z0: f64) -> Result<Self, ErrorsJSL> {
        check_plain(&x)?;
        check_plain(&y)?;
        Ok(Self::build(DomainKind::Plane { z0 }, vec![x, y]))
    }

    pub fn sphere(radius: f64, theta: Axis, phi: Axis) -> Result<Self, ErrorsJSL> {
        if radius.is_nan() || radius < 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("The sphere radius must be non-negative."));
        }
        check_polar(&theta)?;
        check_azimuth(&phi)?;
        Ok(Self::build(DomainKind::Sphere { radius }, vec![theta, phi]))
    }

    pub fn far_field_sphere(theta: Axis, phi: Axis) -> Result<Self, ErrorsJSL> {
        check_polar(&theta)?;
        check_azimuth(&phi)?;
        Ok(Self::build(DomainKind::FarFieldSphere, vec![theta, phi]))
    }

    /// A far-field sphere with `n_theta` Gauss-Legendre polar nodes and `n_phi` uniform azimuths:
    /// exact for band-limited patterns, never samples the poles.
    pub fn far_field_gauss(n_theta: usize, n_phi: usize) -> Result<Self, ErrorsJSL> {
        Self::far_field_sphere(Axis::gauss_legendre_polar(n_theta)?, Axis::periodic(0.0, TAU, n_phi)?)
    }

    /// A far-field sphere with uniform θ samples including both poles (`n_theta >= 2`), the usual
    /// layout of measured or simulated patterns.
    pub fn far_field_uniform(n_theta: usize, n_phi: usize) -> Result<Self, ErrorsJSL> {
        Self::far_field_sphere(Axis::uniform(0.0, PI, n_theta)?, Axis::periodic(0.0, TAU, n_phi)?)
    }

    pub fn cylinder(radius: f64, phi: Axis, z: Axis) -> Result<Self, ErrorsJSL> {
        if radius.is_nan() || radius < 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("The cylinder radius must be non-negative."));
        }
        check_azimuth(&phi)?;
        check_plain(&z)?;
        Ok(Self::build(DomainKind::Cylinder { radius }, vec![phi, z]))
    }

    pub fn cartesian_box(x: Axis, y: Axis, z: Axis) -> Result<Self, ErrorsJSL> {
        for a in [&x, &y, &z] {
            check_plain(a)?;
        }
        Ok(Self::build(DomainKind::Box, vec![x, y, z]))
    }

    pub fn spherical_shell(r: Axis, theta: Axis, phi: Axis) -> Result<Self, ErrorsJSL> {
        check_radial(&r)?;
        check_polar(&theta)?;
        check_azimuth(&phi)?;
        Ok(Self::build(DomainKind::SphericalShell, vec![r, theta, phi]))
    }

    pub fn cylindrical_volume(rho: Axis, phi: Axis, z: Axis) -> Result<Self, ErrorsJSL> {
        check_radial(&rho)?;
        check_azimuth(&phi)?;
        check_plain(&z)?;
        Ok(Self::build(DomainKind::CylindricalVolume, vec![rho, phi, z]))
    }

    pub fn kind(&self) -> DomainKind {
        self.kind
    }

    pub fn system(&self) -> CoordinateSystem {
        self.kind.system()
    }

    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    pub fn axis(&self, i: usize) -> &Axis {
        &self.axes[i]
    }

    pub fn is_surface(&self) -> bool {
        self.kind.is_surface()
    }

    pub fn is_far_field(&self) -> bool {
        self.kind == DomainKind::FarFieldSphere
    }

    /// Samples per axis.
    pub fn shape(&self) -> Vec<usize> {
        self.axes.iter().map(Axis::len).collect()
    }

    /// Total number of samples.
    pub fn len(&self) -> usize {
        self.axes.iter().map(Axis::len).product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row-major flat index of a multi-index.
    pub fn flatten(&self, index: &[usize]) -> usize {
        index.iter().zip(&self.axes).fold(0, |acc, (&i, a)| acc * a.len() + i)
    }

    /// Multi-index of a flat index.
    pub fn unflatten(&self, mut flat: usize) -> Vec<usize> {
        let mut out = vec![0; self.axes.len()];
        for (slot, a) in out.iter_mut().zip(&self.axes).rev() {
            *slot = flat % a.len();
            flat /= a.len();
        }
        out
    }

    /// The native coordinates `(u₁, u₂, u₃)` of a sample (the fixed coordinate of a surface filled in).
    pub fn native_coordinates(&self, index: &[usize]) -> [f64; 3] {
        let u = |i: usize| self.axes[i].values()[index[i]];
        match self.kind {
            DomainKind::Plane { z0 } => [u(0), u(1), z0],
            DomainKind::Sphere { radius } => [radius, u(0), u(1)],
            DomainKind::FarFieldSphere => [1.0, u(0), u(1)],
            DomainKind::Cylinder { radius } => [radius, u(0), u(1)],
            DomainKind::Box | DomainKind::SphericalShell | DomainKind::CylindricalVolume => [u(0), u(1), u(2)],
        }
    }

    /// The position of a sample in the domain's native system. On the far-field sphere this is
    /// the unit direction (radius 1).
    pub fn position(&self, index: &[usize]) -> Position {
        Position { system: self.system(), coords: self.native_coordinates(index) }
    }

    /// The position of the sample with a given flat index.
    pub fn position_flat(&self, flat: usize) -> Position {
        self.position(&self.unflatten(flat))
    }

    fn compute_weight(&self, index: &[usize]) -> f64 {
        let w: f64 = index.iter().zip(&self.axes).map(|(&i, a)| a.weights()[i]).product();
        let c = self.native_coordinates(index);
        let jacobian = match self.kind {
            DomainKind::Plane { .. } | DomainKind::Box => 1.0,
            DomainKind::Sphere { radius } => radius * radius * c[1].sin(),
            DomainKind::FarFieldSphere => c[1].sin(),
            DomainKind::Cylinder { radius } => radius,
            DomainKind::SphericalShell => c[0] * c[0] * c[1].sin(),
            DomainKind::CylindricalVolume => c[0],
        };
        w * jacobian
    }

    /// Quadrature weights (area, volume or solid-angle elements) in flat order.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// `Σ wᵢ fᵢ`: the integral of samples given in flat order.
    pub fn integrate(&self, values: &[f64]) -> Result<f64, ErrorsJSL> {
        if values.len() != self.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((values.len(), self.len())));
        }
        Ok(values.iter().zip(&self.weights).map(|(v, w)| v * w).sum())
    }

    /// The integral of a function of position over the domain.
    pub fn integrate_fn(&self, f: impl Fn(&Position) -> f64) -> f64 {
        (0..self.len()).map(|i| self.weights[i] * f(&self.position_flat(i))).sum()
    }

    /// The total area, volume or solid angle according to the quadrature rule.
    pub fn measure(&self) -> f64 {
        self.weights.iter().sum()
    }

    /// The index of the polar axis on spherical domains.
    fn polar_axis(&self) -> Option<usize> {
        match self.kind {
            DomainKind::Sphere { .. } | DomainKind::FarFieldSphere => Some(0),
            DomainKind::SphericalShell => Some(1),
            _ => None,
        }
    }

    /// Continues a grid index across a pole on spherical domains: stepping to polar index
    /// `theta_index` (which may be negative or beyond the last sample) at azimuth index
    /// `phi_index` lands on the reflected sample `(θ' , φ + π)`. Requires the polar axis to be
    /// symmetric about the pole being crossed (true for uniform grids that include the pole and
    /// for Gauss-Legendre grids) and the azimuth grid to contain `φ + π`. Returns
    /// `(theta_index, phi_index)` of the equivalent sample, or `None` if it is not on the grid.
    pub fn across_pole(&self, theta_index: isize, phi_index: usize) -> Option<(usize, usize)> {
        let pa = self.polar_axis()?;
        let theta = &self.axes[pa];
        let phi = &self.axes[pa + 1];
        let n = theta.len() as isize;
        let target_theta = if theta_index < 0 {
            // Mirror about θ = 0: the sample at index |i| (or |i| - 1 if the pole itself is not sampled).
            let k = if theta.first().abs() < 1e-12 { -theta_index } else { -theta_index - 1 };
            *theta.values().get(k as usize)?
        } else if theta_index >= n {
            let over = theta_index - (n - 1);
            let k = if (theta.last() - PI).abs() < 1e-12 { n - 1 - over } else { n - over };
            *theta.values().get(usize::try_from(k).ok()?)?
        } else {
            return Some((theta_index as usize, phi_index));
        };
        let ti = theta.values().iter().position(|&v| (v - target_theta).abs() < 1e-9)?;
        let p = phi.values()[phi_index] + PI;
        let pj = phi.nearest(p);
        let found = phi.values()[pj];
        ((phi.reduce(found) - phi.reduce(p)).abs() < 1e-9 || ((phi.reduce(found) - phi.reduce(p)).abs() - TAU).abs() < 1e-9).then_some((ti, pj))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    #[test]
    fn surface_measures() {
        let plane = Domain::plane(Axis::uniform(-1.0, 1.0, 5).unwrap(), Axis::uniform(0.0, 3.0, 4).unwrap(), 2.0).unwrap();
        assert!(close(plane.measure(), 6.0, 1e-14) && plane.is_surface() && plane.shape() == vec![5, 4]);
        let r = 2.5;
        let gauss = Domain::sphere(r, Axis::gauss_legendre_polar(8).unwrap(), Axis::periodic(0.0, TAU, 16).unwrap()).unwrap();
        assert!(close(gauss.measure(), 4.0 * PI * r * r, 1e-13));
        let uniform = Domain::sphere(r, Axis::uniform(0.0, PI, 91).unwrap(), Axis::periodic(0.0, TAU, 72).unwrap()).unwrap();
        assert!(close(uniform.measure(), 4.0 * PI * r * r, 1e-3));
        assert!(close(Domain::far_field_gauss(10, 20).unwrap().measure(), 4.0 * PI, 1e-13));
        let cyl = Domain::cylinder(0.5, Axis::periodic(0.0, TAU, 32).unwrap(), Axis::uniform(0.0, 4.0, 9).unwrap()).unwrap();
        assert!(close(cyl.measure(), TAU * 0.5 * 4.0, 1e-13));
    }

    #[test]
    fn volume_measures() {
        let b = Domain::cartesian_box(Axis::uniform(0.0, 1.0, 3).unwrap(), Axis::uniform(0.0, 2.0, 4).unwrap(), Axis::uniform(-1.0, 2.0, 5).unwrap()).unwrap();
        assert!(close(b.measure(), 6.0, 1e-14) && !b.is_surface() && b.len() == 60);
        let shell = Domain::spherical_shell(Axis::gauss_legendre(1.0, 2.0, 4).unwrap(), Axis::gauss_legendre_polar(6).unwrap(), Axis::periodic(0.0, TAU, 12).unwrap()).unwrap();
        assert!(close(shell.measure(), 4.0 / 3.0 * PI * (8.0 - 1.0), 1e-13));
        let cv = Domain::cylindrical_volume(Axis::gauss_legendre(0.0, 2.0, 3).unwrap(), Axis::periodic(0.0, TAU, 8).unwrap(), Axis::uniform(0.0, 3.0, 2).unwrap()).unwrap();
        assert!(close(cv.measure(), PI * 4.0 * 3.0, 1e-13));
    }

    #[test]
    fn weights_integrate_functions_with_the_right_jacobian() {
        // ∫ cos²θ dΩ = 4π/3, ∫ sin²θ cos²φ dΩ = 4π/3, and spherical harmonics are orthogonal.
        let s = Domain::far_field_gauss(8, 16).unwrap();
        assert!(close(s.integrate_fn(|p| p.coords[1].cos().powi(2)), 4.0 * PI / 3.0, 1e-13));
        assert!(close(s.integrate_fn(|p| (p.coords[1].sin() * p.coords[2].cos()).powi(2)), 4.0 * PI / 3.0, 1e-13));
        let y10 = |p: &Position| p.coords[1].cos();
        let y20 = |p: &Position| 3.0 * p.coords[1].cos().powi(2) - 1.0;
        assert!(s.integrate_fn(|p| y10(p) * y20(p)).abs() < 1e-13);
        // ∫ x² over a ball of radius 1 is 4π/15.
        let ball = Domain::spherical_shell(Axis::gauss_legendre(0.0, 1.0, 6).unwrap(), Axis::gauss_legendre_polar(6).unwrap(), Axis::periodic(0.0, TAU, 12).unwrap()).unwrap();
        assert!(close(ball.integrate_fn(|p| p.to_cartesian()[0].powi(2)), 4.0 * PI / 15.0, 1e-12));
        // ∫ ρ² over a cylinder of radius 1 and height 2 is π.
        let cv = Domain::cylindrical_volume(Axis::gauss_legendre(0.0, 1.0, 4).unwrap(), Axis::periodic(0.0, TAU, 4).unwrap(), Axis::uniform(0.0, 2.0, 2).unwrap()).unwrap();
        assert!(close(cv.integrate_fn(|p| p.coords[0].powi(2)), PI, 1e-12));
        let values: Vec<f64> = (0..s.len()).map(|i| y10(&s.position_flat(i)).powi(2)).collect();
        assert!(close(s.integrate(&values).unwrap(), 4.0 * PI / 3.0, 1e-13));
        assert!(s.integrate(&values[1..]).is_err());
    }

    #[test]
    fn indexing_positions_and_poles() {
        let d = Domain::far_field_uniform(7, 8).unwrap(); // θ = 0, 30°, ..., 180°; φ every 45°
        for f in [0, 5, 17, d.len() - 1] {
            assert_eq!(d.flatten(&d.unflatten(f)), f);
        }
        let p = d.position(&[3, 2]);
        assert_eq!(p.system, CoordinateSystem::Spherical);
        assert!(close(p.coords[0], 1.0, 1e-15) && close(p.coords[1], PI / 2.0, 1e-12) && close(p.coords[2], PI / 2.0, 1e-12));
        assert!(d.is_far_field() && d.weights()[0] == 0.0); // the pole sample carries no solid angle
        // One step past the north pole at φ = 45° lands on θ = 30°, φ = 225°.
        assert_eq!(d.across_pole(-1, 1), Some((1, 5)));
        assert_eq!(d.across_pole(7, 0), Some((5, 4))); // one past the south pole
        assert_eq!(d.across_pole(3, 6), Some((3, 6))); // inside: unchanged
        // Gauss-Legendre grids do not sample the poles: index -1 mirrors to index 0.
        let g = Domain::far_field_gauss(4, 6).unwrap();
        assert_eq!(g.across_pole(-1, 0), Some((0, 3)));
        assert_eq!(g.across_pole(4, 1), Some((3, 4)));
        // An odd number of azimuths has no sample at φ + π.
        assert_eq!(Domain::far_field_uniform(5, 5).unwrap().across_pole(-1, 0), None);
        let plane = Domain::plane(Axis::uniform(0.0, 1.0, 2).unwrap(), Axis::uniform(0.0, 1.0, 2).unwrap(), 0.0).unwrap();
        assert_eq!(plane.across_pole(-1, 0), None);
        assert_eq!(plane.native_coordinates(&[1, 0]), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn validation() {
        let az = || Axis::periodic(0.0, TAU, 8).unwrap();
        let polar = || Axis::uniform(0.0, PI, 5).unwrap();
        assert!(Domain::sphere(-1.0, polar(), az()).is_err());
        assert!(Domain::sphere(1.0, Axis::uniform(0.0, 4.0, 5).unwrap(), az()).is_err()); // θ beyond π
        assert!(Domain::sphere(1.0, polar(), Axis::uniform(0.0, TAU, 8).unwrap()).is_err()); // φ not periodic
        assert!(Domain::sphere(1.0, polar(), Axis::periodic(0.0, 1.0, 8).unwrap()).is_err()); // wrong period
        assert!(Domain::cylinder(1.0, az(), az()).is_err());
        assert!(Domain::spherical_shell(Axis::uniform(-1.0, 1.0, 3).unwrap(), polar(), az()).is_err());
        assert!(Domain::cartesian_box(az(), polar(), polar()).is_err());
        assert!(Domain::plane(polar(), az(), 0.0).is_err());
    }
}
