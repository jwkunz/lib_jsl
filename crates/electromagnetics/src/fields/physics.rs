//! Physical operations on fields, with quantity checks: plane waves, the Poynting vector, power
//! flow, far-field radiation intensity and directivity, fields from potentials, wave impedance.

use crate::conventions::{C64, Medium};
use crate::coordinates::polarization::PolarizationBasis;
use crate::coordinates::systems::CoordinateSystem;
use crate::fields::domain::{Domain, DomainKind};
use crate::fields::field_container::{FieldQuantity, HarmonicField, Rank, TimeField};
use lib_jsl_core::ErrorsJSL;
use ndarray::{Array2, Array3};
use std::f64::consts::PI;
use std::sync::Arc;

fn expect(field_quantity: FieldQuantity, wanted: FieldQuantity, what: &'static str) -> Result<(), ErrorsJSL> {
    if field_quantity != wanted {
        return Err(ErrorsJSL::InvalidInputRange(what));
    }
    Ok(())
}

fn cross(a: [C64; 3], b: [C64; 3]) -> [C64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn cartesian_field(domain: Arc<Domain>, quantity: FieldQuantity, template: &HarmonicField, values: Vec<[C64; 3]>) -> Result<HarmonicField, ErrorsJSL> {
    let arr = Array2::from_shape_fn((domain.len(), 3), |(i, k)| values[i][k]);
    HarmonicField::new(domain, quantity, Rank::Vector(PolarizationBasis::CARTESIAN), template.frequency(), template.convention(), arr)
}

/// A uniform plane wave `E = E₀ p̂ e^{-jk k̂·r}` and `H = k̂ × E / η` on a (non-far-field) domain,
/// both in Cartesian components. `direction` is the propagation direction (normalised here) and
/// `polarization` the Cartesian electric polarization (its component along `direction` is
/// removed, then it is normalised).
pub fn plane_wave(domain: Arc<Domain>, direction: [f64; 3], polarization: [C64; 3], amplitude: C64, frequency: f64, medium: &Medium, convention: crate::conventions::TimeConvention) -> Result<(HarmonicField, HarmonicField), ErrorsJSL> {
    if domain.is_far_field() {
        return Err(ErrorsJSL::InvalidInputRange("Plane waves are defined on spatial domains, not the far-field sphere."));
    }
    let n = (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]).sqrt();
    if n == 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("The propagation direction must be non-zero."));
    }
    let k_hat = direction.map(|v| v / n);
    let along: C64 = (0..3).map(|i| polarization[i] * k_hat[i]).sum();
    let p: [C64; 3] = std::array::from_fn(|i| polarization[i] - along * k_hat[i]);
    let pn = p.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    if pn == 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("The polarization must have a component transverse to the direction."));
    }
    let p = p.map(|v| v / pn);
    let k = medium.wavenumber(frequency, convention);
    let eta = medium.impedance(convention);
    let k_c = k_hat.map(|v| C64::new(v, 0.0));
    let e = HarmonicField::vector_from_fn(domain.clone(), FieldQuantity::ElectricField, PolarizationBasis::CARTESIAN, frequency, convention, |pos| {
        let r = pos.to_cartesian();
        let phase = (-convention.unit() * k * (k_hat[0] * r[0] + k_hat[1] * r[1] + k_hat[2] * r[2])).exp();
        p.map(|v| v * amplitude * phase)
    })?;
    let h_values: Vec<[C64; 3]> = (0..e.len()).map(|i| cross(k_c, e.vector(i)).map(|v| v / eta)).collect();
    let h = cartesian_field(domain, FieldQuantity::MagneticField, &e, h_values)?;
    Ok((e, h))
}

/// The complex Poynting vector `S = ½ E × H*` (its real part is the time-averaged power density),
/// in Cartesian components. Requires an electric and a magnetic field on the same domain.
pub fn poynting(e: &HarmonicField, h: &HarmonicField) -> Result<HarmonicField, ErrorsJSL> {
    expect(e.quantity(), FieldQuantity::ElectricField, "The first field must be the electric field E.")?;
    expect(h.quantity(), FieldQuantity::MagneticField, "The second field must be the magnetic field H.")?;
    if !(Arc::ptr_eq(e.domain(), h.domain()) || e.domain() == h.domain()) || e.convention() != h.convention() || e.is_scalar() || h.is_scalar() {
        return Err(ErrorsJSL::InvalidInputRange("E and H must be vector fields on the same domain and convention."));
    }
    let s: Vec<[C64; 3]> = (0..e.len()).map(|i| cross(e.cartesian(i), h.cartesian(i).map(|v| v.conj())).map(|v| v * 0.5)).collect();
    cartesian_field(e.domain().clone(), FieldQuantity::PowerDensity, e, s)
}

/// The outward unit normal of a surface domain at a sample (`+ẑ`, `r̂` or `ρ̂`).
fn surface_normal(domain: &Domain, sample: usize) -> Result<[f64; 3], ErrorsJSL> {
    let p = domain.position_flat(sample);
    match domain.kind() {
        DomainKind::Plane { .. } => Ok([0.0, 0.0, 1.0]),
        DomainKind::Sphere { .. } => Ok(p.frame(CoordinateSystem::Spherical)[0]),
        DomainKind::Cylinder { .. } => Ok(p.frame(CoordinateSystem::Cylindrical)[0]),
        _ => Err(ErrorsJSL::InvalidInputRange("Power flow needs a plane, sphere or cylinder surface.")),
    }
}

/// The time-averaged power `∫ Re(S)·n̂ dA` through a surface (normal `+ẑ` for planes, outward for
/// spheres and cylinders).
pub fn power_through_surface(e: &HarmonicField, h: &HarmonicField) -> Result<f64, ErrorsJSL> {
    let s = poynting(e, h)?;
    let d = s.domain();
    (0..d.len())
        .map(|i| {
            let n = surface_normal(d, i)?;
            let v = s.vector(i);
            Ok(d.weights()[i] * (0..3).map(|k| v[k].re * n[k]).sum::<f64>())
        })
        .sum()
}

/// The instantaneous Poynting vector `S(t) = E(t) × H(t)` of time-domain fields.
pub fn poynting_time(e: &TimeField, h: &TimeField) -> Result<TimeField, ErrorsJSL> {
    expect(e.quantity(), FieldQuantity::ElectricField, "The first field must be the electric field E.")?;
    expect(h.quantity(), FieldQuantity::MagneticField, "The second field must be the magnetic field H.")?;
    let (Rank::Vector(be), Rank::Vector(bh)) = (e.rank(), h.rank()) else {
        return Err(ErrorsJSL::InvalidInputRange("E and H must be vector fields."));
    };
    if !(Arc::ptr_eq(e.domain(), h.domain()) || e.domain() == h.domain()) || e.time_axis() != h.time_axis() {
        return Err(ErrorsJSL::InvalidInputRange("E and H must share a domain and time axis."));
    }
    let (steps, n, _) = e.values().dim();
    let mut out = Array3::<f64>::zeros((steps, n, 3));
    for i in 0..n {
        let p = e.domain().position_flat(i);
        for t in 0..steps {
            let ev = be.convert_real([e.values()[[t, i, 0]], e.values()[[t, i, 1]], e.values()[[t, i, 2]]], PolarizationBasis::CARTESIAN, &p)?;
            let hv = bh.convert_real([h.values()[[t, i, 0]], h.values()[[t, i, 1]], h.values()[[t, i, 2]]], PolarizationBasis::CARTESIAN, &p)?;
            let s = [ev[1] * hv[2] - ev[2] * hv[1], ev[2] * hv[0] - ev[0] * hv[2], ev[0] * hv[1] - ev[1] * hv[0]];
            for k in 0..3 {
                out[[t, i, k]] = s[k];
            }
        }
    }
    TimeField::new(e.domain().clone(), FieldQuantity::PowerDensity, Rank::Vector(PolarizationBasis::CARTESIAN), e.time_axis(), out)
}

fn require_far_field(e: &HarmonicField) -> Result<(), ErrorsJSL> {
    expect(e.quantity(), FieldQuantity::ElectricField, "Far-field operations need the electric field E.")?;
    if !e.domain().is_far_field() || e.is_scalar() {
        return Err(ErrorsJSL::InvalidInputRange("Far-field operations need an E field on the far-field sphere."));
    }
    Ok(())
}

/// The radiation intensity `U = |E_ff|² / (2η)` (W/sr) of a far-field pattern `E_ff = r e^{jkr} E`.
pub fn radiation_intensity(e: &HarmonicField, medium: &Medium) -> Result<HarmonicField, ErrorsJSL> {
    require_far_field(e)?;
    let eta = medium.impedance(e.convention()).re;
    let intensity = e.norm_squared();
    let values = Array2::from_shape_fn((e.len(), 1), |(i, _)| C64::new(intensity[i] / (2.0 * eta), 0.0));
    HarmonicField::new(e.domain().clone(), FieldQuantity::Generic, Rank::Scalar, e.frequency(), e.convention(), values)
}

/// The total radiated power `∫ U dΩ` of a far-field pattern.
pub fn radiated_power(e: &HarmonicField, medium: &Medium) -> Result<f64, ErrorsJSL> {
    Ok(radiation_intensity(e, medium)?.integrate()?.re)
}

/// The directivity pattern `D(Ω) = 4π U(Ω) / P_rad` (dimensionless).
pub fn directivity(e: &HarmonicField) -> Result<HarmonicField, ErrorsJSL> {
    let u = radiation_intensity(e, &Medium::VACUUM)?;
    let p = u.integrate()?.re;
    if p <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("The pattern radiates no power."));
    }
    Ok(u.scaled(C64::new(4.0 * PI / p, 0.0)))
}

/// The largest value of the directivity pattern on its grid.
pub fn peak_directivity(e: &HarmonicField) -> Result<f64, ErrorsJSL> {
    Ok(directivity(e)?.values().iter().map(|v| v.re).fold(0.0, f64::max))
}

/// The fields of the potentials: `E = -∇Φ - ∂A/∂t` (with `∂/∂t → ±jω` per convention) and
/// `B = ∇ × A`, on a volume domain. Returns `(E, B)` in the domain's linear basis.
pub fn fields_from_potentials(phi: &HarmonicField, a: &HarmonicField) -> Result<(HarmonicField, HarmonicField), ErrorsJSL> {
    expect(phi.quantity(), FieldQuantity::ScalarPotential, "The first field must be the scalar potential Φ.")?;
    expect(a.quantity(), FieldQuantity::VectorPotential, "The second field must be the vector potential A.")?;
    let grad = phi.gradient()?;
    let a_native = a.to_basis(grad.basis().expect("gradients are vectors"))?;
    let dadt = a_native.time_derivative();
    let mut e = grad.scaled(C64::new(-1.0, 0.0));
    *e.values_mut() = e.values() - dadt.values();
    let e = e.with_quantity(FieldQuantity::ElectricField)?;
    let b = a.curl()?.with_quantity(FieldQuantity::MagneticFluxDensity)?;
    Ok((e, b))
}

/// The wave impedance `|E| / |H|` at every sample (Ω); `∞` where `H` vanishes.
pub fn wave_impedance(e: &HarmonicField, h: &HarmonicField) -> Result<Vec<f64>, ErrorsJSL> {
    expect(e.quantity(), FieldQuantity::ElectricField, "The first field must be the electric field E.")?;
    expect(h.quantity(), FieldQuantity::MagneticField, "The second field must be the magnetic field H.")?;
    if e.len() != h.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((e.len(), h.len())));
    }
    Ok(e.norm_squared().iter().zip(h.norm_squared()).map(|(a, b)| if b > 0.0 { (a / b).sqrt() } else { f64::INFINITY }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conventions::TimeConvention;
    use crate::fields::domain::Axis;
    use crate::fields::field_container::TimeAxis;
    use std::f64::consts::TAU;

    fn c(re: f64) -> C64 {
        C64::new(re, 0.0)
    }

    #[test]
    fn plane_wave_power_flux_and_impedance() {
        let f = 3e8;
        let plane = Arc::new(Domain::plane(Axis::uniform(-0.5, 0.5, 11).unwrap(), Axis::uniform(-1.0, 1.0, 11).unwrap(), 0.3).unwrap());
        let (e, h) = plane_wave(plane.clone(), [0.0, 0.0, 1.0], [c(1.0), c(0.0), c(0.0)], c(2.0), f, &Medium::VACUUM, TimeConvention::PlusJ).unwrap();
        let eta = Medium::VACUUM.impedance(TimeConvention::PlusJ).re;
        let want = 4.0 / (2.0 * eta) * plane.measure(); // |E|²/(2η) × area
        assert!((power_through_surface(&e, &h).unwrap() - want).abs() < 1e-12 * want);
        assert!(wave_impedance(&e, &h).unwrap().iter().all(|z| (z - eta).abs() < 1e-9));
        // Through a closed sphere a plane wave carries no net power, in either convention.
        let sphere = Arc::new(Domain::sphere(0.4, Axis::gauss_legendre_polar(16).unwrap(), Axis::periodic(0.0, TAU, 32).unwrap()).unwrap());
        for conv in [TimeConvention::PlusJ, TimeConvention::MinusI] {
            let (es, hs) = plane_wave(sphere.clone(), [1.0, 1.0, 0.0], [c(0.0), c(0.0), c(1.0)], c(1.0), f, &Medium::VACUUM, conv).unwrap();
            assert!(power_through_surface(&es, &hs).unwrap().abs() < 1e-12);
        }
        // Errors: wrong quantities, far-field domains, bad polarization.
        assert!(poynting(&h, &e).is_err());
        assert!(plane_wave(Arc::new(Domain::far_field_gauss(2, 4).unwrap()), [0.0, 0.0, 1.0], [c(1.0), c(0.0), c(0.0)], c(1.0), f, &Medium::VACUUM, TimeConvention::PlusJ).is_err());
        assert!(plane_wave(plane.clone(), [0.0, 0.0, 1.0], [c(0.0), c(0.0), c(1.0)], c(1.0), f, &Medium::VACUUM, TimeConvention::PlusJ).is_err());
        assert!(plane_wave(plane, [0.0; 3], [c(1.0), c(0.0), c(0.0)], c(1.0), f, &Medium::VACUUM, TimeConvention::PlusJ).is_err());
    }

    #[test]
    fn instantaneous_poynting_averages_to_the_complex_one() {
        let f = 1e8;
        let plane = Arc::new(Domain::plane(Axis::uniform(0.0, 1.0, 2).unwrap(), Axis::uniform(0.0, 1.0, 2).unwrap(), 0.0).unwrap());
        let (e, h) = plane_wave(plane, [0.0, 0.0, 1.0], [c(0.0), c(1.0), c(0.0)], c(3.0), f, &Medium::VACUUM, TimeConvention::PlusJ).unwrap();
        let t = TimeAxis::new(0.0, 1.0 / (f * 64.0), 64).unwrap();
        let s_t = poynting_time(&e.to_time_series(t).unwrap(), &h.to_time_series(t).unwrap()).unwrap();
        let mean: f64 = (0..64).map(|k| s_t.at(k, 0)[2]).sum::<f64>() / 64.0;
        assert!((mean - poynting(&e, &h).unwrap().vector(0)[2].re).abs() < 1e-12);
    }

    #[test]
    fn far_field_intensity_power_and_directivity_of_a_short_dipole() {
        // An odd node count puts a sample exactly on the equator, where the peak is.
        let d = Arc::new(Domain::far_field_gauss(13, 24).unwrap());
        let e = HarmonicField::vector_from_fn(d, FieldQuantity::ElectricField, PolarizationBasis::SPHERICAL, 1e9, TimeConvention::PlusJ, |p| [c(0.0), c(10.0 * p.coords[1].sin()), c(0.0)]).unwrap();
        let eta = Medium::VACUUM.impedance(TimeConvention::PlusJ).re;
        // P = ∫ 100 sin²θ / (2η) dΩ = 100 · (8π/3) / (2η).
        assert!((radiated_power(&e, &Medium::VACUUM).unwrap() - 100.0 * 8.0 * PI / 3.0 / (2.0 * eta)).abs() < 1e-10);
        assert!((peak_directivity(&e).unwrap() - 1.5).abs() < 1e-10);
        assert!((directivity(&e).unwrap().integrate().unwrap().re - 4.0 * PI).abs() < 1e-10);
        let zero = e.scaled(c(0.0));
        assert!(directivity(&zero).is_err());
        assert!(radiated_power(&e.clone().with_quantity(FieldQuantity::MagneticField).unwrap(), &Medium::VACUUM).is_err());
    }

    #[test]
    fn potentials_give_the_expected_fields() {
        let axis = || Axis::uniform(-1.0, 1.0, 5).unwrap();
        let d = Arc::new(Domain::cartesian_box(axis(), axis(), axis()).unwrap());
        let f = 1e3;
        let omega = TAU * f;
        // Φ = 2x, A = B₀/2 (-y, x, 0): E = -2x̂ - jωA, B = B₀ ẑ.
        let phi = HarmonicField::scalar_from_fn(d.clone(), FieldQuantity::ScalarPotential, f, TimeConvention::PlusJ, |p| c(2.0 * p.coords[0])).unwrap();
        let b0 = 0.3;
        let a = HarmonicField::vector_from_fn(d.clone(), FieldQuantity::VectorPotential, PolarizationBasis::CARTESIAN, f, TimeConvention::PlusJ, |p| [c(-b0 / 2.0 * p.coords[1]), c(b0 / 2.0 * p.coords[0]), c(0.0)]).unwrap();
        let (e, b) = fields_from_potentials(&phi, &a).unwrap();
        assert_eq!((e.quantity(), b.quantity()), (FieldQuantity::ElectricField, FieldQuantity::MagneticFluxDensity));
        let i = d.flatten(&[3, 1, 2]);
        let [x, y, _] = d.position_flat(i).to_cartesian();
        let ev = e.vector(i);
        assert!((ev[0] - (c(-2.0) - C64::new(0.0, omega) * (-b0 / 2.0 * y))).norm() < 1e-9);
        assert!((ev[1] - (-C64::new(0.0, omega) * (b0 / 2.0 * x))).norm() < 1e-9);
        assert!((b.vector(i)[2] - c(b0)).norm() < 1e-9);
        assert!(fields_from_potentials(&a, &phi).is_err());
    }
}
