//! Field containers in action: power conservation, Maxwell's equations from vector calculus,
//! basis-independent overlap integrals, and rigid transforms of patterns.
//!
//! Run with `cargo run --release -p lib_jsl_electromagnetics --example fields_demo`.

use lib_jsl_core::constants::em::C0;
use lib_jsl_electromagnetics::algorithms::overlap_integral;
use lib_jsl_electromagnetics::antenna_models::{Antenna, HertzianDipole};
use lib_jsl_electromagnetics::conventions::{C64, Medium, TimeConvention};
use lib_jsl_electromagnetics::coordinates::{PolarizationBasis, RigidTransform};
use lib_jsl_electromagnetics::fields::physics::{peak_directivity, plane_wave, power_through_surface};
use lib_jsl_electromagnetics::fields::{Axis, Domain, Extrapolation, FieldQuantity, HarmonicField, InterpolationMethod};
use std::f64::consts::{PI, TAU};
use std::sync::Arc;

fn main() {
    let f = 3e8; // λ = 1 m
    let conv = TimeConvention::PlusJ;

    // 1. Power conservation: the Poynting flux of a Hertzian dipole's exact near field through a
    //    sphere of any radius equals the far-field radiated power.
    let dipole = HertzianDipole::new(0.05);
    println!("== Hertzian dipole (l = λ/20): power through spheres vs far-field power ==");
    println!("  far-field radiated power  {:.6e} W per A²", dipole.radiated_power(f));
    for r in [0.02, 0.1, 0.5, 2.0, 10.0] {
        let sphere = Arc::new(Domain::sphere(r, Axis::gauss_legendre_polar(12).unwrap(), Axis::periodic(0.0, TAU, 8).unwrap()).unwrap());
        let (e, h) = dipole.near_field_containers(sphere, f, conv).unwrap();
        println!("  r = {r:>5} λ: Poynting flux {:.6e} W", power_through_surface(&e, &h).unwrap());
    }

    // 2. Maxwell's equations from the containers' vector calculus: for a plane wave in a box,
    //    ∇·E = 0 and ∇×E = -jωμ₀H.
    let axis = || Axis::uniform(-0.25, 0.25, 21).unwrap();
    let cube = Arc::new(Domain::cartesian_box(axis(), axis(), axis()).unwrap());
    let (e, h) = plane_wave(cube.clone(), [1.0, 1.0, 0.0], [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)], C64::new(1.0, 0.0), f, &Medium::VACUUM, conv).unwrap();
    let div = e.divergence().unwrap();
    let curl = e.curl().unwrap().to_basis(PolarizationBasis::CARTESIAN).unwrap();
    let mu0 = lib_jsl_core::constants::em::MU_0;
    let faraday = h.scaled(-conv.time_derivative(TAU * f) * mu0);
    let max_div = div.values().iter().map(|v| v.norm()).fold(0.0, f64::max);
    let max_err = (0..cube.len()).map(|i| (0..3).map(|k| (curl.vector(i)[k] - faraday.vector(i)[k]).norm()).sum::<f64>()).fold(0.0, f64::max);
    println!("\n== plane wave in a 0.5 λ cube (21³ samples) ==");
    println!("  max |∇·E| = {max_div:.2e} V/m²   max |∇×E + jωμ₀H| = {max_err:.2e} V/m² (|∇×E| ≈ {:.1})", TAU * f / C0);

    // 3. Overlap integrals are independent of the basis and grid: Gaussian beam coupling.
    let plane = |n: usize| Arc::new(Domain::plane(Axis::gauss_legendre(-4.0, 4.0, n).unwrap(), Axis::gauss_legendre(-4.0, 4.0, n).unwrap(), 0.0).unwrap());
    let beam = |d: Arc<Domain>, w: f64, x0: f64| HarmonicField::vector_from_fn(d, FieldQuantity::ElectricField, PolarizationBasis::CARTESIAN, f, conv, |p| {
        let [x, y, _] = p.to_cartesian();
        [C64::new((-((x - x0).powi(2) + y * y) / (w * w)).exp(), 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)]
    }).unwrap();
    let reference = beam(plane(50), 1.0, 0.0);
    println!("\n== Gaussian beam coupling (waist 1) against offset copies on a different grid and basis ==");
    for offset in [0.0, 0.5, 1.0, 1.5] {
        let other = beam(plane(70), 1.0, offset).to_basis(PolarizationBasis::CYLINDRICAL).unwrap();
        let o = overlap_integral(&reference, &other, InterpolationMethod::Cubic).unwrap();
        println!("  offset {offset:.1}: efficiency {:.5} (closed form {:.5})", o.efficiency, (-(offset * offset)).exp());
    }

    // 4. Rigid transforms of a far-field pattern: rotation keeps the directivity, translation only
    //    adds a phase ramp.
    let sphere = Arc::new(Domain::far_field_uniform(91, 180).unwrap());
    let pattern = dipole.far_field_pattern(sphere.clone(), f, conv).unwrap();
    let tilted = pattern.transformed(&RigidTransform::rotation_about([1.0, 0.0, 0.0], 0.7), InterpolationMethod::Cubic, Extrapolation::Error).unwrap();
    let moved = pattern.transformed(&RigidTransform::translation([0.25, 0.0, 0.0]), InterpolationMethod::Cubic, Extrapolation::Error).unwrap();
    let along_x = sphere.flatten(&[45, 0]);
    println!("\n== rigid transforms of the dipole pattern ==");
    println!("  peak directivity: original {:.4}, rotated 40° {:.4}", peak_directivity(&pattern).unwrap(), peak_directivity(&tilted).unwrap());
    println!("  translated λ/4 along x: phase change towards +x {:.1}° (expected 90°)", (moved.vector(along_x)[1] / pattern.vector(along_x)[1]).arg().to_degrees());
    let _ = PI;
}
