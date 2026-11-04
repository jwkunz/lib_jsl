//! Finite-difference time-domain simulation: reflection from a dielectric in 1D, and a pulse in a
//! 2D domain with a perfectly matched layer.
//!
//! Run with `cargo run --release -p lib_jsl_electromagnetics --example fdtd`.

use lib_jsl_electromagnetics::algorithms::{Fdtd1D, FdtdTmz, gaussian_pulse};

fn main() {
    // 1D: a Gaussian pulse meets glass (ε_r = 2.25) at cell 500.
    let mut line = Fdtd1D::new(1000, 1e-3, 1.0).unwrap();
    line.set_material(500, 1000, 2.25, 0.0, 1.0);
    let mut peak_incident: f64 = 0.0;
    let mut peak_reflected: f64 = 0.0;
    // The incident pulse passes the probe (cell 300) near step 260; the reflection returns near 660.
    for n in 0..1000 {
        line.step(Some((100, gaussian_pulse(n, 60.0, 15.0))));
        let v = line.ez[300];
        if n < 500 { peak_incident = peak_incident.max(v) } else if v.abs() > peak_reflected.abs() { peak_reflected = v }
    }
    println!("1D: reflection {:.4} (Fresnel (1 − 1.5)/(1 + 1.5) = -0.2000)", peak_reflected / peak_incident);
    // 2D: a pulse from the centre of a 120 × 120 grid with a 10-cell PML.
    let mut grid = FdtdTmz::new(120, 120, 1e-3, 0.7, 10).unwrap();
    let mut peak: f64 = 0.0;
    for n in 0..600 {
        grid.step(Some((60, 60, gaussian_pulse(n, 30.0, 10.0))));
        peak = peak.max(grid.interior_energy(10));
        if n % 150 == 149 {
            println!("2D: step {:3}, energy left inside the PML boundary {:.2e} of its peak", n + 1, grid.interior_energy(10) / peak);
        }
    }
}
