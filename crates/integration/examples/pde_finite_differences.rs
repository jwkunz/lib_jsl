//! Finite differences for the heat, wave and Poisson equations.
//!
//! Run with `cargo run --release -p lib_jsl_integration --example pde_finite_differences`.

use lib_jsl_integration::pde::finite_differences::{HeatScheme, heat_equation_1d, poisson_2d, wave_equation_1d};

fn main() {
    // Heat: a rod at 0 °C with its ends held at 0 °C and 100 °C, after one time unit.
    let n = 51;
    let u0 = vec![0.0; n];
    let rod = heat_equation_1d(&u0, 1.0, 0.1, 0.01, 100, 0.0, 100.0, HeatScheme::CrankNicolson).unwrap();
    println!("rod temperature at x = 0.5 after t = 1: {:.3} °C", rod[n / 2]);
    // The explicit scheme refuses steps beyond its stability limit r = αΔt/Δx² ≤ 1/2.
    println!("explicit with r = 2.5: {:?}", heat_equation_1d(&u0, 1.0, 0.1, 0.01, 100, 0.0, 100.0, HeatScheme::Explicit).err());

    // Wave: a plucked string returns to its initial shape after one period 2L/c.
    let pluck: Vec<f64> = (0..n).map(|i| { let x = i as f64 / (n - 1) as f64; if x < 0.3 { x / 0.3 } else { (1.0 - x) / 0.7 } }).collect();
    let dx = 1.0 / (n - 1) as f64;
    let after_period = wave_equation_1d(&pluck, &vec![0.0; n], 1.0, 1.0, dx, 2 * (n - 1)).unwrap();
    let drift = pluck.iter().zip(&after_period).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("plucked string after one period (Courant number 1): max change {drift:.1e}");

    // Poisson: −∇²u = 1 on the unit square with u = 0 on the edges (e.g. a loaded membrane).
    let u = poisson_2d(|_, _| 1.0, |_, _| 0.0, 1.0, 1.0, 65, 65).unwrap();
    println!("membrane deflection at the centre: {:.6} (series value 0.0736713...)", u[32][32]);
}
