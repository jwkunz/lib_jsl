//! Finite elements in one and two dimensions.
//!
//! Run with `cargo run --release -p lib_jsl_integration --example pde_finite_elements`.

use lib_jsl_integration::pde::finite_elements::{EndCondition, TriangleMesh, fem_1d, fem_poisson_2d};

fn main() {
    // A cooling fin: −u'' + 4u = 0 with the base held at 100 and an insulated tip (zero flux).
    // Exact: u = 100 cosh(2(1 − x)) / cosh 2.
    let nodes: Vec<f64> = (0..=20).map(|i| i as f64 / 20.0).collect();
    let u = fem_1d(|_| 1.0, |_| 4.0, |_| 0.0, &nodes, EndCondition::Dirichlet(100.0), EndCondition::Flux(0.0)).unwrap();
    let tip_exact = 100.0 / 2f64.cosh();
    println!("fin tip temperature: {:.4} (exact {tip_exact:.4})", u[20]);

    // A loaded square membrane: −∇²u = 1, u = 0 on the edges, on 2 × 32² triangles.
    let mesh = TriangleMesh::rectangle(1.0, 1.0, 32, 32);
    let fixed = mesh.box_boundary_nodes();
    let u = fem_poisson_2d(&mesh, |_, _| 1.0, |_, _| 1.0, |_, _| 0.0, &fixed).unwrap();
    let centre = mesh.nodes.iter().position(|p| (p[0] - 0.5).abs() < 1e-12 && (p[1] - 0.5).abs() < 1e-12).unwrap();
    println!("membrane deflection at the centre: {:.6} (series value 0.0736713...)", u[centre]);

    // The same membrane with a stiffer right half: k = 1 for x < ½ and 5 beyond.
    let u = fem_poisson_2d(&mesh, |x, _| if x < 0.5 { 1.0 } else { 5.0 }, |_, _| 1.0, |_, _| 0.0, &fixed).unwrap();
    println!("with a stiff right half: centre deflection {:.6}", u[centre]);
}
