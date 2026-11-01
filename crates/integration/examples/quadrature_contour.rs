//! Contour integrals in the complex plane: residues, Cauchy's formula and winding numbers.
//!
//! Run with `cargo run -p lib_jsl_integration --example quadrature_contour`.

use lib_jsl_integration::complex_path_integrals::{C64, Contour, PathSegment, cauchy_derivative, contour_integral, residue, winding_number};
use std::f64::consts::PI;

fn main() {
    let i = C64::new(0.0, 1.0);

    // Cauchy's theorem: an analytic function integrates to zero around a closed path.
    let square = Contour::rectangle(C64::new(-1.0, -1.0), C64::new(1.0, 1.0));
    let r = contour_integral(|z| z.exp() * z.sin(), &square, 1e-12).unwrap();
    println!("∮ eᶻ sin z dz around a square = {:.1e} ({} evaluations)", r.value.norm(), r.evaluations);

    // A pole inside: ∮ dz/z = 2πi.
    let r = contour_integral(|z| z.inv(), &Contour::circle(C64::new(0.2, 0.1), 1.0), 1e-12).unwrap();
    println!("∮ dz/z = {:.12}  (2πi = {:.12}i)", r.value, 2.0 * PI);

    // The residue of 1/(z² + 1) at i is 1/(2i).
    let res = residue(|z| (z * z + 1.0).inv(), i, 0.5, 1e-12).unwrap();
    println!("Res(1/(z²+1), i) = {res:.12}");

    // A real integral by closing the contour in the upper half-plane: ∫ dx/(1+x²) = π.
    let r_big = 50.0;
    let semicircle = Contour::new(vec![
        PathSegment::line(C64::new(-r_big, 0.0), C64::new(r_big, 0.0)),
        PathSegment::arc(C64::new(0.0, 0.0), r_big, 0.0, PI),
    ]);
    let r = contour_integral(|z| (z * z + 1.0).inv(), &semicircle, 1e-12).unwrap();
    println!("closed semicircle: {:.12} (π = {PI:.12})", r.value.re);

    // Cauchy's formula for derivatives: the 5th derivative of e^z at 0 is 1.
    let d = cauchy_derivative(|z| z.exp(), C64::new(0.0, 0.0), 5, 2.0, 1e-13).unwrap();
    println!("f⁽⁵⁾(0) for f = eᶻ: {d:.14}");

    // Winding numbers of a polygon.
    let triangle = [C64::new(0.0, 0.0), C64::new(2.0, 0.0), C64::new(1.0, 2.0)];
    println!("winding number around (1, 0.5): {:?}, around (3, 3): {:?}", winding_number(&triangle, C64::new(1.0, 0.5)), winding_number(&triangle, C64::new(3.0, 3.0)));
}
