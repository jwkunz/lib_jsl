//! Two-port network parameters and their conversions: S, Z, Y, ABCD and T.
//!
//! Scattering parameters are referred to a real impedance `z0` on both ports (power waves
//! `a = (V + z0 I)/(2√z0)`, `b = (V − z0 I)/(2√z0)`). Cascades multiply ABCD (or T) matrices.

use crate::conventions::C64;
use lib_jsl_core::ErrorsJSL;

/// A 2 × 2 complex matrix.
pub type Matrix2 = [[C64; 2]; 2];

fn det(m: &Matrix2) -> C64 {
    m[0][0] * m[1][1] - m[0][1] * m[1][0]
}

fn singular(d: C64) -> Result<(), ErrorsJSL> {
    if d.norm() < 1e-300 {
        return Err(ErrorsJSL::RuntimeError("The conversion is singular for this network."));
    }
    Ok(())
}

fn c(v: f64) -> C64 {
    C64::new(v, 0.0)
}

/// The product of two 2 × 2 matrices.
pub fn multiply(a: &Matrix2, b: &Matrix2) -> Matrix2 {
    [[a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]], [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]]]
}

/// Z (impedance) parameters to S parameters.
pub fn z_to_s(z: &Matrix2, z0: f64) -> Result<Matrix2, ErrorsJSL> {
    let (a, b, cc, d) = (z[0][0], z[0][1], z[1][0], z[1][1]);
    let delta = (a + z0) * (d + z0) - b * cc;
    singular(delta)?;
    Ok([[((a - z0) * (d + z0) - b * cc) / delta, c(2.0 * z0) * b / delta], [c(2.0 * z0) * cc / delta, ((a + z0) * (d - z0) - b * cc) / delta]])
}

/// S parameters to Z parameters.
pub fn s_to_z(s: &Matrix2, z0: f64) -> Result<Matrix2, ErrorsJSL> {
    let (a, b, cc, d) = (s[0][0], s[0][1], s[1][0], s[1][1]);
    let one = c(1.0);
    let delta = (one - a) * (one - d) - b * cc;
    singular(delta)?;
    Ok([[c(z0) * ((one + a) * (one - d) + b * cc) / delta, c(2.0 * z0) * b / delta], [c(2.0 * z0) * cc / delta, c(z0) * ((one - a) * (one + d) + b * cc) / delta]])
}

/// The matrix inverse (Z ↔ Y).
pub fn invert(m: &Matrix2) -> Result<Matrix2, ErrorsJSL> {
    let d = det(m);
    singular(d)?;
    Ok([[m[1][1] / d, -m[0][1] / d], [-m[1][0] / d, m[0][0] / d]])
}

/// S parameters to ABCD (chain) parameters.
pub fn s_to_abcd(s: &Matrix2, z0: f64) -> Result<Matrix2, ErrorsJSL> {
    let (s11, s12, s21, s22) = (s[0][0], s[0][1], s[1][0], s[1][1]);
    singular(s21)?;
    let one = c(1.0);
    let two = c(2.0) * s21;
    Ok([
        [((one + s11) * (one - s22) + s12 * s21) / two, c(z0) * ((one + s11) * (one + s22) - s12 * s21) / two],
        [((one - s11) * (one - s22) - s12 * s21) / (two * z0), ((one - s11) * (one + s22) + s12 * s21) / two],
    ])
}

/// ABCD parameters to S parameters.
pub fn abcd_to_s(m: &Matrix2, z0: f64) -> Result<Matrix2, ErrorsJSL> {
    let (a, b, cc, d) = (m[0][0], m[0][1], m[1][0] * z0, m[1][1]);
    let b = b / z0;
    let denominator = a + b + cc + d;
    singular(denominator)?;
    Ok([[(a + b - cc - d) / denominator, c(2.0) * det(m) / denominator], [c(2.0) / denominator, (-a + b - cc + d) / denominator]])
}

/// S parameters to scattering transfer (T) parameters, `[b₁, a₁]ᵀ = T [a₂, b₂]ᵀ`, which
/// multiply for cascades.
pub fn s_to_t(s: &Matrix2) -> Result<Matrix2, ErrorsJSL> {
    singular(s[1][0])?;
    let s21 = s[1][0];
    Ok([[-det(s) / s21, s[0][0] / s21], [-s[1][1] / s21, c(1.0) / s21]])
}

/// T parameters back to S parameters.
pub fn t_to_s(t: &Matrix2) -> Result<Matrix2, ErrorsJSL> {
    singular(t[1][1])?;
    let t22 = t[1][1];
    Ok([[t[0][1] / t22, det(t) / t22], [c(1.0) / t22, -t[1][0] / t22]])
}

/// The S parameters of two networks in cascade (port 2 of `first` to port 1 of `second`).
pub fn cascade(first: &Matrix2, second: &Matrix2, z0: f64) -> Result<Matrix2, ErrorsJSL> {
    abcd_to_s(&multiply(&s_to_abcd(first, z0)?, &s_to_abcd(second, z0)?), z0)
}

/// Whether `S = Sᵀ` (reciprocal network) to the tolerance.
pub fn is_reciprocal(s: &Matrix2, tolerance: f64) -> bool {
    (s[0][1] - s[1][0]).norm() <= tolerance
}

/// Whether `SᴴS = I` (lossless network) to the tolerance.
pub fn is_lossless(s: &Matrix2, tolerance: f64) -> bool {
    let col = |j: usize, k: usize| s[0][j].conj() * s[0][k] + s[1][j].conj() * s[1][k];
    (col(0, 0) - 1.0).norm() <= tolerance && (col(1, 1) - 1.0).norm() <= tolerance && col(0, 1).norm() <= tolerance
}

/// Insertion loss `−20 log₁₀|S₂₁|` in dB.
pub fn insertion_loss_db(s: &Matrix2) -> f64 {
    -20.0 * s[1][0].norm().log10()
}

/// ABCD matrices of the elementary sections.
pub fn series_impedance(z: C64) -> Matrix2 {
    [[c(1.0), z], [c(0.0), c(1.0)]]
}

pub fn shunt_admittance(y: C64) -> Matrix2 {
    [[c(1.0), c(0.0)], [y, c(1.0)]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::networks::transmission_lines::TransmissionLine;

    fn close(a: &Matrix2, b: &Matrix2, tol: f64) -> bool {
        (0..2).all(|i| (0..2).all(|j| (a[i][j] - b[i][j]).norm() < tol))
    }

    #[test]
    fn elementary_networks_have_textbook_s_parameters() {
        let z0 = 50.0;
        let z = C64::new(30.0, 40.0);
        let s = abcd_to_s(&series_impedance(z), z0).unwrap();
        let d = z + 2.0 * z0;
        assert!((s[0][0] - z / d).norm() < 1e-14 && (s[1][0] - 2.0 * z0 / d).norm() < 1e-14);
        assert!(is_reciprocal(&s, 1e-14) && !is_lossless(&s, 1e-6));
        // A 6 dB matched attenuator: S₂₁ = ½, S₁₁ = 0.
        let attenuator = [[c(0.0), c(0.5)], [c(0.5), c(0.0)]];
        assert!((insertion_loss_db(&attenuator) - 6.0206).abs() < 1e-4);
        // A lossless line section is lossless and reciprocal with |S₂₁| = 1 when matched.
        let line = TransmissionLine::lossless(z0, 1.0).unwrap();
        let s_line = abcd_to_s(&line.abcd(0.123, 1e9), z0).unwrap();
        assert!(is_lossless(&s_line, 1e-12) && s_line[0][0].norm() < 1e-12);
    }

    #[test]
    fn conversions_round_trip_and_cascades_agree() {
        let z0 = 50.0;
        let s = [[C64::new(0.2, -0.1), C64::new(0.6, 0.3)], [C64::new(0.55, 0.25), C64::new(-0.15, 0.2)]];
        let z = s_to_z(&s, z0).unwrap();
        assert!(close(&z_to_s(&z, z0).unwrap(), &s, 1e-12));
        let y = invert(&z).unwrap();
        assert!(close(&invert(&y).unwrap(), &z, 1e-9));
        assert!(close(&abcd_to_s(&s_to_abcd(&s, z0).unwrap(), z0).unwrap(), &s, 1e-12));
        assert!(close(&t_to_s(&s_to_t(&s).unwrap()).unwrap(), &s, 1e-12));
        // Two line sections in cascade equal one longer section; T matrices multiply too.
        let line = TransmissionLine::lossless(75.0, 0.8).unwrap();
        let (a, b) = (abcd_to_s(&line.abcd(0.1, 1e9), z0).unwrap(), abcd_to_s(&line.abcd(0.25, 1e9), z0).unwrap());
        let whole = abcd_to_s(&line.abcd(0.35, 1e9), z0).unwrap();
        assert!(close(&cascade(&a, &b, z0).unwrap(), &whole, 1e-12));
        assert!(close(&t_to_s(&multiply(&s_to_t(&a).unwrap(), &s_to_t(&b).unwrap())).unwrap(), &whole, 1e-12));
        assert!(s_to_t(&[[c(1.0), c(0.0)], [c(0.0), c(1.0)]]).is_err());
    }
}
