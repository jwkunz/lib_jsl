//! Smith chart geometry: the map between normalised impedance and reflection coefficient, the
//! constant-resistance and constant-reactance circles, and impedance loci along a line.

use crate::conventions::C64;

/// The reflection coefficient of the normalised impedance `z = Z/Z₀`: `Γ = (z − 1)/(z + 1)`.
pub fn impedance_to_gamma(z: C64) -> C64 {
    if z.re.is_infinite() || z.im.is_infinite() {
        return C64::new(1.0, 0.0);
    }
    (z - 1.0) / (z + 1.0)
}

/// The normalised impedance at reflection coefficient `Γ`: `z = (1 + Γ)/(1 − Γ)`.
pub fn gamma_to_impedance(gamma: C64) -> C64 {
    (1.0 + gamma) / (1.0 - gamma)
}

/// The normalised admittance at `Γ` (the impedance at `−Γ`): reading the chart as an admittance
/// chart is a rotation by 180°.
pub fn gamma_to_admittance(gamma: C64) -> C64 {
    gamma_to_impedance(-gamma)
}

/// The circle of constant normalised resistance `r`: centre `(r/(1+r), 0)`, radius `1/(1+r)`.
pub fn resistance_circle(r: f64) -> (C64, f64) {
    (C64::new(r / (1.0 + r), 0.0), 1.0 / (1.0 + r))
}

/// The circle of constant normalised reactance `x ≠ 0`: centre `(1, 1/x)`, radius `1/|x|`.
pub fn reactance_circle(x: f64) -> (C64, f64) {
    (C64::new(1.0, 1.0 / x), 1.0 / x.abs())
}

/// The circle of constant VSWR through `Γ`: centred on the chart with radius `|Γ|`.
pub fn vswr_circle(gamma: C64) -> (C64, f64) {
    (C64::new(0.0, 0.0), gamma.norm())
}

/// The reflection coefficient seen `d` wavelengths from the load towards the generator on a
/// lossless line: `Γ(d) = Γ_L e^{−j4πd}` (clockwise rotation, once per half wavelength).
pub fn towards_generator(gamma_load: C64, wavelengths: f64) -> C64 {
    gamma_load * C64::from_polar(1.0, -4.0 * std::f64::consts::PI * wavelengths)
}

/// The locus of `Γ` along a lossless line from the load, at the given distances (wavelengths).
pub fn impedance_locus(z_load: C64, distances: &[f64]) -> Vec<C64> {
    let g = impedance_to_gamma(z_load);
    distances.iter().map(|&d| towards_generator(g, d)).collect()
}

/// Points of a circle (centre, radius) clipped to the unit disc, for plotting chart grids.
pub fn circle_points(centre: C64, radius: f64, samples: usize) -> Vec<C64> {
    (0..samples)
        .map(|k| centre + C64::from_polar(radius, std::f64::consts::TAU * k as f64 / samples as f64))
        .filter(|p| p.norm() <= 1.0 + 1e-12)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chart_circles_contain_the_mapped_impedances() {
        for r in [0.0, 0.5, 1.0, 3.0] {
            let (centre, radius) = resistance_circle(r);
            for x in [-5.0, -0.3, 0.0, 0.7, 20.0] {
                let g = impedance_to_gamma(C64::new(r, x));
                assert!(((g - centre).norm() - radius).abs() < 1e-12, "r={r} x={x}");
                if x != 0.0 {
                    let (cx, rx) = reactance_circle(x);
                    assert!(((g - cx).norm() - rx).abs() < 1e-12);
                }
            }
        }
        assert_eq!(impedance_to_gamma(C64::new(1.0, 0.0)), C64::new(0.0, 0.0));
        assert_eq!(impedance_to_gamma(C64::new(f64::INFINITY, 0.0)), C64::new(1.0, 0.0));
        let z = C64::new(0.4, -1.3);
        assert!((gamma_to_impedance(impedance_to_gamma(z)) - z).norm() < 1e-12);
        assert!((gamma_to_admittance(impedance_to_gamma(z)) - 1.0 / z).norm() < 1e-12);
    }

    #[test]
    fn rotation_towards_the_generator_inverts_at_a_quarter_wave() {
        let z = C64::new(2.0, 0.5);
        let quarter = gamma_to_impedance(towards_generator(impedance_to_gamma(z), 0.25));
        assert!((quarter - 1.0 / z).norm() < 1e-12);
        let locus = impedance_locus(z, &[0.0, 0.1, 0.5]);
        assert!((locus[2] - locus[0]).norm() < 1e-12 && (locus[1].norm() - locus[0].norm()).abs() < 1e-12);
        assert!((vswr_circle(locus[0]).1 - impedance_to_gamma(z).norm()).abs() < 1e-15);
        assert!(circle_points(C64::new(0.0, 0.0), 1.0, 36).len() == 36);
    }
}
