//! The method of moments for straight thin-wire antennas (Hallén's integral equation).
//!
//! A centre-fed straight wire of length `L` and radius `a ≪ λ` along z carries a current `I(z)`
//! with `I(±L/2) = 0`. Hallén's equation (engineering convention `e^{+jωt}`) for a delta-gap
//! voltage `V` is
//!
//! `∫ I(z') G(z − z') dz' = C cos kz − (jV / 2η) sin k|z|`,  `G = e^{−jkR} / (4πR)`,
//! `R = √((z − z')² + a²)` (the reduced kernel),
//!
//! with an unknown constant `C`. The current is expanded in triangle functions on `N` interior
//! nodes and the equation matched at those nodes and at the wire end, giving a square system for
//! the node currents and `C`. The near-singular `1/R` part of the kernel is integrated
//! analytically and the smooth remainder by Gauss-Legendre quadrature.

use crate::conventions::{C64, Medium, TimeConvention};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_core::constants::em::C0;
use std::f64::consts::{PI, TAU};

/// A centre-fed straight wire dipole along z.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WireDipole {
    pub length: f64,
    pub radius: f64,
}

/// The moment-method solution at one frequency for a 1 V delta-gap excitation.
#[derive(Debug, Clone)]
pub struct WireSolution {
    pub frequency: f64,
    /// Node positions along the wire (m), ends excluded (where the current is zero).
    pub nodes: Vec<f64>,
    /// Node currents (A) for 1 V across the gap, in the `e^{+jωt}` convention.
    pub currents: Vec<C64>,
    /// The input impedance `V / I(0)` (Ω).
    pub input_impedance: C64,
    length: f64,
}

const GAUSS_8: [(f64, f64); 8] = [
    (-0.960_289_856_497_536_2, 0.101_228_536_290_376_26),
    (-0.796_666_477_413_626_7, 0.222_381_034_453_374_47),
    (-0.525_532_409_916_329, 0.313_706_645_877_887_3),
    (-0.183_434_642_495_649_8, 0.362_683_783_378_362),
    (0.183_434_642_495_649_8, 0.362_683_783_378_362),
    (0.525_532_409_916_329, 0.313_706_645_877_887_3),
    (0.796_666_477_413_626_7, 0.222_381_034_453_374_47),
    (0.960_289_856_497_536_2, 0.101_228_536_290_376_26),
];

/// `∫_{u0}^{u1} (α + βu) / √(u² + a²) du`.
fn linear_over_r(alpha: f64, beta: f64, u0: f64, u1: f64, a: f64) -> f64 {
    let f = |u: f64| alpha * (u / a).asinh() + beta * (u * u + a * a).sqrt();
    f(u1) - f(u0)
}

/// Solves `A x = b` for a dense complex system by Gaussian elimination with partial pivoting.
fn solve_complex(mut a: Vec<Vec<C64>>, mut b: Vec<C64>) -> Result<Vec<C64>, ErrorsJSL> {
    let n = b.len();
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| a[i][k].norm().total_cmp(&a[j][k].norm())).expect("non-empty");
        if a[p][k].norm() < 1e-300 {
            return Err(ErrorsJSL::RuntimeError("The moment matrix is singular."));
        }
        a.swap(k, p);
        b.swap(k, p);
        for i in k + 1..n {
            let m = a[i][k] / a[k][k];
            let (top, rest) = a.split_at_mut(i);
            rest[0][k..].iter_mut().zip(&top[k][k..]).for_each(|(x, y)| *x -= m * y);
            b[i] = b[i] - m * b[k];
        }
    }
    for i in (0..n).rev() {
        let s: C64 = (i + 1..n).map(|j| a[i][j] * b[j]).sum();
        b[i] = (b[i] - s) / a[i][i];
    }
    Ok(b)
}

impl WireDipole {
    pub fn new(length: f64, radius: f64) -> Result<Self, ErrorsJSL> {
        if length.is_nan() || radius.is_nan() || length <= 0.0 || radius <= 0.0 || radius * 20.0 > length {
            return Err(ErrorsJSL::InvalidInputRange("Need a thin wire: 0 < 20 a < L."));
        }
        Ok(WireDipole { length, radius })
    }

    /// Solves for the current with `interior_nodes` (odd, so a node sits at the feed) at
    /// `frequency`. Segments should stay longer than a few radii for the thin-wire kernel.
    pub fn solve(&self, frequency: f64, interior_nodes: usize) -> Result<WireSolution, ErrorsJSL> {
        if interior_nodes < 3 || interior_nodes.is_multiple_of(2) || frequency.is_nan() || frequency <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("Need an odd number (>= 3) of interior nodes and a positive frequency."));
        }
        let n = interior_nodes;
        let (h, a) = (self.length / 2.0, self.radius);
        let k = TAU * frequency / C0;
        let eta = Medium::VACUUM.impedance(TimeConvention::PlusJ).re;
        let delta = self.length / (n + 1) as f64;
        let node = |i: usize| -h + i as f64 * delta; // i = 0 ..= n + 1, ends at 0 and n + 1
        // Match at the interior nodes and at one end: midpoint matching against triangle bases
        // decouples alternate nodes and makes the current oscillate.
        let matches: Vec<f64> = (1..=n).map(node).chain(std::iter::once(h)).collect();
        let j = C64::new(0.0, 1.0);
        let mut matrix = vec![vec![C64::new(0.0, 0.0); n + 1]; n + 1];
        let mut rhs = vec![C64::new(0.0, 0.0); n + 1];
        for (m, &z) in matches.iter().enumerate() {
            for (col, entry) in matrix[m].iter_mut().take(n).enumerate() {
                let (left, centre, right) = (node(col), node(col + 1), node(col + 2));
                // Analytic part: ∫ Λ(z') / (4πR) dz' on both halves of the triangle.
                let mut value = C64::new(
                    (linear_over_r((z - left) / delta, 1.0 / delta, left - z, centre - z, a) + linear_over_r((right - z) / delta, -1.0 / delta, centre - z, right - z, a)) / (4.0 * PI),
                    0.0,
                );
                // Smooth part: (e^{−jkR} − 1) / (4πR), finite as R → a.
                for (lo, hi, rising) in [(left, centre, true), (centre, right, false)] {
                    let (mid, half) = (0.5 * (lo + hi), 0.5 * (hi - lo));
                    for &(x, w) in &GAUSS_8 {
                        let zp = mid + half * x;
                        let shape = if rising { (zp - lo) / delta } else { (hi - zp) / delta };
                        let r = ((z - zp).powi(2) + a * a).sqrt();
                        value += w * half * shape * ((-j * k * r).exp() - 1.0) / (4.0 * PI * r);
                    }
                }
                *entry = value;
            }
            matrix[m][n] = C64::new(-(k * z).cos(), 0.0);
            rhs[m] = -j / (2.0 * eta) * (k * z.abs()).sin();
        }
        let x = solve_complex(matrix, rhs)?;
        let currents = x[..n].to_vec();
        let feed = currents[n / 2];
        Ok(WireSolution { frequency, nodes: (1..=n).map(node).collect(), currents, input_impedance: C64::new(1.0, 0.0) / feed, length: self.length })
    }
}

impl WireSolution {
    /// The far field `E_θ` (V, with `e^{−jkr}/r` removed) at polar angle `θ` for the 1 V feed:
    /// `jηk sinθ / (4π) ∫ I(z') e^{jkz' cosθ} dz'`, integrating the piecewise-linear current
    /// exactly segment by segment with Gauss quadrature.
    pub fn far_field_theta(&self, theta: f64, convention: TimeConvention) -> C64 {
        let k = TAU * self.frequency / C0;
        let eta = Medium::VACUUM.impedance(TimeConvention::PlusJ).re;
        let h = self.length / 2.0;
        let mut z: Vec<f64> = vec![-h];
        z.extend(&self.nodes);
        z.push(h);
        let mut i: Vec<C64> = vec![C64::new(0.0, 0.0)];
        i.extend(&self.currents);
        i.push(C64::new(0.0, 0.0));
        let mut integral = C64::new(0.0, 0.0);
        for s in 0..z.len() - 1 {
            let (mid, half) = (0.5 * (z[s] + z[s + 1]), 0.5 * (z[s + 1] - z[s]));
            for &(x, w) in &GAUSS_8 {
                let current = i[s] + (i[s + 1] - i[s]) * (0.5 * (x + 1.0));
                integral += w * half * current * C64::from_polar(1.0, k * (mid + half * x) * theta.cos());
            }
        }
        let value = C64::new(0.0, eta * k * theta.sin() / (4.0 * PI)) * integral;
        TimeConvention::PlusJ.convert(value, convention)
    }

    /// The radiated power (W) for the 1 V feed by integrating `|E_θ|² / (2η)` over the sphere.
    pub fn radiated_power(&self) -> f64 {
        let eta = Medium::VACUUM.impedance(TimeConvention::PlusJ).re;
        let steps = 720;
        let d = PI / steps as f64;
        (0..steps)
            .map(|s| {
                let theta = (s as f64 + 0.5) * d;
                self.far_field_theta(theta, TimeConvention::PlusJ).norm_sqr() / (2.0 * eta) * TAU * theta.sin() * d
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const F: f64 = 300e6;

    fn lambda() -> f64 {
        C0 / F
    }

    #[test]
    fn half_wave_dipole_impedance_and_power_balance() {
        let dipole = WireDipole::new(0.5 * lambda(), 1e-4 * lambda()).unwrap();
        let solution = dipole.solve(F, 61).unwrap();
        let z = solution.input_impedance;
        // Thin-wire limit 73 + j42.5 Ω; a finite radius raises both a little.
        assert!(z.re > 70.0 && z.re < 90.0 && z.im > 30.0 && z.im < 55.0, "{z}");
        // Power conservation: the radiated power equals the power delivered, ½ Re(Z) |I|².
        let delivered = 0.5 * (C64::new(1.0, 0.0) / z.conj()).re;
        assert!((solution.radiated_power() / delivered - 1.0).abs() < 0.02, "{} vs {delivered}", solution.radiated_power());
        // The current is close to the sinusoid cos(kz) of a half-wave dipole.
        let k = TAU / lambda();
        let feed = solution.currents[30];
        for (z, i) in solution.nodes.iter().zip(&solution.currents).step_by(6) {
            assert!(((i / feed).norm() - (k * z).cos()).abs() < 0.1, "z={z}");
        }
    }

    #[test]
    fn resonance_falls_just_short_of_half_a_wavelength_and_converges() {
        let reactance = |fraction: f64| WireDipole::new(fraction * lambda(), 1e-3 * lambda()).unwrap().solve(F, 41).unwrap().input_impedance.im;
        assert!(reactance(0.45) < 0.0 && reactance(0.5) > 0.0);
        // Bisect the resonant length.
        let (mut lo, mut hi) = (0.45, 0.5);
        for _ in 0..30 {
            let mid = 0.5 * (lo + hi);
            if reactance(mid) < 0.0 { lo = mid } else { hi = mid }
        }
        assert!(lo > 0.46 && lo < 0.49, "{lo}");
        let resonant = WireDipole::new(lo * lambda(), 1e-3 * lambda()).unwrap().solve(F, 41).unwrap().input_impedance;
        assert!(resonant.re > 55.0 && resonant.re < 75.0, "{resonant}");
        // Refining the mesh changes the impedance little.
        let d = WireDipole::new(0.5 * lambda(), 1e-3 * lambda()).unwrap();
        let (coarse, fine) = (d.solve(F, 41).unwrap().input_impedance, d.solve(F, 81).unwrap().input_impedance);
        assert!((coarse - fine).norm() < 0.03 * fine.norm(), "{coarse} {fine}");
        assert!(d.solve(F, 40).is_err() && WireDipole::new(1.0, 0.1).is_err());
    }

    #[test]
    fn short_dipole_matches_the_hertzian_radiation_resistance() {
        // A short dipole's triangular current gives R = 20π² (L/λ)² (a quarter of the uniform case).
        // The delta-gap feed adds a gap capacitance that grows slowly as the mesh is refined, which
        // lowers R a little below the ideal value, so allow 15% on a coarse mesh.
        let l = 0.05 * lambda();
        let solution = WireDipole::new(l, 1e-4 * lambda()).unwrap().solve(F, 11).unwrap();
        let expected = 20.0 * PI * PI * 0.05f64.powi(2);
        assert!((solution.input_impedance.re / expected - 1.0).abs() < 0.15, "{} vs {expected}", solution.input_impedance.re);
        // R scales as (L/λ)².
        let half = WireDipole::new(l / 2.0, 0.5e-4 * lambda()).unwrap().solve(F, 11).unwrap();
        assert!((solution.input_impedance.re / half.input_impedance.re / 4.0 - 1.0).abs() < 0.05);
        assert!(solution.input_impedance.im < -100.0); // strongly capacitive
        let peak = solution.far_field_theta(std::f64::consts::FRAC_PI_2, TimeConvention::PlusJ);
        assert!((solution.far_field_theta(0.3, TimeConvention::MinusI) - solution.far_field_theta(0.3, TimeConvention::PlusJ).conj()).norm() < 1e-12 * peak.norm());
    }
}
