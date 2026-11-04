//! Spherical wave expansions of radiated fields.
//!
//! Outside a sphere enclosing all sources, the field is a sum of outgoing vector spherical wave
//! functions (engineering convention `e^{+jωt}`, spherical Hankel functions `h_n = h_n^{(2)}`):
//!
//! `E = Σ_{n,m} α_nm M_nm + β_nm N_nm`, `M_nm = h_n(kr) Φ_nm`,
//! `N_nm = (1/k)∇×M_nm = −√(n(n+1)) h_n(kr)/(kr) Y_nm r̂ − (1/kr) (kr h_n)' Ψ_nm`,
//!
//! where `Y_nm` are the orthonormal spherical harmonics, `Ψ_nm = r∇Y_nm / √(n(n+1))` and
//! `Φ_nm = r̂ × Ψ_nm` the orthonormal tangential vector spherical harmonics. The far-field
//! pattern (with `e^{−jkr}/r` removed) is `F = Σ (j^{n+1}/k) α_nm Φ_nm + (j^{n+2}/k) β_nm Ψ_nm`,
//! so projecting a pattern onto `Φ_nm` and `Ψ_nm` gives the wave amplitudes, from which the
//! field follows at any radius outside the minimum sphere. A source of electrical radius
//! `k r₀` needs degrees up to about `k r₀ + 10`.
//!
//! See the book entry [Spherical waves and near-field to far-field transformation](https://jwkunz.github.io/lib_jsl/book/em-waves/near-to-far.html).

use crate::conventions::{C64, Medium, TimeConvention};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_core::constants::em::C0;
use lib_jsl_special_functions::{bessel_spherical, bessel_spherical_y, legendre_with_derivative, spherical_harmonic};
use std::f64::consts::{PI, TAU};

/// The tangential vector spherical harmonics `(Ψ_nm, Φ_nm)` as `[θ, φ]` components (`n ≥ 1`,
/// `|m| ≤ n`). Poles are approached rather than evaluated (θ is nudged by 1e-9).
pub fn vector_spherical_harmonics(n: u32, m: i32, theta: f64, phi: f64) -> ([C64; 2], [C64; 2]) {
    let theta = theta.clamp(1e-9, PI - 1e-9);
    let (nf, mf) = (n as f64, m as f64);
    // ∂Y/∂θ = ½ [√((n−m)(n+m+1)) e^{−jφ} Y_n^{m+1} − √((n+m)(n−m+1)) e^{jφ} Y_n^{m−1}].
    let up = if m < n as i32 { ((nf - mf) * (nf + mf + 1.0)).sqrt() * C64::from_polar(1.0, -phi) * spherical_harmonic(n, m + 1, theta, phi) } else { C64::new(0.0, 0.0) };
    let down = if m > -(n as i32) { ((nf + mf) * (nf - mf + 1.0)).sqrt() * C64::from_polar(1.0, phi) * spherical_harmonic(n, m - 1, theta, phi) } else { C64::new(0.0, 0.0) };
    let d_theta = 0.5 * (up - down);
    let d_phi_over_sin = C64::new(0.0, mf) * spherical_harmonic(n, m, theta, phi) / theta.sin();
    let norm = 1.0 / (nf * (nf + 1.0)).sqrt();
    let psi = [d_theta * norm, d_phi_over_sin * norm];
    // r̂ × (a θ̂ + b φ̂) = a φ̂ − b θ̂.
    let phi_vec = [-psi[1], psi[0]];
    (psi, phi_vec)
}

/// The outgoing spherical Hankel function `h_n^{(2)}(x) = j_n(x) − j y_n(x)` and
/// `(x h_n(x))'`.
pub(crate) fn hankel(n: u32, x: f64) -> (C64, C64) {
    let h = |k: i32| C64::new(bessel_spherical(k, x), -bessel_spherical_y(k, x));
    let hn = h(n as i32);
    (hn, x * h(n as i32 - 1) - n as f64 * hn)
}

/// Gauss-Legendre nodes and weights on [−1, 1] by Newton's method.
pub(crate) fn gauss_legendre(count: usize) -> Vec<(f64, f64)> {
    (0..count)
        .map(|i| {
            let mut x = (PI * (i as f64 + 0.75) / (count as f64 + 0.5)).cos();
            for _ in 0..100 {
                let (p, dp) = legendre_with_derivative(count as u32, x);
                let dx = p / dp;
                x -= dx;
                if dx.abs() < 1e-16 {
                    break;
                }
            }
            let dp = legendre_with_derivative(count as u32, x).1;
            (x, 2.0 / ((1.0 - x * x) * dp * dp))
        })
        .collect()
}

/// A spherical wave expansion truncated at degree `N`.
#[derive(Debug, Clone)]
pub struct SphericalWaveExpansion {
    pub frequency: f64,
    pub max_degree: u32,
    /// TE amplitudes `α_nm`, indexed by [`SphericalWaveExpansion::index`].
    pub te: Vec<C64>,
    /// TM amplitudes `β_nm`.
    pub tm: Vec<C64>,
}

impl SphericalWaveExpansion {
    /// The storage index of `(n, m)`: `n² − 1 + n + m` (degrees from 1).
    pub fn index(n: u32, m: i32) -> usize {
        ((n * n) as i64 - 1 + n as i64 + m as i64) as usize
    }

    fn wavenumber(&self) -> f64 {
        TAU * self.frequency / C0
    }

    /// Expands a far-field pattern `[F_θ, F_φ]` (V, `e^{−jkr}/r` removed, in `convention`) up to
    /// degree `max_degree`, projecting by Gauss-Legendre quadrature in θ and the trapezoid rule
    /// in φ (exact for patterns band-limited to that degree).
    pub fn from_far_field<F>(pattern: F, frequency: f64, max_degree: u32, convention: TimeConvention) -> Result<Self, ErrorsJSL>
    where
        F: Fn(f64, f64) -> [C64; 2],
    {
        if max_degree == 0 || frequency.is_nan() || frequency <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("Need a positive frequency and maximum degree."));
        }
        let size = Self::index(max_degree, max_degree as i32) + 1;
        let (mut a, mut b) = (vec![C64::new(0.0, 0.0); size], vec![C64::new(0.0, 0.0); size]);
        let thetas = gauss_legendre(max_degree as usize + 2);
        let phis = 2 * max_degree as usize + 2;
        // ANCHOR: swe_projection
        for &(x, w) in &thetas {
            let theta = x.acos();
            for p in 0..phis {
                let phi = TAU * p as f64 / phis as f64;
                let f = pattern(theta, phi).map(|v| convention.convert(v, TimeConvention::PlusJ));
                let weight = w * TAU / phis as f64;
                for n in 1..=max_degree {
                    for m in -(n as i32)..=n as i32 {
                        let (psi, phv) = vector_spherical_harmonics(n, m, theta, phi);
                        let k = Self::index(n, m);
                        a[k] += weight * (f[0] * psi[0].conj() + f[1] * psi[1].conj());
                        b[k] += weight * (f[0] * phv[0].conj() + f[1] * phv[1].conj());
                    }
                }
            }
        }
        // ANCHOR_END: swe_projection
        let k = TAU * frequency / C0;
        let j = C64::new(0.0, 1.0);
        let mut te = vec![C64::new(0.0, 0.0); size];
        let mut tm = vec![C64::new(0.0, 0.0); size];
        for n in 1..=max_degree {
            for m in -(n as i32)..=n as i32 {
                let i = Self::index(n, m);
                te[i] = k * b[i] / j.powu(n + 1);
                tm[i] = k * a[i] / j.powu(n + 2);
            }
        }
        Ok(SphericalWaveExpansion { frequency, max_degree, te, tm })
    }

    /// The far-field pattern `[F_θ, F_φ]` in `convention`.
    pub fn far_field(&self, theta: f64, phi: f64, convention: TimeConvention) -> [C64; 2] {
        let k = self.wavenumber();
        let j = C64::new(0.0, 1.0);
        let mut f = [C64::new(0.0, 0.0); 2];
        for n in 1..=self.max_degree {
            for m in -(n as i32)..=n as i32 {
                let i = Self::index(n, m);
                let (psi, phv) = vector_spherical_harmonics(n, m, theta, phi);
                let (ca, cb) = (self.te[i] * j.powu(n + 1) / k, self.tm[i] * j.powu(n + 2) / k);
                for c in 0..2 {
                    f[c] += ca * phv[c] + cb * psi[c];
                }
            }
        }
        f.map(|v| TimeConvention::PlusJ.convert(v, convention))
    }

    /// The electric field `[E_r, E_θ, E_φ]` at a point outside the minimum sphere.
    pub fn field_at(&self, r: f64, theta: f64, phi: f64, convention: TimeConvention) -> [C64; 3] {
        let kr = self.wavenumber() * r;
        let mut e = [C64::new(0.0, 0.0); 3];
        for n in 1..=self.max_degree {
            let (h, dh) = hankel(n, kr);
            let nn = (n as f64 * (n as f64 + 1.0)).sqrt();
            for m in -(n as i32)..=n as i32 {
                let i = Self::index(n, m);
                let (psi, phv) = vector_spherical_harmonics(n, m, theta, phi);
                e[0] -= self.tm[i] * nn * h / kr * spherical_harmonic(n, m, theta.clamp(1e-9, PI - 1e-9), phi);
                for c in 0..2 {
                    e[c + 1] += self.te[i] * h * phv[c] - self.tm[i] * dh / kr * psi[c];
                }
            }
        }
        e.map(|v| TimeConvention::PlusJ.convert(v, convention))
    }

    /// The radiated power `(1/2η) ∫ |F|² dΩ` from the coefficients (Parseval).
    pub fn radiated_power(&self) -> f64 {
        let k = self.wavenumber();
        let eta = Medium::VACUUM.impedance(TimeConvention::PlusJ).re;
        self.te.iter().chain(&self.tm).map(|c| c.norm_sqr()).sum::<f64>() / (k * k * 2.0 * eta)
    }

    /// The fraction of power carried by each degree `n = 1..=N`.
    pub fn power_by_degree(&self) -> Vec<f64> {
        let total: f64 = self.te.iter().chain(&self.tm).map(|c| c.norm_sqr()).sum();
        (1..=self.max_degree)
            .map(|n| (-(n as i32)..=n as i32).map(|m| { let i = Self::index(n, m); self.te[i].norm_sqr() + self.tm[i].norm_sqr() }).sum::<f64>() / total)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::antenna_models::{Antenna, HalfWaveDipole, HertzianDipole};
    use crate::coordinates::systems::Position;

    const F: f64 = 1e9;

    #[test]
    fn vector_harmonics_are_orthonormal_and_match_finite_differences() {
        let nodes = gauss_legendre(12);
        let modes = [(1u32, 0i32), (1, 1), (2, -1), (3, 2)];
        for &(n1, m1) in &modes {
            for &(n2, m2) in &modes {
                let (mut pp, mut ff, mut pf) = (C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0));
                for &(x, w) in &nodes {
                    for p in 0..16 {
                        let (t, ph) = (x.acos(), TAU * p as f64 / 16.0);
                        let (a1, b1) = vector_spherical_harmonics(n1, m1, t, ph);
                        let (a2, b2) = vector_spherical_harmonics(n2, m2, t, ph);
                        let dw = w * TAU / 16.0;
                        pp += dw * (a1[0] * a2[0].conj() + a1[1] * a2[1].conj());
                        ff += dw * (b1[0] * b2[0].conj() + b1[1] * b2[1].conj());
                        pf += dw * (a1[0] * b2[0].conj() + a1[1] * b2[1].conj());
                    }
                }
                let want = if (n1, m1) == (n2, m2) { 1.0 } else { 0.0 };
                assert!((pp - want).norm() < 1e-12 && (ff - want).norm() < 1e-12 && pf.norm() < 1e-12, "({n1},{m1}) ({n2},{m2})");
            }
        }
        // The θ derivative identity against a finite difference.
        let (t, ph, h) = (0.7, 0.4, 1e-6);
        let numeric = (spherical_harmonic(3, 2, t + h, ph) - spherical_harmonic(3, 2, t - h, ph)) / (2.0 * h);
        let (psi, _) = vector_spherical_harmonics(3, 2, t, ph);
        assert!((psi[0] * 12f64.sqrt() - numeric).norm() < 1e-8);
    }

    #[test]
    fn a_hertzian_dipole_is_a_single_tm_mode_and_its_near_field_is_reconstructed() {
        let dipole = HertzianDipole::new(0.01);
        let swe = SphericalWaveExpansion::from_far_field(|t, p| dipole.far_field(t, p, F, TimeConvention::PlusJ), F, 4, TimeConvention::PlusJ).unwrap();
        let power = swe.power_by_degree();
        assert!(power[0] > 1.0 - 1e-12 && swe.te.iter().all(|c| c.norm() < 1e-12 * swe.tm[SphericalWaveExpansion::index(1, 0)].norm()));
        // The exact near field, reactive terms included, from the far-field coefficients alone.
        let lambda = C0 / F;
        for &(r, t, p) in &[(0.2 * lambda, 0.6, 0.3), (0.5 * lambda, 1.4, 2.0), (3.0 * lambda, 2.5, -1.0)] {
            let e = swe.field_at(r, t, p, TimeConvention::PlusJ);
            let (exact, _) = dipole.near_field(&Position::spherical(r, t, p), F, TimeConvention::PlusJ);
            let scale = exact.iter().map(|v| v.norm()).fold(0.0, f64::max);
            for c in 0..3 {
                assert!((e[c] - exact[c]).norm() < 1e-9 * scale, "r={r} component {c}: {} vs {}", e[c], exact[c]);
            }
        }
        assert!((swe.radiated_power() / dipole.radiated_power(F) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn half_wave_dipole_pattern_converges_and_conventions_conjugate() {
        let antenna = HalfWaveDipole::new(F);
        let pattern = |t: f64, p: f64| antenna.far_field(t, p, F, TimeConvention::PlusJ);
        let swe = SphericalWaveExpansion::from_far_field(pattern, F, 15, TimeConvention::PlusJ).unwrap();
        for &(t, p) in &[(0.4, 0.0), (1.2, 1.0), (2.0, 4.0)] {
            let (a, b) = (swe.far_field(t, p, TimeConvention::PlusJ), pattern(t, p));
            assert!((a[0] - b[0]).norm() < 1e-6 * b[0].norm().max(1.0), "θ={t}");
        }
        assert!((swe.radiated_power() / antenna.radiated_power(F) - 1.0).abs() < 1e-6);
        // Only odd TM degrees with m = 0 (a z-directed, symmetric current).
        assert!(swe.tm[SphericalWaveExpansion::index(2, 0)].norm() < 1e-9 * swe.tm[SphericalWaveExpansion::index(1, 0)].norm());
        let minus = swe.far_field(1.0, 0.5, TimeConvention::MinusI);
        assert!((minus[0] - swe.far_field(1.0, 0.5, TimeConvention::PlusJ)[0].conj()).norm() < 1e-15);
        assert!(SphericalWaveExpansion::from_far_field(pattern, F, 0, TimeConvention::PlusJ).is_err());
    }
}
