//! Near-field to far-field transformation from measured or simulated tangential fields.
//!
//! * [`planar_near_to_far`]: tangential `E_x, E_y` sampled on a plane `z = z₀` in front of the
//!   antenna give its plane-wave spectrum `A(k_x, k_y) = e^{jk_z z₀} ∬ E_t e^{j(k_x x + k_y y)} dx dy`,
//!   and the far field follows by stationary phase,
//!   `F_θ = (jk/2π)(A_x cos φ + A_y sin φ)`, `F_φ = (jk cos θ/2π)(A_y cos φ − A_x sin φ)`
//!   (forward half-space only; the scan must be sampled at `≤ λ/2`). Truncating the scan
//!   abruptly rings like a rectangular window; tapering the outer part with [`scan_taper`]
//!   suppresses it.
//! * [`spherical_near_to_far`]: tangential `E_θ, E_φ` on a sphere of radius `r₀` enclosing the
//!   antenna are projected onto the vector spherical harmonics, and dividing by the radial wave
//!   functions gives the spherical wave amplitudes, hence the far field in every direction.
//!
//! The engineering convention `e^{+jωt}` is used internally; inputs and outputs are converted.

use crate::conventions::{C64, TimeConvention};
use crate::fields::spherical_waves::{SphericalWaveExpansion, gauss_legendre, hankel, vector_spherical_harmonics};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_core::constants::em::C0;
use ndarray::Array2;
use std::f64::consts::{FRAC_PI_2, TAU};

/// Raised-cosine edge weights for scan coordinates: 1 in the inner `1 − fraction` of the
/// half-extent, rolling smoothly to 0 at the edges. Multiply the sampled fields by
/// `w(x_i) w(y_j)` to suppress truncation ripple in [`planar_near_to_far`].
pub fn scan_taper(coordinates: &[f64], fraction: f64) -> Vec<f64> {
    let half = coordinates.iter().fold(0.0f64, |m, c| m.max(c.abs())).max(f64::MIN_POSITIVE);
    let start = 1.0 - fraction.clamp(0.0, 1.0);
    coordinates
        .iter()
        .map(|c| {
            let r = c.abs() / half;
            if r <= start || fraction <= 0.0 { 1.0 } else { 0.5 * (1.0 + (std::f64::consts::PI * (r - start) / (1.0 - start)).cos()) }
        })
        .collect()
}

/// The far field `[F_θ, F_φ]` (V, `e^{−jkr}/r` removed) in each direction `(θ, φ)` with
/// `θ < π/2`, from `E_x[[i, j]]`, `E_y[[i, j]]` sampled at `(xs[i], ys[j], z0)` on a uniform grid.
#[allow(clippy::too_many_arguments)]
pub fn planar_near_to_far(xs: &[f64], ys: &[f64], ex: &Array2<C64>, ey: &Array2<C64>, z0: f64, frequency: f64, directions: &[(f64, f64)], convention: TimeConvention) -> Result<Vec<[C64; 2]>, ErrorsJSL> {
    if xs.len() < 2 || ys.len() < 2 || ex.dim() != (xs.len(), ys.len()) || ey.dim() != ex.dim() {
        return Err(ErrorsJSL::InvalidInputRange("The field arrays must match the grid of at least 2 × 2 points."));
    }
    if directions.iter().any(|&(t, _)| !(0.0..FRAC_PI_2).contains(&t)) {
        return Err(ErrorsJSL::InvalidInputRange("Planar transformation covers only the forward half-space (θ < π/2)."));
    }
    let k = TAU * frequency / C0;
    let (dx, dy) = (xs[1] - xs[0], ys[1] - ys[0]);
    let to_plus = |v: C64| convention.convert(v, TimeConvention::PlusJ);
    let j = C64::new(0.0, 1.0);
    Ok(directions
        .iter()
        .map(|&(theta, phi)| {
            let (kx, ky, kz) = (k * theta.sin() * phi.cos(), k * theta.sin() * phi.sin(), k * theta.cos());
            // Separable phase factors.
            let px: Vec<C64> = xs.iter().map(|&x| C64::from_polar(1.0, kx * x)).collect();
            let py: Vec<C64> = ys.iter().map(|&y| C64::from_polar(1.0, ky * y)).collect();
            let (mut ax, mut ay) = (C64::new(0.0, 0.0), C64::new(0.0, 0.0));
            for (i, fx) in px.iter().enumerate() {
                for (jj, fy) in py.iter().enumerate() {
                    let w = fx * fy;
                    ax += to_plus(ex[[i, jj]]) * w;
                    ay += to_plus(ey[[i, jj]]) * w;
                }
            }
            let shift = C64::from_polar(dx * dy, kz * z0);
            let (ax, ay) = (ax * shift, ay * shift);
            let (s, c) = phi.sin_cos();
            let f_theta = j * k / TAU * (ax * c + ay * s);
            let f_phi = j * k * theta.cos() / TAU * (ay * c - ax * s);
            [f_theta, f_phi].map(|v| TimeConvention::PlusJ.convert(v, convention))
        })
        .collect())
}

/// The spherical wave expansion (and hence the far field) of an antenna from the tangential
/// field `[E_θ, E_φ]` on a sphere of radius `r0` enclosing it, truncated at `max_degree`
/// (about `k r_min + 10` for a minimum sphere of radius `r_min`). `field(θ, φ)` is sampled at the
/// Gauss-Legendre × uniform quadrature points of that degree.
pub fn spherical_near_to_far<F>(field: F, r0: f64, frequency: f64, max_degree: u32, convention: TimeConvention) -> Result<SphericalWaveExpansion, ErrorsJSL>
where
    F: Fn(f64, f64) -> [C64; 2],
{
    if max_degree == 0 || r0.is_nan() || r0 <= 0.0 || frequency.is_nan() || frequency <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("Need a positive radius, frequency and maximum degree."));
    }
    let size = SphericalWaveExpansion::index(max_degree, max_degree as i32) + 1;
    let (mut a, mut b) = (vec![C64::new(0.0, 0.0); size], vec![C64::new(0.0, 0.0); size]);
    let thetas = gauss_legendre(max_degree as usize + 2);
    let phis = 2 * max_degree as usize + 2;
    for &(x, w) in &thetas {
        let theta = x.acos();
        for p in 0..phis {
            let phi = TAU * p as f64 / phis as f64;
            let e = field(theta, phi).map(|v| convention.convert(v, TimeConvention::PlusJ));
            let weight = w * TAU / phis as f64;
            for n in 1..=max_degree {
                for m in -(n as i32)..=n as i32 {
                    let (psi, phv) = vector_spherical_harmonics(n, m, theta, phi);
                    let i = SphericalWaveExpansion::index(n, m);
                    a[i] += weight * (e[0] * psi[0].conj() + e[1] * psi[1].conj());
                    b[i] += weight * (e[0] * phv[0].conj() + e[1] * phv[1].conj());
                }
            }
        }
    }
    let kr = TAU * frequency / C0 * r0;
    let (mut te, mut tm) = (vec![C64::new(0.0, 0.0); size], vec![C64::new(0.0, 0.0); size]);
    for n in 1..=max_degree {
        let (h, dh) = hankel(n, kr);
        for m in -(n as i32)..=n as i32 {
            let i = SphericalWaveExpansion::index(n, m);
            // ⟨E, Φ⟩ = α h_n(kr₀) and ⟨E, Ψ⟩ = −β (kr₀ h_n)' / (kr₀).
            te[i] = b[i] / h;
            tm[i] = -a[i] * kr / dh;
        }
    }
    Ok(SphericalWaveExpansion { frequency, max_degree, te, tm })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use crate::antenna_models::HertzianDipole;
    use crate::coordinates::systems::Position;
    use crate::conventions::Medium;

    const F: f64 = 1e9;

    fn lambda() -> f64 {
        C0 / F
    }

    /// The exact Cartesian E of a unit-current dipole of length `l` along `axis` (unit) at the
    /// Cartesian point `p`, built from the z-dipole by a rotation taking ẑ to `axis`.
    fn dipole_cartesian(l: f64, axis: [f64; 3], p: [f64; 3]) -> [C64; 3] {
        // An orthonormal frame (u, v, axis).
        let a = axis;
        let helper = if a[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
        let cross = |x: [f64; 3], y: [f64; 3]| [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
        let u = { let c = cross(helper, a); let n = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt(); [c[0] / n, c[1] / n, c[2] / n] };
        let v = cross(a, u);
        let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
        let local = [dot(p, u), dot(p, v), dot(p, a)];
        let (r, theta, phi) = { let r = dot(local, local).sqrt(); (r, (local[2] / r).acos(), local[1].atan2(local[0])) };
        let (e, _) = HertzianDipole::new(l).near_field(&Position::spherical(r, theta, phi), F, TimeConvention::PlusJ);
        let (st, ct, sp, cp) = (theta.sin(), theta.cos(), phi.sin(), phi.cos());
        let lx = e[0] * st * cp + e[1] * ct * cp - e[2] * sp;
        let ly = e[0] * st * sp + e[1] * ct * sp + e[2] * cp;
        let lz = e[0] * ct - e[1] * st;
        [0, 1, 2].map(|c| lx * u[c] + ly * v[c] + lz * a[c])
    }

    /// The analytic far field of that dipole: `−(jηkl/4π) (â − (â·r̂) r̂)` in θ, φ components.
    fn dipole_far(l: f64, axis: [f64; 3], theta: f64, phi: f64) -> [C64; 2] {
        let k = TAU / lambda();
        let eta = Medium::VACUUM.impedance(TimeConvention::PlusJ).re;
        let c = C64::new(0.0, -eta * k * l / (4.0 * PI));
        let t_hat = [theta.cos() * phi.cos(), theta.cos() * phi.sin(), -theta.sin()];
        let p_hat = [-phi.sin(), phi.cos(), 0.0];
        let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
        [c * dot(axis, t_hat), c * dot(axis, p_hat)]
    }

    #[test]
    fn spherical_scan_recovers_an_offset_dipole_exactly() {
        // A dipole tilted and displaced 0.3λ from the origin excites many spherical modes.
        let l = 0.01;
        let axis = { let a = [0.3f64, -0.5, 0.8]; let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt(); [a[0] / n, a[1] / n, a[2] / n] };
        let offset = [0.3 * lambda(), 0.0, 0.1 * lambda()];
        let r0 = 0.8 * lambda();
        let field = |theta: f64, phi: f64| {
            let p = Position::spherical(r0, theta, phi).to_cartesian();
            let e = dipole_cartesian(l, axis, [p[0] - offset[0], p[1] - offset[1], p[2] - offset[2]]);
            let t_hat = [theta.cos() * phi.cos(), theta.cos() * phi.sin(), -theta.sin()];
            let p_hat = [-phi.sin(), phi.cos(), 0.0];
            [e[0] * t_hat[0] + e[1] * t_hat[1] + e[2] * t_hat[2], e[0] * p_hat[0] + e[1] * p_hat[1]]
        };
        let swe = spherical_near_to_far(field, r0, F, 14, TimeConvention::PlusJ).unwrap();
        let k = TAU / lambda();
        for &(t, p) in &[(0.3f64, 0.2f64), (1.1, 2.5), (2.2, -1.0), (1.6, 4.0)] {
            let r_hat = [t.sin() * p.cos(), t.sin() * p.sin(), t.cos()];
            let phase = C64::from_polar(1.0, k * (r_hat[0] * offset[0] + r_hat[1] * offset[1] + r_hat[2] * offset[2]));
            let want = dipole_far(l, axis, t, p).map(|v| v * phase);
            let got = swe.far_field(t, p, TimeConvention::PlusJ);
            let scale = (want[0].norm_sqr() + want[1].norm_sqr()).sqrt().max(1e-3);
            assert!((got[0] - want[0]).norm() < 1e-6 * scale && (got[1] - want[1]).norm() < 1e-6 * scale, "({t}, {p}): {got:?} vs {want:?}");
        }
        assert!(spherical_near_to_far(field, 0.0, F, 5, TimeConvention::PlusJ).is_err());
    }

    #[test]
    fn planar_scan_recovers_the_forward_pattern() {
        // An x-directed dipole 2λ below a 24λ × 24λ scan plane sampled at λ/4, tapered over
        // the outer 30% (without the taper the truncation ripple is about 10%).
        let (l, axis) = (0.01, [1.0, 0.0, 0.0]);
        let z0 = 2.0 * lambda();
        let n = 97;
        let coords: Vec<f64> = (0..n).map(|i| (i as f64 - 48.0) * lambda() / 4.0).collect();
        let w = scan_taper(&coords, 0.3);
        let mut ex = Array2::zeros((n, n));
        let mut ey = Array2::zeros((n, n));
        for (i, &x) in coords.iter().enumerate() {
            for (j, &y) in coords.iter().enumerate() {
                let e = dipole_cartesian(l, axis, [x, y, z0]);
                ex[[i, j]] = e[0] * w[i] * w[j];
                ey[[i, j]] = e[1] * w[i] * w[j];
            }
        }
        let directions = [(0.0, 0.0), (0.3, 0.0), (0.4, FRAC_PI_2), (0.5, 0.8)];
        let far = planar_near_to_far(&coords, &coords, &ex, &ey, z0, F, &directions, TimeConvention::PlusJ).unwrap();
        let peak = dipole_far(l, axis, 0.0, 0.0)[0].norm();
        for (&(t, p), got) in directions.iter().zip(&far) {
            let want = dipole_far(l, axis, t, p);
            assert!((got[0] - want[0]).norm() < 0.01 * peak && (got[1] - want[1]).norm() < 0.01 * peak, "({t}, {p}): {got:?} vs {want:?}");
        }
        // Conventions conjugate consistently.
        let minus = planar_near_to_far(&coords, &coords, &ex.mapv(|v: C64| v.conj()), &ey.mapv(|v: C64| v.conj()), z0, F, &directions[..1], TimeConvention::MinusI).unwrap();
        assert!((minus[0][0] - far[0][0].conj()).norm() < 1e-9 * peak);
        assert!(planar_near_to_far(&coords, &coords, &ex, &ey, z0, F, &[(2.0, 0.0)], TimeConvention::PlusJ).is_err());
        assert!(w[48] == 1.0 && w[0] < 1e-12 && scan_taper(&coords, 0.0).iter().all(|&v| v == 1.0));
    }
}
