//! Signal models for arrays: polarization states, point sources, distributed (brightness /
//! noise-temperature) scenes, and the covariance matrices they induce.
//!
//! Polarizations are expressed in the `(θ̂, φ̂)` basis of the arrival direction. Powers are
//! available powers in watts over the scene bandwidth, so `R` has units of W.
//!
//! See the book entry [Arrays, scenes and beamformers](https://jwkunz.github.io/lib_jsl/book/em-arrays/beamforming.html).

pub mod covariance;
pub mod distributed;
pub mod point_source;

pub use covariance::{Scene, sample_covariance, synthesize_snapshots};
pub use distributed::{BrightnessScene, SkyPolarization};
pub use point_source::PointSource;

use crate::conventions::{C64, TimeConvention};
use std::f64::consts::FRAC_1_SQRT_2;

/// A 2×2 coherency (polarization) matrix `J = ⟨p pᴴ⟩` in the `(θ̂, φ̂)` basis, normalised to unit
/// trace (it describes the *shape* of the polarization; power is carried separately).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coherency(pub [[C64; 2]; 2]);

impl Coherency {
    /// A fully polarized state with Jones vector `p` (normalised here).
    pub fn from_jones(p: [C64; 2]) -> Self {
        let n = p[0].norm_sqr() + p[1].norm_sqr();
        let p = p.map(|v| v / n.sqrt().max(1e-300));
        Coherency([[p[0] * p[0].conj(), p[0] * p[1].conj()], [p[1] * p[0].conj(), p[1] * p[1].conj()]])
    }

    /// Unpolarized: `½ I`.
    pub fn unpolarized() -> Self {
        let h = C64::new(0.5, 0.0);
        let z = C64::new(0.0, 0.0);
        Coherency([[h, z], [z, h]])
    }

    /// Linear polarization along θ̂.
    pub fn theta() -> Self {
        Self::from_jones([C64::new(1.0, 0.0), C64::new(0.0, 0.0)])
    }

    /// Linear polarization along φ̂.
    pub fn phi() -> Self {
        Self::from_jones([C64::new(0.0, 0.0), C64::new(1.0, 0.0)])
    }

    /// A wave arriving from the direction with right-hand circular polarization (IEEE, with
    /// respect to its own propagation, i.e. `-r̂`).
    pub fn rhcp_incoming(convention: TimeConvention) -> Self {
        // θ̂ × φ̂ = r̂ points against the propagation, so handedness is reversed in this basis.
        Self::from_jones([C64::new(FRAC_1_SQRT_2, 0.0), convention.unit() * FRAC_1_SQRT_2])
    }

    /// A wave arriving from the direction with left-hand circular polarization.
    pub fn lhcp_incoming(convention: TimeConvention) -> Self {
        Self::from_jones([C64::new(FRAC_1_SQRT_2, 0.0), -convention.unit() * FRAC_1_SQRT_2])
    }

    pub fn trace(&self) -> f64 {
        self.0[0][0].re + self.0[1][1].re
    }

    /// The degree of polarization `√(1 - 4 det J)` (0 unpolarized, 1 fully polarized).
    pub fn degree_of_polarization(&self) -> f64 {
        let det = (self.0[0][0] * self.0[1][1] - self.0[0][1] * self.0[1][0]).re;
        (1.0 - 4.0 * det).max(0.0).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coherency_states() {
        assert!((Coherency::unpolarized().degree_of_polarization()).abs() < 1e-12);
        assert!((Coherency::theta().degree_of_polarization() - 1.0).abs() < 1e-12);
        let r = Coherency::rhcp_incoming(TimeConvention::PlusJ);
        assert!((r.trace() - 1.0).abs() < 1e-12 && (r.degree_of_polarization() - 1.0).abs() < 1e-12);
        // Right- and left-hand states are orthogonal: tr(J_R J_L) = 0.
        let l = Coherency::lhcp_incoming(TimeConvention::PlusJ);
        let tr: C64 = (0..2).flat_map(|i| (0..2).map(move |k| (i, k))).map(|(i, k)| r.0[i][k] * l.0[k][i]).sum();
        assert!(tr.norm() < 1e-12);
        assert!((Coherency::from_jones([C64::new(3.0, 0.0), C64::new(0.0, 4.0)]).trace() - 1.0).abs() < 1e-12);
    }
}
