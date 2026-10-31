//! Polarization bases: how the three components of a field vector are defined at a point.
//!
//! - [`PolarizationBasis::Linear`]`(system)`: the unit vectors of `system` (x̂ŷẑ, r̂θ̂φ̂ or ρ̂φ̂ẑ),
//!   components in that order.
//! - [`PolarizationBasis::Circular`]`(system)`: right- and left-hand circular components formed
//!   from the two unit vectors transverse to the system's *longitudinal* direction (ẑ, r̂ or ρ̂),
//!   followed by the longitudinal component. Components are ordered `[R, L, longitudinal]`.
//!
//! Handedness follows the IEEE definition (right-hand: the field rotates counter-clockwise when
//! viewed along the longitudinal direction, i.e. the right-hand rule about it) in either time
//! convention, which is why circular conversions take a [`TimeConvention`]. Every conversion goes
//! through Cartesian components at the sample's [`Position`].

use super::systems::{CoordinateSystem, Position};
use crate::conventions::{C64, TimeConvention};
use lib_jsl_core::ErrorsJSL;
use std::f64::consts::FRAC_1_SQRT_2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolarizationBasis {
    Linear(CoordinateSystem),
    Circular(CoordinateSystem),
}

impl PolarizationBasis {
    pub const CARTESIAN: PolarizationBasis = PolarizationBasis::Linear(CoordinateSystem::Cartesian);
    pub const SPHERICAL: PolarizationBasis = PolarizationBasis::Linear(CoordinateSystem::Spherical);
    pub const CYLINDRICAL: PolarizationBasis = PolarizationBasis::Linear(CoordinateSystem::Cylindrical);

    pub fn system(self) -> CoordinateSystem {
        match self {
            PolarizationBasis::Linear(s) | PolarizationBasis::Circular(s) => s,
        }
    }

    pub fn is_circular(self) -> bool {
        matches!(self, PolarizationBasis::Circular(_))
    }

    /// Indices into the system's frame of the transverse pair `(a, b)` (with `â × b̂` along the
    /// longitudinal direction) and of the longitudinal vector.
    fn transverse_axes(system: CoordinateSystem) -> ([usize; 2], usize) {
        match system {
            CoordinateSystem::Cartesian => ([0, 1], 2),   // x̂ × ŷ = ẑ
            CoordinateSystem::Spherical => ([1, 2], 0),   // θ̂ × φ̂ = r̂
            CoordinateSystem::Cylindrical => ([1, 2], 0), // φ̂ × ẑ = ρ̂
        }
    }

    /// The basis vectors at `position`, as complex Cartesian components (`vectors[i]` is the i-th
    /// basis vector). Component `i` of a field in this basis is `vectors[i]ᴴ · E`.
    pub fn vectors(self, position: &Position, convention: TimeConvention) -> [[C64; 3]; 3] {
        let frame = position.frame(self.system());
        let real = |v: [f64; 3]| v.map(|x| C64::new(x, 0.0));
        match self {
            PolarizationBasis::Linear(_) => frame.map(real),
            PolarizationBasis::Circular(s) => {
                let ([a, b], l) = Self::transverse_axes(s);
                // RHCP for e^{+jωt}: (â - j b̂)/√2; the other convention conjugates.
                let j = convention.unit();
                let r: [C64; 3] = std::array::from_fn(|k| (C64::new(frame[a][k], 0.0) - j * frame[b][k]) * FRAC_1_SQRT_2);
                let left: [C64; 3] = std::array::from_fn(|k| (C64::new(frame[a][k], 0.0) + j * frame[b][k]) * FRAC_1_SQRT_2);
                [r, left, real(frame[l])]
            }
        }
    }

    /// Converts components in this basis to Cartesian components.
    pub fn to_cartesian(self, components: [C64; 3], position: &Position, convention: TimeConvention) -> [C64; 3] {
        let v = self.vectors(position, convention);
        std::array::from_fn(|k| (0..3).map(|i| components[i] * v[i][k]).sum())
    }

    /// Converts Cartesian components to components in this basis.
    pub fn from_cartesian(self, cartesian: [C64; 3], position: &Position, convention: TimeConvention) -> [C64; 3] {
        let v = self.vectors(position, convention);
        std::array::from_fn(|i| (0..3).map(|k| v[i][k].conj() * cartesian[k]).sum())
    }

    /// Converts components from this basis to `target` at `position`.
    pub fn convert(self, components: [C64; 3], target: PolarizationBasis, position: &Position, convention: TimeConvention) -> [C64; 3] {
        if self == target {
            return components;
        }
        target.from_cartesian(self.to_cartesian(components, position, convention), position, convention)
    }

    /// Real (time-domain) vectors can only be expressed in linear bases.
    pub fn convert_real(self, components: [f64; 3], target: PolarizationBasis, position: &Position) -> Result<[f64; 3], ErrorsJSL> {
        if self.is_circular() || target.is_circular() {
            return Err(ErrorsJSL::InvalidInputRange("Circular bases need complex (harmonic) field values."));
        }
        let (from, to) = (position.frame(self.system()), position.frame(target.system()));
        let cart: [f64; 3] = std::array::from_fn(|k| (0..3).map(|i| components[i] * from[i][k]).sum());
        Ok(std::array::from_fn(|i| (0..3).map(|k| to[i][k] * cart[k]).sum()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    const ALL: [PolarizationBasis; 6] = [
        PolarizationBasis::Linear(CoordinateSystem::Cartesian),
        PolarizationBasis::Linear(CoordinateSystem::Spherical),
        PolarizationBasis::Linear(CoordinateSystem::Cylindrical),
        PolarizationBasis::Circular(CoordinateSystem::Cartesian),
        PolarizationBasis::Circular(CoordinateSystem::Spherical),
        PolarizationBasis::Circular(CoordinateSystem::Cylindrical),
    ];

    fn close(a: [C64; 3], b: [C64; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).norm() < 1e-12)
    }

    fn c(re: f64, im: f64) -> C64 {
        C64::new(re, im)
    }

    #[test]
    fn every_basis_is_unitary_and_round_trips() {
        let e = [c(1.0, 0.5), c(-0.3, 2.0), c(0.7, -1.1)];
        let power: f64 = e.iter().map(|v| v.norm_sqr()).sum();
        for pos in [Position::spherical(2.0, 0.8, -2.1), Position::cylindrical(1.0, 0.4, 3.0), Position::cartesian(1.0, 2.0, 3.0)] {
            for conv in [TimeConvention::PlusJ, TimeConvention::MinusI] {
                for a in ALL {
                    let in_a = a.from_cartesian(e, &pos, conv);
                    assert!((in_a.iter().map(|v| v.norm_sqr()).sum::<f64>() - power).abs() < 1e-12, "{a:?} preserves power");
                    assert!(close(a.to_cartesian(in_a, &pos, conv), e), "{a:?} round trip");
                    for b in ALL {
                        let there = a.convert(in_a, b, &pos, conv);
                        assert!(close(b.convert(there, a, &pos, conv), in_a), "{a:?} <-> {b:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn linear_components_match_textbook_values() {
        // On the equator at φ = 0: ẑ = -θ̂ and x̂ = r̂.
        let p = Position::spherical(1.0, FRAC_PI_2, 0.0);
        let z_hat = [c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)];
        assert!(close(PolarizationBasis::SPHERICAL.from_cartesian(z_hat, &p, TimeConvention::PlusJ), [c(0.0, 0.0), c(-1.0, 0.0), c(0.0, 0.0)]));
        let x_hat = [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)];
        assert!(close(PolarizationBasis::SPHERICAL.from_cartesian(x_hat, &p, TimeConvention::PlusJ), [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)]));
        // A purely radial field in cylindrical coordinates at φ = 90° points along +y.
        let q = Position::cylindrical(2.0, FRAC_PI_2, 0.0);
        let cart = PolarizationBasis::CYLINDRICAL.to_cartesian([c(3.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)], &q, TimeConvention::PlusJ);
        assert!(close(cart, [c(0.0, 0.0), c(3.0, 0.0), c(0.0, 0.0)]));
        // Real vectors convert the same way; circular bases are rejected for real data.
        let r = PolarizationBasis::CARTESIAN.convert_real([0.0, 0.0, 1.0], PolarizationBasis::SPHERICAL, &p).unwrap();
        assert!((r[1] + 1.0).abs() < 1e-12);
        assert!(PolarizationBasis::CARTESIAN.convert_real([1.0, 0.0, 0.0], PolarizationBasis::Circular(CoordinateSystem::Cartesian), &p).is_err());
    }

    #[test]
    fn rhcp_rotates_by_the_right_hand_rule_in_both_conventions() {
        // A pure R component along +z must rotate x̂ → ŷ (counter-clockwise about +z).
        let pos = Position::cartesian(0.0, 0.0, 0.0);
        let omega = 1.0;
        for conv in [TimeConvention::PlusJ, TimeConvention::MinusI] {
            let e = PolarizationBasis::Circular(CoordinateSystem::Cartesian).to_cartesian([c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)], &pos, conv);
            let at = |t: f64| [conv.instantaneous(e[0], omega, t), conv.instantaneous(e[1], omega, t)];
            let (a, b) = (at(0.0), at(0.1));
            let cross = a[0] * b[1] - a[1] * b[0];
            assert!(cross > 0.0, "{conv:?}: RHCP must turn counter-clockwise about +z");
            // The left-hand vector turns the other way.
            let l = PolarizationBasis::Circular(CoordinateSystem::Cartesian).to_cartesian([c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)], &pos, conv);
            let lt = |t: f64| [conv.instantaneous(l[0], omega, t), conv.instantaneous(l[1], omega, t)];
            let (la, lb) = (lt(0.0), lt(0.1));
            assert!(la[0] * lb[1] - la[1] * lb[0] < 0.0);
        }
        // In the spherical circular basis an outgoing wave along +x (θ = 90°, φ = 0) has its
        // transverse pair (θ̂, φ̂) = (-ẑ, ŷ): R must rotate counter-clockwise about +x as well.
        let p = Position::spherical(1.0, FRAC_PI_2, 0.0);
        let e = PolarizationBasis::Circular(CoordinateSystem::Spherical).to_cartesian([c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)], &p, TimeConvention::PlusJ);
        let at = |t: f64| [TimeConvention::PlusJ.instantaneous(e[1], 1.0, t), TimeConvention::PlusJ.instantaneous(e[2], 1.0, t)];
        let (a, b) = (at(0.0), at(0.1));
        assert!(a[0] * b[1] - a[1] * b[0] > 0.0); // y → z rotation is counter-clockwise about +x
    }
}
