//! Cartesian, spherical and cylindrical coordinates: point conversion, scale factors and local
//! unit-vector frames.
//!
//! Spherical coordinates are `(r, θ, φ)` with θ the polar angle from +z in `[0, π]` and φ the
//! azimuth from +x; cylindrical coordinates are `(ρ, φ, z)`. A [`Position`] keeps the coordinates
//! it was given, so angles survive conversions between the angular systems and the local frame stays
//! well defined at the origin and on the polar axis (where Cartesian input could not determine it).

/// A three-dimensional coordinate system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoordinateSystem {
    /// `(x, y, z)`
    Cartesian,
    /// `(r, θ, φ)`
    Spherical,
    /// `(ρ, φ, z)`
    Cylindrical,
}

/// A point expressed in a particular coordinate system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    pub system: CoordinateSystem,
    pub coords: [f64; 3],
}

impl Position {
    pub fn cartesian(x: f64, y: f64, z: f64) -> Self {
        Position { system: CoordinateSystem::Cartesian, coords: [x, y, z] }
    }

    pub fn spherical(r: f64, theta: f64, phi: f64) -> Self {
        Position { system: CoordinateSystem::Spherical, coords: [r, theta, phi] }
    }

    pub fn cylindrical(rho: f64, phi: f64, z: f64) -> Self {
        Position { system: CoordinateSystem::Cylindrical, coords: [rho, phi, z] }
    }

    /// The Cartesian coordinates `[x, y, z]`.
    pub fn to_cartesian(&self) -> [f64; 3] {
        let [a, b, c] = self.coords;
        match self.system {
            CoordinateSystem::Cartesian => [a, b, c],
            CoordinateSystem::Spherical => {
                let (st, ct) = b.sin_cos();
                let (sp, cp) = c.sin_cos();
                [a * st * cp, a * st * sp, a * ct]
            }
            CoordinateSystem::Cylindrical => {
                let (sp, cp) = b.sin_cos();
                [a * cp, a * sp, c]
            }
        }
    }

    /// The same point expressed in `system`. Angles are carried over directly between the
    /// spherical and cylindrical systems (they share φ), so no information is lost on the axis.
    pub fn in_system(&self, system: CoordinateSystem) -> Position {
        use CoordinateSystem::*;
        if system == self.system {
            return *self;
        }
        let coords = match (self.system, system) {
            (Spherical, Cylindrical) => {
                let [r, t, p] = self.coords;
                [r * t.sin(), p, r * t.cos()]
            }
            (Cylindrical, Spherical) => {
                let [rho, p, z] = self.coords;
                [rho.hypot(z), rho.atan2(z), p]
            }
            (_, Cartesian) => self.to_cartesian(),
            (Cartesian, Spherical) => {
                let [x, y, z] = self.coords;
                let rho = x.hypot(y);
                [rho.hypot(z), rho.atan2(z), y.atan2(x)]
            }
            (Cartesian, Cylindrical) => {
                let [x, y, z] = self.coords;
                [x.hypot(y), y.atan2(x), z]
            }
            _ => unreachable!("identical systems handled above"),
        };
        Position { system, coords }
    }

    /// The scale factors `(h₁, h₂, h₃)` of `system` at this point, so that
    /// `ds² = Σ hᵢ² duᵢ²`: `(1, 1, 1)`, `(1, r, r sin θ)` or `(1, ρ, 1)`.
    pub fn scale_factors(&self, system: CoordinateSystem) -> [f64; 3] {
        let p = self.in_system(system);
        match system {
            CoordinateSystem::Cartesian => [1.0, 1.0, 1.0],
            CoordinateSystem::Spherical => [1.0, p.coords[0], p.coords[0] * p.coords[1].sin()],
            CoordinateSystem::Cylindrical => [1.0, p.coords[0], 1.0],
        }
    }

    /// The orthonormal, right-handed unit vectors `(û₁, û₂, û₃)` of `system` at this point, each
    /// as Cartesian components (`frame[i]` is `ûᵢ`). Built from the angles, so it is defined even
    /// where the position alone is degenerate (r = 0, ρ = 0, the poles).
    pub fn frame(&self, system: CoordinateSystem) -> [[f64; 3]; 3] {
        let p = self.in_system(system);
        match system {
            CoordinateSystem::Cartesian => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            CoordinateSystem::Spherical => {
                let (st, ct) = p.coords[1].sin_cos();
                let (sp, cp) = p.coords[2].sin_cos();
                [[st * cp, st * sp, ct], [ct * cp, ct * sp, -st], [-sp, cp, 0.0]]
            }
            CoordinateSystem::Cylindrical => {
                let (sp, cp) = p.coords[1].sin_cos();
                [[cp, sp, 0.0], [-sp, cp, 0.0], [0.0, 0.0, 1.0]]
            }
        }
    }

    /// The Euclidean distance to another point.
    pub fn distance(&self, other: &Position) -> f64 {
        let (a, b) = (self.to_cartesian(), other.to_cartesian());
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    const SYSTEMS: [CoordinateSystem; 3] = [CoordinateSystem::Cartesian, CoordinateSystem::Spherical, CoordinateSystem::Cylindrical];

    fn close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < tol)
    }

    #[test]
    fn conversions_round_trip_between_every_pair() {
        let pts = [Position::cartesian(1.0, -2.0, 0.5), Position::spherical(3.0, 2.2, -1.1), Position::cylindrical(0.7, 2.9, -4.0)];
        for p in pts {
            for s in SYSTEMS {
                for t in SYSTEMS {
                    let back = p.in_system(s).in_system(t).to_cartesian();
                    assert!(close(back, p.to_cartesian(), 1e-12), "{p:?} via {s:?} to {t:?}");
                }
            }
        }
        let s = Position::cartesian(0.0, 1.0, 1.0).in_system(CoordinateSystem::Spherical);
        assert!(close(s.coords, [2f64.sqrt(), PI / 4.0, FRAC_PI_2], 1e-12));
        assert!((Position::cartesian(0.0, 0.0, 0.0).distance(&Position::spherical(2.0, 1.0, 3.0)) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn frames_are_orthonormal_right_handed_and_match_coordinate_derivatives() {
        let pts = [Position::spherical(2.0, 0.6, 1.3), Position::cylindrical(1.5, -2.0, 0.4), Position::cartesian(0.3, 0.4, -0.2)];
        for p in pts {
            for s in SYSTEMS {
                let f = p.frame(s);
                let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
                for i in 0..3 {
                    for j in 0..3 {
                        assert!((dot(f[i], f[j]) - f64::from(i == j)).abs() < 1e-12);
                    }
                }
                let cross = [f[0][1] * f[1][2] - f[0][2] * f[1][1], f[0][2] * f[1][0] - f[0][0] * f[1][2], f[0][0] * f[1][1] - f[0][1] * f[1][0]];
                assert!(close(cross, f[2], 1e-12), "right-handed {s:?}");
                // ∂r/∂uᵢ = hᵢ ûᵢ (central differences in the native coordinates).
                let native = p.in_system(s);
                let h = p.scale_factors(s);
                for i in 0..3 {
                    let step = 1e-6;
                    let mut a = native;
                    let mut b = native;
                    a.coords[i] += step;
                    b.coords[i] -= step;
                    let (ca, cb) = (a.to_cartesian(), b.to_cartesian());
                    let d: [f64; 3] = std::array::from_fn(|k| (ca[k] - cb[k]) / (2.0 * step));
                    assert!(close(d, f[i].map(|v| v * h[i]), 1e-6), "{s:?} axis {i}");
                }
            }
        }
    }

    #[test]
    fn angles_survive_the_axis_and_the_origin() {
        // On the z axis Cartesian input cannot fix φ, but a spherical position keeps it.
        let pole = Position::spherical(1.0, 0.0, 1.2);
        let f = pole.frame(CoordinateSystem::Spherical);
        assert!(close(f[2], [-(1.2f64).sin(), 1.2f64.cos(), 0.0], 1e-12)); // φ̂ follows the stored φ
        assert!(close(pole.in_system(CoordinateSystem::Cylindrical).coords, [0.0, 1.2, 1.0], 1e-12));
        let origin = Position::spherical(0.0, 0.5, 0.25);
        assert_eq!(origin.to_cartesian(), [0.0, 0.0, 0.0]);
        assert!(close(origin.frame(CoordinateSystem::Spherical)[0], Position::spherical(1.0, 0.5, 0.25).frame(CoordinateSystem::Spherical)[0], 1e-15));
        assert_eq!(pole.scale_factors(CoordinateSystem::Spherical)[2], 0.0);
    }
}
