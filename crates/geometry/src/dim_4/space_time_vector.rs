//! Events and four-vectors in Minkowski space.
//!
//! A [`SpaceTimeVector4D`] holds `(t, x, y, z)` with the time component `t` expressed in the same
//! length unit as the spatial ones (that is, `t` is really `c·time`, with `c = 1` in natural
//! units; use [`SpaceTimeVector4D::from_time`] to convert from seconds and a speed of light).
//! The metric signature is `(-, +, +, +)`: the invariant interval `-t² + x² + y² + z²` is negative
//! for timelike separations, positive for spacelike, and zero on the light cone.

use std::ops::{Add, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq, Default)]

pub struct SpaceTimeVector4D {
    pub t: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// How a four-vector sits relative to the light cone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]

pub enum Causality {
    Timelike,
    Lightlike,
    Spacelike,
}

impl SpaceTimeVector4D {
    pub const fn new(t: f64, x: f64, y: f64, z: f64) -> Self {
        SpaceTimeVector4D { t, x, y, z }
    }

    /// From a time in seconds (or any unit) and speed of light `c` in length-per-time units.
    pub fn from_time(time: f64, c: f64, x: f64, y: f64, z: f64) -> Self {
        SpaceTimeVector4D::new(c * time, x, y, z)
    }

    /// The components as `[t, x, y, z]`.
    pub const fn components(&self) -> [f64; 4] {
        [self.t, self.x, self.y, self.z]
    }

    /// The spatial part `[x, y, z]`.
    pub const fn spatial(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// The Minkowski inner product `-t t' + x x' + y y' + z z'`.
    pub fn dot(&self, other: &Self) -> f64 {
        -self.t * other.t + self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// The squared interval `self · self`: negative timelike, zero lightlike, positive spacelike.
    pub fn interval_squared(&self) -> f64 {
        self.dot(self)
    }

    /// Classifies the vector, treating `|interval²| <= tol` as lightlike.
    pub fn causality(&self, tol: f64) -> Causality {
        let s = self.interval_squared();
        if s.abs() <= tol {
            Causality::Lightlike
        } else if s < 0.0 {
            Causality::Timelike
        } else {
            Causality::Spacelike
        }
    }

    /// The proper time (as `c·τ` length) between two events separated by this timelike vector;
    /// `None` if the separation is not timelike.
    pub fn proper_time(&self) -> Option<f64> {
        let s = self.interval_squared();
        (s < 0.0).then(|| (-s).sqrt())
    }

    /// The proper distance between events separated by this spacelike vector; `None` otherwise.
    pub fn proper_distance(&self) -> Option<f64> {
        let s = self.interval_squared();
        (s > 0.0).then(|| s.sqrt())
    }

    /// The Euclidean magnitude of the spatial part.
    pub fn spatial_norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// The Lorentz factor `1 / sqrt(1 - β²)` for a velocity `beta` (units of `c`); `None` if
    /// `|beta| >= 1`.
    pub fn gamma(beta: [f64; 3]) -> Option<f64> {
        let b2 = beta[0] * beta[0] + beta[1] * beta[1] + beta[2] * beta[2];
        (b2 < 1.0).then(|| 1.0 / (1.0 - b2).sqrt())
    }

    /// The 4×4 matrix (rows/columns ordered `t, x, y, z`) of the boost to a frame moving with
    /// velocity `beta` (units of `c`) relative to the current one: components of a vector in the
    /// current frame map to its components in the moving frame. `None` if `|beta| >= 1`.
    pub fn boost_matrix(beta: [f64; 3]) -> Option<[[f64; 4]; 4]> {
        let g = Self::gamma(beta)?;
        let b2 = beta[0] * beta[0] + beta[1] * beta[1] + beta[2] * beta[2];
        let mut m = [[0.0; 4]; 4];
        m[0][0] = g;
        for i in 0..3 {
            m[0][i + 1] = -g * beta[i];
            m[i + 1][0] = -g * beta[i];
            for j in 0..3 {
                let delta = if i == j { 1.0 } else { 0.0 };
                m[i + 1][j + 1] = delta + if b2 > 0.0 { (g - 1.0) * beta[i] * beta[j] / b2 } else { 0.0 };
            }
        }
        Some(m)
    }

    /// Applies a 4×4 matrix to the components.
    pub fn transformed(&self, m: &[[f64; 4]; 4]) -> Self {
        let v = self.components();
        let r: [f64; 4] = std::array::from_fn(|i| (0..4).map(|j| m[i][j] * v[j]).sum());
        SpaceTimeVector4D::new(r[0], r[1], r[2], r[3])
    }

    /// This vector as seen from a frame moving with velocity `beta` (units of `c`).
    pub fn boosted(&self, beta: [f64; 3]) -> Option<Self> {
        Self::boost_matrix(beta).map(|m| self.transformed(&m))
    }

    /// The four-velocity `γ (1, v)` of a particle with 3-velocity `beta` (units of `c`); its
    /// interval is always `-1`. `None` if `|beta| >= 1`.
    pub fn four_velocity(beta: [f64; 3]) -> Option<Self> {
        let g = Self::gamma(beta)?;
        Some(SpaceTimeVector4D::new(g, g * beta[0], g * beta[1], g * beta[2]))
    }

    /// The rapidity `atanh(β)` for motion along one axis; `None` if `|beta| >= 1`.
    pub fn rapidity(beta: f64) -> Option<f64> {
        (beta.abs() < 1.0).then(|| beta.atanh())
    }

    /// The relativistic composition of two collinear velocities (units of `c`).
    pub fn add_collinear_velocities(u: f64, v: f64) -> f64 {
        (u + v) / (1.0 + u * v)
    }
}

impl Add for SpaceTimeVector4D {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        SpaceTimeVector4D::new(self.t + o.t, self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl Sub for SpaceTimeVector4D {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        SpaceTimeVector4D::new(self.t - o.t, self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Neg for SpaceTimeVector4D {
    type Output = Self;
    fn neg(self) -> Self {
        SpaceTimeVector4D::new(-self.t, -self.x, -self.y, -self.z)
    }
}
impl Mul<f64> for SpaceTimeVector4D {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        SpaceTimeVector4D::new(self.t * s, self.x * s, self.y * s, self.z * s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_and_causality() {
        let timelike = SpaceTimeVector4D::new(5.0, 3.0, 0.0, 0.0);
        assert_eq!(timelike.interval_squared(), -16.0);
        assert_eq!(timelike.proper_time(), Some(4.0));
        assert_eq!(timelike.causality(1e-12), Causality::Timelike);
        let spacelike = SpaceTimeVector4D::new(1.0, 2.0, 2.0, 0.0);
        assert_eq!(spacelike.proper_distance(), Some(7.0f64.sqrt()));
        assert_eq!(spacelike.causality(1e-12), Causality::Spacelike);
        let light = SpaceTimeVector4D::new(5.0, 3.0, 4.0, 0.0);
        assert_eq!(light.causality(1e-12), Causality::Lightlike);
        assert!(light.proper_time().is_none() && light.proper_distance().is_none());
        assert_eq!(SpaceTimeVector4D::from_time(2.0, 3.0, 1.0, 0.0, 0.0).t, 6.0);
    }

    #[test]
    fn boosts_preserve_the_interval_and_dilate_time() {
        let v = SpaceTimeVector4D::new(3.0, 1.0, -2.0, 0.5);
        for beta in [[0.6, 0.0, 0.0], [0.0, -0.3, 0.4], [0.2, 0.3, -0.5]] {
            let b = v.boosted(beta).unwrap();
            assert!((b.interval_squared() - v.interval_squared()).abs() < 1e-12, "{beta:?}");
        }
        // Time dilation: a clock at rest in frame S' at x' = 0 ticks 1 unit of proper time while
        // S sees γ units. Boosting the event (1,0,0,0) into a frame moving at β gives (γ, -γβ, ...).
        let tick = SpaceTimeVector4D::new(1.0, 0.0, 0.0, 0.0);
        let seen = tick.boosted([0.6, 0.0, 0.0]).unwrap();
        assert!((seen.t - 1.25).abs() < 1e-12 && (seen.x + 0.75).abs() < 1e-12);
        // Boost followed by the opposite boost is the identity.
        let back = v.boosted([0.2, 0.3, -0.5]).unwrap().boosted([-0.2, -0.3, 0.5]).unwrap();
        assert!((back - v).components().iter().all(|c| c.abs() < 1e-12));
        assert!(v.boosted([0.8, 0.8, 0.0]).is_none() && SpaceTimeVector4D::boost_matrix([1.0, 0.0, 0.0]).is_none());
        assert_eq!(v.boosted([0.0; 3]).unwrap(), v);
    }

    #[test]
    fn four_velocity_rapidity_and_velocity_addition() {
        let u = SpaceTimeVector4D::four_velocity([0.6, 0.0, 0.0]).unwrap();
        assert!((u.interval_squared() + 1.0).abs() < 1e-12 && (u.t - 1.25).abs() < 1e-12);
        // Seen from its own rest frame the four-velocity is (1, 0, 0, 0).
        let rest = u.boosted([0.6, 0.0, 0.0]).unwrap();
        assert!((rest.t - 1.0).abs() < 1e-12 && rest.spatial_norm() < 1e-12);
        // Rapidities add; so do velocities relativistically, and never exceed c.
        let (r1, r2) = (SpaceTimeVector4D::rapidity(0.5).unwrap(), SpaceTimeVector4D::rapidity(0.7).unwrap());
        assert!(((r1 + r2).tanh() - SpaceTimeVector4D::add_collinear_velocities(0.5, 0.7)).abs() < 1e-12);
        assert!(SpaceTimeVector4D::add_collinear_velocities(0.99, 0.99) < 1.0);
        assert!(SpaceTimeVector4D::rapidity(1.0).is_none());
        let sum = SpaceTimeVector4D::new(1.0, 2.0, 3.0, 4.0) + SpaceTimeVector4D::new(1.0, 1.0, 1.0, 1.0) * 2.0 - SpaceTimeVector4D::new(0.0, 1.0, 0.0, 0.0);
        assert_eq!(sum, SpaceTimeVector4D::new(3.0, 3.0, 5.0, 6.0));
        assert_eq!(-SpaceTimeVector4D::new(1.0, 0.0, 0.0, 0.0), SpaceTimeVector4D::new(-1.0, 0.0, 0.0, 0.0));
    }
}
