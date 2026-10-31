//! Field containers: scalar or vector samples on a [`Domain`], harmonic or time-varying, tagged
//! with the physical quantity they hold.
//!
//! - [`HarmonicField`]: complex peak phasors at one frequency, with a [`TimeConvention`]. Values
//!   are stored as a `(samples, components)` array, one component for scalar fields and three
//!   (in the field's [`PolarizationBasis`]) for vector fields.
//! - [`TimeField`]: real samples on a uniform [`TimeAxis`], stored as `(steps, samples, components)`.
//!   Vector time fields use linear bases only (circular components are complex by nature).
//!
//! Every sample's position comes from the domain, so a field never stores positions itself.
//! Domains are shared through `Arc`, which makes fields cheap to derive from one another.

use crate::conventions::{C64, TimeConvention};
use crate::coordinates::polarization::PolarizationBasis;
use crate::coordinates::systems::Position;
use crate::fields::domain::Domain;
use lib_jsl_core::ErrorsJSL;
use ndarray::{Array2, Array3};
use std::f64::consts::TAU;
use std::sync::Arc;

/// The physical quantity a field holds, with its SI units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldQuantity {
    /// E, V/m
    ElectricField,
    /// H, A/m
    MagneticField,
    /// D, C/m²
    ElectricFluxDensity,
    /// B, T
    MagneticFluxDensity,
    /// J, A/m² (volume) or A/m (surface)
    ElectricCurrentDensity,
    /// M, V/m² (volume) or V/m (surface)
    MagneticCurrentDensity,
    /// A, Wb/m
    VectorPotential,
    /// Φ, V
    ScalarPotential,
    /// S (Poynting vector), W/m²
    PowerDensity,
    /// Brightness or noise temperature, K
    Temperature,
    /// Anything else (no unit or rank checks)
    Generic,
}

impl FieldQuantity {
    pub fn units(self) -> &'static str {
        match self {
            FieldQuantity::ElectricField => "V/m",
            FieldQuantity::MagneticField => "A/m",
            FieldQuantity::ElectricFluxDensity => "C/m²",
            FieldQuantity::MagneticFluxDensity => "T",
            FieldQuantity::ElectricCurrentDensity => "A/m²",
            FieldQuantity::MagneticCurrentDensity => "V/m²",
            FieldQuantity::VectorPotential => "Wb/m",
            FieldQuantity::ScalarPotential => "V",
            FieldQuantity::PowerDensity => "W/m²",
            FieldQuantity::Temperature => "K",
            FieldQuantity::Generic => "",
        }
    }

    /// Whether the quantity is inherently scalar (`Some(true)`), vector (`Some(false)`), or either.
    pub fn is_scalar(self) -> Option<bool> {
        match self {
            FieldQuantity::ScalarPotential | FieldQuantity::Temperature => Some(true),
            FieldQuantity::Generic => None,
            _ => Some(false),
        }
    }
}

/// Scalar, or vector with components in a polarization basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rank {
    Scalar,
    Vector(PolarizationBasis),
}

impl Rank {
    pub fn components(self) -> usize {
        match self {
            Rank::Scalar => 1,
            Rank::Vector(_) => 3,
        }
    }

    pub fn basis(self) -> Option<PolarizationBasis> {
        match self {
            Rank::Scalar => None,
            Rank::Vector(b) => Some(b),
        }
    }
}

fn check_rank(quantity: FieldQuantity, rank: Rank) -> Result<(), ErrorsJSL> {
    match (quantity.is_scalar(), rank) {
        (Some(true), Rank::Vector(_)) => Err(ErrorsJSL::InvalidInputRange("This quantity is a scalar field.")),
        (Some(false), Rank::Scalar) => Err(ErrorsJSL::InvalidInputRange("This quantity is a vector field.")),
        _ => Ok(()),
    }
}

/// Complex peak phasors at one frequency on a domain.
#[derive(Debug, Clone, PartialEq)]
pub struct HarmonicField {
    domain: Arc<Domain>,
    quantity: FieldQuantity,
    rank: Rank,
    frequency: f64,
    convention: TimeConvention,
    values: Array2<C64>,
}

impl HarmonicField {
    /// A field from a `(samples, components)` array.
    pub fn new(domain: Arc<Domain>, quantity: FieldQuantity, rank: Rank, frequency: f64, convention: TimeConvention, values: Array2<C64>) -> Result<Self, ErrorsJSL> {
        check_rank(quantity, rank)?;
        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("The frequency must be positive and finite."));
        }
        if values.dim() != (domain.len(), rank.components()) {
            return Err(ErrorsJSL::IncompatibleArraySizes((values.nrows(), domain.len())));
        }
        Ok(HarmonicField { domain, quantity, rank, frequency, convention, values })
    }

    /// A zero field.
    pub fn zeros(domain: Arc<Domain>, quantity: FieldQuantity, rank: Rank, frequency: f64, convention: TimeConvention) -> Result<Self, ErrorsJSL> {
        let values = Array2::zeros((domain.len(), rank.components()));
        Self::new(domain, quantity, rank, frequency, convention, values)
    }

    /// A scalar field sampled from a function of position.
    pub fn scalar_from_fn(domain: Arc<Domain>, quantity: FieldQuantity, frequency: f64, convention: TimeConvention, f: impl Fn(&Position) -> C64) -> Result<Self, ErrorsJSL> {
        let values = Array2::from_shape_fn((domain.len(), 1), |(i, _)| f(&domain.position_flat(i)));
        Self::new(domain, quantity, Rank::Scalar, frequency, convention, values)
    }

    /// A vector field sampled from a function returning components in `basis`.
    pub fn vector_from_fn(domain: Arc<Domain>, quantity: FieldQuantity, basis: PolarizationBasis, frequency: f64, convention: TimeConvention, f: impl Fn(&Position) -> [C64; 3]) -> Result<Self, ErrorsJSL> {
        let mut values = Array2::zeros((domain.len(), 3));
        for i in 0..domain.len() {
            let v = f(&domain.position_flat(i));
            for k in 0..3 {
                values[[i, k]] = v[k];
            }
        }
        Self::new(domain, quantity, Rank::Vector(basis), frequency, convention, values)
    }

    pub fn domain(&self) -> &Arc<Domain> {
        &self.domain
    }

    pub fn quantity(&self) -> FieldQuantity {
        self.quantity
    }

    pub fn rank(&self) -> Rank {
        self.rank
    }

    pub fn basis(&self) -> Option<PolarizationBasis> {
        self.rank.basis()
    }

    pub fn is_scalar(&self) -> bool {
        self.rank == Rank::Scalar
    }

    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    /// The angular frequency `2πf`.
    pub fn omega(&self) -> f64 {
        TAU * self.frequency
    }

    pub fn convention(&self) -> TimeConvention {
        self.convention
    }

    pub fn values(&self) -> &Array2<C64> {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut Array2<C64> {
        &mut self.values
    }

    pub fn len(&self) -> usize {
        self.values.nrows()
    }

    pub fn is_empty(&self) -> bool {
        self.values.nrows() == 0
    }

    /// The value of a scalar field at a sample.
    pub fn scalar(&self, sample: usize) -> C64 {
        self.values[[sample, 0]]
    }

    /// The components of a vector field at a sample, in the field's basis.
    pub fn vector(&self, sample: usize) -> [C64; 3] {
        [self.values[[sample, 0]], self.values[[sample, 1]], self.values[[sample, 2]]]
    }

    /// The Cartesian components of a vector field at a sample.
    pub fn cartesian(&self, sample: usize) -> [C64; 3] {
        match self.rank {
            Rank::Vector(b) => b.to_cartesian(self.vector(sample), &self.domain.position_flat(sample), self.convention),
            Rank::Scalar => [self.scalar(sample), C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
        }
    }

    /// The same field with a different quantity tag (rank rules still apply).
    pub fn with_quantity(mut self, quantity: FieldQuantity) -> Result<Self, ErrorsJSL> {
        check_rank(quantity, self.rank)?;
        self.quantity = quantity;
        Ok(self)
    }

    /// The same physical field expressed in another time convention (components are conjugated;
    /// circular components stay consistent because the circular basis vectors conjugate too).
    pub fn with_convention(&self, convention: TimeConvention) -> Self {
        let mut out = self.clone();
        if convention != self.convention {
            out.values.mapv_inplace(|v| v.conj());
            out.convention = convention;
        }
        out
    }

    /// The field multiplied by a complex factor.
    pub fn scaled(&self, factor: C64) -> Self {
        let mut out = self.clone();
        out.values.mapv_inplace(|v| v * factor);
        out
    }

    fn check_compatible(&self, other: &HarmonicField) -> Result<(), ErrorsJSL> {
        if !(Arc::ptr_eq(&self.domain, &other.domain) || self.domain == other.domain) {
            return Err(ErrorsJSL::InvalidInputRange("Fields live on different domains (resample first)."));
        }
        if self.rank != other.rank || self.convention != other.convention || (self.frequency - other.frequency).abs() > 1e-12 * self.frequency {
            return Err(ErrorsJSL::InvalidInputRange("Fields differ in rank, basis, convention or frequency."));
        }
        Ok(())
    }

    /// The sample-wise sum of two compatible fields of the same quantity (superposition).
    pub fn add(&self, other: &HarmonicField) -> Result<Self, ErrorsJSL> {
        self.check_compatible(other)?;
        if self.quantity != other.quantity {
            return Err(ErrorsJSL::InvalidInputRange("Cannot add different physical quantities."));
        }
        let mut out = self.clone();
        out.values = &self.values + &other.values;
        Ok(out)
    }

    /// `|F|²` at every sample (`Σ|cᵢ|²`, which no unitary basis change alters).
    pub fn norm_squared(&self) -> Vec<f64> {
        self.values.rows().into_iter().map(|r| r.iter().map(|v| v.norm_sqr()).sum()).collect()
    }

    /// `∫ |F|² dA` (or dV, dΩ) with the domain's quadrature weights.
    pub fn integrate_norm_squared(&self) -> f64 {
        self.norm_squared().iter().zip(self.domain.weights()).map(|(v, w)| v * w).sum()
    }

    /// `∫ F dA` for a scalar field.
    pub fn integrate(&self) -> Result<C64, ErrorsJSL> {
        if !self.is_scalar() {
            return Err(ErrorsJSL::InvalidInputRange("Only scalar fields integrate to a number."));
        }
        Ok(self.values.column(0).iter().zip(self.domain.weights()).map(|(v, w)| v * *w).sum())
    }

    /// The inner product `∫ F · G* dA` of two fields on the same domain, computed in Cartesian
    /// components (so the two fields may use different bases).
    pub fn inner_product(&self, other: &HarmonicField) -> Result<C64, ErrorsJSL> {
        if !(Arc::ptr_eq(&self.domain, &other.domain) || self.domain == other.domain) {
            return Err(ErrorsJSL::InvalidInputRange("Fields live on different domains (resample first)."));
        }
        if self.is_scalar() != other.is_scalar() || self.convention != other.convention {
            return Err(ErrorsJSL::InvalidInputRange("Inner products need matching rank and time convention."));
        }
        Ok((0..self.len())
            .map(|i| {
                let (a, b) = (self.cartesian(i), other.cartesian(i));
                let dot: C64 = if self.is_scalar() { a[0] * b[0].conj() } else { (0..3).map(|k| a[k] * b[k].conj()).sum() };
                dot * self.domain.weights()[i]
            })
            .sum())
    }

    /// The real time signal `Re{X e^{±jωt}}` on a time axis (linear bases only).
    pub fn to_time_series(&self, time: TimeAxis) -> Result<TimeField, ErrorsJSL> {
        if self.basis().is_some_and(|b| b.is_circular()) {
            return Err(ErrorsJSL::InvalidInputRange("Convert circular components to a linear basis first."));
        }
        let (n, c) = self.values.dim();
        let omega = self.omega();
        let values = Array3::from_shape_fn((time.count, n, c), |(t, i, k)| self.convention.instantaneous(self.values[[i, k]], omega, time.time(t)));
        TimeField::new(self.domain.clone(), self.quantity, self.rank, time, values)
    }
}

/// A uniform time axis `start + k·step`, `k = 0..count`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeAxis {
    pub start: f64,
    pub step: f64,
    pub count: usize,
}

impl TimeAxis {
    pub fn new(start: f64, step: f64, count: usize) -> Result<Self, ErrorsJSL> {
        if !start.is_finite() || !step.is_finite() || step <= 0.0 || count == 0 {
            return Err(ErrorsJSL::InvalidInputRange("A time axis needs a finite start, a positive step and at least one sample."));
        }
        Ok(TimeAxis { start, step, count })
    }

    pub fn time(&self, k: usize) -> f64 {
        self.start + self.step * k as f64
    }

    pub fn times(&self) -> Vec<f64> {
        (0..self.count).map(|k| self.time(k)).collect()
    }
}

/// Real field samples on a domain over a uniform time axis.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeField {
    domain: Arc<Domain>,
    quantity: FieldQuantity,
    rank: Rank,
    time: TimeAxis,
    values: Array3<f64>,
}

impl TimeField {
    /// A field from a `(steps, samples, components)` array; vector fields need a linear basis.
    pub fn new(domain: Arc<Domain>, quantity: FieldQuantity, rank: Rank, time: TimeAxis, values: Array3<f64>) -> Result<Self, ErrorsJSL> {
        check_rank(quantity, rank)?;
        if rank.basis().is_some_and(|b| b.is_circular()) {
            return Err(ErrorsJSL::InvalidInputRange("Time-varying vector fields need a linear basis."));
        }
        if values.dim() != (time.count, domain.len(), rank.components()) {
            return Err(ErrorsJSL::IncompatibleArraySizes((values.dim().1, domain.len())));
        }
        Ok(TimeField { domain, quantity, rank, time, values })
    }

    /// A field sampled from a function of position and time returning `rank.components()` values.
    pub fn from_fn(domain: Arc<Domain>, quantity: FieldQuantity, rank: Rank, time: TimeAxis, f: impl Fn(&Position, f64) -> Vec<f64>) -> Result<Self, ErrorsJSL> {
        let (n, c) = (domain.len(), rank.components());
        let mut values = Array3::zeros((time.count, n, c));
        for i in 0..n {
            let p = domain.position_flat(i);
            for t in 0..time.count {
                let v = f(&p, time.time(t));
                if v.len() != c {
                    return Err(ErrorsJSL::IncompatibleArraySizes((v.len(), c)));
                }
                for k in 0..c {
                    values[[t, i, k]] = v[k];
                }
            }
        }
        Self::new(domain, quantity, rank, time, values)
    }

    pub fn domain(&self) -> &Arc<Domain> {
        &self.domain
    }

    pub fn quantity(&self) -> FieldQuantity {
        self.quantity
    }

    pub fn rank(&self) -> Rank {
        self.rank
    }

    pub fn basis(&self) -> Option<PolarizationBasis> {
        self.rank.basis()
    }

    pub fn time_axis(&self) -> TimeAxis {
        self.time
    }

    pub fn values(&self) -> &Array3<f64> {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut Array3<f64> {
        &mut self.values
    }

    /// The components at one time step and sample.
    pub fn at(&self, step: usize, sample: usize) -> Vec<f64> {
        (0..self.rank.components()).map(|k| self.values[[step, sample, k]]).collect()
    }

    /// The harmonic component at `frequency` by projection over the time axis:
    /// `X = (2/N) Σ x(t) e^{∓jωt}`, exact when the axis spans a whole number of periods.
    /// `frequency` must be positive.
    pub fn harmonic_at(&self, frequency: f64, convention: TimeConvention) -> Result<HarmonicField, ErrorsJSL> {
        let (steps, n, c) = self.values.dim();
        let omega = TAU * frequency;
        let mut out = Array2::<C64>::zeros((n, c));
        for t in 0..steps {
            let phase = C64::from_polar(2.0 / steps as f64, -convention.sign() * omega * self.time.time(t));
            for i in 0..n {
                for k in 0..c {
                    out[[i, k]] += phase * self.values[[t, i, k]];
                }
            }
        }
        HarmonicField::new(self.domain.clone(), self.quantity, self.rank, frequency, convention, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coordinates::systems::CoordinateSystem;
    use crate::fields::domain::Axis;
    use std::f64::consts::PI;

    fn sphere() -> Arc<Domain> {
        Arc::new(Domain::far_field_gauss(8, 16).unwrap())
    }

    fn c(re: f64, im: f64) -> C64 {
        C64::new(re, im)
    }

    #[test]
    fn construction_rank_rules_and_accessors() {
        let d = sphere();
        let e = HarmonicField::vector_from_fn(d.clone(), FieldQuantity::ElectricField, PolarizationBasis::SPHERICAL, 1e9, TimeConvention::PlusJ, |p| [c(0.0, 0.0), c(p.coords[1].sin(), 0.0), c(0.0, 1.0)]).unwrap();
        assert_eq!((e.len(), e.basis(), e.quantity().units()), (d.len(), Some(PolarizationBasis::SPHERICAL), "V/m"));
        assert!((e.omega() - TAU * 1e9).abs() < 1e-3);
        // A θ̂ vector at the equator is -ẑ in Cartesian components.
        let eq = (0..d.len()).find(|&i| (d.position_flat(i).coords[1] - PI / 2.0).abs() < 0.2).unwrap();
        let cart = e.cartesian(eq);
        assert!(cart[2].re < -0.9);
        assert!(HarmonicField::zeros(d.clone(), FieldQuantity::Temperature, Rank::Vector(PolarizationBasis::CARTESIAN), 1.0, TimeConvention::PlusJ).is_err());
        assert!(HarmonicField::zeros(d.clone(), FieldQuantity::ElectricField, Rank::Scalar, 1.0, TimeConvention::PlusJ).is_err());
        assert!(HarmonicField::zeros(d.clone(), FieldQuantity::Generic, Rank::Scalar, 0.0, TimeConvention::PlusJ).is_err());
        assert!(HarmonicField::new(d.clone(), FieldQuantity::Generic, Rank::Scalar, 1.0, TimeConvention::PlusJ, Array2::zeros((3, 1))).is_err());
        assert!(e.clone().with_quantity(FieldQuantity::MagneticField).is_ok() && e.clone().with_quantity(FieldQuantity::ScalarPotential).is_err());
    }

    #[test]
    fn integrals_inner_products_and_superposition() {
        let d = sphere();
        let one = HarmonicField::scalar_from_fn(d.clone(), FieldQuantity::Generic, 1.0, TimeConvention::PlusJ, |_| c(1.0, 0.0)).unwrap();
        assert!((one.integrate().unwrap().re - 4.0 * PI).abs() < 1e-12);
        // ∫|ẑ|² dΩ = 4π regardless of the basis the field is stored in.
        let z_cart = HarmonicField::vector_from_fn(d.clone(), FieldQuantity::ElectricField, PolarizationBasis::CARTESIAN, 1.0, TimeConvention::PlusJ, |_| [c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)]).unwrap();
        let z_sph = HarmonicField::vector_from_fn(d.clone(), FieldQuantity::ElectricField, PolarizationBasis::SPHERICAL, 1.0, TimeConvention::PlusJ, |p| {
            let (s, cth) = p.coords[1].sin_cos();
            [c(cth, 0.0), c(-s, 0.0), c(0.0, 0.0)]
        })
        .unwrap();
        assert!((z_cart.integrate_norm_squared() - 4.0 * PI).abs() < 1e-12);
        assert!((z_cart.inner_product(&z_sph).unwrap() - c(4.0 * PI, 0.0)).norm() < 1e-12);
        let double = z_cart.add(&z_cart).unwrap();
        assert!((double.integrate_norm_squared() - 16.0 * PI).abs() < 1e-11);
        assert!((z_cart.scaled(c(0.0, 2.0)).integrate_norm_squared() - 16.0 * PI).abs() < 1e-11);
        assert!(z_cart.add(&z_sph).is_err()); // different bases: convert first
        assert!(z_cart.integrate().is_err());
        let other = Arc::new(Domain::far_field_gauss(4, 8).unwrap());
        let elsewhere = HarmonicField::zeros(other, FieldQuantity::ElectricField, Rank::Vector(PolarizationBasis::CARTESIAN), 1.0, TimeConvention::PlusJ).unwrap();
        assert!(z_cart.inner_product(&elsewhere).is_err());
    }

    #[test]
    fn time_conventions_and_time_series_describe_the_same_signal() {
        let d = Arc::new(Domain::plane(Axis::uniform(0.0, 1.0, 3).unwrap(), Axis::uniform(0.0, 1.0, 2).unwrap(), 0.0).unwrap());
        let f = 50.0;
        let h = HarmonicField::scalar_from_fn(d.clone(), FieldQuantity::ScalarPotential, f, TimeConvention::PlusJ, |p| C64::from_polar(1.0 + p.coords[0], 0.4)).unwrap();
        let t = TimeAxis::new(0.0, 1.0 / (f * 40.0), 40).unwrap(); // exactly one period
        let series = h.to_time_series(t).unwrap();
        let other = h.with_convention(TimeConvention::MinusI);
        assert_ne!(other.values(), h.values());
        let series2 = other.to_time_series(t).unwrap();
        assert!(series.values().iter().zip(series2.values().iter()).all(|(a, b)| (a - b).abs() < 1e-12));
        assert!((series.at(0, 2)[0] - 1.5 * 0.4f64.cos()).abs() < 1e-12); // sample 2 is x = 0.5, y = 0
        // Projecting the time series back recovers the phasors, in either convention.
        for conv in [TimeConvention::PlusJ, TimeConvention::MinusI] {
            let back = series.harmonic_at(f, conv).unwrap();
            let want = h.with_convention(conv);
            assert!(back.values().iter().zip(want.values().iter()).all(|(a, b)| (a - b).norm() < 1e-12), "{conv:?}");
        }
    }

    #[test]
    fn time_field_rules() {
        let d = Arc::new(Domain::cartesian_box(Axis::uniform(0.0, 1.0, 2).unwrap(), Axis::uniform(0.0, 1.0, 2).unwrap(), Axis::uniform(0.0, 1.0, 2).unwrap()).unwrap());
        let t = TimeAxis::new(0.0, 0.1, 5).unwrap();
        let tf = TimeField::from_fn(d.clone(), FieldQuantity::ElectricField, Rank::Vector(PolarizationBasis::CARTESIAN), t, |p, time| vec![p.coords[0] * time, 0.0, 1.0]).unwrap();
        assert_eq!(tf.at(4, 7), vec![0.4, 0.0, 1.0]);
        assert_eq!(tf.time_axis().times().len(), 5);
        assert!(TimeField::from_fn(d.clone(), FieldQuantity::Generic, Rank::Vector(PolarizationBasis::Circular(CoordinateSystem::Cartesian)), t, |_, _| vec![0.0; 3]).is_err());
        assert!(TimeField::from_fn(d.clone(), FieldQuantity::Generic, Rank::Scalar, t, |_, _| vec![0.0; 2]).is_err());
        assert!(TimeAxis::new(0.0, 0.0, 3).is_err() && TimeAxis::new(0.0, 1.0, 0).is_err());
        let circ = HarmonicField::zeros(d, FieldQuantity::ElectricField, Rank::Vector(PolarizationBasis::Circular(CoordinateSystem::Cartesian)), 1.0, TimeConvention::PlusJ).unwrap();
        assert!(circ.to_time_series(t).is_err());
    }
}
