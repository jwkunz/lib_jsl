//! Globally adaptive Gauss-Kronrod quadrature with the 7-point Gauss and 15-point Kronrod pair.
//!
//! See the book entry [Adaptive Gauss-Kronrod quadrature](https://jwkunz.github.io/lib_jsl/book/integration/gauss-kronrod.html).

use crate::can_integrate::CanIntegrate1D;
use lib_jsl_core::ErrorsJSL;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ANCHOR: constants
/// Weights of the 7-point Gauss-Legendre rule at `0`, `±KRONROD_NODES_15[2]`, `[4]` and `[6]`
/// (QUADPACK `qk15`, to full double precision).
#[allow(clippy::excessive_precision)] // QUADPACK's published digits, kept verbatim
pub(crate) const GAUSS_LEGENDRE_WEIGHTS_7: [f64; 4] = [
    0.417_959_183_673_469_387_755_102_040_816_327,
    0.381_830_050_505_118_944_950_369_775_488_975,
    0.279_705_391_489_276_667_901_467_771_423_780,
    0.129_484_966_168_869_693_270_611_432_679_082,
];

/// The non-negative nodes of the 15-point Kronrod rule, from the centre outwards. The entries
/// with even index are the nodes of the 7-point Gauss rule; the others are the Kronrod extension.
#[allow(clippy::excessive_precision)] // QUADPACK's published digits, kept verbatim
pub(crate) const KRONROD_NODES_15: [f64; 8] = [
    0.0,
    0.207_784_955_007_898_467_600_689_403_773_245,
    0.405_845_151_377_397_166_906_606_412_076_961,
    0.586_087_235_467_691_130_294_144_845_693_013,
    0.741_531_185_599_394_439_863_864_773_280_788,
    0.864_864_423_359_769_072_789_712_788_640_926,
    0.949_107_912_342_758_524_526_189_684_047_851,
    0.991_455_371_120_812_639_206_854_697_526_329,
];

/// Weights of the 15-point Kronrod rule, matching [`KRONROD_NODES_15`].
#[allow(clippy::excessive_precision)] // QUADPACK's published digits, kept verbatim
pub(crate) const KRONROD_WEIGHTS_15: [f64; 8] = [
    0.209_482_141_084_727_828_012_999_174_891_714,
    0.204_432_940_075_298_892_414_161_999_234_649,
    0.190_350_578_064_785_409_913_256_402_421_014,
    0.169_004_726_639_267_902_826_583_426_598_550,
    0.140_653_259_715_525_918_745_189_590_510_238,
    0.104_790_010_322_250_183_839_876_322_541_518,
    0.063_092_092_629_978_553_290_700_663_189_204,
    0.022_935_322_010_529_224_963_732_008_058_970,
];
// ANCHOR_END: constants

/// The Kronrod estimate of `∫ₐᵇ f`, an estimate of its error, and whether every sample was finite.
#[derive(Debug, Clone, Copy)]
struct Panel {
    a: f64,
    b: f64,
    value: f64,
    error: f64,
}

// ANCHOR: gauss_kronrod_panel
/// Applies the G7-K15 pair on `[a, b]` with QUADPACK's error estimate.
fn gauss_kronrod<F: Fn(f64) -> f64>(a: f64, b: f64, f: &F) -> Result<Panel, ErrorsJSL> {
    let centre = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let f_centre = f(centre);
    let mut kronrod = f_centre * KRONROD_WEIGHTS_15[0];
    let mut gauss = f_centre * GAUSS_LEGENDRE_WEIGHTS_7[0];
    let mut absolute = kronrod.abs();
    let mut pairs = [(0.0, 0.0); 8];
    for j in 1..8 {
        let dx = half * KRONROD_NODES_15[j];
        let (lo, hi) = (f(centre - dx), f(centre + dx));
        pairs[j] = (lo, hi);
        kronrod += KRONROD_WEIGHTS_15[j] * (lo + hi);
        absolute += KRONROD_WEIGHTS_15[j] * (lo.abs() + hi.abs());
        if j % 2 == 0 {
            gauss += GAUSS_LEGENDRE_WEIGHTS_7[j / 2] * (lo + hi);
        }
    }
    // ∫|f − mean| over the panel, which calibrates the raw difference |K − G|.
    let mean = 0.5 * kronrod;
    let mut deviation = KRONROD_WEIGHTS_15[0] * (f_centre - mean).abs();
    for (j, &(lo, hi)) in pairs.iter().enumerate().skip(1) {
        deviation += KRONROD_WEIGHTS_15[j] * ((lo - mean).abs() + (hi - mean).abs());
    }
    let (value, absolute, deviation) = (kronrod * half, absolute * half.abs(), deviation * half.abs());
    let mut error = ((kronrod - gauss) * half).abs();
    if deviation != 0.0 && error != 0.0 {
        error = deviation * (200.0 * error / deviation).powf(1.5).min(1.0);
    }
    if absolute > f64::MIN_POSITIVE / (50.0 * f64::EPSILON) {
        error = error.max(50.0 * f64::EPSILON * absolute);
    }
    if !value.is_finite() || !error.is_finite() {
        return Err(ErrorsJSL::RuntimeError("Gauss-Kronrod: the integrand is not finite at a sample point"));
    }
    Ok(Panel { a, b, value, error })
}
// ANCHOR_END: gauss_kronrod_panel

/// Orders panels by error estimate, for the priority queue.
struct ByError(Panel);

impl PartialEq for ByError {
    fn eq(&self, other: &Self) -> bool {
        self.0.error == other.0.error
    }
}
impl Eq for ByError {}
impl PartialOrd for ByError {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ByError {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.error.total_cmp(&other.0.error)
    }
}

/// Globally adaptive Gauss-Kronrod quadrature (the strategy of QUADPACK's `qag` with its 15-point
/// rule): the interval with the largest error estimate is bisected until the estimates sum to
/// less than the tolerance.
///
/// Each panel is integrated with the 15-point Kronrod rule, which contains the 7-point Gauss rule;
/// their difference, calibrated as in QUADPACK, estimates the error of the (far more accurate)
/// Kronrod value. The rule is exact for polynomials of degree 22 on each panel, so smooth
/// integrands need very few panels, while kinks, peaks and integrable endpoint singularities
/// draw the bisections towards themselves.
///
/// The tolerance is absolute, raised if necessary to the rounding-error level
/// `50 ε ∫|f|`; [`integrate`](CanIntegrate1D::integrate) fails if it cannot be met within
/// [`max_subintervals`](Self::max_subintervals) panels, and
/// [`integrate_with_error`](Self::integrate_with_error) returns the estimate either way.
pub struct AdaptiveGL7K15IntegrationEngine {
    tolerance: f64,
    lower_bound: f64,
    upper_bound: f64,
    /// The largest number of panels (default 1000).
    pub max_subintervals: usize,
}

impl AdaptiveGL7K15IntegrationEngine {
    /// An engine on `[lower_bound, upper_bound]` aiming for an absolute error below `tolerance`.
    pub fn new(tolerance: f64, lower_bound: f64, upper_bound: f64) -> Self {
        AdaptiveGL7K15IntegrationEngine { tolerance, lower_bound, upper_bound, max_subintervals: 1000 }
    }

    /// Changes the error tolerance.
    pub fn set_tolerance(&mut self, x: f64) {
        self.tolerance = x;
    }

    /// The integral, its error estimate and the number of evaluations of `f`. The estimate may
    /// exceed the tolerance if `max_subintervals` was reached; a non-finite sample is an error.
    pub fn integrate_with_error<F: Fn(f64) -> f64>(&self, f: F) -> Result<(f64, f64, usize), ErrorsJSL> {
        // ANCHOR: global_adaptivity
        let first = gauss_kronrod(self.lower_bound, self.upper_bound, &f)?;
        let (mut value, mut error) = (first.value, first.error);
        let mut queue = BinaryHeap::from([ByError(first)]);
        let mut evaluations = 15;
        while error > self.tolerance.max(50.0 * f64::EPSILON * value.abs()) && queue.len() < self.max_subintervals {
            let worst = queue.pop().expect("at least one panel").0;
            let middle = 0.5 * (worst.a + worst.b);
            if middle <= worst.a.min(worst.b) || middle >= worst.a.max(worst.b) {
                // The panel cannot be split further in floating point.
                queue.push(ByError(worst));
                break;
            }
            let left = gauss_kronrod(worst.a, middle, &f)?;
            let right = gauss_kronrod(middle, worst.b, &f)?;
            evaluations += 30;
            value += left.value + right.value - worst.value;
            error += left.error + right.error - worst.error;
            queue.push(ByError(left));
            queue.push(ByError(right));
        }
        // ANCHOR_END: global_adaptivity
        // Re-sum to remove the rounding accumulated by the running updates.
        let value = queue.iter().map(|p| p.0.value).sum();
        let error = queue.iter().map(|p| p.0.error).sum();
        Ok((value, error, evaluations))
    }
}

impl CanIntegrate1D for AdaptiveGL7K15IntegrationEngine {
    fn integrate<F>(&self, f: F) -> Result<f64, ErrorsJSL>
    where
        F: Fn(f64) -> f64,
    {
        let (value, error, _) = self.integrate_with_error(f)?;
        if error > self.tolerance.max(50.0 * f64::EPSILON * value.abs()) {
            return Err(ErrorsJSL::RuntimeError("Gauss-Kronrod: the tolerance was not met within the subinterval limit"));
        }
        Ok(value)
    }

    fn set_lower_bound(&mut self, x: f64) {
        self.lower_bound = x;
    }

    fn set_upper_bound(&mut self, x: f64) {
        self.upper_bound = x;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_gl7k15_adaptive_integration() {
        let dut = AdaptiveGL7K15IntegrationEngine::new(1e-12, PI / 2.0, 0.0);
        assert!((dut.integrate(f64::sin).unwrap() + 1.0).abs() < 1e-12);
        let dut = AdaptiveGL7K15IntegrationEngine::new(1e-12, -PI / 2.0, PI / 2.0);
        assert!(dut.integrate(f64::sin).unwrap().abs() < 1e-12);
    }

    #[test]
    fn the_pair_has_the_expected_degrees_of_exactness() {
        // On [−1, 1]: Kronrod is exact to degree 22 (3·7 + 1), Gauss to degree 13 (2·7 − 1).
        let exact = |k: i32| if k % 2 == 1 { 0.0 } else { 2.0 / (k + 1) as f64 };
        for k in 0..=23 {
            let p = gauss_kronrod(-1.0, 1.0, &|x: f64| x.powi(k)).unwrap();
            if k <= 22 {
                assert!((p.value - exact(k)).abs() < 2e-16, "Kronrod, degree {k}: {}", p.value - exact(k));
            }
        }
        let gauss = |k: i32| {
            let (kn, g) = (KRONROD_NODES_15, GAUSS_LEGENDRE_WEIGHTS_7);
            let x = [-kn[6], -kn[4], -kn[2], 0.0, kn[2], kn[4], kn[6]];
            let w = [g[3], g[2], g[1], g[0], g[1], g[2], g[3]];
            x.iter().zip(w).map(|(x, w)| w * x.powi(k)).sum::<f64>()
        };
        for k in 0..=13 {
            assert!((gauss(k) - exact(k)).abs() < 1e-15, "Gauss, degree {k}: {}", gauss(k) - exact(k));
        }
        assert!((gauss(14) - exact(14)).abs() > 1e-6);
    }

    #[test]
    fn difficult_integrands_and_failures() {
        // An integrable endpoint singularity: ∫₀¹ ln x dx = −1.
        let dut = AdaptiveGL7K15IntegrationEngine::new(1e-10, 0.0, 1.0);
        let (v, err, _) = dut.integrate_with_error(f64::ln).unwrap();
        assert!((v + 1.0).abs() < 1e-10 && err < 1e-10, "{v} {err}");
        // A tolerance below rounding level is raised rather than looping forever.
        let tiny = AdaptiveGL7K15IntegrationEngine::new(1e-30, 0.0, 1.0);
        assert!((tiny.integrate(f64::exp).unwrap() - (std::f64::consts::E - 1.0)).abs() < 1e-15);
        // A non-finite sample is an error, as is an unreachable tolerance.
        assert!(AdaptiveGL7K15IntegrationEngine::new(1e-8, -1.0, 1.0).integrate(|x| 1.0 / x).is_err());
        let mut few = AdaptiveGL7K15IntegrationEngine::new(1e-14, 0.0, 1.0);
        few.max_subintervals = 3;
        assert!(few.integrate(|x: f64| x.powf(-0.9)).is_err());
    }
}
