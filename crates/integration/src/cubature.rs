//! Multidimensional integration over hyperrectangles.

use crate::gaussian_quadrature::gauss_legendre;
use lib_jsl_core::ErrorsJSL;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// The outcome of a cubature.
#[derive(Debug, Clone)]
pub struct CubatureResult {
    pub value: f64,
    /// Estimated absolute error.
    pub error: f64,
    pub evaluations: usize,
    /// Whether the requested tolerance was met within the evaluation budget.
    pub converged: bool,
}

fn check_box(lower: &[f64], upper: &[f64]) -> Result<usize, ErrorsJSL> {
    if lower.len() != upper.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((lower.len(), upper.len())));
    }
    if lower.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("The box needs at least one dimension."));
    }
    if lower.iter().chain(upper).any(|v| !v.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("The box bounds must be finite."));
    }
    Ok(lower.len())
}

/// The tensor-product Gauss-Legendre rule with `points_per_axis` nodes along every axis: exact
/// for polynomials of degree `2·points − 1` in each variable, at a cost of `pointsᵈ` evaluations
/// (practical up to about five dimensions).
pub fn tensor_gauss_legendre<F: Fn(&[f64]) -> f64>(f: F, lower: &[f64], upper: &[f64], points_per_axis: usize) -> Result<f64, ErrorsJSL> {
    let d = check_box(lower, upper)?;
    let rule = gauss_legendre(points_per_axis)?;
    let half: Vec<f64> = (0..d).map(|i| 0.5 * (upper[i] - lower[i])).collect();
    let mid: Vec<f64> = (0..d).map(|i| 0.5 * (upper[i] + lower[i])).collect();
    let total = points_per_axis.checked_pow(d as u32).ok_or(ErrorsJSL::InvalidInputRange("Too many cubature points."))?;
    let mut x = vec![0.0; d];
    let mut index = vec![0usize; d];
    let mut sum = 0.0;
    for _ in 0..total {
        let mut w = 1.0;
        for k in 0..d {
            x[k] = mid[k] + half[k] * rule.nodes[index[k]];
            w *= rule.weights[index[k]];
        }
        sum += w * f(&x);
        // Odometer increment.
        for slot in index.iter_mut() {
            *slot += 1;
            if *slot < points_per_axis {
                break;
            }
            *slot = 0;
        }
    }
    Ok(sum * half.iter().product::<f64>())
}

/// A subregion with its degree-7 estimate, error estimate and preferred split axis.
struct Region {
    centre: Vec<f64>,
    half: Vec<f64>,
    value: f64,
    error: f64,
    split_axis: usize,
}

impl PartialEq for Region {
    fn eq(&self, other: &Self) -> bool {
        self.error == other.error
    }
}
impl Eq for Region {}
impl PartialOrd for Region {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Region {
    fn cmp(&self, other: &Self) -> Ordering {
        self.error.total_cmp(&other.error)
    }
}

/// Applies the Genz-Malik degree-7 rule with its embedded degree-5 rule on one box. Returns
/// `(value, error, split_axis)` and the number of evaluations.
fn genz_malik<F: Fn(&[f64]) -> f64>(f: &F, centre: &[f64], half: &[f64]) -> (f64, f64, usize, usize) {
    let n = centre.len();
    let nf = n as f64;
    let (l2, l3, l4, l5) = ((9.0f64 / 70.0).sqrt(), (9.0f64 / 10.0).sqrt(), (9.0f64 / 10.0).sqrt(), (9.0f64 / 19.0).sqrt());
    let w = [
        (12824.0 - 9120.0 * nf + 400.0 * nf * nf) / 19683.0,
        980.0 / 6561.0,
        (1820.0 - 400.0 * nf) / 19683.0,
        200.0 / 19683.0,
        6859.0 / 19683.0 / 2f64.powi(n as i32),
    ];
    let w5 = [(729.0 - 950.0 * nf + 50.0 * nf * nf) / 729.0, 245.0 / 486.0, (265.0 - 100.0 * nf) / 1458.0, 25.0 / 729.0];
    let mut x = centre.to_vec();
    let mut evaluations = 0;
    let mut eval = |x: &[f64]| {
        evaluations += 1;
        f(x)
    };
    let f0 = eval(&x);
    let (mut s2, mut s3, mut s4, mut s5) = (0.0, 0.0, 0.0, 0.0);
    let mut split_axis = 0;
    let mut largest_difference = -1.0;
    for i in 0..n {
        let mut pair = |lambda: f64, x: &mut Vec<f64>| {
            x[i] = centre[i] + lambda * half[i];
            let a = eval(x);
            x[i] = centre[i] - lambda * half[i];
            let b = eval(x);
            x[i] = centre[i];
            a + b
        };
        let p2 = pair(l2, &mut x);
        let p3 = pair(l3, &mut x);
        s2 += p2;
        s3 += p3;
        // Fourth-difference estimate of roughness along this axis.
        let difference = (p2 - 2.0 * f0 - (l2 * l2 / (l3 * l3)) * (p3 - 2.0 * f0)).abs();
        if difference > largest_difference * (1.0 + 1e-12) {
            largest_difference = difference;
            split_axis = i;
        }
    }
    for i in 0..n {
        for j in i + 1..n {
            for (si, sj) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                x[i] = centre[i] + si * l4 * half[i];
                x[j] = centre[j] + sj * l4 * half[j];
                s4 += eval(&x);
            }
            x[i] = centre[i];
            x[j] = centre[j];
        }
    }
    for corner in 0..(1usize << n) {
        for k in 0..n {
            let sign = if corner >> k & 1 == 1 { 1.0 } else { -1.0 };
            x[k] = centre[k] + sign * l5 * half[k];
        }
        s5 += eval(&x);
    }
    let volume: f64 = half.iter().map(|h| 2.0 * h).product();
    let seventh = volume * (w[0] * f0 + w[1] * s2 + w[2] * s3 + w[3] * s4 + w[4] * s5);
    let fifth = volume * (w5[0] * f0 + w5[1] * s2 + w5[2] * s3 + w5[3] * s4);
    (seventh, (seventh - fifth).abs(), split_axis, evaluations)
}

/// Options for [`adaptive_cubature`].
#[derive(Debug, Clone, Copy)]
pub struct CubatureOptions {
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub max_evaluations: usize,
}

impl Default for CubatureOptions {
    fn default() -> Self {
        CubatureOptions { absolute_tolerance: 1e-12, relative_tolerance: 1e-10, max_evaluations: 2_000_000 }
    }
}

/// Globally adaptive cubature over a hyperrectangle with the Genz-Malik degree-7 rule
/// (`2ⁿ + 2n² + 2n + 1` points per region) and its embedded degree-5 rule for error estimation.
/// The region with the largest error is repeatedly bisected along the axis with the largest
/// fourth divided difference, until the summed error meets
/// `max(absolute, relative · |value|)` or the evaluation budget is spent. Suited to smooth
/// integrands in two to about ten dimensions; for higher dimensions use quasi-Monte Carlo.
pub fn adaptive_cubature<F: Fn(&[f64]) -> f64>(f: F, lower: &[f64], upper: &[f64], options: CubatureOptions) -> Result<CubatureResult, ErrorsJSL> {
    let n = check_box(lower, upper)?;
    if n > 20 {
        return Err(ErrorsJSL::InvalidInputRange("The Genz-Malik rule needs 2^n points; use quasi-Monte Carlo above 20 dimensions."));
    }
    let centre: Vec<f64> = (0..n).map(|i| 0.5 * (lower[i] + upper[i])).collect();
    let half: Vec<f64> = (0..n).map(|i| 0.5 * (upper[i] - lower[i])).collect();
    let (value, error, split_axis, mut evaluations) = genz_malik(&f, &centre, &half);
    let mut heap = BinaryHeap::new();
    heap.push(Region { centre, half, value, error, split_axis });
    let (mut total, mut total_error) = (value, error);
    let per_region = evaluations;
    let tolerance = |v: f64| options.absolute_tolerance.max(options.relative_tolerance * v.abs());
    while total_error > tolerance(total) && evaluations + 2 * per_region <= options.max_evaluations {
        let region = heap.pop().expect("the heap always holds a region");
        let axis = region.split_axis;
        let mut half = region.half.clone();
        half[axis] *= 0.5;
        total -= region.value;
        total_error -= region.error;
        for sign in [-1.0, 1.0] {
            let mut centre = region.centre.clone();
            centre[axis] += sign * half[axis];
            let (value, error, split_axis, used) = genz_malik(&f, &centre, &half);
            evaluations += used;
            total += value;
            total_error += error;
            heap.push(Region { centre, half: half.clone(), value, error, split_axis });
        }
    }
    // Re-sum to shed the rounding accumulated by the running updates.
    let value: f64 = heap.iter().map(|r| r.value).sum();
    let error: f64 = heap.iter().map(|r| r.error).sum();
    Ok(CubatureResult { value, error, evaluations, converged: error <= tolerance(value) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{E, PI};

    #[test]
    fn genz_malik_is_exact_for_degree_seven_polynomials() {
        for n in 2..6 {
            let centre = vec![0.3; n];
            let half = vec![0.7; n];
            // ∫ over [−0.4, 1]^n of x₀⁶ x₁ + x₀³ x₁⁴ (degree 7) with the remaining axes constant.
            let f = |x: &[f64]| x[0].powi(6) * x[1] + x[0].powi(3) * x[1].powi(4);
            let m = |k: i32| (1f64.powi(k + 1) - (-0.4f64).powi(k + 1)) / (k + 1) as f64;
            let rest = 1.4f64.powi(n as i32 - 2);
            let want = (m(6) * m(1) + m(3) * m(4)) * rest;
            let (value, _, _, used) = genz_malik(&f, &centre, &half);
            assert!((value - want).abs() < 1e-13, "n={n}: {value} {want}");
            assert_eq!(used, (1 << n) + 2 * n * n + 2 * n + 1);
        }
    }

    #[test]
    fn adaptive_cubature_reaches_its_tolerance_on_smooth_and_peaked_integrands() {
        let r = adaptive_cubature(|x| (x[0] + x[1] + x[2]).exp(), &[0.0; 3], &[1.0; 3], CubatureOptions::default()).unwrap();
        assert!(r.converged && (r.value - (E - 1.0).powi(3)).abs() < 1e-9, "{r:?}");
        // A narrow Gaussian peak off-centre: ∫_{R²} e^{−((x−a)² + (y−b)²)/(2σ²)} = 2πσ².
        let s = 0.05;
        let peak = |x: &[f64]| (-((x[0] - 0.3).powi(2) + (x[1] + 0.2).powi(2)) / (2.0 * s * s)).exp();
        let r = adaptive_cubature(peak, &[-1.0, -1.0], &[1.0, 1.0], CubatureOptions::default()).unwrap();
        assert!(r.converged && (r.value - 2.0 * PI * s * s).abs() < 1e-10, "{r:?}");
        // Five dimensions: Π cos(x_i) over [0, 1]^5 = sin(1)^5.
        let r = adaptive_cubature(|x| x.iter().map(|v| v.cos()).product(), &[0.0; 5], &[1.0; 5], CubatureOptions::default()).unwrap();
        assert!((r.value - 1f64.sin().powi(5)).abs() < 1e-10, "{r:?}");
    }

    #[test]
    fn tensor_rule_one_dimension_and_validation() {
        let v = tensor_gauss_legendre(|x| x[0] * x[0] * x[1].exp(), &[0.0, 0.0], &[2.0, 1.0], 8).unwrap();
        assert!((v - 8.0 / 3.0 * (E - 1.0)).abs() < 1e-13);
        let r = adaptive_cubature(|x| x[0].sin(), &[0.0], &[PI], CubatureOptions::default()).unwrap();
        assert!((r.value - 2.0).abs() < 1e-10, "{r:?}");
        assert!(adaptive_cubature(|x| x[0], &[0.0], &[1.0, 2.0], CubatureOptions::default()).is_err());
        assert!(tensor_gauss_legendre(|x| x[0], &[], &[], 3).is_err());
        // A tiny budget stops early and reports non-convergence.
        let tight = CubatureOptions { max_evaluations: 50, ..Default::default() };
        assert!(!adaptive_cubature(peak_2d, &[-1.0, -1.0], &[1.0, 1.0], tight).unwrap().converged);
    }

    fn peak_2d(x: &[f64]) -> f64 {
        (-(x[0] * x[0] + x[1] * x[1]) * 400.0).exp()
    }
}
