//! Gaussian quadrature rules: Gauss-Legendre, Gauss-Hermite and generalized Gauss-Laguerre.
//!
//! An n-point Gauss rule integrates `w(x) p(x)` exactly for every polynomial `p` of degree up to
//! `2n − 1`. Nodes are the zeros of the orthogonal polynomial of degree n, found by Newton's
//! method on the three-term recurrence from asymptotic starting guesses (as in Numerical
//! Recipes), so rules with hundreds of points are cheap and accurate to rounding.

use crate::can_integrate::CanIntegrate1D;
use lib_jsl_core::ErrorsJSL;
use lib_jsl_special_functions::ln_gamma;
use std::f64::consts::PI;

/// Nodes and weights of a quadrature rule.
#[derive(Debug, Clone)]
pub struct QuadratureRule {
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
}

impl QuadratureRule {
    /// `Σ w_i f(x_i)`.
    pub fn apply<F: Fn(f64) -> f64>(&self, f: F) -> f64 {
        self.nodes.iter().zip(&self.weights).map(|(&x, &w)| w * f(x)).sum()
    }
}

fn check(n: usize) -> Result<(), ErrorsJSL> {
    if n == 0 {
        return Err(ErrorsJSL::InvalidInputRange("A quadrature rule needs at least one point."));
    }
    Ok(())
}

/// The n-point Gauss-Legendre rule for `∫_{−1}^{1} f(x) dx`, nodes ascending.
///
/// The nodes are the zeros of the Legendre polynomial `P_n` and the weights are
/// `w_i = 2 / ((1 − x_i²) P_n'(x_i)²)`. The rule is exact for polynomials of degree `2n − 1`.
/// Map it to `[a, b]` with `x = (a + b)/2 + (b − a)/2 · t`, or use [`GaussLegendreEngine`].
///
/// ```
/// use lib_jsl_integration::gaussian_quadrature::gauss_legendre;
///
/// let rule = gauss_legendre(3).unwrap();
/// // Three points integrate x⁴ exactly (degree 4 ≤ 2·3 − 1): ∫ x⁴ dx over [−1, 1] = 2/5.
/// assert!((rule.apply(|x| x.powi(4)) - 0.4).abs() < 1e-15);
/// // The weights sum to the length of the interval.
/// assert!((rule.weights.iter().sum::<f64>() - 2.0).abs() < 1e-15);
/// ```
pub fn gauss_legendre(n: usize) -> Result<QuadratureRule, ErrorsJSL> {
    check(n)?;
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    // ANCHOR: legendre_newton
    // Nodes come in ± pairs, so only the non-negative half is computed.
    for i in 0..n.div_ceil(2) {
        // Asymptotic first guess for the i-th zero (Tricomi).
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut derivative = 1.0;
        for _ in 0..100 {
            // Bonnet's recurrence: j P_j = (2j − 1) x P_{j−1} − (j − 1) P_{j−2}.
            let (mut p1, mut p2) = (1.0, 0.0);
            for j in 1..=n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2 * j - 1) as f64 * z * p2 - (j - 1) as f64 * p3) / j as f64;
            }
            // P_n'(x) = n (x P_n − P_{n−1}) / (x² − 1), then a Newton step.
            derivative = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            let dz = p1 / derivative;
            z -= dz;
            if dz.abs() <= 1e-16 {
                break;
            }
        }
        nodes[i] = -z;
        nodes[n - 1 - i] = z;
        let w = 2.0 / ((1.0 - z * z) * derivative * derivative);
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    // ANCHOR_END: legendre_newton
    if n % 2 == 1 {
        nodes[n / 2] = 0.0;
    }
    Ok(QuadratureRule { nodes, weights })
}

/// The n-point Gauss-Hermite rule for `∫_{−∞}^{∞} e^{−x²} f(x) dx` (physicists' weight), nodes
/// ascending. For the probabilists' weight `e^{−x²/2}` scale nodes by √2 and weights by √2.
pub fn gauss_hermite(n: usize) -> Result<QuadratureRule, ErrorsJSL> {
    check(n)?;
    let pim4 = PI.powf(-0.25);
    let mut z_values = vec![0.0; n];
    let mut w_values = vec![0.0; n];
    let nf = n as f64;
    let mut z = 0.0;
    for i in 0..n.div_ceil(2) {
        z = match i {
            0 => (2.0 * nf + 1.0).sqrt() - 1.85575 * (2.0 * nf + 1.0).powf(-1.0 / 6.0),
            1 => z - 1.14 * nf.powf(0.426) / z,
            2 => 1.86 * z - 0.86 * z_values[0],
            3 => 1.91 * z - 0.91 * z_values[1],
            _ => 2.0 * z - z_values[i - 2],
        };
        let mut derivative = 1.0;
        for _ in 0..100 {
            // Orthonormal Hermite recurrence (stable for large n).
            let (mut p1, mut p2) = (pim4, 0.0);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = z * (2.0 / (jf + 1.0)).sqrt() * p2 - (jf / (jf + 1.0)).sqrt() * p3;
            }
            derivative = (2.0 * nf).sqrt() * p2;
            let dz = p1 / derivative;
            z -= dz;
            if dz.abs() <= 1e-15 * z.abs().max(1.0) {
                break;
            }
        }
        z_values[i] = z;
        w_values[i] = 2.0 / (derivative * derivative);
    }
    // Store ascending: the computed zeros are the positive ones, largest first.
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        nodes[n - 1 - i] = z_values[i];
        nodes[i] = -z_values[i];
        weights[i] = w_values[i];
        weights[n - 1 - i] = w_values[i];
    }
    if n % 2 == 1 {
        nodes[n / 2] = 0.0;
    }
    Ok(QuadratureRule { nodes, weights })
}

/// The n-point generalized Gauss-Laguerre rule for `∫_0^∞ x^α e^{−x} f(x) dx` (`α > −1`), nodes
/// ascending.
pub fn gauss_laguerre(n: usize, alpha: f64) -> Result<QuadratureRule, ErrorsJSL> {
    check(n)?;
    if alpha.is_nan() || alpha <= -1.0 {
        return Err(ErrorsJSL::InvalidInputRange("The Laguerre exponent must exceed -1."));
    }
    let nf = n as f64;
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let mut z = 0.0;
    let log_norm = ln_gamma(alpha + nf) - ln_gamma(nf);
    for i in 0..n {
        z = match i {
            0 => (1.0 + alpha) * (3.0 + 0.92 * alpha) / (1.0 + 2.4 * nf + 1.8 * alpha),
            1 => z + (15.0 + 6.25 * alpha) / (1.0 + 0.9 * alpha + 2.5 * nf),
            _ => {
                let ai = (i - 1) as f64;
                z + ((1.0 + 2.55 * ai) / (1.9 * ai) + 1.26 * ai * alpha / (1.0 + 3.5 * ai)) * (z - nodes[i - 2]) / (1.0 + 0.3 * alpha)
            }
        };
        let (mut derivative, mut previous) = (1.0, 0.0);
        for _ in 0..100 {
            let (mut p1, mut p2) = (1.0, 0.0);
            for j in 1..=n {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = ((2.0 * jf - 1.0 + alpha - z) * p2 - (jf - 1.0 + alpha) * p3) / jf;
            }
            derivative = (nf * p1 - (nf + alpha) * p2) / z;
            previous = p2;
            let dz = p1 / derivative;
            z -= dz;
            if dz.abs() <= 1e-15 * z.abs().max(1e-300) {
                break;
            }
        }
        nodes[i] = z;
        weights[i] = -log_norm.exp() / (derivative * nf * previous);
    }
    Ok(QuadratureRule { nodes, weights })
}

/// Integrates `f` over `[a, b]` with a fixed n-point Gauss-Legendre rule.
#[derive(Debug, Clone)]
pub struct GaussLegendreEngine {
    rule: QuadratureRule,
    lower_bound: f64,
    upper_bound: f64,
}

impl GaussLegendreEngine {
    pub fn new(number_of_points: usize, lower_bound: f64, upper_bound: f64) -> Result<Self, ErrorsJSL> {
        Ok(GaussLegendreEngine { rule: gauss_legendre(number_of_points)?, lower_bound, upper_bound })
    }
}

impl CanIntegrate1D for GaussLegendreEngine {
    fn set_lower_bound(&mut self, x: f64) {
        self.lower_bound = x;
    }
    fn set_upper_bound(&mut self, x: f64) {
        self.upper_bound = x;
    }
    fn integrate<F>(&self, f: F) -> Result<f64, ErrorsJSL>
    where
        F: Fn(f64) -> f64,
    {
        let half = 0.5 * (self.upper_bound - self.lower_bound);
        let mid = 0.5 * (self.upper_bound + self.lower_bound);
        Ok(half * self.rule.apply(|t| f(mid + half * t)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lib_jsl_special_functions::gamma;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn legendre_rules_are_exact_to_degree_2n_minus_1() {
        for n in [1, 2, 3, 5, 10, 64, 200] {
            let rule = gauss_legendre(n).unwrap();
            assert!(close(rule.weights.iter().sum(), 2.0, 1e-14), "n={n}");
            assert!(rule.nodes.windows(2).all(|w| w[0] < w[1]));
            for k in 0..(2 * n).min(30) {
                let exact = if k % 2 == 1 { 0.0 } else { 2.0 / (k + 1) as f64 };
                assert!(close(rule.apply(|x| x.powi(k as i32)), exact, 1e-13), "n={n} k={k}");
            }
        }
        // One degree higher is no longer exact.
        let r3 = gauss_legendre(3).unwrap();
        assert!((r3.apply(|x| x.powi(6)) - 2.0 / 7.0).abs() > 1e-3);
        let engine = GaussLegendreEngine::new(20, 0.0, PI).unwrap();
        assert!(close(engine.integrate(f64::sin).unwrap(), 2.0, 1e-15));
        assert!(gauss_legendre(0).is_err());
    }

    #[test]
    fn hermite_rules_integrate_gaussian_moments() {
        for n in [1, 2, 5, 20, 100] {
            let rule = gauss_hermite(n).unwrap();
            assert!(close(rule.weights.iter().sum(), PI.sqrt(), 1e-13), "n={n}");
            assert!(rule.nodes.windows(2).all(|w| w[0] < w[1]));
            // ∫ e^{−x²} x^{2k} = Γ(k + ½).
            for k in 0..n.min(10) {
                assert!(close(rule.apply(|x| x.powi(2 * k as i32)), gamma(k as f64 + 0.5), 1e-12), "n={n} k={k}");
            }
        }
        let rule = gauss_hermite(30).unwrap();
        assert!(close(rule.apply(f64::cos), PI.sqrt() * (-0.25f64).exp(), 1e-14));
    }

    #[test]
    fn laguerre_rules_integrate_gamma_moments() {
        for &alpha in &[0.0, -0.5, 1.5, 4.0] {
            for n in [1, 3, 8, 40] {
                let rule = gauss_laguerre(n, alpha).unwrap();
                assert!(rule.nodes.windows(2).all(|w| w[0] < w[1]) && rule.nodes[0] > 0.0);
                // ∫ x^α e^{−x} x^k = Γ(α + k + 1).
                for k in 0..(2 * n).min(12) {
                    let want = gamma(alpha + k as f64 + 1.0);
                    assert!(close(rule.apply(|x| x.powi(k as i32)), want, 1e-11 * want.max(1.0)), "α={alpha} n={n} k={k}");
                }
            }
        }
        // ∫_0^∞ e^{−x} sin x dx = ½.
        assert!(close(gauss_laguerre(40, 0.0).unwrap().apply(f64::sin), 0.5, 1e-12));
        assert!(gauss_laguerre(4, -1.0).is_err());
    }
}
