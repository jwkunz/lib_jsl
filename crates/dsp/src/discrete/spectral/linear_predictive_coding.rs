//! Linear predictive coding (LPC).
//!
//! LPC models a signal as the output of an all-pole filter driven by a white excitation: each
//! sample is predicted from the previous `order` samples, `x̂[n] = -Σ_{k=1}^{p} a[k] x[n-k]`, and the
//! coefficients minimise the prediction error power. Writing `A(z) = 1 + a[1] z^-1 + … + a[p] z^-p`,
//! the *analysis* filter `A(z)` whitens the signal into the residual, and the *synthesis* filter
//! `1 / A(z)` rebuilds the signal from it. The all-pole spectrum `σ² / |A(e^{jω})|²` is a smooth
//! estimate of the signal's spectral envelope, which is why LPC underlies speech coding.
//!
//! The coefficients here come from the *autocorrelation method*: the autocorrelation sequence is
//! computed (implicitly windowing the signal with zeros outside the record) and the resulting
//! Toeplitz normal equations are solved with the Levinson-Durbin recursion, which also yields the
//! reflection (PARCOR) coefficients. The autocorrelation method always gives a stable synthesis
//! filter (all reflection coefficients have magnitude below one).

use lib_jsl_core::ErrorsJSL;

/// The result of an LPC analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct LpcResult {
    /// `A(z)` coefficients `[1, a1, …, ap]` (length `order + 1`).
    pub coefficients: Vec<f64>,
    /// Reflection (PARCOR) coefficients `k1 … kp`; `|k| < 1` means the synthesis filter is stable.
    pub reflection: Vec<f64>,
    /// Prediction error power after the final stage (`r[0] ∏(1 - k²)`).
    pub error_power: f64,
}

/// Biased autocorrelation estimate `r[k] = Σ_n x[n] x[n+k]` for lags `0..=max_lag`.
pub fn autocorrelation(x: &[f64], max_lag: usize) -> Vec<f64> {
    (0..=max_lag).map(|k| x.iter().zip(x.iter().skip(k)).map(|(a, b)| a * b).sum()).collect()
}

/// Solves the Toeplitz normal equations for an autocorrelation sequence `r[0..=order]` with the
/// Levinson-Durbin recursion.
///
/// Fails if `r[0]` is not positive or the sequence is not positive-definite to the requested order.
pub fn levinson_durbin(r: &[f64], order: usize) -> Result<LpcResult, ErrorsJSL> {
    if order == 0 || r.len() <= order {
        return Err(ErrorsJSL::InvalidInputRange("Levinson-Durbin needs order >= 1 and order + 1 autocorrelation lags"));
    }
    if !(r[0] > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Zero-lag autocorrelation must be positive"));
    }
    let mut a = vec![0.0; order + 1];
    a[0] = 1.0;
    let mut err = r[0];
    let mut reflection = Vec::with_capacity(order);
    for m in 1..=order {
        let acc: f64 = (0..m).map(|k| a[k] * r[m - k]).sum();
        let k = -acc / err;
        if !k.is_finite() || k.abs() >= 1.0 + 1e-12 {
            return Err(ErrorsJSL::RuntimeError("Autocorrelation sequence is not positive-definite"));
        }
        let prev = a.clone();
        for i in 1..m {
            a[i] = prev[i] + k * prev[m - i];
        }
        a[m] = k;
        reflection.push(k);
        err *= 1.0 - k * k;
    }
    Ok(LpcResult { coefficients: a, reflection, error_power: err })
}

/// LPC analysis of `x` by the autocorrelation method.
///
/// `order` must be at least 1 and less than the signal length. An all-zero signal is rejected
/// because it has no defined predictor.
pub fn linear_predictive_coding(x: &[f64], order: usize) -> Result<LpcResult, ErrorsJSL> {
    if order == 0 || order >= x.len() {
        return Err(ErrorsJSL::InvalidInputRange("LPC order must satisfy 1 <= order < signal length"));
    }
    levinson_durbin(&autocorrelation(x, order), order)
}

/// The prediction residual `e = A(z) x` (the analysis filter), same length as `x`, assuming zero
/// initial conditions.
pub fn lpc_residual(x: &[f64], coefficients: &[f64]) -> Vec<f64> {
    (0..x.len())
        .map(|n| coefficients.iter().enumerate().filter(|(k, _)| *k <= n).map(|(k, a)| a * x[n - k]).sum())
        .collect()
}

/// Runs an excitation through the synthesis filter `1 / A(z)`, the inverse of [`lpc_residual`].
pub fn lpc_synthesize(excitation: &[f64], coefficients: &[f64]) -> Vec<f64> {
    let mut y = Vec::with_capacity(excitation.len());
    for (n, &e) in excitation.iter().enumerate() {
        let feedback: f64 = (1..coefficients.len()).filter(|&k| k <= n).map(|k| coefficients[k] * y[n - k]).sum();
        y.push(e - feedback);
    }
    y
}

/// Converts reflection coefficients to `A(z)` coefficients (the step-up recursion).
pub fn reflection_to_lpc(reflection: &[f64]) -> Vec<f64> {
    let mut a = vec![1.0];
    for &k in reflection {
        let mut next = a.clone();
        next.push(0.0);
        for i in 1..=a.len() {
            next[i] += k * a[a.len() - i];
        }
        a = next;
    }
    a
}

/// Converts `A(z)` coefficients to reflection coefficients (the step-down recursion). Fails if a
/// stage has `|k| >= 1` (an unstable or degenerate filter).
pub fn lpc_to_reflection(coefficients: &[f64]) -> Result<Vec<f64>, ErrorsJSL> {
    if coefficients.is_empty() || (coefficients[0] - 1.0).abs() > 1e-12 {
        return Err(ErrorsJSL::InvalidInputRange("LPC coefficients must start with 1"));
    }
    let mut a = coefficients.to_vec();
    let mut k = vec![0.0; a.len() - 1];
    for m in (1..a.len()).rev() {
        let km = a[m];
        if km.abs() >= 1.0 {
            return Err(ErrorsJSL::RuntimeError("Filter is not strictly stable: |reflection| >= 1"));
        }
        k[m - 1] = km;
        let denom = 1.0 - km * km;
        let prev = a.clone();
        for i in 1..m {
            a[i] = (prev[i] - km * prev[m - i]) / denom;
        }
        a.truncate(m);
    }
    Ok(k)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }

    /// Gaussian-ish unit-variance noise from a sum of uniforms.
    fn noise(n: usize, seed: &mut u64) -> Vec<f64> {
        (0..n).map(|_| (0..12).map(|_| lcg(seed)).sum::<f64>()).collect()
    }

    /// Synthesizes an AR process with `A(z) = 1 + a1 z^-1 + …`.
    fn ar_process(a: &[f64], n: usize, seed: &mut u64) -> Vec<f64> {
        lpc_synthesize(&noise(n + 500, seed), a)[500..].to_vec()
    }

    #[test]
    fn levinson_solves_a_known_second_order_system() {
        // r = [1, 0.5, 0.2]: closed form k1 = -r1/r0, a2 = (r1^2 - r0 r2) / (r0^2 - r1^2).
        let res = levinson_durbin(&[1.0, 0.5, 0.2], 2).unwrap();
        let a2 = (0.25 - 0.2) / (1.0 - 0.25);
        let a1 = -0.5 * (1.0 + a2);
        assert!((res.coefficients[1] - a1).abs() < 1e-12 && (res.coefficients[2] - a2).abs() < 1e-12);
        assert!((res.reflection[0] + 0.5).abs() < 1e-12 && (res.reflection[1] - a2).abs() < 1e-12);
        // Error power equals r0 + a1 r1 + a2 r2.
        assert!((res.error_power - (1.0 + a1 * 0.5 + a2 * 0.2)).abs() < 1e-12);
    }

    #[test]
    fn recovers_the_coefficients_of_an_ar_process() {
        let mut seed = 42;
        let truth = [1.0, -1.5, 0.9]; // poles at radius ~0.95
        let x = ar_process(&truth, 50_000, &mut seed);
        let est = linear_predictive_coding(&x, 2).unwrap();
        assert!((est.coefficients[1] - truth[1]).abs() < 0.02, "{:?}", est.coefficients);
        assert!((est.coefficients[2] - truth[2]).abs() < 0.02);
        assert!((est.error_power / x.len() as f64 - 1.0).abs() < 0.1, "innovation power {}", est.error_power / x.len() as f64);
        assert!(est.reflection.iter().all(|k| k.abs() < 1.0));
    }

    #[test]
    fn a_pure_sinusoid_is_perfectly_predictable_with_two_coefficients() {
        let w = 0.3;
        let x: Vec<f64> = (0..400).map(|n| (w * n as f64).sin()).collect();
        let res = linear_predictive_coding(&x, 2).unwrap();
        // x[n] = 2 cos(w) x[n-1] - x[n-2]  =>  a = [1, -2cos w, 1]
        assert!((res.coefficients[1] + 2.0 * w.cos()).abs() < 0.02);
        assert!((res.coefficients[2] - 1.0).abs() < 0.02);
        let e = lpc_residual(&x, &res.coefficients);
        let tail: f64 = e[10..].iter().map(|v| v * v).sum::<f64>() / (e.len() - 10) as f64;
        assert!(tail < 1e-3, "residual power {tail}");
    }

    #[test]
    fn analysis_and_synthesis_are_inverses() {
        let mut seed = 7;
        let x = noise(300, &mut seed);
        let a = [1.0, -0.6, 0.3, -0.1];
        let e = lpc_residual(&x, &a);
        let back = lpc_synthesize(&e, &a);
        assert!(x.iter().zip(&back).all(|(p, q)| (p - q).abs() < 1e-10));
        // The residual of the true model is white with the innovation variance.
        let synth = lpc_synthesize(&x, &a);
        let again = lpc_residual(&synth, &a);
        assert!(x.iter().zip(&again).all(|(p, q)| (p - q).abs() < 1e-10));
    }

    #[test]
    fn reflection_conversions_round_trip() {
        let mut seed = 3;
        let x = ar_process(&[1.0, -0.8, 0.5, -0.2], 5000, &mut seed);
        let res = linear_predictive_coding(&x, 6).unwrap();
        let rebuilt = reflection_to_lpc(&res.reflection);
        assert!(rebuilt.iter().zip(&res.coefficients).all(|(p, q)| (p - q).abs() < 1e-12));
        let k = lpc_to_reflection(&res.coefficients).unwrap();
        assert!(k.iter().zip(&res.reflection).all(|(p, q)| (p - q).abs() < 1e-9));
        assert!(lpc_to_reflection(&[1.0, -1.5, 0.6, 2.0]).is_err());
        assert!(lpc_to_reflection(&[2.0, 1.0]).is_err());
    }

    #[test]
    fn prediction_error_decreases_with_order() {
        let mut seed = 11;
        let x = ar_process(&[1.0, -1.2, 0.8, -0.3, 0.1], 20_000, &mut seed);
        let errs: Vec<f64> = (1..=6).map(|p| linear_predictive_coding(&x, p).unwrap().error_power).collect();
        assert!(errs.windows(2).all(|w| w[1] <= w[0] + 1e-9), "{errs:?}");
    }

    #[test]
    fn invalid_arguments() {
        assert!(linear_predictive_coding(&[1.0, 2.0, 3.0], 0).is_err());
        assert!(linear_predictive_coding(&[1.0, 2.0, 3.0], 3).is_err());
        assert!(linear_predictive_coding(&[0.0; 20], 2).is_err());
        assert!(levinson_durbin(&[1.0, 0.5], 2).is_err());
        assert!(levinson_durbin(&[1.0, 1.5, 0.0], 2).is_err());
    }
}
