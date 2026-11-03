use super::{extend, len_guard, truncate};

/// The modified Bessel function of the first kind of order zero, to full double precision: the
/// power series `Σ (x²/4)^k / (k!)²` for `|x| < 30`, and above it the asymptotic series
/// `eˣ / √(2πx) · Σ ((2k − 1)!!)² / (k! (8x)^k)`, whose terms shrink until `k ≈ 2x`.
pub(crate) fn i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 30.0 {
        let q = 0.25 * ax * ax;
        let (mut term, mut sum, mut k) = (1.0f64, 1.0f64, 0.0f64);
        while term > 1e-17 * sum {
            k += 1.0;
            term *= q / (k * k);
            sum += term;
        }
        sum
    } else {
        let (mut term, mut sum, mut k) = (1.0f64, 1.0f64, 0.0f64);
        while term.abs() > 1e-17 * sum && k < 60.0 {
            k += 1.0;
            term *= (2.0 * k - 1.0).powi(2) / (k * 8.0 * ax);
            sum += term;
        }
        ax.exp() / (2.0 * std::f64::consts::PI * ax).sqrt() * sum
    }
}

/// The Kaiser window is defined by a specific mathematical formula that involves the modified Bessel function of the first kind. The coefficients are computed based on the position within the window, the total length, and the beta parameter, which controls the shape of the window and its spectral properties.
pub fn kaiser(m: usize, beta: f64, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let n: Vec<f64> = (0..m2).map(|i| i as f64).collect();
    let alpha = (m2 as f64 - 1.0) / 2.0;
    let i0b = i0(beta);
    let mut w = vec![0.0; m2];
    for (i, wn) in w.iter_mut().enumerate() {
        let t = 1.0 - ((n[i] - alpha) / alpha).powi(2);
        *wn = i0(beta * t.max(0.0).sqrt()) / i0b;
    }
    truncate(w, needs_trunc)
}

/// Kaiser's empirical `β` giving a stopband attenuation of `a` dB in a window-method FIR filter
/// (as SciPy's `kaiser_beta`). Larger `β` lowers the side lobes and widens the main lobe.
pub fn kaiser_beta(a: f64) -> f64 {
    if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a > 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    }
}

/// Kaiser's estimate of the stopband attenuation in dB reached by a Kaiser-window FIR filter with
/// `numtaps` taps and a transition `width` given as a fraction of the Nyquist frequency (as
/// SciPy's `kaiser_atten`).
pub fn kaiser_atten(numtaps: usize, width: f64) -> f64 {
    2.285 * (numtaps as f64 - 1.0) * std::f64::consts::PI * width + 7.95
}

/// The number of taps Kaiser's formula predicts for an attenuation of `at` dB with a transition
/// `width` given as a fraction of the Nyquist frequency (the inverse of [`kaiser_atten`]; the tap
/// count of SciPy's `kaiserord`).
pub fn kaiser_estimate_numtaps(at: f64, width: f64) -> usize {
    ((at - 7.95) / (2.285 * std::f64::consts::PI * width)).ceil() as usize + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn bessel_i0_is_accurate_across_the_switch_to_the_asymptotic_series() {
        // Values of I0 from mpmath to 17 digits.
        for (x, want) in [(0.5, 1.0634833707413236), (6.0, 67.23440697647797), (29.9, 708478330489.0145), (30.1, 862432920031.7792), (60.0, 5.894077055609801e24)] {
            assert!((i0(x) / want - 1.0).abs() < 4e-15, "{x}: {} against {want}", i0(x));
        }
    }

    #[test]
    fn test_kaiser() {
        assert_window(include_str!("../test_data/kaiser.csv"), kaiser(7, 14.0, true));
    }
}
