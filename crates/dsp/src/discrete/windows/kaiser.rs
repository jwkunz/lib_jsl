use super::{extend, len_guard, truncate};

/// The modified Bessel function of the first kind, order 0, is used in the computation of the Kaiser window coefficients. It is defined as an infinite series and can be computed using a numerical approximation for practical purposes.
fn i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = x / 3.75;
        let t2 = t * t;
        1.0 + t2
            * (3.515_622_9
                + t2
                    * (3.089_942_4
                        + t2
                            * (1.206_749_2
                                + t2 * (0.265_973_2 + t2 * (0.036_076_8 + t2 * 0.004_581_3)))))
    } else {
        let t = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.398_942_28
                + t * (0.013_285_92
                    + t * (0.002_253_19
                        + t * (-0.001_575_65
                            + t * (0.009_162_81
                                + t * (-0.020_577_06
                                    + t * (0.026_355_37
                                        + t * (-0.016_476_33 + t * 0.003_923_77))))))))
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

/// This function is used to compute the beta parameter for the Kaiser window based on the desired attenuation in decibels. The beta parameter controls the shape of the window and its spectral properties, with higher values leading to a narrower main lobe and lower side lobes in the frequency response.
pub fn kaiser_beta(a: f64) -> f64 {
    if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a > 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    }
}

/// This function is used to compute the attenuation in decibels for the Kaiser window based on the desired beta parameter. The attenuation in decibels is a measure of how much the side lobes of the window are suppressed compared to the main lobe, with higher values indicating better suppression of side lobes.
pub fn kaiser_atten(numtaps: usize, width: f64) -> f64 {
    2.285 * (numtaps as f64 - 1.0) * std::f64::consts::PI * width + 7.95
}

/// This is a method to estimate the number of taps needed for a Kaiser window based on the desired attenuation in decibels and the width of the main lobe. The number of taps determines the length of the window and its spectral properties, with more taps leading to a narrower main lobe and better suppression of side lobes.
pub fn kaiser_estimate_numtaps(at: f64, width: f64) -> usize {
    ((at - 7.95) / (2.285 * std::f64::consts::PI * width)).ceil() as usize + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_kaiser() {
        assert_window(include_str!("../test_data/kaiser.csv"), kaiser(7, 14.0, true));
    }
}
