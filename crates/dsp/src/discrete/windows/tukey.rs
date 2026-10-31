use std::f64::consts::PI;

use super::{boxcar::boxcar, extend, hann::hann, len_guard, truncate};

/// The Tukey window, also known as the tapered cosine window, is defined by a specific mathematical formula that combines a cosine taper with a flat top. The coefficients are computed based on the position within the window, the total length, and the alpha parameter, which controls the shape of the window and its spectral properties.
pub fn tukey(m: usize, alpha: f64, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    if alpha <= 0.0 {
        return boxcar(m, sym);
    }
    if alpha >= 1.0 {
        return hann(m, sym);
    }
    let (m2, needs_trunc) = extend(m, sym);
    let mut w = vec![0.0; m2];
    let denom = m2 as f64 - 1.0;
    for (n, wn) in w.iter_mut().enumerate() {
        let x = n as f64 / denom;
        *wn = if x < alpha / 2.0 {
            0.5 * (1.0 + (2.0 * PI / alpha * (x - alpha / 2.0)).cos())
        } else if x <= 1.0 - alpha / 2.0 {
            1.0
        } else {
            0.5 * (1.0 + (2.0 * PI / alpha * (x - 1.0 + alpha / 2.0)).cos())
        };
    }
    truncate(w, needs_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_tukey() {
        assert_window(include_str!("../test_data/tukey.csv"), tukey(7, 0.5, true));
    }
}
