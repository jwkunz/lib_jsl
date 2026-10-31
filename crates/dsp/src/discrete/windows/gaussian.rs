use super::{extend, len_guard, truncate};

/// The Gaussian window is defined by a specific mathematical formula that involves an exponential function based on the distance from the center of the window. The coefficients are computed based on the position within the window, the total length, and the standard deviation parameter, which controls the shape of the window and its spectral properties.
pub fn gaussian(m: usize, std: f64, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let mut w = vec![0.0; m2];
    let c = (m2 as f64 - 1.0) / 2.0;
    for (i, wn) in w.iter_mut().enumerate() {
        let n = i as f64 - c;
        *wn = (-0.5 * (n / std).powi(2)).exp();
    }
    truncate(w, needs_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_gaussian() {
        assert_window(include_str!("../test_data/gaussian.csv"), gaussian(7, 1.0, true));
    }
}
