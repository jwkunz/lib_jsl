use super::{extend, len_guard, truncate};

/// The exponential window is defined by a specific mathematical formula that involves an exponential function based on the distance from the center of the window. The coefficients are computed based on the position within the window, the total length, and the parameters `center` and `tau`, which control the shape of the window and its spectral properties.
pub fn exponential(m: usize, center: Option<f64>, tau: f64, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let c = center.unwrap_or((m2 as f64 - 1.0) / 2.0);
    let mut w = vec![0.0; m2];
    for (i, wn) in w.iter_mut().enumerate() {
        *wn = (-(i as f64 - c).abs() / tau).exp();
    }
    truncate(w, needs_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_exponential() {
        assert_window(include_str!("../test_data/exponential.csv"), exponential(7, None, 1.0, true));
    }
}
