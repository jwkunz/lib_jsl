use std::f64::consts::PI;

use super::{extend, len_guard, truncate};

/// The cosine window is defined by a specific mathematical formula that involves the sine function. The coefficients are computed based on the position within the window and the total length, with special handling for symmetric and periodic windows to ensure proper shape and spectral properties.
pub fn cosine(m: usize, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let mut w = vec![0.0; m2];
    for (i, wn) in w.iter_mut().enumerate() {
        *wn = (PI / m2 as f64 * (i as f64 + 0.5)).sin();
    }
    truncate(w, needs_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_cosine() {
        assert_window(include_str!("../test_data/cosine.csv"), cosine(7, true));
    }
}
