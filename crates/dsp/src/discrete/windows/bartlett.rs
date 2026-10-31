use super::{extend, len_guard, truncate};

/// The Bartlett window, also known as the triangular window, is defined by a linear tapering of the coefficients from the center to the edges. The coefficients are computed based on the position within the window and the total length, with special handling for even and odd lengths to ensure proper symmetry.
pub fn bartlett(m: usize, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let mut w = vec![0.0; m2];
    let alpha = (m2 as f64 - 1.0) / 2.0;
    for (n, wn) in w.iter_mut().enumerate() {
        *wn = 1.0 - ((n as f64 - alpha) / alpha).abs();
    }
    truncate(w, needs_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_bartlett() {
        assert_window(include_str!("../test_data/bartlett.csv"), bartlett(7, true));
    }
}
