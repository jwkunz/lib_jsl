use super::general_cosine::general_cosine;

/// The Lanczos window is defined by the sinc function, which provides a smooth tapering of the coefficients based on the distance from the center of the window. The coefficients are computed based on the position within the window and the total length, with special handling for the edges to ensure proper symmetry and smoothness.
pub fn blackman(m: usize, sym: bool) -> Vec<f64> {
    general_cosine(m, &[0.42, 0.5, 0.08], sym)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_blackman() {
        assert_window(include_str!("../test_data/blackman.csv"), blackman(7, true));
    }
}
