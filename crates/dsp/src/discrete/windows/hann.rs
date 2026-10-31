use super::hamming::general_hamming;

/// The Hanning window, also known as the Hann window, is defined by a specific set of coefficients that provide a smooth tapering of the coefficients based on the position within the window. The coefficients are computed using the general cosine function with specific values to achieve the desired shape and spectral properties.
pub fn hann(m: usize, sym: bool) -> Vec<f64> {
    general_hamming(m, 0.5, sym)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_hann() {
        assert_window(include_str!("../test_data/hann.csv"), hann(7, true));
    }
}
