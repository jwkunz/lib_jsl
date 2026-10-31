use super::general_cosine::general_cosine;

/// The Hamming window is defined by a specific set of coefficients that provide a smooth tapering of the coefficients based on the position within the window. The coefficients are computed using the general cosine function with specific values to achieve the desired shape and spectral properties.
pub fn general_hamming(m: usize, alpha: f64, sym: bool) -> Vec<f64> {
    general_cosine(m, &[alpha, 1.0 - alpha], sym)
}

/// The Hanning window, also known as the Hann window, is defined by a specific set of coefficients that provide a smooth tapering of the coefficients based on the position within the window. The coefficients are computed using the general cosine function with specific values to achieve the desired shape and spectral properties.
pub fn hamming(m: usize, sym: bool) -> Vec<f64> {
    general_hamming(m, 0.54, sym)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_hamming() {
        assert_window(include_str!("../test_data/hamming.csv"), hamming(7, true));
    }

    #[test]
    fn test_general_hamming() {
        assert_window(include_str!("../test_data/general_hamming.csv"), general_hamming(7, 0.54, true));
    }
}
