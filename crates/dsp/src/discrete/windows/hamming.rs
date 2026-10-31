use super::general_cosine::general_cosine;

/// The general Hamming window is a two-term cosine window with coefficients `[alpha, 1 - alpha]`.
pub fn general_hamming(m: usize, alpha: f64, sym: bool) -> Vec<f64> {
    general_cosine(m, &[alpha, 1.0 - alpha], sym)
}

/// The Hamming window is the general Hamming window with `alpha = 0.54`, which places a zero near the first side lobe of the rectangular window's spectrum.
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
