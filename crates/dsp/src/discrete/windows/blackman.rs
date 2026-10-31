use super::general_cosine::general_cosine;

/// The Blackman window is a three-term general cosine window with coefficients `[0.42, 0.5, 0.08]`, giving lower side lobes than the Hann and Hamming windows at the cost of a wider main lobe.
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
