use super::general_cosine::general_cosine;

/// The Nuttall window is defined by a specific set of coefficients that provide a smooth tapering of the coefficients based on the position within the window. The coefficients are computed using the general cosine function with specific values to achieve the desired shape and spectral properties.
pub fn nuttall(m: usize, sym: bool) -> Vec<f64> {
    general_cosine(m, &[0.363_581_9, 0.489_177_5, 0.136_599_5, 0.010_641_1], sym)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_nuttall() {
        assert_window(include_str!("../test_data/nuttall.csv"), nuttall(7, true));
    }
}
