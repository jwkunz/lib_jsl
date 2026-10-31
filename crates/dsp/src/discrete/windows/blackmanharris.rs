use super::general_cosine::general_cosine;

/// The Blackman-Harris window is defined by a specific set of coefficients that provide a smooth tapering of the coefficients based on the position within the window. The coefficients are computed using the general cosine function with specific values to achieve the desired shape and spectral properties.
pub fn blackmanharris(m: usize, sym: bool) -> Vec<f64> {
    general_cosine(m, &[0.358_75, 0.488_29, 0.141_28, 0.011_68], sym)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_blackmanharris() {
        assert_window(include_str!("../test_data/blackmanharris.csv"), blackmanharris(7, true));
    }
}
