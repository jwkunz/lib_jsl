use super::general_cosine::general_cosine;

/// The Flattop window is defined by a specific set of coefficients that provide a flat top shape to the window, which is useful for certain applications where a flat frequency response is desired. The coefficients are computed using the general cosine function with specific values to achieve the desired shape and spectral properties.
pub fn flattop(m: usize, sym: bool) -> Vec<f64> {
    general_cosine(
        m,
        &[0.215_578_95, 0.416_631_58, 0.277_263_158, 0.083_578_947, 0.006_947_368],
        sym,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_flattop() {
        assert_window(include_str!("../test_data/flattop.csv"), flattop(7, true));
    }
}
