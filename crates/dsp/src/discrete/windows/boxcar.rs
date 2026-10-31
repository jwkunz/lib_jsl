/// The boxcar window, also known as the rectangular window, is defined by a constant value of 1 across the entire window. It does not taper the edges and is equivalent to applying no window at all, which can lead to significant spectral leakage in Fourier analysis.
pub fn boxcar(m: usize, _sym: bool) -> Vec<f64> {
    vec![1.0; m]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_boxcar() {
        assert_window(include_str!("../test_data/boxcar.csv"), boxcar(7, true));
    }
}
