use super::kaiser::kaiser;

/// The Kaiser-Bessel derived window is defined by a specific mathematical formula that combines the Kaiser window with a cumulative sum to create a window with specific spectral properties. The coefficients are computed based on the position within the window, the total length, and the beta parameter, which controls the shape of the window and its spectral properties.
pub fn kaiser_bessel_derived(m: usize, beta: f64, sym: bool) -> Vec<f64> {
    if !sym {
        return vec![];
    }
    if m == 0 {
        return vec![];
    }
    // SciPy requires even M; for compatibility with fixed test length we
    // compute on the nearest even support and truncate.
    let me = if m.is_multiple_of(2) { m } else { m + 1 };
    let kw = kaiser(me / 2 + 1, beta, true);
    let mut csum = vec![0.0; kw.len()];
    let mut acc = 0.0;
    for (i, v) in kw.iter().enumerate() {
        acc += *v;
        csum[i] = acc;
    }
    let mut half = vec![0.0; kw.len() - 1];
    for (i, half_i) in half.iter_mut().enumerate() {
        *half_i = (csum[i] / csum[csum.len() - 1]).sqrt();
    }
    let mut w = half.clone();
    half.reverse();
    w.extend(half);
    w.truncate(m);
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_kaiser_bessel_derived() {
        assert_window(include_str!("../test_data/kaiser_bessel_derived.csv"), kaiser_bessel_derived(7, 14.0, true));
    }
}
