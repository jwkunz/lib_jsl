use super::kaiser::kaiser;

/// The Kaiser-Bessel derived window of even length `m`: the square root of the normalised running
/// sum of a Kaiser window of length `m/2 + 1`, mirrored. It satisfies the Princen-Bradley
/// condition `w[n]² + w[n + m/2]² = 1`, which makes it suitable for the MDCT. Only the symmetric
/// form exists (`sym = false` returns an empty vector). SciPy rejects odd lengths; here an odd
/// `m` gives the window of length `m + 1` with its last sample dropped, which is not symmetric.
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
        assert_window(include_str!("../test_data/kaiser_bessel_derived.csv"), kaiser_bessel_derived(8, 14.0, true));
    }
}
