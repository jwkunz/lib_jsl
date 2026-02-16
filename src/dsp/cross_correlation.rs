use crate::{
    dsp::convolve::{convolve, ConvolveMethod, ConvolveMode},
    prelude::{ErrorsJSL, IsAnalytic},
};

#[derive(Clone, Copy, Debug)]
pub enum CorrelationMode {
    Full,
    Same,
    Valid,
}

#[inline]
fn map_mode(mode: CorrelationMode) -> ConvolveMode {
    match mode {
        CorrelationMode::Full => ConvolveMode::Full,
        CorrelationMode::Same => ConvolveMode::Same,
        CorrelationMode::Valid => ConvolveMode::Valid,
    }
}

/// 1D cross-correlation, similar to scipy.signal.correlate.
///
/// The implementation follows the standard relation:
/// correlate(in1, in2) = convolve(in1, reverse(conj(in2))).
pub fn cross_correlate<T: IsAnalytic>(
    in1: &[T],
    in2: &[T],
    mode: CorrelationMode,
    method: ConvolveMethod,
) -> Result<Vec<T>, ErrorsJSL> {
    if in1.is_empty() || in2.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("Input arrays must be non-empty"));
    }

    let mut rhs = Vec::with_capacity(in2.len());
    for x in in2.iter().rev() {
        rhs.push(x.f_conj());
    }
    convolve(in1, &rhs, map_mode(mode), method)
}

/// Lag indices for 1D cross-correlation output, similar to
/// scipy.signal.correlation_lags.
pub fn cross_correlation_lags(in1_len: usize, in2_len: usize, mode: CorrelationMode) -> Vec<isize> {
    if in1_len == 0 || in2_len == 0 {
        return vec![];
    }

    let full: Vec<isize> = (-(in2_len as isize - 1)..=(in1_len as isize - 1)).collect();

    match mode {
        CorrelationMode::Full => full,
        CorrelationMode::Same => {
            let start = (in2_len - 1) / 2;
            full[start..start + in1_len].to_vec()
        }
        CorrelationMode::Valid => {
            if in1_len >= in2_len {
                (0..=(in1_len - in2_len)).map(|x| x as isize).collect()
            } else {
                (-(in2_len as isize - in1_len as isize)..=0).collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_csv_f64(csv: &str) -> Vec<f64> {
        csv.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<f64>().unwrap())
            .collect()
    }

    fn parse_csv_i64(csv: &str) -> Vec<isize> {
        csv.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<isize>().unwrap())
            .collect()
    }

    #[test]
    fn test_correlate_doc_example_full_same_valid_golden() {
        let in1 = [1.0, 2.0, 3.0];
        let in2 = [0.0, 1.0, 0.5];

        let full = cross_correlate(&in1, &in2, CorrelationMode::Full, ConvolveMethod::Direct).unwrap();
        let same = cross_correlate(&in1, &in2, CorrelationMode::Same, ConvolveMethod::Direct).unwrap();
        let valid = cross_correlate(&in1, &in2, CorrelationMode::Valid, ConvolveMethod::Direct).unwrap();

        let g_full = parse_csv_f64(include_str!("test_data/correlate_doc_full.csv"));
        let g_same = parse_csv_f64(include_str!("test_data/correlate_doc_same.csv"));
        let g_valid = parse_csv_f64(include_str!("test_data/correlate_doc_valid.csv"));

        assert_eq!(full, g_full);
        assert_eq!(same, g_same);
        assert_eq!(valid, g_valid);
    }

    #[test]
    fn test_correlation_lags_doc_example() {
        let full = cross_correlation_lags(3, 3, CorrelationMode::Full);
        let same = cross_correlation_lags(3, 3, CorrelationMode::Same);
        let valid = cross_correlation_lags(3, 3, CorrelationMode::Valid);

        let g_full = parse_csv_i64(include_str!("test_data/correlation_lags_doc_full.csv"));
        let g_same = parse_csv_i64(include_str!("test_data/correlation_lags_doc_same.csv"));
        let g_valid = parse_csv_i64(include_str!("test_data/correlation_lags_doc_valid.csv"));

        assert_eq!(full, g_full);
        assert_eq!(same, g_same);
        assert_eq!(valid, g_valid);
    }

    #[test]
    fn test_correlate_fft_matches_direct_real() {
        let in1 = [0.2, -1.0, 0.5, 2.0, -0.7, 0.3, 1.1];
        let in2 = [1.0, -0.25, 0.75, -0.5];
        let yd = cross_correlate(&in1, &in2, CorrelationMode::Full, ConvolveMethod::Direct).unwrap();
        let yf = cross_correlate(&in1, &in2, CorrelationMode::Full, ConvolveMethod::Fft).unwrap();
        assert_eq!(yd.len(), yf.len());
        for (a, b) in yd.iter().zip(yf.iter()) {
            assert!((a - b).abs() < 1e-9, "direct={a}, fft={b}");
        }
    }

    #[test]
    fn test_correlate_fft_matches_direct_complex() {
        use num::Complex;
        let in1 = [
            Complex::new(1.0, 1.0),
            Complex::new(2.0, -1.0),
            Complex::new(0.5, 0.25),
        ];
        let in2 = [Complex::new(0.0, 1.0), Complex::new(1.0, 0.0)];
        let yd = cross_correlate(&in1, &in2, CorrelationMode::Full, ConvolveMethod::Direct).unwrap();
        let yf = cross_correlate(&in1, &in2, CorrelationMode::Full, ConvolveMethod::Fft).unwrap();
        assert_eq!(yd.len(), yf.len());
        for (a, b) in yd.iter().zip(yf.iter()) {
            assert!((a.re - b.re).abs() < 1e-9 && (a.im - b.im).abs() < 1e-9);
        }
    }
}
