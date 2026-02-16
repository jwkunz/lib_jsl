/// This module implements 1D convolution, similar to scipy.signal.convolve.
/// It supports three modes: 'full', 'same', and 'valid', and can use direct
/// or FFT-based convolution for analytic scalar types.

use crate::{
    ffts::{
        best_fft::BestFft,
        fft_enginer_trait::{FfftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
    },
    prelude::{ErrorsJSL, IsAnalytic},
};
use num::Complex;

#[derive(Clone, Copy, Debug)]
pub enum ConvolveMode {
    /// 'full' returns the convolution at each point of overlap, with an output size of in1.len() + in2.len() - 1.
    Full,
    /// 'same' returns the convolution at points where the signals overlap completely, with an output size of in1.len() (the same as the first input).
    Same,
    /// 'valid' returns the convolution at points where the signals overlap completely without zero-padding, with an output size of max(in1.len(), in2.len()) - min(in1.len(), in2.len()) + 1.
    Valid,
}

#[derive(Clone, Copy, Debug)]
pub enum ConvolveMethod {
    /// 'auto' chooses the method based on the input sizes.
    Auto,
    /// 'direct' computes the convolution using a straightforward nested loop approach, which is efficient for small input sizes.
    Direct,
    /// 'fft' computes convolution using the Fast Fourier Transform.
    Fft,
}

fn direct_full<T: IsAnalytic>(in1: &[T], in2: &[T]) -> Vec<T> {
    let mut out = vec![T::zero(); in1.len() + in2.len() - 1];
    for (i, &x) in in1.iter().enumerate() {
        for (j, &h) in in2.iter().enumerate() {
            out[i + j] += x * h;
        }
    }
    out
}

fn fft_full<T: IsAnalytic>(in1: &[T], in2: &[T]) -> Result<Vec<T>, ErrorsJSL> {
    let n = in1.len() + in2.len() - 1;
    let nfft = n.next_power_of_two();

    let mut a = vec![Complex::new(0.0, 0.0); nfft];
    let mut b = vec![Complex::new(0.0, 0.0); nfft];
    for (i, &v) in in1.iter().enumerate() {
        a[i] = v.to_complex();
    }
    for (i, &v) in in2.iter().enumerate() {
        b[i] = v.to_complex();
    }

    let mut fft = BestFft::new();
    fft.plan(
        nfft,
        FftScaleFactor::None,
        FftDirection::Forward,
        FftOrdering::Standard,
    )?;
    let a_fft = fft.execute(&a)?;
    let b_fft = fft.execute(&b)?;

    let mut c_fft = vec![Complex::new(0.0, 0.0); nfft];
    for i in 0..nfft {
        c_fft[i] = a_fft[i] * b_fft[i];
    }

    let mut ifft = BestFft::new();
    ifft.plan(
        nfft,
        FftScaleFactor::N,
        FftDirection::Inverse,
        FftOrdering::Standard,
    )?;
    let c = ifft.execute(&c_fft)?;
    Ok(c.into_iter().take(n).map(T::from_complex).collect())
}

fn apply_mode<T: IsAnalytic>(full: Vec<T>, in1_len: usize, in2_len: usize, mode: ConvolveMode) -> Vec<T> {
    match mode {
        ConvolveMode::Full => full,
        ConvolveMode::Same => {
            let start = (in2_len - 1) / 2;
            full[start..start + in1_len].to_vec()
        }
        ConvolveMode::Valid => {
            if in1_len >= in2_len {
                let start = in2_len - 1;
                let len = in1_len - in2_len + 1;
                full[start..start + len].to_vec()
            } else {
                let start = in1_len - 1;
                let len = in2_len - in1_len + 1;
                full[start..start + len].to_vec()
            }
        }
    }
}

/// 1D convolution, similar to scipy.signal.convolve.
pub fn convolve<T: IsAnalytic>(
    in1: &[T],
    in2: &[T],
    mode: ConvolveMode,
    method: ConvolveMethod,
) -> Result<Vec<T>, ErrorsJSL> {
    if in1.is_empty() || in2.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("Input arrays must be non-empty"));
    }

    let full = match method {
        ConvolveMethod::Direct => direct_full(in1, in2),
        ConvolveMethod::Fft => fft_full(in1, in2)?,
        ConvolveMethod::Auto => {
            let work = in1.len().saturating_mul(in2.len());
            if work <= 4096 {
                direct_full(in1, in2)
            } else {
                fft_full(in1, in2)?
            }
        }
    };
    Ok(apply_mode(full, in1.len(), in2.len(), mode))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_csv(csv: &str) -> Vec<f64> {
        csv.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<f64>().unwrap())
            .collect()
    }

    #[test]
    fn test_convolve_doc_example_full_golden() {
        let in1 = [1.0, 2.0, 3.0];
        let in2 = [0.0, 1.0, 0.5];
        let y = convolve(&in1, &in2, ConvolveMode::Full, ConvolveMethod::Direct).unwrap();
        let golden = parse_csv(include_str!("test_data/convolve_doc_full.csv"));
        assert_eq!(y.len(), golden.len());
        for (a, g) in y.iter().zip(golden.iter()) {
            assert!((a - g).abs() < 1e-12, "actual={a}, golden={g}");
        }
    }

    #[test]
    fn test_convolve_doc_example_same_and_valid_golden() {
        let in1 = [1.0, 2.0, 3.0];
        let in2 = [0.0, 1.0, 0.5];
        let same = convolve(&in1, &in2, ConvolveMode::Same, ConvolveMethod::Direct).unwrap();
        let valid = convolve(&in1, &in2, ConvolveMode::Valid, ConvolveMethod::Direct).unwrap();
        let golden_same = parse_csv(include_str!("test_data/convolve_doc_same.csv"));
        let golden_valid = parse_csv(include_str!("test_data/convolve_doc_valid.csv"));
        assert_eq!(same, golden_same);
        assert_eq!(valid, golden_valid);
    }

    #[test]
    fn test_convolve_fft_matches_direct() {
        let in1 = [0.2, -1.0, 0.5, 2.0, -0.7, 0.3, 1.1];
        let in2 = [1.0, -0.25, 0.75, -0.5];
        let yd = convolve(&in1, &in2, ConvolveMode::Full, ConvolveMethod::Direct).unwrap();
        let yf = convolve(&in1, &in2, ConvolveMode::Full, ConvolveMethod::Fft).unwrap();
        assert_eq!(yd.len(), yf.len());
        for (a, b) in yd.iter().zip(yf.iter()) {
            assert!((a - b).abs() < 1e-9, "direct={a}, fft={b}");
        }
    }

    #[test]
    fn test_convolve_generic_complex_fft_matches_direct() {
        use num::Complex;
        let in1 = [
            Complex::new(1.0, 1.0),
            Complex::new(2.0, -1.0),
            Complex::new(0.5, 0.25),
        ];
        let in2 = [Complex::new(0.0, 1.0), Complex::new(1.0, 0.0)];
        let yd = convolve(&in1, &in2, ConvolveMode::Full, ConvolveMethod::Direct).unwrap();
        let yf = convolve(&in1, &in2, ConvolveMode::Full, ConvolveMethod::Fft).unwrap();
        assert_eq!(yd.len(), yf.len());
        for (a, b) in yd.iter().zip(yf.iter()) {
            assert!((a.re - b.re).abs() < 1e-9 && (a.im - b.im).abs() < 1e-9);
        }
    }
}
