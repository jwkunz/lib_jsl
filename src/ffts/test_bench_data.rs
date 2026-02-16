/// Test bench data for validating FFT implementations against known inputs and outputs.
/// This module provides functions to load precomputed input and golden output data for FFT tests, allowing us to verify the correctness of our FFT implementations against established benchmarks. 
/// The data is stored in binary format, with each complex number represented as 16 bytes (8 bytes for the real part and 8 bytes for the imaginary part) in little-endian format. The functions in this module parse the binary 
use num::Complex;

fn parse_complex_bin(bytes: &[u8]) -> Vec<Complex<f64>> {
    assert!(bytes.len() % 16 == 0, "binary complex data must be 16-byte aligned");

    bytes
        .chunks_exact(16)
        .map(|chunk| {
            let mut re = [0_u8; 8];
            let mut im = [0_u8; 8];
            re.copy_from_slice(&chunk[0..8]);
            im.copy_from_slice(&chunk[8..16]);
            Complex::new(f64::from_le_bytes(re), f64::from_le_bytes(im))
        })
        .collect()
}

pub(crate) fn fft_gaussian_32768_input() -> Vec<Complex<f64>> {
    parse_complex_bin(include_bytes!("test_data/fft_gaussian_32768_input.bin"))
}

pub(crate) fn fft_gaussian_32768_golden() -> Vec<Complex<f64>> {
    parse_complex_bin(include_bytes!("test_data/fft_gaussian_32768_golden.bin"))
}

pub(crate) fn fft_gaussian_63_input() -> Vec<Complex<f64>> {
    parse_complex_bin(include_bytes!("test_data/fft_gaussian_63_input.bin"))
}

pub(crate) fn fft_gaussian_63_golden() -> Vec<Complex<f64>> {
    parse_complex_bin(include_bytes!("test_data/fft_gaussian_63_golden.bin"))
}
