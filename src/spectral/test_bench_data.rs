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
