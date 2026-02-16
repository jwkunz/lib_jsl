use num::Complex;

fn parse_complex_csv(csv: &str) -> Vec<Complex<f64>> {
    csv.lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let (re, im) = line
                .split_once(',')
                .expect("each row must be formatted as `re,im`");
            Complex::new(
                re.parse::<f64>().expect("real part must parse as f64"),
                im.parse::<f64>().expect("imag part must parse as f64"),
            )
        })
        .collect()
}

pub(crate) fn fft_gaussian_1024_input() -> Vec<Complex<f64>> {
    parse_complex_csv(include_str!("test_data/fft_gaussian_1024_input.csv"))
}

pub(crate) fn fft_gaussian_1024_golden() -> Vec<Complex<f64>> {
    parse_complex_csv(include_str!("test_data/fft_gaussian_1024_golden.csv"))
}
