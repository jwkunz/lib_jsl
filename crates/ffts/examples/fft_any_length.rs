//! FFTs of any length (Bluestein) and the fractional Fourier transform.
//!
//! Run with `cargo run -p lib_jsl_ffts --example fft_any_length`.

use lib_jsl_ffts::{Fft, FftDirection, FftEngine1D, FftOrdering, FftScaleFactor, frft};
use num_complex::Complex;

fn main() {
    // 1000 samples (not a power of two): the default Fft switches to Bluestein's algorithm.
    let n = 1000;
    let x: Vec<Complex<f64>> = (0..n).map(|k| Complex::new((2.0 * std::f64::consts::PI * 37.0 * k as f64 / n as f64).cos(), 0.0)).collect();
    let mut fft = Fft::new();
    fft.plan(n, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).unwrap();
    let y = fft.execute(&x).unwrap();
    println!("|X_37| = {:.6} (n/2 = {}), largest other bin {:.1e}", y[37].norm(), n / 2, (0..n).filter(|&k| k != 37 && k != n - 37).map(|k| y[k].norm()).fold(0.0, f64::max));

    // A prime length works the same way.
    let p = 997;
    let ones = vec![Complex::new(1.0, 0.0); p];
    let mut fp = Fft::new();
    fp.plan(p, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).unwrap();
    println!("DFT of {p} ones: X_0 = {:.6}", fp.execute(&ones).unwrap()[0].re);

    // The fractional Fourier transform of order 1 is the unitary DFT; order 0.5 is "half way".
    let g: Vec<f64> = (0..64).map(|k| { let c = if k < 32 { k as f64 } else { k as f64 - 64.0 }; (-std::f64::consts::PI * (c / 8.0).powi(2)).exp() }).collect();
    let half = frft(&g, 0.5).unwrap();
    // This Gaussian is (approximately) an eigenfunction of every order: its magnitude barely changes.
    println!("Gaussian: |F^0.5 g| at 0 = {:.4} (g(0) = {:.4})", half[0].norm(), g[0]);
}
