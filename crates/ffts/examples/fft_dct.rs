//! Discrete cosine and sine transforms: compression by truncating DCT coefficients.
//!
//! Run with `cargo run -p lib_jsl_ffts --example fft_dct`.

use lib_jsl_ffts::{TrigNormalization, TrigTransformType, dct, dst, idct};

fn main() {
    // A smooth signal on [0, 1]: its DCT-II coefficients decay quickly.
    let n = 64;
    let x: Vec<f64> = (0..n).map(|k| { let t = (k as f64 + 0.5) / n as f64; (2.0 * t).exp() * (3.0 * t).cos() }).collect();
    let c = dct(&x, TrigTransformType::II, TrigNormalization::Ortho).unwrap();
    // Keep only the largest coefficients (orthonormal: the energy of the rest is the squared error).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| c[b].abs().total_cmp(&c[a].abs()));
    let scale = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    for keep in [4, 8, 16] {
        let mut kept = vec![0.0; n];
        for &k in &order[..keep] {
            kept[k] = c[k];
        }
        let approx = idct(&kept, TrigTransformType::II, TrigNormalization::Ortho).unwrap();
        let err = approx.iter().zip(&x).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        println!("{keep:>2} of {n} DCT coefficients: largest error {:.1e} of the signal's size", err / scale);
    }

    // The DST-I diagonalises the second-difference matrix with zero boundary values.
    let s = dst(&[0.0, 1.0, 0.0, 0.0, 0.0], TrigTransformType::I, TrigNormalization::Backward).unwrap();
    println!("DST-I of e₁: {s:.4?}");
}
