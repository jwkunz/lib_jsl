//! Digital modulation: mapping bits to constellations, soft demapping, OFDM and FM.

use lib_jsl_dsp::discrete::digital_communications::constellations::Constellation;
use lib_jsl_dsp::discrete::digital_communications::fm::{fm_demodulate, fm_modulate};
use lib_jsl_dsp::discrete::digital_communications::ofdm::{Ofdm, OfdmParams};
use num_complex::Complex;

fn main() {
    let qam = Constellation::qam(16).unwrap();
    let bits = [1, 0, 1, 1, 0, 0, 1, 0];
    let symbols = qam.map_bits(&bits).unwrap();
    println!("16-QAM symbols {}", symbols.iter().map(|c| format!("{c:.3}")).collect::<Vec<_>>().join(", "));
    let noisy: Vec<Complex<f64>> = symbols.iter().map(|s| s + Complex::new(0.05, -0.08)).collect();
    println!("hard decisions {:?}", qam.demap_hard(&noisy));
    let llr = qam.demap_soft(&noisy, 0.02).unwrap();
    println!("log-likelihood ratios (positive favours 0) {:.1?}", &llr[..4]);
    let params = OfdmParams::wifi_like();
    println!("OFDM: FFT {}, prefix {}, {} data subcarriers, {} samples per symbol", params.fft_size(), params.cp_len(), params.num_data(), params.symbol_len());
    let mut ofdm = Ofdm::new(params.clone()).unwrap();
    let data: Vec<Complex<f64>> = (0..params.num_data()).map(|k| qam.points()[k % 16]).collect();
    let tx = ofdm.modulate(&data).unwrap();
    let rx = ofdm.demodulate(&tx).unwrap();
    println!("OFDM round trip error {:.1e}", rx.data.iter().zip(&data).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max));
    let message: Vec<f64> = (0..480).map(|n| (2.0 * std::f64::consts::PI * 300.0 * n as f64 / 48000.0).sin()).collect();
    let fm = fm_modulate(&message, 3000.0, 48000.0).unwrap();
    let back = fm_demodulate(&fm, 3000.0, 48000.0).unwrap();
    println!("FM: |baseband| = {:.3} everywhere; message error {:.1e}", fm[100].norm(), back.iter().zip(&message).skip(5).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max));
}
