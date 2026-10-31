//! Orthogonal frequency-division multiplexing (OFDM).
//!
//! OFDM sends data on many narrow, orthogonal subcarriers at once. The transmitter places one
//! constellation symbol on each active subcarrier of an `N`-point frequency grid, takes an inverse
//! FFT to get `N` time samples, and prepends a **cyclic prefix** (a copy of the last `cp_len`
//! samples). Because the prefix makes the channel's linear convolution look *circular* over the
//! symbol body, a multipath channel whose delay spread is shorter than the prefix turns into a
//! simple complex gain `H[k]` on each subcarrier: the receiver drops the prefix, takes an FFT, and
//! equalizes every subcarrier with one complex division.
//!
//! The channel gains come either from a known **training symbol** ([`Ofdm::estimate_channel`]),
//! which gives an exact per-subcarrier estimate, or from the **pilot** subcarriers carried in every
//! symbol, interpolated in frequency across the data subcarriers ([`Ofdm::demodulate`]). Pilot
//! interpolation needs the pilots dense enough that the channel is smooth between them: roughly
//! `pilot spacing < N / delay spread`.
//!
//! Scaling is chosen so the time-domain samples have unit average power when the subcarrier
//! symbols have unit power.

use crate::{
    ffts::{
        best_fft::BestFft,
        fft_engine_trait::{FfftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
    },
    prelude::ErrorsJSL,
};
use num::Complex;

/// The subcarrier layout of an OFDM system. Subcarriers are numbered by signed index `k` in
/// `-N/2 .. N/2` (index 0 is DC and is never used); negative indices are the upper half of the FFT.
#[derive(Clone, Debug, PartialEq)]
pub struct OfdmParams {
    fft_size: usize,
    cp_len: usize,
    data: Vec<isize>,
    pilots: Vec<isize>,
    pilot_values: Vec<Complex<f64>>,
}

impl OfdmParams {
    /// A layout with `fft_size` (a power of two, at least 8), a cyclic prefix of `cp_len` samples
    /// (`< fft_size`), the signed indices of the data and pilot subcarriers, and the known value
    /// sent on each pilot every symbol. Indices must be distinct, nonzero, and within `±fft_size/2`.
    pub fn new(fft_size: usize, cp_len: usize, data: Vec<isize>, pilots: Vec<isize>, pilot_values: Vec<Complex<f64>>) -> Result<Self, ErrorsJSL> {
        if fft_size < 8 || !fft_size.is_power_of_two() {
            return Err(ErrorsJSL::InvalidInputRange("OFDM FFT size must be a power of two, at least 8"));
        }
        if cp_len >= fft_size {
            return Err(ErrorsJSL::InvalidInputRange("The cyclic prefix must be shorter than the FFT size"));
        }
        if data.is_empty() {
            return Err(ErrorsJSL::InvalidInputRange("At least one data subcarrier is required"));
        }
        if pilots.len() != pilot_values.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((pilots.len(), pilot_values.len())));
        }
        let half = (fft_size / 2) as isize;
        let mut all: Vec<isize> = data.iter().chain(&pilots).copied().collect();
        if all.iter().any(|&k| k == 0 || k < -half || k >= half) {
            return Err(ErrorsJSL::InvalidInputRange("Subcarrier indices must be nonzero and within -N/2 .. N/2"));
        }
        all.sort_unstable();
        if all.windows(2).any(|w| w[0] == w[1]) {
            return Err(ErrorsJSL::InvalidInputRange("Subcarrier indices must be distinct across data and pilots"));
        }
        // Keep the pilots sorted by index (carrying their values) for interpolation.
        let mut pv: Vec<(isize, Complex<f64>)> = pilots.into_iter().zip(pilot_values).collect();
        pv.sort_by_key(|p| p.0);
        Ok(OfdmParams { fft_size, cp_len, data, pilots: pv.iter().map(|p| p.0).collect(), pilot_values: pv.iter().map(|p| p.1).collect() })
    }

    /// A 64-point layout in the style of IEEE 802.11a/g: 48 data subcarriers `±1..±26`, pilots at
    /// `±7` and `±21` (BPSK `+1, +1, +1, −1`), and a 16-sample prefix.
    pub fn wifi_like() -> Self {
        let pilots = vec![-21isize, -7, 7, 21];
        let data: Vec<isize> = (-26..=26).filter(|k| *k != 0 && !pilots.contains(k)).collect();
        let values = [1.0, 1.0, 1.0, -1.0].iter().map(|&v| Complex::new(v, 0.0)).collect();
        OfdmParams::new(64, 16, data, pilots, values).expect("valid layout")
    }

    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    pub fn cp_len(&self) -> usize {
        self.cp_len
    }

    /// Number of data symbols carried by each OFDM symbol.
    pub fn num_data(&self) -> usize {
        self.data.len()
    }

    /// Samples per OFDM symbol including the prefix.
    pub fn symbol_len(&self) -> usize {
        self.fft_size + self.cp_len
    }

    fn bin(&self, k: isize) -> usize {
        k.rem_euclid(self.fft_size as isize) as usize
    }

    fn active_count(&self) -> usize {
        self.data.len() + self.pilots.len()
    }
}

/// The result of demodulating a stream of OFDM symbols.
#[derive(Clone, Debug, PartialEq)]
pub struct OfdmDemodulated {
    /// Equalized data symbols, `num_data` per OFDM symbol, in subcarrier-list order.
    pub data: Vec<Complex<f64>>,
    /// The channel estimate used for each data subcarrier of each OFDM symbol (same layout).
    pub channel: Vec<Complex<f64>>,
}

/// An OFDM modulator/demodulator for a fixed layout.
pub struct Ofdm {
    params: OfdmParams,
    fwd: BestFft,
    inv: BestFft,
}

impl Ofdm {
    pub fn new(params: OfdmParams) -> Result<Self, ErrorsJSL> {
        let mut fwd = BestFft::new();
        fwd.plan(params.fft_size, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard)?;
        let mut inv = BestFft::new();
        inv.plan(params.fft_size, FftScaleFactor::N, FftDirection::Inverse, FftOrdering::Standard)?;
        Ok(Ofdm { params, fwd, inv })
    }

    pub fn params(&self) -> &OfdmParams {
        &self.params
    }

    /// Time samples (with prefix) for one OFDM symbol from its data symbols.
    fn modulate_symbol(&mut self, data: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
        let p = &self.params;
        let mut x = vec![Complex::new(0.0, 0.0); p.fft_size];
        for (&k, &v) in p.data.iter().zip(data) {
            x[p.bin(k)] = v;
        }
        for (&k, &v) in p.pilots.iter().zip(&p.pilot_values) {
            x[p.bin(k)] = v;
        }
        let scale = p.fft_size as f64 / (p.active_count() as f64).sqrt();
        let body: Vec<Complex<f64>> = self.inv.execute(&x)?.into_iter().map(|v| v * scale).collect();
        let mut out = Vec::with_capacity(p.symbol_len());
        out.extend_from_slice(&body[p.fft_size - p.cp_len..]);
        out.extend_from_slice(&body);
        Ok(out)
    }

    /// Modulates data symbols into a time-domain signal. The number of symbols must be a multiple
    /// of [`num_data`](OfdmParams::num_data).
    pub fn modulate(&mut self, symbols: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
        let nd = self.params.num_data();
        if symbols.is_empty() || symbols.len() % nd != 0 {
            return Err(ErrorsJSL::IncompatibleArraySizes((symbols.len(), nd)));
        }
        let mut out = Vec::with_capacity(symbols.len() / nd * self.params.symbol_len());
        for chunk in symbols.chunks(nd) {
            out.extend(self.modulate_symbol(chunk)?);
        }
        Ok(out)
    }

    /// Frequency-domain values on every subcarrier index of one received OFDM symbol (with prefix).
    fn to_frequency(&mut self, symbol: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
        let p = &self.params;
        let y = self.fwd.execute(&symbol[p.cp_len..])?;
        let scale = (p.active_count() as f64).sqrt() / p.fft_size as f64;
        Ok(y.into_iter().map(|v| v * scale).collect())
    }

    fn check_stream(&self, rx: &[Complex<f64>]) -> Result<usize, ErrorsJSL> {
        let len = self.params.symbol_len();
        if rx.is_empty() || rx.len() % len != 0 {
            return Err(ErrorsJSL::IncompatibleArraySizes((rx.len(), len)));
        }
        Ok(rx.len() / len)
    }

    /// Per-data-subcarrier channel estimate for one symbol, interpolated linearly in frequency
    /// from the pilots (holding the outermost pilot's value beyond them).
    fn pilot_channel(&self, freq: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let p = &self.params;
        let h_pilot: Vec<Complex<f64>> = p.pilots.iter().zip(&p.pilot_values).map(|(&k, &v)| freq[p.bin(k)] / v).collect();
        p.data
            .iter()
            .map(|&k| {
                let idx = p.pilots.partition_point(|&q| q < k);
                if idx == 0 {
                    h_pilot[0]
                } else if idx == p.pilots.len() {
                    h_pilot[idx - 1]
                } else {
                    let (k0, k1) = (p.pilots[idx - 1] as f64, p.pilots[idx] as f64);
                    let t = (k as f64 - k0) / (k1 - k0);
                    h_pilot[idx - 1] * (1.0 - t) + h_pilot[idx] * t
                }
            })
            .collect()
    }

    /// Demodulates a stream of OFDM symbols (each `symbol_len` samples, prefix included),
    /// equalizing with a pilot-derived channel estimate per symbol. Without pilots there is
    /// nothing to estimate from and the channel is taken to be 1.
    pub fn demodulate(&mut self, rx: &[Complex<f64>]) -> Result<OfdmDemodulated, ErrorsJSL> {
        let count = self.check_stream(rx)?;
        let len = self.params.symbol_len();
        let mut data = Vec::with_capacity(count * self.params.num_data());
        let mut channel = Vec::with_capacity(data.capacity());
        for s in 0..count {
            let freq = self.to_frequency(&rx[s * len..(s + 1) * len])?;
            let h = if self.params.pilots.is_empty() { vec![Complex::new(1.0, 0.0); self.params.num_data()] } else { self.pilot_channel(&freq) };
            for (&k, &hk) in self.params.data.iter().zip(&h) {
                data.push(freq[self.params.bin(k)] / hk);
            }
            channel.extend(h);
        }
        Ok(OfdmDemodulated { data, channel })
    }

    /// Estimates the channel on every data subcarrier from one received training OFDM symbol (with
    /// prefix) and the `num_data` known symbols it carried: `H[k] = Y[k] / X[k]`.
    pub fn estimate_channel(&mut self, training_rx: &[Complex<f64>], known: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
        if training_rx.len() != self.params.symbol_len() || known.len() != self.params.num_data() {
            return Err(ErrorsJSL::IncompatibleArraySizes((training_rx.len(), self.params.symbol_len())));
        }
        if known.iter().any(|v| v.norm() < 1e-12) {
            return Err(ErrorsJSL::InvalidInputRange("Training symbols must be nonzero"));
        }
        let freq = self.to_frequency(training_rx)?;
        Ok(self.params.data.iter().zip(known).map(|(&k, &x)| freq[self.params.bin(k)] / x).collect())
    }

    /// Demodulates using a fixed per-data-subcarrier channel estimate (e.g. from a training symbol).
    pub fn demodulate_with_channel(&mut self, rx: &[Complex<f64>], channel: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
        let count = self.check_stream(rx)?;
        let len = self.params.symbol_len();
        if channel.len() != self.params.num_data() {
            return Err(ErrorsJSL::IncompatibleArraySizes((channel.len(), self.params.num_data())));
        }
        let mut data = Vec::with_capacity(count * channel.len());
        for s in 0..count {
            let freq = self.to_frequency(&rx[s * len..(s + 1) * len])?;
            for (&k, &hk) in self.params.data.iter().zip(channel) {
                data.push(freq[self.params.bin(k)] / hk);
            }
        }
        Ok(data)
    }
}

/// Modulates `symbols` (a multiple of the layout's data-subcarrier count) into an OFDM signal.
pub fn ofdm_modulate(symbols: &[Complex<f64>], params: &OfdmParams) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
    Ofdm::new(params.clone())?.modulate(symbols)
}

/// Demodulates an OFDM signal, equalizing with the pilots, and returns the data symbols.
pub fn ofdm_demodulate(rx: &[Complex<f64>], params: &OfdmParams) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
    Ok(Ofdm::new(params.clone())?.demodulate(rx)?.data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::digital_communications::constellations::Constellation;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }

    fn gauss(seed: &mut u64) -> f64 {
        (0..12).map(|_| lcg(seed)).sum::<f64>()
    }

    fn data_symbols(c: &Constellation, n: usize, seed: &mut u64) -> Vec<Complex<f64>> {
        (0..n).map(|_| c.points()[((lcg(seed) + 0.5) * c.size() as f64) as usize % c.size()]).collect()
    }

    /// Linear convolution with a channel impulse response, truncated to the input length.
    fn apply_channel(x: &[Complex<f64>], h: &[Complex<f64>]) -> Vec<Complex<f64>> {
        (0..x.len()).map(|n| h.iter().enumerate().filter(|(k, _)| *k <= n).map(|(k, c)| c * x[n - k]).sum()).collect()
    }

    /// 64-point layout with a pilot on every 4th subcarrier: dense enough to interpolate a mild channel.
    fn dense_pilot_params() -> OfdmParams {
        let pilots: Vec<isize> = (-24..=24).step_by(4).filter(|k| *k != 0).collect();
        let data: Vec<isize> = (-26..=26).filter(|k| *k != 0 && !pilots.contains(k)).collect();
        let values = vec![Complex::new(1.0, 0.0); pilots.len()];
        OfdmParams::new(64, 16, data, pilots, values).unwrap()
    }

    #[test]
    fn layout_of_the_wifi_like_configuration() {
        let p = OfdmParams::wifi_like();
        assert_eq!((p.fft_size(), p.cp_len(), p.num_data(), p.symbol_len()), (64, 16, 48, 80));
    }

    #[test]
    fn noiseless_round_trip_and_unit_power() {
        let c = Constellation::qam(16).unwrap();
        let mut seed = 3;
        let params = OfdmParams::wifi_like();
        let syms = data_symbols(&c, 48 * 50, &mut seed);
        let tx = ofdm_modulate(&syms, &params).unwrap();
        assert_eq!(tx.len(), 50 * 80);
        let power = tx.iter().map(|v| v.norm_sqr()).sum::<f64>() / tx.len() as f64;
        assert!((power - 1.0).abs() < 0.1, "average sample power {power}");
        let back = ofdm_demodulate(&tx, &params).unwrap();
        assert_eq!(back.len(), syms.len());
        assert!(back.iter().zip(&syms).all(|(a, b)| (a - b).norm() < 1e-10));
    }

    #[test]
    fn the_cyclic_prefix_copies_the_symbol_tail() {
        let mut seed = 5;
        let params = OfdmParams::wifi_like();
        let c = Constellation::qpsk();
        let tx = ofdm_modulate(&data_symbols(&c, 48, &mut seed), &params).unwrap();
        for i in 0..16 {
            assert!((tx[i] - tx[64 + i]).norm() < 1e-12, "prefix sample {i}");
        }
    }

    #[test]
    fn a_single_subcarrier_is_a_pure_tone() {
        // One data symbol on one subcarrier (plus the pilots, which we zero by using none).
        let params = OfdmParams::new(32, 4, vec![3], vec![], vec![]).unwrap();
        let tx = ofdm_modulate(&[Complex::new(1.0, 0.0)], &params).unwrap();
        for n in 0..32 {
            let want = Complex::from_polar(1.0, std::f64::consts::TAU * 3.0 * n as f64 / 32.0);
            assert!((tx[4 + n] - want).norm() < 1e-12, "sample {n}");
        }
    }

    #[test]
    fn multipath_within_the_prefix_is_removed_by_per_subcarrier_equalization() {
        let c = Constellation::qam(16).unwrap();
        let mut seed = 7;
        let params = OfdmParams::wifi_like();
        let mut ofdm = Ofdm::new(params.clone()).unwrap();
        let h = [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.35, 0.2), Complex::new(0.0, 0.0), Complex::new(-0.15, 0.1)];
        // Training symbol then payload; the channel is applied to the whole stream.
        let training = data_symbols(&c, 48, &mut seed);
        let payload = data_symbols(&c, 48 * 30, &mut seed);
        let mut all = training.clone();
        all.extend(&payload);
        let rx = apply_channel(&ofdm.modulate(&all).unwrap(), &h);
        let est = ofdm.estimate_channel(&rx[..80], &training).unwrap();
        let eq = ofdm.demodulate_with_channel(&rx[80..], &est).unwrap();
        assert!(eq.iter().zip(&payload).all(|(a, b)| (a - b).norm() < 1e-9), "exact equalization with a training-symbol estimate");
        // Without equalization the channel wrecks the symbols.
        let raw = ofdm.demodulate_with_channel(&rx[80..], &vec![Complex::new(1.0, 0.0); 48]).unwrap();
        let wrong = raw.iter().zip(&payload).filter(|(a, b)| c.slice(**a) != **b).count();
        assert!(wrong > payload.len() / 4);
    }

    #[test]
    fn pilot_interpolation_equalizes_a_mild_channel_and_tolerates_noise() {
        let c = Constellation::qpsk();
        let mut seed = 11;
        let params = dense_pilot_params();
        let mut ofdm = Ofdm::new(params.clone()).unwrap();
        let nd = params.num_data();
        let syms = data_symbols(&c, nd * 60, &mut seed);
        let h = [Complex::new(0.9, 0.2), Complex::new(0.25, -0.1)];
        let mut rx = apply_channel(&ofdm.modulate(&syms).unwrap(), &h);
        for v in rx.iter_mut() {
            *v += Complex::new(gauss(&mut seed), gauss(&mut seed)) * 0.02;
        }
        let out = ofdm.demodulate(&rx).unwrap();
        let errors = out.data.iter().zip(&syms).filter(|(a, b)| c.slice(**a) != **b).count();
        assert_eq!(errors, 0);
        assert_eq!(out.channel.len(), out.data.len());
        // The estimated channel has the shape of the true frequency response at a data subcarrier.
        let k = params.data[10] as f64;
        let truth: Complex<f64> = h.iter().enumerate().map(|(n, c)| c * Complex::from_polar(1.0, -std::f64::consts::TAU * k * n as f64 / 64.0)).sum();
        assert!((out.channel[10] - truth).norm() < 0.05, "{} vs {truth}", out.channel[10]);
    }

    #[test]
    fn a_channel_longer_than_the_prefix_breaks_orthogonality() {
        let c = Constellation::qpsk();
        let mut seed = 13;
        let params = OfdmParams::new(64, 4, (-26..=26).filter(|k| *k != 0 && k % 4 != 0).collect(), (-24..=24).step_by(4).filter(|k| *k != 0).collect(), vec![Complex::new(1.0, 0.0); 12]).unwrap();
        let mut ofdm = Ofdm::new(params.clone()).unwrap();
        let syms = data_symbols(&c, params.num_data() * 40, &mut seed);
        // 12-sample delay spread against a 4-sample prefix: heavy inter-symbol interference.
        let mut h = vec![Complex::new(0.0, 0.0); 13];
        h[0] = Complex::new(1.0, 0.0);
        h[12] = Complex::new(0.8, 0.0);
        let rx = apply_channel(&ofdm.modulate(&syms).unwrap(), &h);
        let out = ofdm.demodulate(&rx).unwrap();
        let errors = out.data.iter().zip(&syms).filter(|(a, b)| c.slice(**a) != **b).count();
        assert!(errors > syms.len() / 20, "only {errors} errors out of {}", syms.len());
    }

    #[test]
    fn invalid_layouts_and_inputs() {
        let one = Complex::new(1.0, 0.0);
        assert!(OfdmParams::new(60, 4, vec![1], vec![], vec![]).is_err(), "FFT size must be a power of two");
        assert!(OfdmParams::new(64, 64, vec![1], vec![], vec![]).is_err(), "prefix must be shorter than the FFT");
        assert!(OfdmParams::new(64, 8, vec![], vec![], vec![]).is_err(), "no data subcarriers");
        assert!(OfdmParams::new(64, 8, vec![0], vec![], vec![]).is_err(), "DC is not allowed");
        assert!(OfdmParams::new(64, 8, vec![40], vec![], vec![]).is_err(), "out of range");
        assert!(OfdmParams::new(64, 8, vec![1, 2], vec![2], vec![one]).is_err(), "pilot overlaps data");
        assert!(OfdmParams::new(64, 8, vec![1], vec![5], vec![]).is_err(), "pilot value count mismatch");
        assert!(OfdmParams::new(64, 8, vec![1, 1], vec![], vec![]).is_err(), "duplicate indices");
        let params = OfdmParams::wifi_like();
        assert!(ofdm_modulate(&[one; 47], &params).is_err(), "not a multiple of the data count");
        assert!(ofdm_modulate(&[], &params).is_err());
        assert!(ofdm_demodulate(&[one; 79], &params).is_err(), "not a multiple of the symbol length");
        let mut ofdm = Ofdm::new(params).unwrap();
        assert!(ofdm.estimate_channel(&[one; 80], &[Complex::new(0.0, 0.0); 48]).is_err(), "zero training symbols");
        assert!(ofdm.demodulate_with_channel(&[one; 80], &[one; 3]).is_err());
    }
}
