//! A simple MP3-style perceptual audio codec.
//!
//! This is a small, readable codec in the spirit of MPEG-1 Layer III, intended as a worked
//! example rather than a production encoder. It is **not bitstream-compatible with MP3**: it uses
//! its own container, and the standard's polyphase window and Huffman tables are not reproduced.
//! What it does implement is the shape of the Layer III pipeline:
//!
//! 1. **Transform.** Audio is cut into granules of 576 samples and transformed with a
//!    sine-windowed MDCT (50% overlap, perfect reconstruction) into 576 spectral lines. Layer III
//!    uses a polyphase filterbank followed by a smaller MDCT; here a single MDCT stands in.
//!    There is no block switching, so transients can pre-echo.
//! 2. **Psychoacoustic model.** The lines are grouped into the 22 Layer III long-block
//!    scalefactor bands. A [`PsychoacousticModel`] returns, per band, the noise energy that can
//!    be hidden under the signal. [`GenericPsychoacousticModel`] is a deliberately simple default:
//!    inter-band spreading, a masking offset that depends on how tonal the band is, and the
//!    absolute threshold of hearing. Any other model can be supplied through
//!    [`mp3_encode_with_model`].
//! 3. **Quantization.** Each band gets a step size `2^(e/4)` (1.5 dB resolution, like Layer III
//!    scalefactors) chosen so quantization noise sits at the masking threshold; values are
//!    quantized with Layer III's power law `q = round(|x/step|^0.75 - 0.0946)`.
//! 4. **Rate control.** A search over a global noise scale finds the best quality that fits the
//!    granule's bit budget; unused bits accumulate in a reservoir for later granules.
//! 5. **Entropy coding.** Per band, an Exp-Golomb code whose order is chosen to minimise the bit
//!    count (a stand-in for Layer III's Huffman table selection), with trailing all-zero bands
//!    omitted. Stereo files use mid/side coding when the channels are similar.
//!
//! Samples are interleaved 16-bit PCM.
//!
//! See the book entry [Audio files](https://jwkunz.github.io/lib_jsl/book/foundations/audio.html#a-perceptual-codec).

use lib_jsl_core::ErrorsJSL;
use std::f64::consts::PI;
use std::sync::OnceLock;

/// Spectral lines per granule (and hop size in samples), as in Layer III.
pub const GRANULE: usize = 576;

/// Layer III long-block scalefactor band edges (in spectral lines) for 576-line granules.
pub const BAND_EDGES: [usize; 23] = [
    0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 52, 62, 74, 90, 110, 134, 162, 196, 238, 288, 342, 418, 576,
];
const NUM_BANDS: usize = BAND_EDGES.len() - 1;

const MAGIC: &[u8; 4] = b"JMP3";
const VERSION: u8 = 1;
const HEADER_LEN: usize = 14;

/// Decoded audio: interleaved 16-bit PCM.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mp3Audio {
    pub sample_rate: u32,
    pub channels: usize,
    /// Interleaved samples, `frames * channels` long.
    pub samples: Vec<i16>,
}

// ------------------------------------------------------ psychoacoustics

/// Decides how much quantization noise each band can hide.
pub trait PsychoacousticModel {
    /// Given one granule's `GRANULE` MDCT coefficients (in sample-amplitude units, orthonormal
    /// scaling) and the sampling rate, returns the maximum *noise energy* (sum of squares over the
    /// band's lines) that is inaudible, for each of the [`BAND_EDGES`] bands.
    fn thresholds(&self, coefs: &[f64], sample_rate: u32) -> [f64; NUM_BANDS];
}

/// A simple, generic masking model.
///
/// Per band: take the signal energy, spread it to neighbouring bands with an asymmetric slope
/// (masking extends further upward in frequency than downward), lower it by an offset that
/// interpolates between `noise_offset_db` (noise-like bands) and `tonal_offset_db` (tonal bands,
/// which mask less), and never go below the absolute threshold of hearing.
#[derive(Clone, Debug)]
pub struct GenericPsychoacousticModel {
    /// Masker-to-mask offset for noise-like bands, in dB.
    pub noise_offset_db: f64,
    /// Masker-to-mask offset for strongly tonal bands, in dB.
    pub tonal_offset_db: f64,
    /// Spreading toward higher bands, in dB per band.
    pub upward_slope_db: f64,
    /// Spreading toward lower bands, in dB per band.
    pub downward_slope_db: f64,
}

impl Default for GenericPsychoacousticModel {
    fn default() -> Self {
        GenericPsychoacousticModel {
            noise_offset_db: 6.0,
            tonal_offset_db: 18.0,
            upward_slope_db: 10.0,
            downward_slope_db: 25.0,
        }
    }
}

/// Absolute threshold of hearing in dB SPL (Terhardt's approximation), `f` in Hz.
pub fn absolute_threshold_db(f: f64) -> f64 {
    let k = (f / 1000.0).max(0.02);
    3.64 * k.powf(-0.8) - 6.5 * (-0.6 * (k - 3.3).powi(2)).exp() + 1e-3 * k.powi(4)
}

impl PsychoacousticModel for GenericPsychoacousticModel {
    fn thresholds(&self, coefs: &[f64], sample_rate: u32) -> [f64; NUM_BANDS] {
        let mut masking = [0.0f64; NUM_BANDS];
        let mut ath = [0.0f64; NUM_BANDS];
        for b in 0..NUM_BANDS {
            let (lo, hi) = (BAND_EDGES[b], BAND_EDGES[b + 1]);
            let width = (hi - lo) as f64;
            let energy: f64 = coefs[lo..hi].iter().map(|c| c * c).sum();
            // Tonality from the spectral flatness of the band: ~0 for noise, ~1 for a sinusoid.
            let mean_log = coefs[lo..hi].iter().map(|c| (c * c + 1e-9).ln()).sum::<f64>() / width;
            let flatness_db = 10.0 * (mean_log.exp() / (energy / width + 1e-9)).log10();
            let tonality = (-flatness_db / 20.0).clamp(0.0, 1.0);
            let offset = self.noise_offset_db + (self.tonal_offset_db - self.noise_offset_db) * tonality;
            masking[b] = energy * 10f64.powf(-offset / 10.0);
            let centre_hz = 0.5 * (lo + hi) as f64 * sample_rate as f64 / (2.0 * GRANULE as f64);
            // One unit of line energy is taken as 0 dB SPL.
            ath[b] = width * 10f64.powf(absolute_threshold_db(centre_hz).min(70.0) / 10.0);
        }
        let mut out = [0.0; NUM_BANDS];
        for (b, o) in out.iter_mut().enumerate() {
            let mut spread = 0.0;
            for (j, &m) in masking.iter().enumerate() {
                let d = b as f64 - j as f64; // positive: target above the masker
                let att = if d >= 0.0 { self.upward_slope_db * d } else { self.downward_slope_db * -d };
                spread += m * 10f64.powf(-att / 10.0);
            }
            *o = spread.max(ath[b]);
        }
        out
    }
}

// ------------------------------------------------------------------ MDCT

/// Orthonormal DCT-IV matrix, `C[k][n] = sqrt(2/N) cos(π/N (n + 1/2)(k + 1/2))`.
fn dct4_matrix() -> &'static [f64] {
    static M: OnceLock<Vec<f64>> = OnceLock::new();
    M.get_or_init(|| {
        let scale = (2.0 / GRANULE as f64).sqrt();
        let mut m = vec![0.0; GRANULE * GRANULE];
        for k in 0..GRANULE {
            for n in 0..GRANULE {
                m[k * GRANULE + n] = scale * (PI / GRANULE as f64 * (n as f64 + 0.5) * (k as f64 + 0.5)).cos();
            }
        }
        m
    })
}

fn sine_window() -> &'static [f64] {
    static W: OnceLock<Vec<f64>> = OnceLock::new();
    W.get_or_init(|| (0..2 * GRANULE).map(|n| (PI * (n as f64 + 0.5) / (2 * GRANULE) as f64).sin()).collect())
}

fn dct4(input: &[f64]) -> Vec<f64> {
    let m = dct4_matrix();
    (0..GRANULE)
        .map(|k| m[k * GRANULE..(k + 1) * GRANULE].iter().zip(input).map(|(a, b)| a * b).sum())
        .collect()
}

/// Forward MDCT of a block of `2 * GRANULE` samples into `GRANULE` coefficients.
fn mdct(block: &[f64]) -> Vec<f64> {
    let w = sine_window();
    let z: Vec<f64> = block.iter().zip(w).map(|(x, w)| x * w).collect();
    let h = GRANULE / 2;
    let mut u = vec![0.0; GRANULE];
    for i in 0..h {
        u[i] = -z[3 * h - 1 - i] - z[3 * h + i];
        u[h + i] = z[i] - z[GRANULE - 1 - i];
    }
    dct4(&u)
}

/// Inverse MDCT: returns `2 * GRANULE` windowed samples ready for overlap-add.
fn imdct(coefs: &[f64]) -> Vec<f64> {
    let v = dct4(coefs);
    let h = GRANULE / 2;
    let w = sine_window();
    let mut out = vec![0.0; 2 * GRANULE];
    for i in 0..h {
        out[i] = v[h + i];
        out[h + i] = -v[GRANULE - 1 - i];
        out[GRANULE + i] = -v[h - 1 - i];
        out[3 * h + i] = -v[i];
    }
    for (o, w) in out.iter_mut().zip(w) {
        *o *= w;
    }
    out
}

// ------------------------------------------------------------ bit output

/// A bit sink that can either store bits or just count them.
struct BitWriter {
    bytes: Vec<u8>,
    acc: u32,
    nbits: u32,
    total: usize,
    counting_only: bool,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter { bytes: Vec::new(), acc: 0, nbits: 0, total: 0, counting_only: false }
    }

    fn counter() -> Self {
        BitWriter { counting_only: true, ..Self::new() }
    }

    fn put(&mut self, value: u32, len: u32) {
        self.total += len as usize;
        if self.counting_only || len == 0 {
            return;
        }
        for i in (0..len).rev() {
            self.acc = (self.acc << 1) | ((value >> i) & 1);
            self.nbits += 1;
            if self.nbits == 8 {
                self.bytes.push(self.acc as u8);
                self.acc = 0;
                self.nbits = 0;
            }
        }
    }

    /// Exp-Golomb code of order `k` for `v >= 0`.
    fn put_eg(&mut self, v: u32, k: u32) {
        let x = v as u64 + (1u64 << k);
        let len = 64 - x.leading_zeros();
        self.put(0, len - k - 1);
        // `put` handles at most 32 bits at a time.
        if len > 32 {
            self.put((x >> 32) as u32, len - 32);
            self.put(x as u32, 32);
        } else {
            self.put(x as u32, len);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.put(0, pad);
        }
        self.bytes
    }
}

fn eg_cost(v: u32, k: u32) -> usize {
    let x = v as u64 + (1u64 << k);
    let len = (64 - x.leading_zeros()) as usize;
    2 * len - k as usize - 1
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize, // bit position
}

impl<'a> BitReader<'a> {
    fn bit(&mut self) -> Result<u32, ErrorsJSL> {
        let byte = *self.data.get(self.pos / 8).ok_or(ErrorsJSL::InvalidInputRange("Audio stream ended unexpectedly."))?;
        let b = (byte >> (7 - self.pos % 8)) & 1;
        self.pos += 1;
        Ok(b as u32)
    }

    fn bits(&mut self, n: u32) -> Result<u32, ErrorsJSL> {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.bit()?;
        }
        Ok(v)
    }

    fn get_eg(&mut self, k: u32) -> Result<u32, ErrorsJSL> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 40 {
                return Err(ErrorsJSL::InvalidInputRange("Invalid Exp-Golomb code in audio stream."));
            }
        }
        let n = zeros + k; // bits after the leading one
        let mut x: u64 = 1;
        for _ in 0..n {
            x = (x << 1) | self.bit()? as u64;
        }
        let v = x - (1u64 << k);
        u32::try_from(v).map_err(|_| ErrorsJSL::InvalidInputRange("Audio value out of range."))
    }
}

// ------------------------------------------------------------ quantizer

const STEP_MIN: i32 = -40;
const STEP_MAX: i32 = 110;

fn step_size(e: i32) -> f64 {
    2f64.powf(e as f64 / 4.0)
}

/// Quantized spectrum of one channel in one granule.
struct QuantizedChannel {
    q: [i32; GRANULE],
    exponents: [i32; NUM_BANDS],
}

/// Quantizes a channel with per-band step sizes derived from `thresholds * scale`.
fn quantize_channel(coefs: &[f64], thresholds: &[f64; NUM_BANDS], scale: f64) -> QuantizedChannel {
    let mut out = QuantizedChannel { q: [0; GRANULE], exponents: [0; NUM_BANDS] };
    for b in 0..NUM_BANDS {
        let (lo, hi) = (BAND_EDGES[b], BAND_EDGES[b + 1]);
        let width = (hi - lo) as f64;
        // Uniform quantizer noise is step^2/12 per line.
        let step = (12.0 * thresholds[b] * scale / width).sqrt();
        let e = ((4.0 * step.log2()).round() as i32).clamp(STEP_MIN, STEP_MAX);
        out.exponents[b] = e;
        let inv = 1.0 / step_size(e);
        for (k, &c) in coefs.iter().enumerate().take(hi).skip(lo) {
            let a = (c.abs() * inv).powf(0.75);
            let q = (a - 0.0946 + 0.5).floor().max(0.0) as i32;
            out.q[k] = if c < 0.0 { -q } else { q };
        }
    }
    out
}

fn zigzag_signed(d: i32) -> u32 {
    if d >= 0 { 2 * d as u32 } else { (-2 * d - 1) as u32 }
}

fn unzigzag_signed(u: u32) -> i32 {
    if u.is_multiple_of(2) { (u / 2) as i32 } else { -(u.div_ceil(2) as i32) }
}

const ZERO_BAND: u32 = 7;
const EXP_GOLOMB_ORDERS: u32 = 7; // orders 0..=6

/// Writes one channel's quantized granule.
fn write_channel(w: &mut BitWriter, ch: &QuantizedChannel) {
    let band_nonzero = |b: usize| ch.q[BAND_EDGES[b]..BAND_EDGES[b + 1]].iter().any(|&v| v != 0);
    let coded_bands = (0..NUM_BANDS).rev().find(|&b| band_nonzero(b)).map_or(0, |b| b + 1);
    w.put(coded_bands as u32, 5);
    let mut prev_e = 40;
    for b in 0..coded_bands {
        if !band_nonzero(b) {
            w.put(ZERO_BAND, 3);
            continue;
        }
        let lines = &ch.q[BAND_EDGES[b]..BAND_EDGES[b + 1]];
        let (best_k, _) = (0..EXP_GOLOMB_ORDERS)
            .map(|k| (k, lines.iter().map(|&v| eg_cost(v.unsigned_abs(), k) + (v != 0) as usize).sum::<usize>()))
            .min_by_key(|&(_, c)| c)
            .unwrap();
        w.put(best_k, 3);
        w.put_eg(zigzag_signed(ch.exponents[b] - prev_e), 0);
        prev_e = ch.exponents[b];
        for &v in lines {
            w.put_eg(v.unsigned_abs(), best_k);
            if v != 0 {
                w.put((v < 0) as u32, 1);
            }
        }
    }
}

fn read_channel(r: &mut BitReader) -> Result<Vec<f64>, ErrorsJSL> {
    let coded_bands = r.bits(5)? as usize;
    if coded_bands > NUM_BANDS {
        return Err(ErrorsJSL::InvalidInputRange("Invalid band count in audio stream."));
    }
    let mut coefs = vec![0.0; GRANULE];
    let mut prev_e = 40;
    for b in 0..coded_bands {
        let k = r.bits(3)?;
        if k == ZERO_BAND {
            continue;
        }
        let e = prev_e + unzigzag_signed(r.get_eg(0)?);
        if !(STEP_MIN..=STEP_MAX).contains(&e) {
            return Err(ErrorsJSL::InvalidInputRange("Invalid step size in audio stream."));
        }
        prev_e = e;
        let step = step_size(e);
        for line in coefs.iter_mut().take(BAND_EDGES[b + 1]).skip(BAND_EDGES[b]) {
            let mag = r.get_eg(k)?;
            if mag != 0 {
                let neg = r.bit()? == 1;
                let v = (mag as f64).powf(4.0 / 3.0) * step;
                *line = if neg { -v } else { v };
            }
        }
    }
    Ok(coefs)
}

// --------------------------------------------------------------- encoder

/// Encodes interleaved 16-bit PCM with the [`GenericPsychoacousticModel`].
///
/// `bitrate_kbps` is the target average rate (8 to 320). The encoder tracks the rate closely
/// (unused bits are carried forward in a reservoir). It stops improving quality once the noise is
/// 30 dB below the masking threshold, so very simple content can come in under the target.
pub fn mp3_encode(samples: &[i16], sample_rate: u32, channels: usize, bitrate_kbps: u32) -> Result<Vec<u8>, ErrorsJSL> {
    mp3_encode_with_model(samples, sample_rate, channels, bitrate_kbps, &GenericPsychoacousticModel::default())
}

/// Like [`mp3_encode`] but with a caller-supplied psychoacoustic model.
pub fn mp3_encode_with_model(
    samples: &[i16],
    sample_rate: u32,
    channels: usize,
    bitrate_kbps: u32,
    model: &dyn PsychoacousticModel,
) -> Result<Vec<u8>, ErrorsJSL> {
    if channels != 1 && channels != 2 {
        return Err(ErrorsJSL::InvalidInputRange("Audio coding supports 1 or 2 channels."));
    }
    if !(8000..=192_000).contains(&sample_rate) {
        return Err(ErrorsJSL::InvalidInputRange("Sample rate must be between 8000 and 192000 Hz."));
    }
    if !(8..=320).contains(&bitrate_kbps) {
        return Err(ErrorsJSL::InvalidInputRange("Bitrate must be between 8 and 320 kbps."));
    }
    if !samples.len().is_multiple_of(channels) {
        return Err(ErrorsJSL::IncompatibleArraySizes((samples.len(), channels)));
    }
    let frames = samples.len() / channels;
    if frames > u32::MAX as usize {
        return Err(ErrorsJSL::InvalidInputRange("Audio is too long."));
    }
    let blocks = frames.div_ceil(GRANULE) + 1;

    // De-interleave into padded planes: GRANULE leading zeros so block 0 sees silence first.
    let padded_len = (blocks + 1) * GRANULE;
    let mut planes = vec![vec![0.0f64; padded_len]; channels];
    for (i, frame) in samples.chunks_exact(channels).enumerate() {
        for (c, &s) in frame.iter().enumerate() {
            planes[c][GRANULE + i] = s as f64;
        }
    }

    let avg_bits = bitrate_kbps as f64 * 1000.0 * GRANULE as f64 / sample_rate as f64;
    let mut credit = 0.0f64;
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.push(channels as u8);
    out.extend_from_slice(&sample_rate.to_be_bytes());
    out.extend_from_slice(&(frames as u32).to_be_bytes());
    let mut writer = BitWriter::new();

    for b in 0..blocks {
        let mut spectra: Vec<Vec<f64>> = (0..channels).map(|c| mdct(&planes[c][b * GRANULE..(b + 2) * GRANULE])).collect();
        let mut thresholds: Vec<[f64; NUM_BANDS]> = spectra.iter().map(|s| model.thresholds(s, sample_rate)).collect();
        let mut mid_side = false;
        if channels == 2 {
            let e_l: f64 = spectra[0].iter().map(|x| x * x).sum();
            let e_r: f64 = spectra[1].iter().map(|x| x * x).sum();
            let (mut e_m, mut e_s) = (0.0, 0.0);
            for (&l, &r) in spectra[0].iter().zip(&spectra[1]) {
                let (m, s) = ((l + r) * std::f64::consts::FRAC_1_SQRT_2, (l - r) * std::f64::consts::FRAC_1_SQRT_2);
                e_m += m * m;
                e_s += s * s;
            }
            mid_side = e_l + e_r > 0.0 && e_s < 0.3 * e_m;
            if mid_side {
                let (left, right) = spectra.split_at_mut(1);
                for (a, b) in left[0].iter_mut().zip(right[0].iter_mut()) {
                    let (l, r) = (*a, *b);
                    *a = (l + r) * std::f64::consts::FRAC_1_SQRT_2;
                    *b = (l - r) * std::f64::consts::FRAC_1_SQRT_2;
                }
                // Noise in M and S each must stay under the tighter of the two L/R thresholds.
                let tight: [f64; NUM_BANDS] = std::array::from_fn(|i| thresholds[0][i].min(thresholds[1][i]));
                thresholds = vec![tight, tight];
            }
        }

        let budget = avg_bits + credit.min(2.0 * avg_bits);
        let side_bits = if channels == 2 { 1 } else { 0 };
        let encode_at = |scale_db: f64| -> Vec<QuantizedChannel> {
            let scale = 10f64.powf(scale_db / 10.0);
            spectra.iter().zip(&thresholds).map(|(s, t)| quantize_channel(s, t, scale)).collect()
        };
        let cost_of = |qs: &[QuantizedChannel]| -> usize {
            let mut w = BitWriter::counter();
            for q in qs {
                write_channel(&mut w, q);
            }
            w.total + side_bits
        };
        // Smallest noise scale (best quality) that fits the budget; coarse upper bound if none does.
        const BEST_DB: f64 = -30.0;
        const WORST_DB: f64 = 70.0;
        let chosen = if cost_of(&encode_at(BEST_DB)) as f64 <= budget {
            encode_at(BEST_DB)
        } else {
            let (mut lo, mut hi) = (BEST_DB, WORST_DB);
            for _ in 0..14 {
                let mid = 0.5 * (lo + hi);
                if cost_of(&encode_at(mid)) as f64 <= budget {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            encode_at(hi)
        };

        let before = writer.total;
        if channels == 2 {
            writer.put(mid_side as u32, 1);
        }
        for q in &chosen {
            write_channel(&mut writer, q);
        }
        credit += avg_bits - (writer.total - before) as f64;
        credit = credit.max(0.0);
    }
    out.extend(writer.finish());
    Ok(out)
}

// --------------------------------------------------------------- decoder

/// Decodes a stream produced by [`mp3_encode`].
pub fn mp3_decode(data: &[u8]) -> Result<Mp3Audio, ErrorsJSL> {
    if data.len() < HEADER_LEN || &data[..4] != MAGIC {
        return Err(ErrorsJSL::InvalidInputRange("Not an audio stream produced by mp3_encode."));
    }
    if data[4] != VERSION {
        return Err(ErrorsJSL::InvalidInputRange("Unsupported audio stream version."));
    }
    let channels = data[5] as usize;
    let sample_rate = u32::from_be_bytes([data[6], data[7], data[8], data[9]]);
    let frames = u32::from_be_bytes([data[10], data[11], data[12], data[13]]) as usize;
    if (channels != 1 && channels != 2) || sample_rate == 0 {
        return Err(ErrorsJSL::InvalidInputRange("Invalid audio stream header."));
    }
    let blocks = frames.div_ceil(GRANULE) + 1;
    // Guard against absurd headers before allocating: every block needs at least a few bits.
    if blocks > (data.len() - HEADER_LEN) * 8 {
        return Err(ErrorsJSL::InvalidInputRange("Audio stream is shorter than its header claims."));
    }
    let mut reader = BitReader { data: &data[HEADER_LEN..], pos: 0 };
    let mut planes = vec![vec![0.0f64; (blocks + 1) * GRANULE]; channels];
    for b in 0..blocks {
        let mid_side = channels == 2 && reader.bit()? == 1;
        let mut spectra = Vec::with_capacity(channels);
        for _ in 0..channels {
            spectra.push(read_channel(&mut reader)?);
        }
        if mid_side {
            let (left, right) = spectra.split_at_mut(1);
            for (a, b) in left[0].iter_mut().zip(right[0].iter_mut()) {
                let (m, s) = (*a, *b);
                *a = (m + s) * std::f64::consts::FRAC_1_SQRT_2;
                *b = (m - s) * std::f64::consts::FRAC_1_SQRT_2;
            }
        }
        for (c, spec) in spectra.iter().enumerate() {
            for (i, v) in imdct(spec).into_iter().enumerate() {
                planes[c][b * GRANULE + i] += v;
            }
        }
    }
    let mut samples = Vec::with_capacity(frames * channels);
    for i in 0..frames {
        for plane in &planes {
            samples.push(plane[GRANULE + i].round().clamp(-32768.0, 32767.0) as i16);
        }
    }
    Ok(Mp3Audio { sample_rate, channels, samples })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 44_100;

    fn sine(freq: f64, amp: f64, n: usize) -> Vec<i16> {
        (0..n).map(|i| (amp * (2.0 * PI * freq * i as f64 / SR as f64).sin()).round() as i16).collect()
    }

    fn snr_db(reference: &[i16], decoded: &[i16]) -> f64 {
        let sig: f64 = reference.iter().map(|&x| (x as f64).powi(2)).sum();
        let err: f64 = reference.iter().zip(decoded).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum();
        if err == 0.0 { f64::INFINITY } else { 10.0 * (sig / err).log10() }
    }

    fn noise(n: usize, amp: f64, seed: u64) -> Vec<i16> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (((s >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2.0 * amp) as i16
            })
            .collect()
    }

    #[test]
    fn mdct_is_perfectly_reconstructing() {
        let x: Vec<f64> = (0..4 * GRANULE).map(|i| ((i * 7919) % 2003) as f64 - 1000.0).collect();
        let mut y = vec![0.0; 4 * GRANULE];
        for b in 0..3 {
            let c = mdct(&x[b * GRANULE..(b + 2) * GRANULE]);
            for (i, v) in imdct(&c).into_iter().enumerate() {
                y[b * GRANULE + i] += v;
            }
        }
        // Interior samples (covered by two blocks) are reconstructed exactly.
        for i in GRANULE..3 * GRANULE {
            assert!((x[i] - y[i]).abs() < 1e-8, "sample {i}: {} vs {}", x[i], y[i]);
        }
    }

    #[test]
    fn mdct_preserves_energy() {
        // The windowed lapped transform is orthogonal, so (with silence at both ends) the total
        // coefficient energy equals the total signal energy.
        let mut x = vec![0.0; 6 * GRANULE];
        for (i, v) in x.iter_mut().enumerate().take(4 * GRANULE).skip(GRANULE) {
            *v = (i as f64 * 0.37).sin() * 1000.0 + ((i * 31) % 17) as f64;
        }
        let energy_in: f64 = x.iter().map(|v| v * v).sum();
        let total: f64 = (0..5)
            .map(|b| mdct(&x[b * GRANULE..(b + 2) * GRANULE]).iter().map(|v| v * v).sum::<f64>())
            .sum();
        assert!((total / energy_in - 1.0).abs() < 1e-9, "{total} vs {energy_in}");
    }

    #[test]
    fn quantization_noise_lands_under_the_masking_threshold() {
        // A tone plus broadband noise; quantize at the nominal scale and measure noise per band.
        let pcm: Vec<i16> = sine(1500.0, 9000.0, 4 * GRANULE)
            .iter()
            .zip(noise(4 * GRANULE, 400.0, 11))
            .map(|(&a, b)| a.saturating_add(b))
            .collect();
        let x: Vec<f64> = pcm.iter().map(|&v| v as f64).collect();
        let coefs = mdct(&x[GRANULE..3 * GRANULE]);
        let thresholds = GenericPsychoacousticModel::default().thresholds(&coefs, SR);
        let q = quantize_channel(&coefs, &thresholds, 1.0);
        for b in 0..NUM_BANDS {
            let step = step_size(q.exponents[b]);
            let noise_energy: f64 = (BAND_EDGES[b]..BAND_EDGES[b + 1])
                .map(|k| {
                    let rec = (q.q[k].unsigned_abs() as f64).powf(4.0 / 3.0) * step * (q.q[k].signum() as f64);
                    (coefs[k] - rec).powi(2)
                })
                .sum();
            // Power-law quantization and 1.5 dB step rounding allow a modest overshoot.
            assert!(noise_energy <= 3.0 * thresholds[b], "band {b}: noise {noise_energy} vs threshold {}", thresholds[b]);
        }
    }

    #[test]
    fn exp_golomb_round_trip() {
        let mut w = BitWriter::new();
        let values = [0u32, 1, 2, 3, 7, 8, 100, 65_535, 1_000_000];
        for k in 0..7 {
            for &v in &values {
                w.put_eg(v, k);
            }
        }
        let expected_bits = w.total;
        let bytes = w.finish();
        let mut r = BitReader { data: &bytes, pos: 0 };
        for k in 0..7 {
            for &v in &values {
                assert_eq!(r.get_eg(k).unwrap(), v);
            }
        }
        assert_eq!(r.pos, expected_bits);
        assert_eq!(eg_cost(5, 2), 5);
        for d in -50..50 {
            assert_eq!(unzigzag_signed(zigzag_signed(d)), d);
        }
    }

    #[test]
    fn ath_has_the_expected_shape() {
        // Most sensitive around 3-4 kHz, rising steeply at both ends.
        let low = absolute_threshold_db(50.0);
        let mid = absolute_threshold_db(3500.0);
        let high = absolute_threshold_db(18_000.0);
        assert!(mid < 0.0 && low > 30.0 && high > 40.0);
    }

    #[test]
    fn model_masks_neighbours_of_a_tone_and_respects_the_ath() {
        let mut spec = vec![0.0; GRANULE];
        spec[100] = 20_000.0; // a strong, isolated tone
        let t = GenericPsychoacousticModel::default().thresholds(&spec, SR);
        let tone_band = (0..NUM_BANDS).find(|&b| (BAND_EDGES[b]..BAND_EDGES[b + 1]).contains(&100)).unwrap();
        // The tone's own band tolerates a lot of noise; far-away bands fall back toward the ATH.
        assert!(t[tone_band] > 1e5);
        assert!(t[tone_band] > t[tone_band + 3] && t[tone_band + 3] > 0.0);
        assert!(t[tone_band] > t[tone_band - 3]);
        let silence = GenericPsychoacousticModel::default().thresholds(&vec![0.0; GRANULE], SR);
        assert!(silence.iter().all(|&v| v > 0.0), "ATH keeps thresholds positive");
    }

    #[test]
    fn tone_round_trip() {
        let pcm = sine(1000.0, 12_000.0, 20_000);
        let enc = mp3_encode(&pcm, SR, 1, 128).unwrap();
        let dec = mp3_decode(&enc).unwrap();
        assert_eq!((dec.sample_rate, dec.channels, dec.samples.len()), (SR, 1, pcm.len()));
        let snr = snr_db(&pcm, &dec.samples);
        assert!(snr > 15.0, "snr {snr}");
        assert!(enc.len() * 8 <= (128_000.0 * 20_000.0 / SR as f64 * 1.1) as usize + 400);
    }

    #[test]
    fn higher_bitrate_means_higher_quality_and_bigger_files() {
        let mut pcm = sine(440.0, 8000.0, 30_000);
        let n = noise(30_000, 3000.0, 5);
        for (p, q) in pcm.iter_mut().zip(&n) {
            *p = p.saturating_add(*q);
        }
        let mut last_size = 0;
        let mut last_snr = f64::NEG_INFINITY;
        for kbps in [32u32, 96, 256] {
            let enc = mp3_encode(&pcm, SR, 1, kbps).unwrap();
            let snr = snr_db(&pcm, &mp3_decode(&enc).unwrap().samples);
            assert!(enc.len() > last_size, "{kbps} kbps");
            assert!(snr > last_snr, "{kbps} kbps: snr {snr} <= {last_snr}");
            let limit = kbps as f64 * 1000.0 / 8.0 * 30_000.0 / SR as f64;
            assert!(enc.len() as f64 <= limit * 1.15 + 100.0, "{kbps} kbps: {} bytes vs {limit}", enc.len());
            last_size = enc.len();
            last_snr = snr;
        }
    }

    #[test]
    fn silence_is_tiny_and_exact() {
        let pcm = vec![0i16; 10_000];
        let enc = mp3_encode(&pcm, SR, 1, 128).unwrap();
        assert!(enc.len() < 80, "{} bytes", enc.len());
        assert_eq!(mp3_decode(&enc).unwrap().samples, pcm);
    }

    #[test]
    fn stereo_with_mid_side() {
        let left = sine(660.0, 10_000.0, 12_000);
        let mut inter = Vec::new();
        for &l in &left {
            inter.push(l);
            inter.push(l); // identical channels: pure mid signal
        }
        let enc = mp3_encode(&inter, SR, 2, 128).unwrap();
        let dec = mp3_decode(&enc).unwrap();
        assert_eq!(dec.channels, 2);
        assert_eq!(dec.samples.len(), inter.len());
        assert!(snr_db(&inter, &dec.samples) > 15.0);
        for pair in dec.samples.as_chunks::<2>().0 {
            assert!((pair[0] as i32 - pair[1] as i32).abs() <= 1, "channels diverged");
        }
        // Compare with a genuinely stereo signal: M/S makes the correlated file cheaper.
        let mut wide = Vec::new();
        let other = noise(12_000, 4000.0, 9);
        for (&l, &o) in left.iter().zip(&other) {
            wide.push(l);
            wide.push(o);
        }
        assert!(enc.len() < mp3_encode(&wide, SR, 2, 128).unwrap().len() || enc.len() < 20_000);
    }

    #[test]
    fn lengths_not_a_multiple_of_the_granule() {
        for n in [0usize, 1, 575, 576, 577, 1500] {
            let pcm = sine(500.0, 5000.0, n);
            let dec = mp3_decode(&mp3_encode(&pcm, SR, 1, 96).unwrap()).unwrap();
            assert_eq!(dec.samples.len(), n, "n = {n}");
        }
    }

    #[test]
    fn custom_psychoacoustic_model_is_used() {
        struct Strict;
        impl PsychoacousticModel for Strict {
            fn thresholds(&self, _coefs: &[f64], _sr: u32) -> [f64; NUM_BANDS] {
                [0.01; NUM_BANDS] // demand near-lossless coding everywhere
            }
        }
        let pcm = noise(8000, 5000.0, 3);
        let strict = mp3_encode_with_model(&pcm, SR, 1, 320, &Strict).unwrap();
        let generic = mp3_encode(&pcm, SR, 1, 320).unwrap();
        let s_strict = snr_db(&pcm, &mp3_decode(&strict).unwrap().samples);
        let s_generic = snr_db(&pcm, &mp3_decode(&generic).unwrap().samples);
        assert!(s_strict > s_generic + 3.0, "{s_strict} vs {s_generic}");
    }

    #[test]
    fn rejects_bad_input_and_corrupt_streams() {
        assert!(mp3_encode(&[0; 10], SR, 3, 128).is_err());
        assert!(mp3_encode(&[0; 10], 100, 1, 128).is_err());
        assert!(mp3_encode(&[0; 10], SR, 1, 4).is_err());
        assert!(mp3_encode(&[0; 11], SR, 2, 128).is_err());
        assert!(mp3_decode(&[]).is_err());
        assert!(mp3_decode(b"RIFF0000000000000000").is_err());
        let enc = mp3_encode(&sine(300.0, 9000.0, 5000), SR, 1, 64).unwrap();
        assert!(mp3_decode(&enc[..enc.len() / 2]).is_err());
        let mut bad = enc.clone();
        bad[4] = 99;
        assert!(mp3_decode(&bad).is_err());
        // A header claiming a gigantic length with no data must not allocate wildly.
        let mut huge = enc[..HEADER_LEN].to_vec();
        huge[10..14].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(mp3_decode(&huge).is_err());
    }
}
