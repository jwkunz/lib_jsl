//! Reading and writing RIFF/WAVE audio files.
//!
//! See the book entry [Audio files](https://jwkunz.github.io/lib_jsl/book/foundations/audio.html#wav-files).

use lib_jsl_core::ErrorsJSL;
use std::path::Path;

/// Sample encodings that can be written (and are understood when reading).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavSampleFormat {
    /// 8-bit unsigned PCM.
    Pcm8,
    /// 16-bit signed PCM.
    Pcm16,
    /// 24-bit signed PCM.
    Pcm24,
    /// 32-bit signed PCM.
    Pcm32,
    /// 32-bit IEEE float.
    Float32,
    /// 64-bit IEEE float.
    Float64,
}

impl WavSampleFormat {
    fn bits(self) -> u16 {
        match self {
            WavSampleFormat::Pcm8 => 8,
            WavSampleFormat::Pcm16 => 16,
            WavSampleFormat::Pcm24 => 24,
            WavSampleFormat::Pcm32 | WavSampleFormat::Float32 => 32,
            WavSampleFormat::Float64 => 64,
        }
    }
    fn is_float(self) -> bool {
        matches!(self, WavSampleFormat::Float32 | WavSampleFormat::Float64)
    }
}

/// Decoded audio: interleaved samples normalised to `[-1, 1)` (integer formats) or as stored (float).
#[derive(Debug, Clone, PartialEq)]
pub struct WavAudio {
    pub sample_rate: u32,
    pub channels: u16,
    /// Frame-interleaved samples (`frames * channels` values).
    pub samples: Vec<f32>,
}

impl WavAudio {
    /// Number of sample frames.
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels.max(1) as usize
    }

    /// Samples of one channel.
    pub fn channel(&self, index: usize) -> Vec<f32> {
        self.samples.iter().skip(index).step_by(self.channels as usize).copied().collect()
    }
}

const ERR_FORMAT: ErrorsJSL = ErrorsJSL::InvalidInputRange("Malformed or unsupported WAV file.");

fn u16_at(d: &[u8], p: usize) -> Result<u16, ErrorsJSL> {
    d.get(p..p + 2).map(|b| u16::from_le_bytes([b[0], b[1]])).ok_or(ERR_FORMAT)
}
fn u32_at(d: &[u8], p: usize) -> Result<u32, ErrorsJSL> {
    d.get(p..p + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok_or(ERR_FORMAT)
}

/// Parses a WAV byte stream (PCM 8/16/24/32-bit, IEEE float 32/64-bit, plain or extensible headers).
// ANCHOR: parse_wav
pub fn parse_wav(data: &[u8]) -> Result<WavAudio, ErrorsJSL> {
    if data.get(0..4) != Some(b"RIFF") || data.get(8..12) != Some(b"WAVE") {
        return Err(ERR_FORMAT);
    }
    let mut pos = 12;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (tag, channels, rate, bits)
    let mut payload: Option<&[u8]> = None;
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let size = u32_at(data, pos + 4)? as usize;
        let body_end = (pos + 8 + size).min(data.len()); // tolerate a truncated final chunk
        let body = &data[pos + 8..body_end];
        match id {
            b"fmt " => {
                let mut tag = u16_at(body, 0)?;
                let channels = u16_at(body, 2)?;
                let rate = u32_at(body, 4)?;
                let bits = u16_at(body, 14)?;
                if tag == 0xFFFE {
                    tag = u16_at(body, 24)?; // first two bytes of the sub-format GUID
                }
                fmt = Some((tag, channels, rate, bits));
            }
            b"data" => payload = Some(body),
            _ => {}
        }
        pos += 8 + size + (size & 1);
    }
    let (tag, channels, sample_rate, bits) = fmt.ok_or(ERR_FORMAT)?;
    let body = payload.ok_or(ERR_FORMAT)?;
    if channels == 0 {
        return Err(ERR_FORMAT);
    }
    let bytes = (bits / 8) as usize;
    let samples: Vec<f32> = match (tag, bits) {
        (1, 8) => body.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect(),
        (1, 16) => body.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b) as f32 / 32768.0).collect(),
        (1, 24) => body
            .as_chunks::<3>()
            .0
            .iter()
            .map(|b| (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 8_388_608.0)
            .collect(),
        (1, 32) => body.as_chunks::<4>().0.iter().map(|b| (i32::from_le_bytes(*b) as f64 / 2_147_483_648.0) as f32).collect(),
        (3, 32) => body.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
        (3, 64) => body.as_chunks::<8>().0.iter().map(|b| f64::from_le_bytes(*b) as f32).collect(),
        _ => return Err(ErrorsJSL::InvalidInputRange("Unsupported WAV sample format.")),
    };
    debug_assert!(bytes > 0);
    let whole = samples.len() / channels as usize * channels as usize;
    Ok(WavAudio { sample_rate, channels, samples: samples[..whole].to_vec() })
}
// ANCHOR_END: parse_wav

/// Serialises audio as a canonical WAV file in the requested sample format. Samples outside
/// `[-1, 1)` are clipped for the integer formats (scale is 2^(bits-1)).
pub fn encode_wav(audio: &WavAudio, format: WavSampleFormat) -> Result<Vec<u8>, ErrorsJSL> {
    if audio.channels == 0 || !audio.samples.len().is_multiple_of(audio.channels as usize) {
        return Err(ErrorsJSL::InvalidInputRange("Sample count must be a multiple of the channel count."));
    }
    let bits = format.bits();
    let block = audio.channels * bits / 8;
    let data_len = audio.samples.len() * (bits / 8) as usize;
    if data_len > u32::MAX as usize - 44 {
        return Err(ErrorsJSL::InvalidInputRange("Audio too large for a WAV file."));
    }
    let mut out = Vec::with_capacity(44 + data_len + 1);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len + (data_len & 1)) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&(if format.is_float() { 3u16 } else { 1u16 }).to_le_bytes());
    out.extend_from_slice(&audio.channels.to_le_bytes());
    out.extend_from_slice(&audio.sample_rate.to_le_bytes());
    out.extend_from_slice(&(audio.sample_rate * block as u32).to_le_bytes());
    out.extend_from_slice(&block.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for &s in &audio.samples {
        let c = s.clamp(-1.0, 1.0) as f64;
        match format {
            WavSampleFormat::Pcm8 => out.push((c * 128.0 + 128.0).round().clamp(0.0, 255.0) as u8),
            WavSampleFormat::Pcm16 => out.extend_from_slice(&((c * 32768.0).round().clamp(-32768.0, 32767.0) as i16).to_le_bytes()),
            WavSampleFormat::Pcm24 => out.extend_from_slice(&((c * 8_388_608.0).round().clamp(-8_388_608.0, 8_388_607.0) as i32).to_le_bytes()[..3]),
            WavSampleFormat::Pcm32 => out.extend_from_slice(&((c * 2_147_483_648.0).round().clamp(-2_147_483_648.0, 2_147_483_647.0) as i32).to_le_bytes()),
            WavSampleFormat::Float32 => out.extend_from_slice(&s.to_le_bytes()),
            WavSampleFormat::Float64 => out.extend_from_slice(&(s as f64).to_le_bytes()),
        }
    }
    if data_len & 1 == 1 {
        out.push(0);
    }
    Ok(out)
}

/// Reads a WAV file from disk.
pub fn read_wav_file(path: impl AsRef<Path>) -> Result<WavAudio, ErrorsJSL> {
    let data = std::fs::read(path).map_err(|_| ErrorsJSL::RuntimeError("Could not read the WAV file."))?;
    parse_wav(&data)
}

/// Writes a WAV file to disk.
pub fn write_wav_file(path: impl AsRef<Path>, audio: &WavAudio, format: WavSampleFormat) -> Result<(), ErrorsJSL> {
    std::fs::write(path, encode_wav(audio, format)?).map_err(|_| ErrorsJSL::RuntimeError("Could not write the WAV file."))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(frames: usize, channels: u16) -> WavAudio {
        let samples = (0..frames * channels as usize)
            .map(|i| 0.8 * ((i / channels as usize) as f32 * 0.05 * (1 + i as u32 % channels as u32) as f32).sin())
            .collect();
        WavAudio { sample_rate: 44_100, channels, samples }
    }

    #[test]
    fn round_trips_every_format_within_quantisation() {
        let a = sine(500, 2);
        for (fmt, tol) in [
            (WavSampleFormat::Pcm8, 1.0 / 128.0),
            (WavSampleFormat::Pcm16, 1.0 / 32768.0),
            (WavSampleFormat::Pcm24, 1.0 / 8_388_608.0),
            (WavSampleFormat::Pcm32, 1e-6),
            (WavSampleFormat::Float32, 0.0),
            (WavSampleFormat::Float64, 0.0),
        ] {
            let bytes = encode_wav(&a, fmt).unwrap();
            let b = parse_wav(&bytes).unwrap();
            assert_eq!((b.sample_rate, b.channels, b.samples.len()), (44_100, 2, a.samples.len()), "{fmt:?}");
            let worst = a.samples.iter().zip(&b.samples).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            assert!(worst <= tol, "{fmt:?} worst {worst}");
        }
    }

    #[test]
    fn header_layout_and_channel_access() {
        let a = WavAudio { sample_rate: 8000, channels: 2, samples: vec![0.5, -0.5, 0.25, -0.25] };
        let bytes = encode_wav(&a, WavSampleFormat::Pcm16).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..16], b"WAVEfmt ");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize, bytes.len() - 8);
        let b = parse_wav(&bytes).unwrap();
        assert_eq!(b.frames(), 2);
        assert_eq!(b.channel(1).len(), 2);
        assert!((b.channel(1)[0] + 0.5).abs() < 1e-4);
    }

    #[test]
    fn skips_unknown_chunks_and_rejects_garbage() {
        let a = sine(10, 1);
        let mut bytes = encode_wav(&a, WavSampleFormat::Pcm16).unwrap();
        // Insert an odd-sized LIST chunk (with pad byte) before "data".
        let data_at = bytes.windows(4).position(|w| w == b"data").unwrap();
        let mut extra = b"LIST".to_vec();
        extra.extend_from_slice(&3u32.to_le_bytes());
        extra.extend_from_slice(&[1, 2, 3, 0]);
        bytes.splice(data_at..data_at, extra);
        assert_eq!(parse_wav(&bytes).unwrap().samples.len(), 10);
        assert!(parse_wav(b"not a wav file at all").is_err());
        assert!(parse_wav(&bytes[..30]).is_err());
        assert!(encode_wav(&WavAudio { sample_rate: 1, channels: 2, samples: vec![0.0; 3] }, WavSampleFormat::Pcm16).is_err());
    }

    #[test]
    fn file_round_trip() {
        let path = std::env::temp_dir().join(format!("lib_jsl_wav_test_{}.wav", std::process::id()));
        let a = sine(100, 1);
        write_wav_file(&path, &a, WavSampleFormat::Float32).unwrap();
        assert_eq!(read_wav_file(&path).unwrap(), a);
        std::fs::remove_file(&path).ok();
        assert!(read_wav_file(&path).is_err());
    }
}
