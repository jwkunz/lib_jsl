//! Writing and reading WAV files, and the MP3-style perceptual codec.

use lib_jsl_file_io::{WavAudio, WavSampleFormat, encode_wav, mp3_decode, mp3_encode, parse_wav, read_wav_file, write_wav_file};

fn main() {
    // Half a second of a 440 Hz tone in stereo, the right channel quieter.
    let rate = 44_100;
    let frames = rate as usize / 2;
    let samples: Vec<f32> = (0..frames)
        .flat_map(|i| {
            let s = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin();
            [0.8 * s, 0.4 * s]
        })
        .collect();
    let audio = WavAudio { sample_rate: rate, channels: 2, samples };
    for format in [WavSampleFormat::Pcm8, WavSampleFormat::Pcm16, WavSampleFormat::Pcm24, WavSampleFormat::Float32] {
        let bytes = encode_wav(&audio, format).unwrap();
        let back = parse_wav(&bytes).unwrap();
        let error = audio.samples.iter().zip(&back.samples).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("{format:?}: {} bytes, largest error {error:.1e}", bytes.len());
    }
    let path = std::env::temp_dir().join("lib_jsl_tone.wav");
    write_wav_file(&path, &audio, WavSampleFormat::Pcm16).unwrap();
    let read = read_wav_file(&path).unwrap();
    println!("file: {} Hz, {} channels, {} frames; left channel peak {:.3}", read.sample_rate, read.channels, read.frames(), read.channel(0).iter().fold(0.0f32, |m, v| m.max(v.abs())));

    // The perceptual codec takes 16-bit PCM.
    let pcm: Vec<i16> = audio.samples.iter().map(|&v| (v * 32767.0) as i16).collect();
    for kbps in [64, 128] {
        let coded = mp3_encode(&pcm, rate, 2, kbps).unwrap();
        let decoded = mp3_decode(&coded).unwrap();
        let (signal, noise) = pcm.iter().zip(&decoded.samples).fold((0.0, 0.0), |(s, n), (&a, &b)| (s + (a as f64).powi(2), n + (a as f64 - b as f64).powi(2)));
        println!("{kbps} kbit/s target: {} bytes ({:.0} kbit/s), SNR {:.1} dB", coded.len(), coded.len() as f64 * 8.0 / 0.5 / 1000.0, 10.0 * (signal / noise).log10());
    }
}
