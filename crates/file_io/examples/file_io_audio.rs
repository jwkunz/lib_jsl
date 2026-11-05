//! Writing and reading WAV files in each sample format.

use lib_jsl_file_io::{WavAudio, WavSampleFormat, encode_wav, parse_wav, read_wav_file, write_wav_file};

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
    for format in [WavSampleFormat::Pcm8, WavSampleFormat::Pcm16, WavSampleFormat::Pcm24, WavSampleFormat::Pcm32, WavSampleFormat::Float32] {
        let bytes = encode_wav(&audio, format).unwrap();
        let back = parse_wav(&bytes).unwrap();
        let error = audio.samples.iter().zip(&back.samples).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("{format:?}: {} bytes, largest error {error:.1e}", bytes.len());
    }
    let path = std::env::temp_dir().join("lib_jsl_tone.wav");
    write_wav_file(&path, &audio, WavSampleFormat::Pcm16).unwrap();
    let read = read_wav_file(&path).unwrap();
    println!("file: {} Hz, {} channels, {} frames; left channel peak {:.3}", read.sample_rate, read.channels, read.frames(), read.channel(0).iter().fold(0.0f32, |m, v| m.max(v.abs())));
}
