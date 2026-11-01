//! File formats: codecs and file readers/writers for audio and image data.
//!
//! - [`wav`] reads and writes RIFF/WAVE audio (PCM and float).
//! - [`mp3`] is a small MPEG-1 Layer III encoder/decoder (with [`mp3_tables`] holding the
//!   standard constants it needs).
//! - [`jpeg`] is a baseline JPEG codec, and [`jpeg_files`] reads and writes `.jpg` files with it.
//! - [`csv`] reads and writes delimited text tables, and [`npy`] NumPy `.npy` arrays.
//! - [`png`] and [`ppm`] are lossless image codecs sharing [`png::RasterImage`]; [`deflate`]
//!   provides the zlib compression PNG needs.

pub mod csv;
pub mod deflate;
pub mod jpeg;
pub mod jpeg_files;
pub mod mp3;
pub mod mp3_tables;
pub mod npy;
pub mod png;
pub mod ppm;
pub mod wav;

pub use jpeg::{JpegImage, jpeg_decode, jpeg_encode};
pub use jpeg_files::{read_jpeg_file, write_jpeg_file};
pub use png::{RasterImage, png_decode, png_encode, read_png_file, write_png_file};
pub use ppm::{ppm_decode, ppm_encode, read_ppm_file, write_ppm_file};
pub use csv::{CsvTable, format_csv, format_csv_f64, parse_csv, read_csv_file, write_csv_file};
pub use npy::{NpyArray, NpyData, npy_decode, npy_encode, read_npy_file, write_npy_file};
pub use mp3::{GenericPsychoacousticModel, Mp3Audio, PsychoacousticModel, mp3_decode, mp3_encode, mp3_encode_with_model};
pub use wav::{WavAudio, WavSampleFormat, encode_wav, parse_wav, read_wav_file, write_wav_file};
