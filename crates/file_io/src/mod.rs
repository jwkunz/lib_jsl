//! File formats: codecs and file readers/writers for audio and image data.
//!
//! - [`wav`] reads and writes RIFF/WAVE audio (PCM and float).
//! - [`jpeg`] is a baseline JPEG codec, and [`jpeg_files`] reads and writes `.jpg` files with it.
//! - [`csv`] reads and writes delimited text tables, and [`npy`] NumPy `.npy` arrays.
//! - [`png`] and [`ppm`] are lossless image codecs sharing [`png::RasterImage`]; [`deflate`]
//!   provides the zlib compression PNG and MAT-files need.
//! - [`matrix_market`], [`matlab`], [`harwell_boeing`], [`netcdf`], [`fortran`] and [`arff`]
//!   exchange scientific data with the programs `scipy.io` talks to.
//!
//! See the book entry [Foundations: file formats](https://jwkunz.github.io/lib_jsl/book/foundations/index.html).

pub mod csv;
pub mod deflate;
pub mod jpeg;
pub mod jpeg_files;
pub mod npy;
pub mod png;
pub mod ppm;
pub mod wav;
pub mod matrix_market;
pub mod matlab;
pub mod harwell_boeing;
pub mod netcdf;
pub mod fortran;
pub mod arff;

pub use jpeg::{JpegImage, jpeg_decode, jpeg_encode};
pub use jpeg_files::{read_jpeg_file, write_jpeg_file};
pub use png::{RasterImage, png_decode, png_encode, read_png_file, write_png_file};
pub use ppm::{ppm_decode, ppm_encode, read_ppm_file, write_ppm_file};
pub use csv::{CsvTable, format_csv, format_csv_f64, parse_csv, read_csv_file, write_csv_file};
pub use npy::{NpyArray, NpyData, npy_decode, npy_encode, read_npy_file, write_npy_file};
pub use wav::{WavAudio, WavSampleFormat, encode_wav, parse_wav, read_wav_file, write_wav_file};
pub use matrix_market::{MatrixMarket, MmField, MmFormat, MmInfo, MmSymmetry, mm_decode, mm_encode, mm_info, read_mm_file, write_mm_file};
pub use matlab::{MatClass, MatValue, mat_decode, mat_encode, mat_whos, read_mat_file, write_mat_file};
pub use harwell_boeing::{HarwellBoeing, hb_decode, hb_encode, read_hb_file, write_hb_file};
pub use netcdf::{NcValue, NcVariable, NetCdf, netcdf_decode, netcdf_encode, read_netcdf_file, write_netcdf_file};
pub use fortran::{fortran_records_decode, fortran_records_encode, read_fortran_file, record_from_f32, record_from_f64, record_from_i32, record_to_f32, record_to_f64, record_to_i32, write_fortran_file};
pub use arff::{Arff, ArffType, ArffValue, arff_decode, arff_encode, read_arff_file};
