//! Implementations and shared abstractions for the source coders in this crate.
//!
//! Keeping the codecs together under one submodule makes it easier to navigate
//! the growing set of entropy and dictionary coders while still allowing the
//! crate root to re-export the common entry points.

pub mod arithmetic;
pub mod gray_codes;
pub mod huffman;
pub mod jpeg;
pub mod lz77;
pub mod lz78;
pub mod lzw;
pub mod mp3;
pub mod shannon_fano;
pub mod source_coder;

pub use gray_codes::{gray_code, gray_code_sequence, gray_decode};
pub use jpeg::{jpeg_decode, jpeg_encode, JpegImage};
pub use mp3::{mp3_decode, mp3_encode, mp3_encode_with_model, GenericPsychoacousticModel, Mp3Audio, PsychoacousticModel};
