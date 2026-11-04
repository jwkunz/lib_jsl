//! Implementations and shared abstractions for the source coders in this crate.
//!
//! Keeping the codecs together under one submodule makes it easier to navigate
//! the growing set of entropy and dictionary coders while still allowing the
//! crate root to re-export the common entry points.
//!
//! See the book entry [Entropy coding](https://jwkunz.github.io/lib_jsl/book/information/entropy-coding.html).

pub mod arithmetic;
pub mod gray_codes;
pub mod huffman;
pub mod lz77;
pub mod lz78;
pub mod lzw;
pub mod shannon_fano;
pub mod source_coder;

pub use gray_codes::{gray_code, gray_code_sequence, gray_decode};
