//! Implementations and shared abstractions for the source coders in this crate.
//!
//! Keeping the codecs together under one submodule makes it easier to navigate
//! the growing set of entropy and dictionary coders while still allowing the
//! crate root to re-export the common entry points.

pub mod arithmetic;
pub mod huffman;
pub mod lz77;
pub mod lz78;
pub mod lzw;
pub mod shannon_fano;
pub mod source_coder;
