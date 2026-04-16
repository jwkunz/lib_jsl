//! Shared trait wrappers for the source-coding implementations in this crate.
//!
//! The individual modules such as [`crate::lz77`], [`crate::lz78`], and
//! [`crate::lzw`] intentionally expose plain function-based APIs because that
//! keeps each algorithm easy to read in isolation. This module adds a thin
//! trait layer on top so generic code can work with any of those codecs
//! through one unified interface.
//!
//! # Design
//!
//! The trait uses associated functions rather than instance methods because the
//! current codecs are stateless. Each implementation is represented by a
//! zero-sized wrapper type that simply forwards to the existing module-level
//! functions.

use lib_jsl_core::ErrorsJSL;

use crate::{huffman, lz77, lz78, lzw};

/// Common interface for byte-oriented source coders in this crate.
///
/// A source coder consumes raw bytes and either:
///
/// - produces a compressed byte stream with [`try_compress`], or
/// - reconstructs the original bytes with [`try_decompress`]
///
/// The methods are fallible so callers can handle malformed compressed data or
/// future codec-specific compression failures through the same `Result` type.
pub trait SourceCoder {
    /// Compress a raw byte slice into a codec-specific byte stream.
    fn try_compress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL>;

    /// Decompress a codec-specific byte stream back into its original bytes.
    fn try_decompress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL>;
}

/// Zero-sized wrapper for the sliding-window LZ77 implementation.
pub struct Lz77SourceCoder;

impl SourceCoder for Lz77SourceCoder {
    fn try_compress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
        // The current LZ77 encoder cannot fail, so the trait wrapper simply
        // lifts the existing byte vector into `Ok(...)`.
        Ok(lz77::lz77_compress(input))
    }

    fn try_decompress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
        lz77::try_lz77_decompress(input)
    }
}

/// Zero-sized wrapper for the explicit-dictionary LZ78 implementation.
pub struct Lz78SourceCoder;

impl SourceCoder for Lz78SourceCoder {
    fn try_compress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
        Ok(lz78::lz78_compress(input))
    }

    fn try_decompress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
        lz78::try_lz78_decompress(input)
    }
}

/// Zero-sized wrapper for the Welch-style LZW implementation.
pub struct LzwSourceCoder;

impl SourceCoder for LzwSourceCoder {
    fn try_compress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
        Ok(lzw::lzw_compress(input))
    }

    fn try_decompress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
        lzw::try_lzw_decompress(input)
    }
}

/// Zero-sized wrapper for the Huffman implementation.
pub struct HuffmanSourceCoder;

impl SourceCoder for HuffmanSourceCoder {
    fn try_compress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
        huffman::try_huffman_compress(input)
    }

    fn try_decompress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
        huffman::try_huffman_decompress(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_round_trip<T: SourceCoder>(payload: &[u8]) {
        let compressed = T::try_compress(payload).expect("compression should succeed");
        let decompressed = T::try_decompress(&compressed).expect("decompression should succeed");
        assert_eq!(decompressed, payload);
    }

    #[test]
    fn lz77_trait_wrapper_round_trip() {
        assert_round_trip::<Lz77SourceCoder>(b"TOBEORNOTTOBEORTOBEORNOT");
    }

    #[test]
    fn lz78_trait_wrapper_round_trip() {
        assert_round_trip::<Lz78SourceCoder>(b"ABAABABAABBBBBBBBB");
    }

    #[test]
    fn lzw_trait_wrapper_round_trip() {
        assert_round_trip::<LzwSourceCoder>(b"TOBEORNOTTOBEORTOBEORNOT");
    }

    #[test]
    fn huffman_trait_wrapper_round_trip() {
        assert_round_trip::<HuffmanSourceCoder>(b"this is an example of a huffman tree");
    }

    #[test]
    fn trait_wrappers_support_large_asset_round_trip() {
        let payload = include_bytes!("../test_assets/kjv.txt");

        assert_round_trip::<HuffmanSourceCoder>(payload);
        assert_round_trip::<Lz77SourceCoder>(payload);
        assert_round_trip::<Lz78SourceCoder>(payload);
        assert_round_trip::<LzwSourceCoder>(payload);
    }
}
