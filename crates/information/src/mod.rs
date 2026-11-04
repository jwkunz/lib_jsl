//! Information theory and coding for the `lib_jsl` workspace.
//!
//! - [`source_coders`]: lossless compression. The entropy coders [`huffman`], [`shannon_fano`]
//!   and [`arithmetic`] and the dictionary coders [`lz77`], [`lz78`] and [`lzw`] share the
//!   [`source_coder::SourceCoder`] trait; Gray codes are here too.
//! - [`lfsr`]: linear feedback shift registers, maximum-length (primitive) polynomials and Gold
//!   codes.
//! - [`operators`]: entropy and related measures, the CRC-32 checksums and SHA-256.
//! - [`forward_error_correction`]: error-correcting codes.
//!
//! The source coder modules and the [`source_coder`] trait are also re-exported at the crate root.
//!
//! See the book entry [Information theory](https://jwkunz.github.io/lib_jsl/book/information/index.html).

pub mod forward_error_correction;
pub mod lfsr;
pub mod operators;
pub mod source_coders;

pub use source_coders::arithmetic;
pub use source_coders::huffman;
pub use source_coders::lz77;
pub use source_coders::lz78;
pub use source_coders::lzw;
pub use source_coders::shannon_fano;
pub use source_coders::source_coder;
