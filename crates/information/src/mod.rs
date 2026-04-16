//! Information theory routines for the `lib_jsl` workspace.
//!
//! The crate currently exposes three classic dictionary-based compression
//! families.
//!
//! - [`arithmetic`] for entropy coding with a static arithmetic interval model
//! - [`huffman`] for entropy coding with a prefix-free binary tree
//! - [`lz77`] for the sliding-window back-reference variant
//! - [`lz78`] for explicit dictionary entry emission
//! - [`lzw`] for the Welch-style evolving code table
//! - [`shannon_fano`] for top-down probability splitting into prefix codes
//!
//! The implementation files now live under [`source_coders`], but the
//! individual codec modules and the shared [`source_coder`] trait layer are
//! re-exported at the crate root for convenience.

pub mod source_coders;

pub use source_coders::arithmetic;
pub use source_coders::huffman;
pub use source_coders::lz77;
pub use source_coders::lz78;
pub use source_coders::lzw;
pub use source_coders::shannon_fano;
pub use source_coders::source_coder;
