//! Information theory routines for the `lib_jsl` workspace.
//!
//! The crate currently exposes three classic dictionary-based compression
//! families:
//!
//! - [`huffman`] for entropy coding with a prefix-free binary tree
//! - [`lz77`] for the sliding-window back-reference variant
//! - [`lz78`] for explicit dictionary entry emission
//! - [`lzw`] for the Welch-style evolving code table
//!
//! The [`source_coder`] module provides a shared trait wrapper so callers can
//! use any of these codecs through one common interface.

pub mod huffman;
pub mod lz77;
pub mod lz78;
pub mod source_coder;
pub mod lzw;
