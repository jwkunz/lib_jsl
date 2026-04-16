//! Information theory routines for the `lib_jsl` workspace.
//!
//! The crate currently exposes three classic dictionary-based compression
//! families:
//!
//! - [`arithmetic`] for entropy coding with a static arithmetic interval model
//! - [`huffman`] for entropy coding with a prefix-free binary tree
//! - [`lz77`] for the sliding-window back-reference variant
//! - [`lz78`] for explicit dictionary entry emission
//! - [`lzw`] for the Welch-style evolving code table
//! - [`shannon_fano`] for top-down probability splitting into prefix codes
//!
//! The [`source_coder`] module provides a shared trait wrapper so callers can
//! use any of these codecs through one common interface.

pub mod arithmetic;
pub mod huffman;
pub mod lz77;
pub mod lz78;
pub mod shannon_fano;
pub mod source_coder;
pub mod lzw;
