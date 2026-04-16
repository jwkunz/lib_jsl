//! Information theory routines for the `lib_jsl` workspace.
//!
//! The crate currently exposes three classic dictionary-based compression
//! families:
//!
//! - [`lz77`] for the sliding-window back-reference variant
//! - [`lz78`] for explicit dictionary entry emission
//! - [`lzw`] for the Welch-style evolving code table

pub mod lz77;
pub mod lz78;
pub mod lzw;
