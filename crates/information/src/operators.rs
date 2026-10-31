//! Information operators and hashing/checksums.

pub mod sha256;
pub mod crc32;
pub mod entropy;

pub use sha256::{sha256, to_hex, Sha256};
pub use crc32::{crc32, crc32c, Crc32};
pub use entropy::{cross_entropy, entropy, entropy_from_probabilities, kl_divergence, mutual_information};
