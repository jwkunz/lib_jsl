//! Forward Error Correction (FEC).
//!
//! See the book entry [Error-correcting codes](https://jwkunz.github.io/lib_jsl/book/coding/index.html).

pub mod ldpc;
pub mod polar;
pub mod hamming;
pub mod reed_solomon;
pub mod convolutional;

pub use hamming::{HammingCode, HammingDecoded, HammingStatus};
pub use reed_solomon::ReedSolomon;
pub use convolutional::ConvolutionalCode;
pub use polar::PolarCode;
pub use ldpc::{LdpcCode, LdpcDecoded};
