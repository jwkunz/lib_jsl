//! Linear Feedback Shift Registers (LFSRs).
//!
//! See the book entry [LFSRs and Gold codes](https://jwkunz.github.io/lib_jsl/book/information/lfsr.html).

pub mod linear_feedback_shift_register;
pub mod maximum_length_polynomials;
pub mod gold_codes;

pub use linear_feedback_shift_register::Lfsr;
pub use maximum_length_polynomials::{
    all_maximum_length_polynomials, is_primitive_polynomial, maximum_length_polynomial,
    maximum_length_polynomials,
};
pub use gold_codes::{gold_codes, gold_correlation_bound};
