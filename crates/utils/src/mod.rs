//! General utilities: terminal plots, unit conversion and floating-point helpers.
//!
//! See the book entry [Shared types, constants and utilities](https://jwkunz.github.io/lib_jsl/book/foundations/core-utils.html).

pub mod terminal_plot;
pub mod unit_conversion;
pub mod machine_epsilon;

pub use machine_epsilon::{machine_epsilon, ulp};
pub use terminal_plot::generate_terminal_plot;
pub use unit_conversion::unit_conversion;
