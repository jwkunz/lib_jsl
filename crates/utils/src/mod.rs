//! General utilities, formatting, and file I/O helpers.

pub mod terminal_plot;
pub mod unit_conversion;
pub mod wav_files;
pub mod jpeg_files;
pub mod machine_epsilon;

pub use terminal_plot::generate_terminal_plot;
pub use unit_conversion::unit_conversion;
pub use wav_files::read_wav_file;
pub use jpeg_files::read_jpeg_file;
pub use machine_epsilon::machine_epsilon;
