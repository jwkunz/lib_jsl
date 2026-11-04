//! Sensor arrays: element positions with far-field patterns, responses and array patterns.
//!
//! See the book entry [Arrays, scenes and beamformers](https://jwkunz.github.io/lib_jsl/book/em-arrays/beamforming.html).

pub mod sensor_array;

pub use sensor_array::{ArrayElement, ElementPattern, SensorArray};
