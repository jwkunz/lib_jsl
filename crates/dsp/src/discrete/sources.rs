//! Signal sources and generators.

pub mod pulse_train;
pub mod ramp;
pub mod triangle;
pub mod cordic;
pub mod sinusoid;
pub mod square;
pub mod noise;

pub use pulse_train::pulse_train;
pub use ramp::ramp;
pub use triangle::triangle;
pub use cordic::cordic_sin_cos;
pub use sinusoid::sinusoid;
pub use square::square;
pub use noise::noise;
