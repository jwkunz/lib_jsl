//! Signal sources and generators.

pub mod nco;
pub mod pulse_train;
pub mod ramp;
pub mod triangle;
pub mod cordic;
pub mod sinusoid;
pub mod square;
pub mod noise;

pub use nco::{Nco, NcoSource, Phase, Waveform};
pub use pulse_train::{pulse_train, PulseTrainWave};
pub use ramp::{ramp, RampWave};
pub use triangle::{triangle, TriangleWave};
pub use cordic::{angle_from_radians, angle_from_turns, angle_to_radians, cordic_sin_cos, Cordic};
pub use sinusoid::{sinusoid, SineKernel, SineWave};
pub use square::{square, SquareWave};
pub use noise::{noise, NoiseKind, NoiseWave};
