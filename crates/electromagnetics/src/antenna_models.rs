//! Antenna models.

pub mod half_wave_dipole;
pub mod dish;
pub mod hertzian_dipole;
pub mod magnetic_loop;
pub mod patch_antenna;

pub use half_wave_dipole::HalfWaveDipole;
pub use dish::Dish;
pub use hertzian_dipole::HertzianDipole;
pub use magnetic_loop::MagneticLoop;
pub use patch_antenna::PatchAntenna;
