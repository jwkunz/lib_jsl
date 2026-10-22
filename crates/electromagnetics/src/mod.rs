//! Electromagnetics, antennas, and field computations.

pub mod antenna_models {
    pub struct HalfWaveDipole {
        pub length: f64,
    }

    pub struct Dish {
        pub diameter: f64,
    }

    pub struct HertzianDipole;
    pub struct MagneticLoop;
    pub struct PatchAntenna;
}

pub mod fields {
    pub struct FieldContainer;
}

pub mod algorithms {
    pub fn overlap_integral() {
        todo!("overlap_integral is not yet implemented")
    }

    pub fn cfm_beamformer() {
        todo!("cfm_beamformer is not yet implemented")
    }

    pub fn hadamard_beamformer() {
        todo!("hadamard_beamformer is not yet implemented")
    }

    pub fn lcmv_beamformer() {
        todo!("lcmv_beamformer is not yet implemented")
    }
}
