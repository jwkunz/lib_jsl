//! Linear feedback shift register (LFSR).

pub struct Lfsr {
    pub state: u64,
    pub taps: u64,
}
