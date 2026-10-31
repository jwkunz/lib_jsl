//! Linear feedback shift register (LFSR).

/// A Fibonacci-style linear feedback shift register of up to 64 stages.
///
/// The register holds the bits `s_t ... s_{t+n-1}` of a sequence in the low `width` bits of
/// `state` (bit 0 is the oldest, `s_t`). Each step emits `s_t` and shifts in the new bit
/// `s_{t+n} = parity(state & taps)`, i.e. `taps` has bit `k` set when `s_{t+k}` participates in
/// the recurrence. For the characteristic polynomial `x^n + c_{n-1} x^{n-1} + ... + c_0`, the
/// `taps` mask is exactly the polynomial with its `x^n` term removed (see
/// [`Lfsr::from_polynomial`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lfsr {
    /// Current register contents (low `width` bits).
    pub state: u64,
    /// Feedback tap mask.
    pub taps: u64,
    /// Number of stages (1..=64).
    pub width: u32,
}

impl Lfsr {
    /// Creates a register. Panics if `width` is 0 or greater than 64.
    pub fn new(width: u32, taps: u64, seed: u64) -> Self {
        assert!((1..=64).contains(&width), "LFSR width must be between 1 and 64");
        let m = Self::mask(width);
        Lfsr { state: seed & m, taps: taps & m, width }
    }

    /// Creates a register from a characteristic polynomial given as a bit mask (bit `k` is the
    /// coefficient of `x^k`; the leading `x^n` bit determines the width). E.g. `x^3 + x + 1` is
    /// `0b1011`.
    pub fn from_polynomial(poly: u64, seed: u64) -> Self {
        assert!(poly > 1, "polynomial must have degree >= 1");
        let width = 63 - poly.leading_zeros() as u32;
        Self::new(width, poly & !(1u64 << width), seed)
    }

    fn mask(width: u32) -> u64 {
        if width == 64 { u64::MAX } else { (1u64 << width) - 1 }
    }

    /// Advances the register one step and returns the output bit.
    pub fn step(&mut self) -> u8 {
        let out = (self.state & 1) as u8;
        let fb = ((self.state & self.taps).count_ones() & 1) as u64;
        self.state = (self.state >> 1) | (fb << (self.width - 1));
        out
    }

    /// Returns the next `n` output bits.
    pub fn bits(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.step()).collect()
    }

    /// Returns the sequence period starting from the current state (the number of steps until the
    /// state repeats), or `None` if the state never returns to itself within `limit` steps (as
    /// happens for the all-zero state's trivial period of 1, which is reported as `Some(1)`).
    pub fn period(&self, limit: u64) -> Option<u64> {
        let mut copy = self.clone();
        let start = copy.state;
        for i in 1..=limit {
            copy.step();
            if copy.state == start {
                return Some(i);
            }
        }
        None
    }
}

impl Iterator for Lfsr {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        Some(self.step())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x3_plus_x_plus_1_sequence() {
        // s_{t+3} = s_t + s_{t+1}; seed s0..s2 = 1,0,0 -> 1 0 0 1 0 1 1 | repeat
        let mut r = Lfsr::from_polynomial(0b1011, 0b001);
        assert_eq!(r.bits(14), vec![1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1]);
    }

    #[test]
    fn period_and_state_space() {
        let r = Lfsr::from_polynomial(0b1011, 1);
        assert_eq!(r.period(100), Some(7));
        // x^4 + x + 1 is primitive: period 15; x^4 + x^2 + 1 = (x^2+x+1)^2 is not.
        assert_eq!(Lfsr::from_polynomial(0b10011, 1).period(100), Some(15));
        assert!(Lfsr::from_polynomial(0b10101, 1).period(100).unwrap() < 15);
        // All-zero state is a fixed point.
        assert_eq!(Lfsr::from_polynomial(0b1011, 0).period(10), Some(1));
    }

    #[test]
    fn balance_of_m_sequence() {
        // An m-sequence has 2^(n-1) ones and 2^(n-1) - 1 zeros per period.
        let ones: u32 = Lfsr::from_polynomial(0b100101, 1).bits(31).iter().map(|&b| b as u32).sum();
        assert_eq!(ones, 16);
    }

    #[test]
    fn iterator_and_width_64() {
        let r = Lfsr::new(64, (1 << 63) | 1, 1);
        assert_eq!(r.take(5).count(), 5);
    }
}
