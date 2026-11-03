//! SplitMix64, used to seed larger generators.
//!
//! See the book entry [Pseudo-random number generators](https://jwkunz.github.io/lib_jsl/book/random/generators.html#seeding-with-splitmix64).

/// SplitMix64 — used only for seeding.
/// Prevents zero seeds
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    // ANCHOR: splitmix
    pub fn next_u64(&mut self) -> u64 {
        let mut z = {
            self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
            self.state
        };

        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        let result = z ^ (z >> 31);
        if result == 0{
            self.next_u64()
        }else{
            result
        }
    }
    // ANCHOR_END: splitmix
}
