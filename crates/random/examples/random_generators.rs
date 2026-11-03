//! Uniform pseudo-random generators: seeding, independent streams and a histogram check.
//!
//! Run with `cargo run -p lib_jsl_random --example random_generators`.

use lib_jsl_random::histogram::Histogram;
use lib_jsl_random::uniform_generator::{UniformGenerator, UniformRNG};

fn main() {
    // The same seed gives the same sequence: simulations are reproducible.
    let mut a = UniformRNG::from_seed(42);
    let mut b = UniformRNG::from_seed(42);
    println!("same seed, same numbers: {}", (0..1000).all(|_| a.next_u64() == b.next_u64()));

    // Independent streams for parallel workers: jump each copy 2¹²⁸ steps ahead of the last.
    let mut streams = vec![UniformRNG::from_seed(7)];
    for _ in 1..4 {
        let mut next = streams.last().unwrap().clone();
        next.jump();
        streams.push(next);
    }
    let firsts: Vec<f64> = streams.iter_mut().map(|s| s.next_f64()).collect();
    println!("first value of each of 4 streams: {firsts:.4?}");

    // A histogram of 100 000 uniforms: each of 10 bins should hold about 10%.
    let mut h = Histogram::new(0.0, 1.0, 10);
    let mut rng = UniformRNG::from_seed(1);
    for _ in 0..100_000 {
        h.count(rng.next_f64()).unwrap();
    }
    let fractions = h.get_counts_normalized().unwrap();
    println!("bin fractions: {fractions:.4?}");
}
