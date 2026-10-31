//! Size sweep: BestFft against RustFFT for power-of-two sizes from 64 to 2^20 points (standard
//! ordering, forward, unscaled).
//!
//! Small sizes live entirely in L1 cache, mid sizes in L2/L3, and the largest exceed the last-level
//! cache, so the sweep shows how each engine scales. Laptop CPUs change clock speed constantly, so
//! instead of timing one engine and then the other, every round times both back to back (in
//! alternating order) and the reported ratio is the median of the per-round ratios, which cancels
//! the drift.
//!
//! Run with `cargo bench -p lib_jsl_ffts --bench fft_sizes`, optionally followed by sizes to run,
//! e.g. `-- 64 1024`.

use std::time::Instant;
mod rust_fft_wrapper;
use lib_jsl_ffts::{
    best_fft::BestFft,
    fft_engine_trait::{FfftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
};
use num::Complex;
use rust_fft_wrapper::RustFftWrapper;

const LOG2_SIZES: [u32; 9] = [6, 8, 10, 12, 14, 15, 16, 18, 20];
const ROUNDS: usize = 41;

fn signal(n: usize) -> Vec<Complex<f64>> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    (0..n).map(|_| Complex::new(next(), next())).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn format_time(seconds: f64) -> String {
    if seconds >= 1e-3 {
        format!("{:.3} ms", seconds * 1e3)
    } else if seconds >= 1e-6 {
        format!("{:.2} us", seconds * 1e6)
    } else {
        format!("{:.0} ns", seconds * 1e9)
    }
}

fn main() {
    let only: Vec<String> = std::env::args().skip(1).filter(|a| !a.starts_with('-')).collect();
    println!("{:>9}  {:>11}  {:>11}  {:>16}", "points", "best_fft", "rustfft", "best/rustfft");
    for log2n in LOG2_SIZES {
        let n = 1usize << log2n;
        if !only.is_empty() && !only.iter().any(|s| s == &n.to_string()) {
            continue;
        }
        let x = signal(n);
        let mut engines: [Box<dyn FfftEngine1D>; 2] = [Box::new(BestFft::new()), Box::new(RustFftWrapper::new())];
        for e in engines.iter_mut() {
            e.plan(n, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).unwrap();
            for _ in 0..3 {
                std::hint::black_box(e.execute(&x).unwrap()); // warm caches and allocator
            }
        }
        // Repeat each timing for roughly a couple of milliseconds so short transforms are measurable.
        let reps = (2_000_000.0 / (n as f64 * (n as f64).log2())).ceil().max(1.0) as usize;
        let mut times = [Vec::with_capacity(ROUNDS), Vec::with_capacity(ROUNDS)];
        for round in 0..ROUNDS {
            for k in 0..2 {
                let i = (k + round) % 2;
                let start = Instant::now();
                for _ in 0..reps {
                    std::hint::black_box(engines[i].execute(std::hint::black_box(&x)).unwrap());
                }
                times[i].push(start.elapsed().as_secs_f64() / reps as f64);
            }
        }
        let ratio = median(times[0].iter().zip(&times[1]).map(|(a, b)| a / b).collect());
        println!(
            "{n:>9}  {:>11}  {:>11}  {ratio:>16.2}",
            format_time(median(times[0].clone())),
            format_time(median(times[1].clone()))
        );
    }
}
