//! Wall-clock timings at sizes beyond the criterion benchmarks, where threading starts to pay:
//! `cargo run --release --manifest-path crates/linalg_bench/Cargo.toml --example large_sizes
//! [--features threading] [-- sizes...]`, optionally with `OPENBLAS_NUM_THREADS` and `MATMUL_NUM_THREADS` set.

use lib_jsl_linalg_bench::*;
use std::hint::black_box;
use std::time::Instant;

/// The fastest of several runs, in milliseconds (robust to frequency scaling and interruptions).
fn time(f: impl Fn()) -> f64 {
    f(); // warm-up
    (0..15)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

/// An operation's name and its two implementations.
type Row<'a> = (&'static str, Box<dyn Fn() + 'a>, Box<dyn Fn() + 'a>);

fn main() {
    let sizes: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().expect("sizes")).collect();
    let sizes = if sizes.is_empty() { vec![512, 1024] } else { sizes };
    for n in sizes {
        let a = random_matrix(n, n, 1);
        let b = random_vector(n, 2);
        let spd = positive_definite(n, 3);
        let s = symmetric(n, 4);
        let rows: [Row; 4] = [
            ("solve", Box::new(|| drop(black_box(ours::solve(&a, &b)))), Box::new(|| drop(black_box(lapack::solve(&a, &b))))),
            ("inverse", Box::new(|| drop(black_box(ours::inverse(&a)))), Box::new(|| drop(black_box(lapack::inverse(&a))))),
            ("cholesky", Box::new(|| drop(black_box(ours::cholesky(&spd)))), Box::new(|| drop(black_box(lapack::cholesky(&spd))))),
            ("eigh", Box::new(|| drop(black_box(ours::eigh(&s)))), Box::new(|| drop(black_box(lapack::eigh(&s))))),
        ];
        for (name, ours_f, lapack_f) in rows.iter() {
            let (o, l) = (time(ours_f), time(lapack_f));
            println!("{name:10} n={n:<5} ours {o:9.3} ms  lapack {l:9.3} ms  ratio {:5.2}", o / l);
        }
    }
}
