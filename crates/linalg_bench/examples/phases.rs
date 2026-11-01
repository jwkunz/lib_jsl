//! Values-only against values-and-vectors timings, to locate the cost inside the eigensolvers.
use lib_jsl_linalg::{Eig, EigVals, EigValsh, Eigh, SVD, UPLO};
use lib_jsl_linalg_bench::*;
use std::hint::black_box;
use std::time::Instant;
fn best(f: impl Fn()) -> f64 {
    f();
    (0..7).map(|_| { let t = Instant::now(); f(); t.elapsed().as_secs_f64() * 1e3 }).fold(f64::INFINITY, f64::min)
}
fn main() {
    for n in [256, 512] {
        let (a, s) = (random_matrix(n, n, 1), symmetric(n, 2));
        println!("n={n}: eigvalsh {:.1} ms, eigh {:.1} ms, eigvals {:.1} ms, eig {:.1} ms, svd values {:.1} ms, svd full {:.1} ms",
            best(|| drop(black_box(s.eigvalsh(UPLO::Lower).unwrap()))),
            best(|| drop(black_box(s.eigh(UPLO::Lower).unwrap()))),
            best(|| drop(black_box(a.eigvals().unwrap()))),
            best(|| drop(black_box(a.eig().unwrap()))),
            best(|| drop(black_box(a.svd(false, false).unwrap()))),
            best(|| drop(black_box(a.svd(true, true).unwrap()))));
    }
}
