//! Raw matrix-multiply rates: matrixmultiply (the kernel lib_jsl_linalg uses) against ndarray's
//! `dot`, which in this package goes through BLAS.
use lib_jsl_linalg_bench::random_matrix;
use std::time::Instant;
fn main() {
    for (m, k, n) in [(256, 256, 256), (240, 16, 240), (240, 32, 240), (128, 128, 128), (512, 512, 512)] {
        let (a, b) = (random_matrix(m, k, 1), random_matrix(k, n, 2));
        let runs = (2e9 / (2.0 * (m * k * n) as f64)) as usize + 1;
        let _ = a.dot(&b);
        let t = Instant::now();
        for _ in 0..runs {
            std::hint::black_box(a.dot(&b));
        }
        let blas = t.elapsed().as_secs_f64() / runs as f64;
        let mut c = vec![0.0; m * n];
        let t = Instant::now();
        for _ in 0..runs {
            // SAFETY: contiguous row-major buffers of the stated shapes.
            unsafe {
                matrixmultiply::dgemm(m, k, n, 1.0, a.as_ptr(), k as isize, 1, b.as_ptr(), n as isize, 1, 0.0, c.as_mut_ptr(), n as isize, 1)
            };
            std::hint::black_box(&c);
        }
        let mm = t.elapsed().as_secs_f64() / runs as f64;
        let flops = 2.0 * (m * k * n) as f64 / 1e9;
        println!("{m}×{k}×{n}: matrixmultiply {:.1} GFLOP/s, BLAS {:.1} GFLOP/s", flops / mm, flops / blas);
    }
}
