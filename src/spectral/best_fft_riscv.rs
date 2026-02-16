use num::Complex;

#[inline]
pub(crate) fn radix2_pass(buffer: &mut [Complex<f64>], twiddles: &[Complex<f64>], size: usize, m: usize) {
    // RISC-V V intrinsics are not stabilized in std for portable Rust yet.
    // Keep this architecture-specific module as a tuned scalar/unrolled kernel.
    let half = m / 2;
    for k in (0..size).step_by(m) {
        let mut j = 0usize;
        while j + 2 <= half {
            let a0 = k + j;
            let a1 = a0 + 1;
            let b0 = a0 + half;
            let b1 = a1 + half;

            let u0 = buffer[a0];
            let u1 = buffer[a1];
            let t0 = twiddles[j] * buffer[b0];
            let t1 = twiddles[j + 1] * buffer[b1];

            buffer[a0] = u0 + t0;
            buffer[b0] = u0 - t0;
            buffer[a1] = u1 + t1;
            buffer[b1] = u1 - t1;
            j += 2;
        }
        while j < half {
            let a = k + j;
            let b = a + half;
            let u = buffer[a];
            let t = twiddles[j] * buffer[b];
            buffer[a] = u + t;
            buffer[b] = u - t;
            j += 1;
        }
    }
}
