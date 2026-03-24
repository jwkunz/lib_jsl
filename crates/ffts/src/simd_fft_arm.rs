use num::Complex;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[inline]
pub(crate) fn radix2_pass(buffer: &mut [Complex<f64>], twiddles: &[Complex<f64>], size: usize, m: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 guarantees NEON support.
        unsafe {
            radix2_pass_neon_f64(buffer, twiddles, size, m);
        }
        return;
    }
    #[allow(unreachable_code)]
    radix2_pass_scalar(buffer, twiddles, size, m);
}

#[cfg(target_arch = "aarch64")]
unsafe fn radix2_pass_neon_f64(
    buffer: &mut [Complex<f64>],
    twiddles: &[Complex<f64>],
    size: usize,
    m: usize,
) {
    let half = m / 2;
    let base_ptr = buffer.as_mut_ptr() as *mut f64;
    let tw_ptr = twiddles.as_ptr() as *const f64;
    let sign = [-1.0_f64, 1.0_f64];
    let signv = vld1q_f64(sign.as_ptr());

    for k in (0..size).step_by(m) {
        for j in 0..half {
            let a = k + j;
            let b = a + half;
            let a_off = 2 * a;
            let b_off = 2 * b;
            let w_off = 2 * j;

            let u = vld1q_f64(base_ptr.add(a_off));
            let v = vld1q_f64(base_ptr.add(b_off));
            let w = vld1q_f64(tw_ptr.add(w_off));

            let wr = vdupq_laneq_f64(w, 0);
            let wi = vdupq_laneq_f64(w, 1);
            let vs = vrev64q_f64(v);
            let t1 = vmulq_f64(v, wr);
            let t2 = vmulq_f64(vs, wi);
            let t = vaddq_f64(t1, vmulq_f64(t2, signv));

            let add = vaddq_f64(u, t);
            let sub = vsubq_f64(u, t);
            vst1q_f64(base_ptr.add(a_off), add);
            vst1q_f64(base_ptr.add(b_off), sub);
        }
    }
}

#[inline]
fn radix2_pass_scalar(buffer: &mut [Complex<f64>], twiddles: &[Complex<f64>], size: usize, m: usize) {
    let half = m / 2;
    for k in (0..size).step_by(m) {
        for j in 0..half {
            let a = k + j;
            let b = a + half;
            let u = buffer[a];
            let t = twiddles[j] * buffer[b];
            buffer[a] = u + t;
            buffer[b] = u - t;
        }
    }
}
