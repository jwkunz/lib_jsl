use num::Complex;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
pub(crate) fn radix2_pass(buffer: &mut [Complex<f64>], twiddles: &[Complex<f64>], size: usize, m: usize) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                radix2_pass_avx(buffer, twiddles, size, m);
            }
            return;
        }
        if std::arch::is_x86_feature_detected!("sse3") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                radix2_pass_sse3(buffer, twiddles, size, m);
            }
            return;
        }
    }
    radix2_pass_scalar(buffer, twiddles, size, m);
}

#[target_feature(enable = "avx")]
unsafe fn radix2_pass_avx(buffer: &mut [Complex<f64>], twiddles: &[Complex<f64>], size: usize, m: usize) {
    let half = m / 2;
    let base_ptr = buffer.as_mut_ptr() as *mut f64;
    let tw_ptr = twiddles.as_ptr() as *const f64;

    unsafe {
        for k in (0..size).step_by(m) {
            let mut j = 0usize;
            while j + 2 <= half {
                let a = k + j;
                let b = a + half;
                let a_off = 2 * a;
                let b_off = 2 * b;
                let w_off = 2 * j;

                let u = _mm256_loadu_pd(base_ptr.add(a_off));
                let v = _mm256_loadu_pd(base_ptr.add(b_off));
                let w = _mm256_loadu_pd(tw_ptr.add(w_off));

                let wr = _mm256_movedup_pd(w);
                let wi = _mm256_permute_pd(w, 0b1111);
                let vs = _mm256_permute_pd(v, 0b0101);
                let t1 = _mm256_mul_pd(v, wr);
                let t2 = _mm256_mul_pd(vs, wi);
                let t = _mm256_addsub_pd(t1, t2);

                let add = _mm256_add_pd(u, t);
                let sub = _mm256_sub_pd(u, t);
                _mm256_storeu_pd(base_ptr.add(a_off), add);
                _mm256_storeu_pd(base_ptr.add(b_off), sub);

                j += 2;
            }
            while j < half {
                let w = twiddles[j];
                let a = k + j;
                let b = a + half;
                let u = buffer[a];
                let t = w * buffer[b];
                buffer[a] = u + t;
                buffer[b] = u - t;
                j += 1;
            }
        }
    }
}

#[target_feature(enable = "sse3")]
unsafe fn radix2_pass_sse3(buffer: &mut [Complex<f64>], twiddles: &[Complex<f64>], size: usize, m: usize) {
    let half = m / 2;
    let base_ptr = buffer.as_mut_ptr() as *mut f64;
    let tw_ptr = twiddles.as_ptr() as *const f64;

    unsafe {
        for k in (0..size).step_by(m) {
            for j in 0..half {
                let a = k + j;
                let b = a + half;
                let a_off = 2 * a;
                let b_off = 2 * b;
                let w_off = 2 * j;

                let u = _mm_loadu_pd(base_ptr.add(a_off));
                let v = _mm_loadu_pd(base_ptr.add(b_off));
                let w = _mm_loadu_pd(tw_ptr.add(w_off));

                let wr = _mm_unpacklo_pd(w, w);
                let wi = _mm_unpackhi_pd(w, w);
                let vs = _mm_shuffle_pd(v, v, 0b01);
                let t1 = _mm_mul_pd(v, wr);
                let t2 = _mm_mul_pd(vs, wi);
                let t = _mm_addsub_pd(t1, t2);

                let add = _mm_add_pd(u, t);
                let sub = _mm_sub_pd(u, t);
                _mm_storeu_pd(base_ptr.add(a_off), add);
                _mm_storeu_pd(base_ptr.add(b_off), sub);
            }
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
