//! BestFft: the crate's high-performance FFT for power-of-two sizes.
//!
//! The transform is an iterative decimation-in-time Cooley-Tukey FFT on bit-reversed data, tuned in
//! five ways compared with a textbook radix-2 loop:
//!
//! 1. **Radix-4 passes.** Pairs of radix-2 stages are fused into one radix-4 pass. Each pass reads and
//!    writes the array once instead of twice, and needs three complex twiddle multiplications per
//!    four outputs instead of four (the `±i` rotations inside a radix-4 butterfly are free). For an
//!    odd `log2(n)` one radix-2 pass comes first.
//! 2. **SIMD.** On x86-64 with AVX2 and FMA (detected at run time) the passes process two complex
//!    values per 256-bit register, with complex multiplication done by `fmaddsub`. Other targets use
//!    a portable scalar implementation of the same algorithm.
//! 3. **Cache blocking.** Every stage whose butterflies span at most `BLOCK` points is run on one
//!    `BLOCK`-point chunk at a time (all those stages for a chunk, then the next chunk), so the chunk
//!    stays in L1/L2 cache. Only the few largest stages make full passes over a big array.
//! 4. **Bandwidth-saving large passes.** Passes that sweep the whole array store only `W^j` and form
//!    `W^2j` and `W^3j` in registers, cutting their twiddle traffic by a factor of three.
//! 5. **Cache-friendly permutation, no extra copies.** The bit-reversal permutation is done in 16×16
//!    tiles that use whole cache lines, writes straight into the output vector with any scaling
//!    folded in, and the transform then runs in place in the output.
//!
//! Twiddle factors are computed once at planning time from exact angles (no recurrences), and laid
//! out per pass in exactly the order the kernels read them.
use num::Complex;
use lib_jsl_core::ErrorsJSL;

use crate::fft_engine_trait::{FftEngine1D, FftDirection, FftOrdering, FftScaleFactor};

/// Points per cache block (16 bytes each): 4096 points = 64 KiB, comfortably inside L2.
const BLOCK: usize = 4096;

/// Bit width of the tiles used for the bit-reversal permutation: 16 × 16 tiles whose rows are 16
/// contiguous points (256 bytes, four cache lines).
const TILE_BITS: usize = 4;

/// Smallest `log2(size)` that uses the tiled permutation. Below it the whole array sits in L1
/// cache, a plain gather is cheaper, and tiling only adds index arithmetic.
const TILED_PERMUTATION_MIN_LOG2: usize = 12;

/// One pass of the transform.
#[derive(Clone, Copy, Debug)]
enum Pass {
    /// Radix-2 butterflies on adjacent pairs (no twiddles).
    Radix2First,
    /// Radix-4 butterflies on adjacent quadruples (no twiddles).
    Radix4First,
    /// Radix-4 butterflies combining four length-`l` sub-transforms; twiddles start at `offset`
    /// (in complex values) in the twiddle table.
    Radix4 { l: usize, offset: usize },
    /// Like `Radix4`, but only `W^j` is stored (`l` values) and `W^2j`, `W^3j` are computed on the
    /// fly. Used for the passes that sweep the whole array, where memory bandwidth, not arithmetic,
    /// is the limit, so a third of the twiddle traffic is worth two extra multiplications.
    Radix4Compact { l: usize, offset: usize },
    /// The first two passes fused into one register-resident codelet: an 8-point (`l = 2`, after a
    /// radix-2 first pass) or 16-point (`l = 4`, after a radix-4 first pass) transform of each
    /// block, using the `Radix4 { l, offset }` twiddles.
    Codelet { l: usize, offset: usize },
}

impl Pass {
    /// Number of points each butterfly group spans.
    fn span(&self) -> usize {
        match *self {
            Pass::Radix2First => 2,
            Pass::Radix4First => 4,
            Pass::Radix4 { l, .. } | Pass::Radix4Compact { l, .. } | Pass::Codelet { l, .. } => 4 * l,
        }
    }
}

#[derive(Debug)]
pub struct BestFft {
    size: usize,
    direction: FftDirection,
    scale: FftScaleFactor,
    ordering: FftOrdering,
    /// `bit_reverse_map[i]` is `i` with its `log2(size)` bits reversed (small sizes only).
    bit_reverse_map: Vec<u32>,
    /// Bit reversal of `TILE_BITS`-bit values, for the tiled permutation of large sizes.
    rev_tile: Vec<u32>,
    /// Bit reversal of the middle `log2(size) - 2·TILE_BITS` index bits; empty for small sizes.
    rev_mid: Vec<u32>,
    passes: Vec<Pass>,
    /// For each radix-4 pass, for each pair of butterfly indices `(j, j + 1)`:
    /// `[W^2j, W^2(j+1), W^j, W^(j+1), W^3j, W^3(j+1)]` with `W = exp(∓2πi / 4l)`.
    twiddles: Vec<Complex<f64>>,
    use_avx: bool,
}

impl BestFft {
    pub fn new() -> Self {
        Self {
            size: 0,
            direction: FftDirection::Forward,
            scale: FftScaleFactor::None,
            ordering: FftOrdering::Standard,
            bit_reverse_map: Vec::new(),
            rev_tile: Vec::new(),
            rev_mid: Vec::new(),
            passes: Vec::new(),
            twiddles: Vec::new(),
            use_avx: false,
        }
    }

    fn inverse(&self) -> bool {
        matches!(self.direction, FftDirection::Inverse)
    }

    /// Runs one pass over `data` (whose length is a multiple of the pass span).
    fn run_pass(&self, data: &mut [Complex<f64>], pass: Pass) {
        let inverse = self.inverse();
        match pass {
            Pass::Radix2First => radix2_first(data),
            Pass::Radix4First => radix4_first(data, inverse),
            Pass::Radix4 { l, offset } => {
                let tw = &self.twiddles[offset..offset + 3 * l];
                #[cfg(target_arch = "x86_64")]
                if self.use_avx {
                    // SAFETY: `use_avx` is only set when AVX2 and FMA were detected at run time; the
                    // kernel reads and writes within `data` (length a multiple of 4l) and `tw`
                    // (length 3l), as checked by the debug assertions inside.
                    unsafe { avx::radix4_pass(data, tw, l, inverse) };
                    return;
                }
                radix4_pass_scalar(data, tw, l, inverse);
            }
            Pass::Radix4Compact { l, offset } => {
                let tw = &self.twiddles[offset..offset + l];
                #[cfg(target_arch = "x86_64")]
                if self.use_avx {
                    // SAFETY: as above; `tw` has length l.
                    unsafe { avx::radix4_pass_compact(data, tw, l, inverse) };
                    return;
                }
                radix4_pass_compact_scalar(data, tw, l, inverse);
            }
            Pass::Codelet { l, offset } => {
                let tw = &self.twiddles[offset..offset + 3 * l];
                #[cfg(target_arch = "x86_64")]
                if self.use_avx {
                    let p = data.as_mut_ptr();
                    // SAFETY: AVX2/FMA detected; the codelet reads each block fully before writing
                    // it, so reading and writing the same buffer is fine.
                    unsafe { avx::codelet(p, None, 1.0, p, data.len(), tw, l, inverse) };
                    return;
                }
                if l == 2 { radix2_first(data) } else { radix4_first(data, inverse) }
                radix4_pass_scalar(data, tw, l, inverse);
            }
        }
    }

    /// Bit-reverse permutes `input` into a new vector, multiplying by `s` on the way.
    ///
    /// For large sizes the index is split into high, middle and low `TILE_BITS`-bit fields; reversal
    /// swaps (and reverses) the high and low fields and reverses the middle one. For each middle
    /// value a 16 × 16 tile is copied whose source rows and destination rows are each 16 contiguous
    /// points, so every cache line touched is fully used, unlike a plain gather whose reads land on
    /// a different cache line every time.
    fn permute(&self, input: &[Complex<f64>], s: f64) -> Vec<Complex<f64>> {
        let n = self.size;
        let mut out: Vec<Complex<f64>> = Vec::with_capacity(n);
        let dst = out.as_mut_ptr();
        if self.rev_mid.is_empty() {
            for (i, &r) in self.bit_reverse_map.iter().enumerate() {
                // SAFETY: i < n = capacity; r < n because it is a bit-reversed index.
                unsafe { dst.add(i).write(*input.get_unchecked(r as usize) * s) };
            }
        } else {
            let tile = 1usize << TILE_BITS;
            let high_shift = n.trailing_zeros() as usize - TILE_BITS;
            for (m, &rm) in self.rev_mid.iter().enumerate() {
                let (src_mid, dst_mid) = ((rm as usize) << TILE_BITS, m << TILE_BITS);
                for a in 0..tile {
                    let dst_row = (a << high_shift) + dst_mid;
                    let src_col = src_mid + self.rev_tile[a] as usize;
                    for (b, &rb) in self.rev_tile.iter().enumerate() {
                        let src = ((rb as usize) << high_shift) + src_col;
                        // SAFETY: (a, m, b) ranges over every index once, and both the destination
                        // and its bit-reversed source are below n.
                        unsafe { dst.add(dst_row + b).write(*input.get_unchecked(src) * s) };
                    }
                }
            }
        }
        // SAFETY: every one of the n slots was written above (the mapping is a bijection).
        unsafe { out.set_len(n) };
        out
    }

    /// Runs the first (twiddle-free) pass while reading the input, so no separate permutation or
    /// copy pass is needed: element `i` of the first pass's input is `input[map[i]] · s` (or
    /// `input[i] · s` without a map). Returns the vector after the first pass.
    fn fused_first_pass(&self, input: &[Complex<f64>], map: Option<&[u32]>, s: f64) -> Vec<Complex<f64>> {
        let n = self.size;
        let mut out: Vec<Complex<f64>> = Vec::with_capacity(n);
        let dst = out.as_mut_ptr();
        let inverse = self.inverse();
        // SAFETY (all branches): every index 0..n of `dst` is written exactly once, and every source
        // index is below n (`map` is a permutation of 0..n).
        match self.passes.first() {
            None => {
                for (i, &v) in input.iter().enumerate() {
                    unsafe { dst.add(i).write(v * s) };
                }
            }
            #[cfg(target_arch = "x86_64")]
            Some(&Pass::Codelet { l, offset }) if self.use_avx => unsafe {
                avx::codelet(input.as_ptr(), map, s, dst, n, &self.twiddles[offset..offset + 3 * l], l, inverse)
            },
            #[cfg(target_arch = "x86_64")]
            Some(Pass::Radix2First) if self.use_avx && n >= 4 => unsafe { avx::first_radix2(input, map, s, dst, n) },
            #[cfg(target_arch = "x86_64")]
            Some(Pass::Radix4First) if self.use_avx => unsafe { avx::first_radix4(input, map, s, dst, n, inverse) },
            Some(&first) => {
                let get = |i: usize| input[map.map_or(i, |m| m[i] as usize)] * s;
                match first {
                    Pass::Radix2First | Pass::Codelet { l: 2, .. } => {
                        for q in 0..n / 2 {
                            let (a, b) = (get(2 * q), get(2 * q + 1));
                            unsafe {
                                dst.add(2 * q).write(a + b);
                                dst.add(2 * q + 1).write(a - b);
                            }
                        }
                    }
                    _ => {
                        for q in 0..n / 4 {
                            let (a, b, c, d) = (get(4 * q), get(4 * q + 1), get(4 * q + 2), get(4 * q + 3));
                            let (t0, t1, t2, t3) = (a + b, a - b, c + d, rotate(c - d, inverse));
                            unsafe {
                                dst.add(4 * q).write(t0 + t2);
                                dst.add(4 * q + 1).write(t1 + t3);
                                dst.add(4 * q + 2).write(t0 - t2);
                                dst.add(4 * q + 3).write(t1 - t3);
                            }
                        }
                    }
                }
            }
        }
        unsafe { out.set_len(n) };
        // Without SIMD a codelet is the radix-2/radix-4 first pass above plus its second pass.
        if let Some(&Pass::Codelet { l, offset }) = self.passes.first()
            && !self.use_avx
        {
            radix4_pass_scalar(&mut out, &self.twiddles[offset..offset + 3 * l], l, inverse);
        }
        out
    }
}

impl Default for BestFft {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn bit_reverse(mut n: usize, bits: usize) -> usize {
    let mut reversed = 0usize;
    for _ in 0..bits {
        reversed = (reversed << 1) | (n & 1);
        n >>= 1;
    }
    reversed
}

/// Multiplies by `-i` (forward transform) or `+i` (inverse transform).
#[inline(always)]
fn rotate(v: Complex<f64>, inverse: bool) -> Complex<f64> {
    if inverse { Complex::new(-v.im, v.re) } else { Complex::new(v.im, -v.re) }
}

/// First pass for odd `log2(n)`: length-2 DFTs of adjacent pairs.
fn radix2_first(data: &mut [Complex<f64>]) {
    let (pairs, _) = data.as_chunks_mut::<2>();
    for pair in pairs {
        let (a, b) = (pair[0], pair[1]);
        pair[0] = a + b;
        pair[1] = a - b;
    }
}

/// First pass for even `log2(n)`: length-4 DFTs of adjacent quadruples (memory order A, B, C, D
/// holds the sub-transforms of the inputs congruent to 0, 2, 1, 3 mod 4).
fn radix4_first(data: &mut [Complex<f64>], inverse: bool) {
    let (quads, _) = data.as_chunks_mut::<4>();
    for q in quads {
        let (a, b, c, d) = (q[0], q[1], q[2], q[3]);
        let (t0, t1, t2, t3) = (a + b, a - b, c + d, rotate(c - d, inverse));
        q[0] = t0 + t2;
        q[1] = t1 + t3;
        q[2] = t0 - t2;
        q[3] = t1 - t3;
    }
}

/// A radix-4 pass combining four length-`l` sub-transforms in every block of `4l` points.
///
/// With `W = exp(∓2πi / 4l)` and the block's four quarters `A, B, C, D` (sub-transforms of the
/// inputs congruent to 0, 2, 1, 3 modulo 4), output `j + q·l` for `q = 0..3` is
/// `A + W^2j B ± …` arranged as two radix-2 layers with a free `±i` rotation.
fn radix4_pass_scalar(data: &mut [Complex<f64>], tw: &[Complex<f64>], l: usize, inverse: bool) {
    for block in data.chunks_exact_mut(4 * l) {
        let (ab, cd) = block.split_at_mut(2 * l);
        let (qa, qb) = ab.split_at_mut(l);
        let (qc, qd) = cd.split_at_mut(l);
        for j in 0..l {
            // Twiddles are stored per pair of j values: [w2_j, w2_j+1, w1_j, w1_j+1, w3_j, w3_j+1].
            let base = 6 * (j / 2) + (j % 2);
            let (w2, w1, w3) = (tw[base], tw[base + 2], tw[base + 4]);
            let a = qa[j];
            let tb = qb[j] * w2;
            let tc = qc[j] * w1;
            let td = qd[j] * w3;
            let (t0, t1, t2, t3) = (a + tb, a - tb, tc + td, rotate(tc - td, inverse));
            qa[j] = t0 + t2;
            qb[j] = t1 + t3;
            qc[j] = t0 - t2;
            qd[j] = t1 - t3;
        }
    }
}

/// `radix4_pass_scalar` with only `W^j` stored; `W^2j` and `W^3j` are computed on the fly.
fn radix4_pass_compact_scalar(data: &mut [Complex<f64>], tw: &[Complex<f64>], l: usize, inverse: bool) {
    for block in data.chunks_exact_mut(4 * l) {
        let (ab, cd) = block.split_at_mut(2 * l);
        let (qa, qb) = ab.split_at_mut(l);
        let (qc, qd) = cd.split_at_mut(l);
        for j in 0..l {
            let w1 = tw[j];
            let w2 = w1 * w1;
            let w3 = w2 * w1;
            let a = qa[j];
            let tb = qb[j] * w2;
            let tc = qc[j] * w1;
            let td = qd[j] * w3;
            let (t0, t1, t2, t3) = (a + tb, a - tb, tc + td, rotate(tc - td, inverse));
            qa[j] = t0 + t2;
            qb[j] = t1 + t3;
            qc[j] = t0 - t2;
            qd[j] = t1 - t3;
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use num::Complex;
    use std::arch::x86_64::*;

    /// Complex product of two pairs of complex values with FMA.
    #[inline(always)]
    unsafe fn cmul(v: __m256d, w: __m256d) -> __m256d {
        // SAFETY: only called from AVX2/FMA-enabled functions, into which it is inlined.
        unsafe {
            let wr = _mm256_movedup_pd(w); // (wr, wr) for each complex
            let wi = _mm256_permute_pd(w, 0b1111); // (wi, wi)
            let vs = _mm256_permute_pd(v, 0b0101); // (vi, vr)
            // even lanes: vr wr - vi wi ; odd lanes: vi wr + vr wi
            _mm256_fmaddsub_pd(v, wr, _mm256_mul_pd(vs, wi))
        }
    }

    /// AVX2/FMA version of `radix4_pass_scalar`, two butterflies (j, j + 1) per iteration.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA; `data.len()` must be a multiple of `4l`, `l` must be
    /// even, and `tw.len()` must be at least `3l`.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn radix4_pass(data: &mut [Complex<f64>], tw: &[Complex<f64>], l: usize, inverse: bool) {
        debug_assert!(l.is_multiple_of(2) && data.len().is_multiple_of(4 * l) && tw.len() >= 3 * l);
        // Multiplying (x, y) by -i gives (y, -x): swap, then negate the odd lanes. By +i: (-y, x).
        let sign = if inverse { _mm256_set_pd(0.0, -0.0, 0.0, -0.0) } else { _mm256_set_pd(-0.0, 0.0, -0.0, 0.0) };
        let n = data.len();
        let base = data.as_mut_ptr() as *mut f64;
        let twp = tw.as_ptr() as *const f64;
        let quarter = 2 * l; // f64 offset between quarters
        let mut k = 0usize;
        unsafe {
            while k < n {
                let pa = base.add(2 * k);
                let (pb, pc, pd) = (pa.add(quarter), pa.add(2 * quarter), pa.add(3 * quarter));
                let mut j = 0usize;
                let mut t = twp;
                while j < l {
                    let o = 2 * j;
                    let a = _mm256_loadu_pd(pa.add(o));
                    let b = _mm256_loadu_pd(pb.add(o));
                    let c = _mm256_loadu_pd(pc.add(o));
                    let d = _mm256_loadu_pd(pd.add(o));
                    let w2 = _mm256_loadu_pd(t);
                    let w1 = _mm256_loadu_pd(t.add(4));
                    let w3 = _mm256_loadu_pd(t.add(8));
                    let tb = cmul(b, w2);
                    let tc = cmul(c, w1);
                    let td = cmul(d, w3);
                    let t0 = _mm256_add_pd(a, tb);
                    let t1 = _mm256_sub_pd(a, tb);
                    let t2 = _mm256_add_pd(tc, td);
                    let t3 = _mm256_xor_pd(_mm256_permute_pd(_mm256_sub_pd(tc, td), 0b0101), sign);
                    _mm256_storeu_pd(pa.add(o), _mm256_add_pd(t0, t2));
                    _mm256_storeu_pd(pb.add(o), _mm256_add_pd(t1, t3));
                    _mm256_storeu_pd(pc.add(o), _mm256_sub_pd(t0, t2));
                    _mm256_storeu_pd(pd.add(o), _mm256_sub_pd(t1, t3));
                    t = t.add(12);
                    j += 2;
                }
                k += 4 * l;
            }
        }
    }

    /// `radix4_pass` with only `W^j` stored (`tw.len() >= l`); `W^2j = (W^j)²` and
    /// `W^3j = W^j · W^2j` are formed in registers.
    ///
    /// # Safety
    /// As for `radix4_pass`, with `tw.len() >= l`.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn radix4_pass_compact(data: &mut [Complex<f64>], tw: &[Complex<f64>], l: usize, inverse: bool) {
        debug_assert!(l.is_multiple_of(2) && data.len().is_multiple_of(4 * l) && tw.len() >= l);
        let sign = if inverse { _mm256_set_pd(0.0, -0.0, 0.0, -0.0) } else { _mm256_set_pd(-0.0, 0.0, -0.0, 0.0) };
        let n = data.len();
        let base = data.as_mut_ptr() as *mut f64;
        let twp = tw.as_ptr() as *const f64;
        let quarter = 2 * l;
        let mut k = 0usize;
        unsafe {
            while k < n {
                let pa = base.add(2 * k);
                let (pb, pc, pd) = (pa.add(quarter), pa.add(2 * quarter), pa.add(3 * quarter));
                let mut j = 0usize;
                while j < l {
                    let o = 2 * j;
                    let w1 = _mm256_loadu_pd(twp.add(o));
                    let w2 = cmul(w1, w1);
                    let w3 = cmul(w2, w1);
                    let a = _mm256_loadu_pd(pa.add(o));
                    let tb = cmul(_mm256_loadu_pd(pb.add(o)), w2);
                    let tc = cmul(_mm256_loadu_pd(pc.add(o)), w1);
                    let td = cmul(_mm256_loadu_pd(pd.add(o)), w3);
                    let t0 = _mm256_add_pd(a, tb);
                    let t1 = _mm256_sub_pd(a, tb);
                    let t2 = _mm256_add_pd(tc, td);
                    let t3 = _mm256_xor_pd(_mm256_permute_pd(_mm256_sub_pd(tc, td), 0b0101), sign);
                    _mm256_storeu_pd(pa.add(o), _mm256_add_pd(t0, t2));
                    _mm256_storeu_pd(pb.add(o), _mm256_add_pd(t1, t3));
                    _mm256_storeu_pd(pc.add(o), _mm256_sub_pd(t0, t2));
                    _mm256_storeu_pd(pd.add(o), _mm256_sub_pd(t1, t3));
                    j += 2;
                }
                k += 4 * l;
            }
        }
    }

    /// Loads source element `i` (through `map`, if any) as a 128-bit vector.
    #[inline(always)]
    unsafe fn load(input: &[Complex<f64>], map: Option<&[u32]>, i: usize) -> __m128d {
        // SAFETY: callers pass i < n and map (if any) holds indices below n = input.len().
        unsafe {
            let j = match map {
                Some(m) => *m.get_unchecked(i) as usize,
                None => i,
            };
            _mm_loadu_pd(input.as_ptr().add(j) as *const f64)
        }
    }

    /// Loads source element `i` (through `map`, if any) from a raw pointer as a 128-bit vector.
    #[inline(always)]
    unsafe fn load_ptr(src: *const Complex<f64>, map: Option<&[u32]>, i: usize) -> __m128d {
        // SAFETY: callers pass i < n, map entries below n, and `src` valid for n reads.
        unsafe {
            let j = match map {
                Some(m) => *m.get_unchecked(i) as usize,
                None => i,
            };
            _mm_loadu_pd(src.add(j) as *const f64)
        }
    }

    /// One radix-4 butterfly on two lanes: inputs `a, b, c, d` (quarters), twiddles for b, c, d.
    #[inline(always)]
    unsafe fn butterfly4([a, b, c, d]: [__m256d; 4], [w2, w1, w3]: [__m256d; 3], sign: __m256d) -> [__m256d; 4] {
        // SAFETY: only inlined into AVX2/FMA functions.
        unsafe {
            let (tb, tc, td) = (cmul(b, w2), cmul(c, w1), cmul(d, w3));
            let (t0, t1, t2) = (_mm256_add_pd(a, tb), _mm256_sub_pd(a, tb), _mm256_add_pd(tc, td));
            let t3 = _mm256_xor_pd(_mm256_permute_pd(_mm256_sub_pd(tc, td), 0b0101), sign);
            [_mm256_add_pd(t0, t2), _mm256_add_pd(t1, t3), _mm256_sub_pd(t0, t2), _mm256_sub_pd(t1, t3)]
        }
    }

    /// Register-resident codelet for the first two passes: for every block of `4l` points
    /// (`l = 2`: 8 points, `l = 4`: 16 points) it reads the block's inputs (element `i` is
    /// `src[map[i]] · s`, or `src[i] · s` without a map), applies the radix-2 (`l = 2`) or
    /// radix-4 (`l = 4`) first layer and the radix-4 second layer in registers, and stores the
    /// block once. `tw` is the second layer's twiddle table (`3l` values).
    ///
    /// # Safety
    /// AVX2/FMA required; `n` a multiple of `4l`; `src` and `dst` valid for `n` elements (they may
    /// be the same buffer: each block is read completely before it is written); `map` (if any) a
    /// permutation of `0..n`.
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn codelet(src: *const Complex<f64>, map: Option<&[u32]>, s: f64, dst: *mut Complex<f64>, n: usize, tw: &[Complex<f64>], l: usize, inverse: bool) {
        debug_assert!((l == 2 || l == 4) && n.is_multiple_of(4 * l) && tw.len() >= 3 * l);
        unsafe {
            let sv = _mm256_set1_pd(s);
            let sign = if inverse { _mm256_set_pd(0.0, -0.0, 0.0, -0.0) } else { _mm256_set_pd(-0.0, 0.0, -0.0, 0.0) };
            let t = tw.as_ptr() as *const f64;
            let out = dst as *mut f64;
            // Builds the two-lane vector (element p, element q) of the block starting at `base`.
            let pair = |base: usize, p: usize, q: usize| _mm256_mul_pd(_mm256_set_m128d(load_ptr(src, map, base + q), load_ptr(src, map, base + p)), sv);
            if l == 2 {
                let (w2, w1, w3) = (_mm256_loadu_pd(t), _mm256_loadu_pd(t.add(4)), _mm256_loadu_pd(t.add(8)));
                for base in (0..n).step_by(8) {
                    // First layer: radix-2 on pairs (0,1), (2,3), (4,5), (6,7), two pairs per vector.
                    let (x, y) = (pair(base, 0, 2), pair(base, 1, 3));
                    let (s01, d01) = (_mm256_add_pd(x, y), _mm256_sub_pd(x, y));
                    let (x2, y2) = (pair(base, 4, 6), pair(base, 5, 7));
                    let (s23, d23) = (_mm256_add_pd(x2, y2), _mm256_sub_pd(x2, y2));
                    // Regroup into quarters: A = outputs of pair 0, B = pair 1, C = pair 2, D = pair 3.
                    let a = _mm256_permute2f128_pd(s01, d01, 0x20);
                    let b = _mm256_permute2f128_pd(s01, d01, 0x31);
                    let c = _mm256_permute2f128_pd(s23, d23, 0x20);
                    let d = _mm256_permute2f128_pd(s23, d23, 0x31);
                    let r = butterfly4([a, b, c, d], [w2, w1, w3], sign);
                    let o = out.add(2 * base);
                    _mm256_storeu_pd(o, r[0]);
                    _mm256_storeu_pd(o.add(4), r[1]);
                    _mm256_storeu_pd(o.add(8), r[2]);
                    _mm256_storeu_pd(o.add(12), r[3]);
                }
            } else {
                let w = [_mm256_loadu_pd(t), _mm256_loadu_pd(t.add(4)), _mm256_loadu_pd(t.add(8)), _mm256_loadu_pd(t.add(12)), _mm256_loadu_pd(t.add(16)), _mm256_loadu_pd(t.add(20))];
                for base in (0..n).step_by(16) {
                    // First layer: radix-4 on quads 0..3, quads (0, 1) and (2, 3) side by side.
                    let q01 = butterfly4_plain(pair(base, 0, 4), pair(base, 1, 5), pair(base, 2, 6), pair(base, 3, 7), sign);
                    let q23 = butterfly4_plain(pair(base, 8, 12), pair(base, 9, 13), pair(base, 10, 14), pair(base, 11, 15), sign);
                    // Second layer for outputs j = 0, 1 and j = 2, 3 of each quad.
                    for (half, (x0, x1), (y0, y1)) in [(0usize, (q01[0], q01[1]), (q23[0], q23[1])), (1, (q01[2], q01[3]), (q23[2], q23[3]))] {
                        let a = _mm256_permute2f128_pd(x0, x1, 0x20);
                        let b = _mm256_permute2f128_pd(x0, x1, 0x31);
                        let c = _mm256_permute2f128_pd(y0, y1, 0x20);
                        let d = _mm256_permute2f128_pd(y0, y1, 0x31);
                        let r = butterfly4([a, b, c, d], [w[3 * half], w[3 * half + 1], w[3 * half + 2]], sign);
                        let o = out.add(2 * (base + 2 * half));
                        _mm256_storeu_pd(o, r[0]);
                        _mm256_storeu_pd(o.add(8), r[1]);
                        _mm256_storeu_pd(o.add(16), r[2]);
                        _mm256_storeu_pd(o.add(24), r[3]);
                    }
                }
            }
        }
    }

    /// A radix-4 butterfly with unit twiddles.
    #[inline(always)]
    unsafe fn butterfly4_plain(a: __m256d, b: __m256d, c: __m256d, d: __m256d, sign: __m256d) -> [__m256d; 4] {
        // SAFETY: only inlined into AVX2/FMA functions.
        unsafe {
            let (t0, t1, t2) = (_mm256_add_pd(a, b), _mm256_sub_pd(a, b), _mm256_add_pd(c, d));
            let t3 = _mm256_xor_pd(_mm256_permute_pd(_mm256_sub_pd(c, d), 0b0101), sign);
            [_mm256_add_pd(t0, t2), _mm256_add_pd(t1, t3), _mm256_sub_pd(t0, t2), _mm256_sub_pd(t1, t3)]
        }
    }

    /// Fused permutation + radix-2 first pass, two butterflies per iteration.
    ///
    /// # Safety
    /// AVX2/FMA required; `n` a multiple of 4, `input.len() == n`, `map` (if any) a permutation of
    /// `0..n`, and `dst` valid for `n` writes.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn first_radix2(input: &[Complex<f64>], map: Option<&[u32]>, s: f64, dst: *mut Complex<f64>, n: usize) {
        unsafe {
            let sv = _mm256_set1_pd(s);
            let out = dst as *mut f64;
            for q in 0..n / 4 {
                let i = 4 * q;
                // x = (e0, e2), y = (e1, e3): the two butterflies side by side.
                let x = _mm256_mul_pd(_mm256_set_m128d(load(input, map, i + 2), load(input, map, i)), sv);
                let y = _mm256_mul_pd(_mm256_set_m128d(load(input, map, i + 3), load(input, map, i + 1)), sv);
                let (sum, diff) = (_mm256_add_pd(x, y), _mm256_sub_pd(x, y));
                _mm256_storeu_pd(out.add(2 * i), _mm256_permute2f128_pd(sum, diff, 0x20));
                _mm256_storeu_pd(out.add(2 * i + 4), _mm256_permute2f128_pd(sum, diff, 0x31));
            }
        }
    }

    /// Fused permutation + radix-4 first pass, one length-4 DFT per iteration.
    ///
    /// # Safety
    /// As for `first_radix2`.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn first_radix4(input: &[Complex<f64>], map: Option<&[u32]>, s: f64, dst: *mut Complex<f64>, n: usize, inverse: bool) {
        unsafe {
            let sv = _mm256_set1_pd(s);
            // Rotate only the upper complex by -i (forward) or +i (inverse), after a swap.
            let sign = if inverse { _mm256_set_pd(0.0, -0.0, 0.0, 0.0) } else { _mm256_set_pd(-0.0, 0.0, 0.0, 0.0) };
            let out = dst as *mut f64;
            for q in 0..n / 4 {
                let i = 4 * q;
                let x = _mm256_mul_pd(_mm256_set_m128d(load(input, map, i + 2), load(input, map, i)), sv); // (a, c)
                let y = _mm256_mul_pd(_mm256_set_m128d(load(input, map, i + 3), load(input, map, i + 1)), sv); // (b, d)
                let sum = _mm256_add_pd(x, y); // (t0, t2)
                let diff = _mm256_sub_pd(x, y); // (t1, c - d)
                let rotated = _mm256_xor_pd(_mm256_permute_pd(diff, 0b0101), sign);
                let qv = _mm256_blend_pd(diff, rotated, 0b1100); // (t1, t3)
                let u = _mm256_permute2f128_pd(sum, qv, 0x20); // (t0, t1)
                let w = _mm256_permute2f128_pd(sum, qv, 0x31); // (t2, t3)
                _mm256_storeu_pd(out.add(2 * i), _mm256_add_pd(u, w));
                _mm256_storeu_pd(out.add(2 * i + 4), _mm256_sub_pd(u, w));
            }
        }
    }
}

impl FftEngine1D for BestFft {
    fn execute(&mut self, input: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
        if input.len() != self.size {
            return Err(ErrorsJSL::InvalidInputRange(
                "Input size must match the planned size",
            ));
        }
        let n = self.size;
        let s = match self.scale {
            FftScaleFactor::None => 1.0,
            FftScaleFactor::SqrtN => 1.0 / (n as f64).sqrt(),
            FftScaleFactor::N => 1.0 / n as f64,
        };

        // Large standard-order transforms permute in cache-friendly tiles first; otherwise the first
        // pass reads the input directly (through the bit-reversal map when needed).
        let (mut out, passes) = match self.ordering {
            FftOrdering::Standard if !self.rev_mid.is_empty() => (self.permute(input, s), &self.passes[..]),
            FftOrdering::Standard => (self.fused_first_pass(input, Some(&self.bit_reverse_map), s), self.passes.get(1..).unwrap_or(&[])),
            FftOrdering::BitReversed => (self.fused_first_pass(input, None, s), self.passes.get(1..).unwrap_or(&[])),
        };

        // Cache-blocked stages first, then the stages that span more than one block.
        let block = BLOCK.min(n);
        let split = passes.iter().position(|p| p.span() > block).unwrap_or(passes.len());
        if split > 0 {
            for chunk in out.chunks_exact_mut(block) {
                for &pass in &passes[..split] {
                    self.run_pass(chunk, pass);
                }
            }
        }
        for &pass in &passes[split..] {
            self.run_pass(&mut out, pass);
        }
        Ok(out)
    }

    fn plan(
        &mut self,
        size: usize,
        scale: FftScaleFactor,
        direction: FftDirection,
        ordering: FftOrdering,
    ) -> Result<(), ErrorsJSL> {
        if !size.is_power_of_two() {
            return Err(ErrorsJSL::InvalidInputRange("Size must be a power of 2"));
        }
        if size > u32::MAX as usize {
            return Err(ErrorsJSL::InvalidInputRange("Size must fit in 32 bits"));
        }

        self.size = size;
        self.scale = scale;
        self.direction = direction;
        self.ordering = ordering;

        let log2n = size.trailing_zeros() as usize;
        if log2n >= TILED_PERMUTATION_MIN_LOG2.max(2 * TILE_BITS) {
            self.bit_reverse_map = Vec::new();
            self.rev_tile = (0..1usize << TILE_BITS).map(|i| bit_reverse(i, TILE_BITS) as u32).collect();
            let mid_bits = log2n - 2 * TILE_BITS;
            self.rev_mid = (0..1usize << mid_bits).map(|i| bit_reverse(i, mid_bits) as u32).collect();
        } else {
            self.bit_reverse_map = (0..size).map(|i| bit_reverse(i, log2n) as u32).collect();
            self.rev_tile = Vec::new();
            self.rev_mid = Vec::new();
        }

        let sign = match direction {
            FftDirection::Forward => -1.0_f64,
            FftDirection::Inverse => 1.0_f64,
        };
        self.passes.clear();
        self.twiddles.clear();
        // Sub-transform length after the first pass.
        let mut l = match log2n {
            0 => return Ok(()),
            _ if log2n % 2 == 1 => {
                self.passes.push(Pass::Radix2First);
                2
            }
            _ => {
                self.passes.push(Pass::Radix4First);
                4
            }
        };
        let block = BLOCK.min(size);
        while l < size {
            let offset = self.twiddles.len();
            let w = |e: usize| {
                let theta = sign * 2.0 * std::f64::consts::PI * e as f64 / (4 * l) as f64;
                Complex::new(theta.cos(), theta.sin())
            };
            if 4 * l > block {
                // A full-array pass: store W^j only.
                self.twiddles.extend((0..l).map(w));
                self.passes.push(Pass::Radix4Compact { l, offset });
            } else {
                for j in (0..l).step_by(2) {
                    self.twiddles.extend_from_slice(&[w(2 * j), w(2 * (j + 1)), w(j), w(j + 1), w(3 * j), w(3 * (j + 1))]);
                }
                self.passes.push(Pass::Radix4 { l, offset });
            }
            l *= 4;
        }

        // Fuse the twiddle-free first pass and the next radix-4 pass into a codelet.
        if let [Pass::Radix2First | Pass::Radix4First, Pass::Radix4 { l, offset }, ..] = self.passes[..]
            && (l == 2 || l == 4)
        {
            self.passes.splice(0..2, [Pass::Codelet { l, offset }]);
        }

        #[cfg(target_arch = "x86_64")]
        {
            self.use_avx = std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
        }
        Ok(())
    }

    fn get_size(&self) -> usize {
        self.size
    }

    fn get_scale_factor(&self) -> FftScaleFactor {
        self.scale
    }

    fn get_direction(&self) -> FftDirection {
        self.direction
    }

    fn get_ordering(&self) -> FftOrdering {
        self.ordering
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::test_bench_data::{fft_gaussian_32768_golden, fft_gaussian_32768_input};

    fn assert_complex_close(actual: Complex<f64>, expected: Complex<f64>, tol: f64) {
        assert!(
            (actual.re - expected.re).abs() < tol && (actual.im - expected.im).abs() < tol,
            "actual={actual:?}, expected={expected:?}, tol={tol}"
        );
    }

    fn bit_reverse_index(mut n: usize, bits: usize) -> usize {
        let mut reversed = 0;
        for _ in 0..bits {
            reversed <<= 1;
            reversed |= n & 1;
            n >>= 1;
        }
        reversed
    }

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    }

    fn random_signal(n: usize, seed: u64) -> Vec<Complex<f64>> {
        let mut s = seed;
        (0..n).map(|_| Complex::new(lcg(&mut s), lcg(&mut s))).collect()
    }

    /// Exact O(n²) DFT.
    fn naive_dft(x: &[Complex<f64>], inverse: bool) -> Vec<Complex<f64>> {
        let n = x.len();
        let sign = if inverse { 1.0 } else { -1.0 };
        (0..n)
            .map(|k| {
                x.iter()
                    .enumerate()
                    .map(|(t, v)| v * Complex::from_polar(1.0, sign * 2.0 * std::f64::consts::PI * ((k * t) % n) as f64 / n as f64))
                    .sum()
            })
            .collect()
    }

    /// Straightforward radix-2 reference with exactly computed twiddles (for large sizes).
    fn reference_fft(x: &[Complex<f64>], inverse: bool) -> Vec<Complex<f64>> {
        let n = x.len();
        let bits = n.trailing_zeros() as usize;
        let mut a: Vec<Complex<f64>> = (0..n).map(|i| x[bit_reverse_index(i, bits)]).collect();
        let sign = if inverse { 1.0 } else { -1.0 };
        let mut m = 2;
        while m <= n {
            for k in (0..n).step_by(m) {
                for j in 0..m / 2 {
                    let w = Complex::from_polar(1.0, sign * 2.0 * std::f64::consts::PI * j as f64 / m as f64);
                    let (u, t) = (a[k + j], w * a[k + j + m / 2]);
                    a[k + j] = u + t;
                    a[k + j + m / 2] = u - t;
                }
            }
            m *= 2;
        }
        a
    }

    fn run(n: usize, direction: FftDirection, ordering: FftOrdering, scale: FftScaleFactor, input: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let mut fft = BestFft::new();
        fft.plan(n, scale, direction, ordering).unwrap();
        fft.execute(input).unwrap()
    }

    #[test]
    fn test_best_fft_standard_ordering() {
        let input = fft_gaussian_32768_input();
        let expected = fft_gaussian_32768_golden();
        let mut fft = BestFft::new();
        fft.plan(
            input.len(),
            FftScaleFactor::None,
            FftDirection::Forward,
            FftOrdering::Standard,
        )
        .unwrap();

        let start = Instant::now();
        let output = fft.execute(&input).unwrap();
        dbg!("best_fft::standard execute elapsed", start.elapsed());

        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert_complex_close(*actual, *expected, 1e-9);
        }
    }

    #[test]
    fn test_best_fft_bit_reversed_ordering() {
        let input = fft_gaussian_32768_input();
        let expected = fft_gaussian_32768_golden();
        let bits = input.len().trailing_zeros() as usize;
        let mut bit_reversed_input = vec![Complex::new(0.0, 0.0); input.len()];
        for (i, value) in input.iter().enumerate() {
            bit_reversed_input[bit_reverse_index(i, bits)] = *value;
        }
        let mut fft = BestFft::new();
        fft.plan(
            input.len(),
            FftScaleFactor::None,
            FftDirection::Forward,
            FftOrdering::BitReversed,
        )
        .unwrap();

        let start = Instant::now();
        let output = fft.execute(&bit_reversed_input).unwrap();
        dbg!("best_fft::bit_reversed execute elapsed", start.elapsed());

        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert_complex_close(*actual, *expected, 1e-9);
        }
    }

    #[test]
    fn matches_the_exact_dft_for_every_small_size_and_both_directions() {
        for log2n in 0..=9 {
            let n = 1usize << log2n;
            let x = random_signal(n, 7 + log2n as u64);
            for (direction, inverse) in [(FftDirection::Forward, false), (FftDirection::Inverse, true)] {
                let got = run(n, direction, FftOrdering::Standard, FftScaleFactor::None, &x);
                let want = naive_dft(&x, inverse);
                for (g, w) in got.iter().zip(&want) {
                    assert!((g - w).norm() < 1e-11 * (n as f64).max(1.0), "n={n} inverse={inverse}: {g} vs {w}");
                }
            }
        }
    }

    #[test]
    fn matches_an_exact_reference_for_large_sizes_spanning_cache_blocks() {
        // Sizes on both sides of the cache block, with even and odd log2.
        for log2n in [10usize, 11, 12, 13, 14, 15, 16, 17] {
            let n = 1usize << log2n;
            let x = random_signal(n, 100 + log2n as u64);
            for (direction, inverse) in [(FftDirection::Forward, false), (FftDirection::Inverse, true)] {
                let got = run(n, direction, FftOrdering::Standard, FftScaleFactor::None, &x);
                let want = reference_fft(&x, inverse);
                let worst = got.iter().zip(&want).map(|(g, w)| (g - w).norm()).fold(0.0, f64::max);
                assert!(worst < 1e-10 * (n as f64).sqrt() * log2n as f64, "n={n} inverse={inverse}: worst error {worst}");
            }
        }
    }

    #[test]
    fn bit_reversed_input_and_scale_factors() {
        for log2n in [1usize, 2, 3, 6, 11, 14] {
            let n = 1usize << log2n;
            let x = random_signal(n, 300 + log2n as u64);
            let reversed: Vec<Complex<f64>> = (0..n).map(|i| x[bit_reverse_index(i, log2n)]).collect();
            let plain = run(n, FftDirection::Forward, FftOrdering::Standard, FftScaleFactor::None, &x);
            let from_reversed = run(n, FftDirection::Forward, FftOrdering::BitReversed, FftScaleFactor::None, &reversed);
            assert!(plain.iter().zip(&from_reversed).all(|(a, b)| (a - b).norm() < 1e-12));
            for (scale, factor) in [(FftScaleFactor::SqrtN, 1.0 / (n as f64).sqrt()), (FftScaleFactor::N, 1.0 / n as f64)] {
                let scaled = run(n, FftDirection::Forward, FftOrdering::Standard, scale, &x);
                assert!(plain.iter().zip(&scaled).all(|(a, b)| (a * factor - b).norm() < 1e-12 * (n as f64)), "n={n} {scale:?}");
                let scaled_rev = run(n, FftDirection::Forward, FftOrdering::BitReversed, scale, &reversed);
                assert!(plain.iter().zip(&scaled_rev).all(|(a, b)| (a * factor - b).norm() < 1e-12 * (n as f64)));
            }
        }
    }

    #[test]
    fn forward_then_inverse_with_n_scaling_round_trips() {
        for log2n in [0usize, 1, 2, 5, 12, 13, 16] {
            let n = 1usize << log2n;
            let x = random_signal(n, 500 + log2n as u64);
            let y = run(n, FftDirection::Forward, FftOrdering::Standard, FftScaleFactor::None, &x);
            let back = run(n, FftDirection::Inverse, FftOrdering::Standard, FftScaleFactor::N, &y);
            let worst = x.iter().zip(&back).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
            assert!(worst < 1e-13 * log2n.max(1) as f64, "n={n}: {worst}");
        }
    }

    #[test]
    fn the_scalar_kernel_agrees_with_the_simd_kernel() {
        // Force the portable path and compare with whatever the CPU selected.
        for log2n in [1usize, 2, 3, 4, 9, 11, 12, 15] {
            let n = 1usize << log2n;
            let x = random_signal(n, 900 + log2n as u64);
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                for ordering in [FftOrdering::Standard, FftOrdering::BitReversed] {
                    for scale in [FftScaleFactor::None, FftScaleFactor::N] {
                        let mut fast = BestFft::new();
                        fast.plan(n, scale, direction, ordering).unwrap();
                        let mut scalar = BestFft::new();
                        scalar.plan(n, scale, direction, ordering).unwrap();
                        scalar.use_avx = false;
                        let (a, b) = (fast.execute(&x).unwrap(), scalar.execute(&x).unwrap());
                        assert!(a.iter().zip(&b).all(|(p, q)| (p - q).norm() < 1e-12 * n as f64), "n={n} {direction:?} {ordering:?} {scale:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn repeated_execution_reuses_the_plan_and_rejects_wrong_sizes() {
        let mut fft = BestFft::new();
        fft.plan(64, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).unwrap();
        let x = random_signal(64, 1);
        let first = fft.execute(&x).unwrap();
        assert_eq!(fft.execute(&x).unwrap(), first);
        assert!(fft.execute(&x[..32]).is_err());
        assert!(fft.plan(48, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).is_err());
        // Replanning to a different size and direction works.
        fft.plan(8, FftScaleFactor::N, FftDirection::Inverse, FftOrdering::Standard).unwrap();
        assert_eq!(fft.get_size(), 8);
        let one = vec![Complex::new(1.0, 0.0); 8];
        let out = fft.execute(&one).unwrap();
        assert!((out[0] - Complex::new(1.0, 0.0)).norm() < 1e-15 && out[1..].iter().all(|v| v.norm() < 1e-15));
    }
}
