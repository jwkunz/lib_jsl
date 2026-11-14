//! Frequency grids, fast transform lengths, N-dimensional real FFTs and N-dimensional DCTs and DSTs,
//! following `scipy.fft`: `fftfreq`, `rfftfreq`, `next_fast_len`, `ifftshift`, `rfftn`/`irfftn`
//! and `dctn`/`idctn`/`dstn`/`idstn`.
//!
//! See the book entry [Real and multidimensional FFTs](https://jwkunz.github.io/lib_jsl/book/fft/real-multidimensional.html#helpers-frequencies-fast-lengths-and-n-d-transforms).

use crate::dct::{TrigNormalization, TrigTransformType, dct, dst, idct, idst};
use crate::fft_engine_trait::FftDirection;
use crate::multidimensional_fft::fft_axis;
use crate::real_fft::{irfft, rfft};
use lib_jsl_core::ErrorsJSL;
use ndarray::{ArrayD, Axis, IxDyn};
use num_complex::Complex;

type C64 = Complex<f64>;

// ANCHOR: fftfreq
/// The sample frequencies of an `n`-point DFT with sample spacing `d` (NumPy's `fftfreq`):
/// `[0, 1, …, ⌈n/2⌉−1, −⌊n/2⌋, …, −1] / (n d)`, in the order the FFT returns its bins.
pub fn fftfreq(n: usize, d: f64) -> Vec<f64> {
    let scale = 1.0 / (n as f64 * d);
    (0..n).map(|k| if k < n.div_ceil(2) { k as f64 } else { k as f64 - n as f64 } * scale).collect()
}

/// The non-negative sample frequencies of an `n`-point real FFT, `[0, 1, …, ⌊n/2⌋] / (n d)`
/// (NumPy's `rfftfreq`).
pub fn rfftfreq(n: usize, d: f64) -> Vec<f64> {
    let scale = 1.0 / (n as f64 * d);
    (0..=n / 2).map(|k| k as f64 * scale).collect()
}
// ANCHOR_END: fftfreq

// ANCHOR: next_fast_len
/// The smallest length `≥ target` whose prime factors are all small, which FFTs handle fastest
/// (SciPy's `next_fast_len`): 2, 3 and 5 for real transforms (`real = true`), and also 7 and 11
/// for complex ones. Found by enumerating the products of powers of the allowed primes.
pub fn next_fast_len(target: usize, real: bool) -> usize {
    if target <= 1 {
        return target.max(1);
    }
    let primes: &[usize] = if real { &[2, 3, 5] } else { &[2, 3, 5, 7, 11] };
    let mut best = usize::MAX;
    // Depth-first over exponents: each partial product that reaches the target is a candidate.
    fn search(product: usize, primes: &[usize], target: usize, best: &mut usize) {
        if product >= target {
            *best = (*best).min(product);
            return;
        }
        let Some((&p, rest)) = primes.split_first() else { return };
        let mut q = product;
        loop {
            search(q, rest, target, best);
            if q >= target || q > usize::MAX / p {
                break;
            }
            q *= p;
        }
    }
    search(1, primes, target, &mut best);
    best
}
// ANCHOR_END: next_fast_len

/// The inverse of [`fftshift`](crate::fftshift): moves the zero-frequency bin from the centre
/// back to index 0 along every axis (they differ for odd lengths).
pub fn ifftshift(data: &ArrayD<C64>) -> ArrayD<C64> {
    let shape = data.shape().to_vec();
    ArrayD::from_shape_fn(data.raw_dim(), |idx| {
        let src: Vec<usize> = (0..shape.len()).map(|d| (idx[d] + shape[d] / 2) % shape[d]).collect();
        data[IxDyn(&src)]
    })
}

/// One-dimensional `fftshift` for a slice of any element type.
pub fn fftshift_1d<T: Clone>(x: &[T]) -> Vec<T> {
    let n = x.len();
    (0..n).map(|k| x[(k + n - n / 2) % n].clone()).collect()
}

/// One-dimensional `ifftshift` for a slice of any element type.
pub fn ifftshift_1d<T: Clone>(x: &[T]) -> Vec<T> {
    let n = x.len();
    (0..n).map(|k| x[(k + n / 2) % n].clone()).collect()
}

// ANCHOR: rfftn
/// The N-dimensional DFT of real data (SciPy's `rfftn`): a real FFT along the last axis, keeping
/// the `⌊n/2⌋ + 1` non-negative frequencies, then complex FFTs along the other axes. The result
/// has the input's shape with the last axis shortened.
pub fn rfftn(data: &ArrayD<f64>) -> Result<ArrayD<C64>, ErrorsJSL> {
    let nd = data.ndim();
    if nd == 0 {
        return Err(ErrorsJSL::InvalidInputRange("The array must have at least one axis."));
    }
    let last = data.len_of(Axis(nd - 1));
    if last == 0 {
        return Err(ErrorsJSL::InvalidInputRange("The last axis must not be empty."));
    }
    let mut shape = data.shape().to_vec();
    shape[nd - 1] = last / 2 + 1;
    let mut out = ArrayD::<C64>::zeros(IxDyn(&shape));
    for (lane, mut target) in data.lanes(Axis(nd - 1)).into_iter().zip(out.lanes_mut(Axis(nd - 1))) {
        let spectrum = rfft(&lane.to_vec())?;
        target.iter_mut().zip(spectrum).for_each(|(t, s)| *t = s);
    }
    for axis in 0..nd - 1 {
        fft_axis(&mut out, axis, FftDirection::Forward)?;
    }
    Ok(out)
}
// ANCHOR_END: rfftn

/// The inverse of [`rfftn`] (SciPy's `irfftn`): `shape` is the shape of the real output, needed
/// because the last axis of length `n` and `n + 1` have the same half spectrum when `n` is even.
/// Scaled so that `irfftn(rfftn(x), x.shape()) = x`.
pub fn irfftn(spectrum: &ArrayD<C64>, shape: &[usize]) -> Result<ArrayD<f64>, ErrorsJSL> {
    let nd = spectrum.ndim();
    if shape.len() != nd || nd == 0 {
        return Err(ErrorsJSL::InvalidInputRange("The output shape must have one entry per axis."));
    }
    let n = shape[nd - 1];
    if spectrum.shape()[..nd - 1] != shape[..nd - 1] || spectrum.len_of(Axis(nd - 1)) != n / 2 + 1 {
        return Err(ErrorsJSL::InvalidInputRange("The spectrum does not match the output shape."));
    }
    let mut work = spectrum.clone();
    for axis in 0..nd - 1 {
        fft_axis(&mut work, axis, FftDirection::Inverse)?;
    }
    let mut out = ArrayD::<f64>::zeros(IxDyn(shape));
    for (lane, mut target) in work.lanes(Axis(nd - 1)).into_iter().zip(out.lanes_mut(Axis(nd - 1))) {
        let x = irfft(&lane.to_vec(), n)?;
        target.iter_mut().zip(x).for_each(|(t, s)| *t = s);
    }
    Ok(out)
}

fn along_every_axis(data: &ArrayD<f64>, f: impl Fn(&[f64]) -> Result<Vec<f64>, ErrorsJSL>) -> Result<ArrayD<f64>, ErrorsJSL> {
    let mut out = data.clone();
    for axis in 0..out.ndim() {
        for mut lane in out.lanes_mut(Axis(axis)) {
            let y = f(&lane.to_vec())?;
            lane.iter_mut().zip(y).for_each(|(t, s)| *t = s);
        }
    }
    Ok(out)
}

// ANCHOR: dctn
/// The N-dimensional DCT (SciPy's `dctn`): the 1-D [`dct`] along every axis in turn, which is
/// valid because the transform is separable.
pub fn dctn(data: &ArrayD<f64>, kind: TrigTransformType, norm: TrigNormalization) -> Result<ArrayD<f64>, ErrorsJSL> {
    along_every_axis(data, |x| dct(x, kind, norm))
}
// ANCHOR_END: dctn

/// The inverse of [`dctn`] with the same type and normalization.
pub fn idctn(data: &ArrayD<f64>, kind: TrigTransformType, norm: TrigNormalization) -> Result<ArrayD<f64>, ErrorsJSL> {
    along_every_axis(data, |x| idct(x, kind, norm))
}

/// The N-dimensional DST (SciPy's `dstn`).
pub fn dstn(data: &ArrayD<f64>, kind: TrigTransformType, norm: TrigNormalization) -> Result<ArrayD<f64>, ErrorsJSL> {
    along_every_axis(data, |x| dst(x, kind, norm))
}

/// The inverse of [`dstn`] with the same type and normalization.
pub fn idstn(data: &ArrayD<f64>, kind: TrigTransformType, norm: TrigNormalization) -> Result<ArrayD<f64>, ErrorsJSL> {
    along_every_axis(data, |x| idst(x, kind, norm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multidimensional_fft::{fftn, fftshift};

    #[test]
    fn frequency_grids_match_numpy() {
        assert_eq!(fftfreq(5, 0.1), vec![0.0, 2.0, 4.0, -4.0, -2.0]);
        assert_eq!(fftfreq(4, 1.0), vec![0.0, 0.25, -0.5, -0.25]);
        assert_eq!(rfftfreq(5, 0.1), vec![0.0, 2.0, 4.0]);
        assert_eq!(rfftfreq(4, 1.0), vec![0.0, 0.25, 0.5]);
    }

    #[test]
    fn fast_lengths() {
        assert_eq!(next_fast_len(1021, true), 1024);
        assert_eq!(next_fast_len(1021, false), 1024);
        assert_eq!(next_fast_len(97, true), 100);
        assert_eq!(next_fast_len(97, false), 98);
        assert_eq!(next_fast_len(13, false), 14);
        assert_eq!(next_fast_len(1, true), 1);
        for t in 2..500 {
            let n = next_fast_len(t, false);
            assert!(n >= t);
            let mut m = n;
            for p in [2, 3, 5, 7, 11] {
                while m.is_multiple_of(p) {
                    m /= p;
                }
            }
            assert_eq!(m, 1);
        }
    }

    #[test]
    fn shifts_invert_each_other() {
        let x: Vec<i32> = (0..7).collect();
        assert_eq!(fftshift_1d(&x), vec![4, 5, 6, 0, 1, 2, 3]);
        assert_eq!(ifftshift_1d(&fftshift_1d(&x)), x);
        let a = ArrayD::from_shape_fn(IxDyn(&[3, 4]), |i| C64::new((i[0] * 4 + i[1]) as f64, 0.0));
        assert_eq!(ifftshift(&fftshift(&a)), a);
    }

    #[test]
    fn rfftn_is_the_half_spectrum_of_fftn() {
        let x = ArrayD::from_shape_fn(IxDyn(&[3, 4, 5]), |i| ((i[0] * 7 + i[1] * 3 + i[2]) as f64).sin());
        let half = rfftn(&x).unwrap();
        assert_eq!(half.shape(), &[3, 4, 3]);
        let full = fftn(&x.mapv(|v| C64::new(v, 0.0)), FftDirection::Forward).unwrap();
        for (idx, v) in half.indexed_iter() {
            assert!((v - full[idx.clone()]).norm() < 1e-12);
        }
        let back = irfftn(&half, &[3, 4, 5]).unwrap();
        assert!(back.iter().zip(x.iter()).all(|(a, b)| (a - b).abs() < 1e-13));
    }

    #[test]
    fn dctn_inverts() {
        let x = ArrayD::from_shape_fn(IxDyn(&[4, 6]), |i| (i[0] as f64 - 1.3) * (i[1] as f64 + 0.2).cos());
        for kind in [TrigTransformType::I, TrigTransformType::II, TrigTransformType::III, TrigTransformType::IV] {
            for norm in [TrigNormalization::Backward, TrigNormalization::Ortho] {
                let y = dctn(&x, kind, norm).unwrap();
                let back = idctn(&y, kind, norm).unwrap();
                assert!(back.iter().zip(x.iter()).all(|(a, b)| (a - b).abs() < 1e-12));
                let y = dstn(&x, kind, norm).unwrap();
                let back = idstn(&y, kind, norm).unwrap();
                assert!(back.iter().zip(x.iter()).all(|(a, b)| (a - b).abs() < 1e-12));
            }
        }
    }
}
