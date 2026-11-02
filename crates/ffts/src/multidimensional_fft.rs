//! Multidimensional FFTs: transforms along one axis, or over every axis of a 2D, 3D or N-D array.
//!
//! See the book entry [Real and multidimensional FFTs](https://jwkunz.github.io/lib_jsl/book/fft/real-multidimensional.html#multidimensional-transforms).

use crate::fft::Fft;
use crate::fft_engine_trait::{FftDirection, FftEngine1D, FftOrdering, FftScaleFactor};
use lib_jsl_core::ErrorsJSL;
use ndarray::{Array2, Array3, ArrayD, Axis};
use num_complex::Complex;

type C64 = Complex<f64>;

// ANCHOR: fft_axis
/// Transforms every 1D lane of `data` along `axis` in place. Inverse transforms are scaled by
/// `1 / len`, so a forward and inverse pass along the same axis is the identity.
pub fn fft_axis(data: &mut ArrayD<C64>, axis: usize, direction: FftDirection) -> Result<(), ErrorsJSL> {
    if axis >= data.ndim() {
        return Err(ErrorsJSL::InvalidInputRange("The axis is out of range."));
    }
    let len = data.len_of(Axis(axis));
    if len == 0 {
        return Ok(());
    }
    let scale = match direction {
        FftDirection::Forward => FftScaleFactor::None,
        FftDirection::Inverse => FftScaleFactor::N,
    };
    let mut fft = Fft::with_plan(len, scale, direction, FftOrdering::Standard)?;
    for mut lane in data.lanes_mut(Axis(axis)) {
        let out = fft.execute(&lane.to_vec())?;
        lane.iter_mut().zip(out).for_each(|(v, o)| *v = o);
    }
    Ok(())
}
// ANCHOR_END: fft_axis

/// The N-dimensional DFT `X[k] = Σ x[n] e^{∓2πi Σ_d k_d n_d / N_d}` over every axis (the inverse
/// is scaled by `1 / Π N_d`). Any lengths are supported.
pub fn fftn(data: &ArrayD<C64>, direction: FftDirection) -> Result<ArrayD<C64>, ErrorsJSL> {
    let mut out = data.clone();
    for axis in 0..out.ndim() {
        fft_axis(&mut out, axis, direction)?;
    }
    Ok(out)
}

/// The 2D DFT of a matrix (rows and columns).
pub fn fft2(data: &Array2<C64>) -> Result<Array2<C64>, ErrorsJSL> {
    fftn(&data.clone().into_dyn(), FftDirection::Forward)?.into_dimensionality().map_err(|_| ErrorsJSL::RuntimeError("dimension"))
}

/// The inverse 2D DFT (scaled by `1 / (rows · columns)`).
pub fn ifft2(data: &Array2<C64>) -> Result<Array2<C64>, ErrorsJSL> {
    fftn(&data.clone().into_dyn(), FftDirection::Inverse)?.into_dimensionality().map_err(|_| ErrorsJSL::RuntimeError("dimension"))
}

/// The 3D DFT of a volume.
pub fn fft3(data: &Array3<C64>) -> Result<Array3<C64>, ErrorsJSL> {
    fftn(&data.clone().into_dyn(), FftDirection::Forward)?.into_dimensionality().map_err(|_| ErrorsJSL::RuntimeError("dimension"))
}

/// The inverse 3D DFT.
pub fn ifft3(data: &Array3<C64>) -> Result<Array3<C64>, ErrorsJSL> {
    fftn(&data.clone().into_dyn(), FftDirection::Inverse)?.into_dimensionality().map_err(|_| ErrorsJSL::RuntimeError("dimension"))
}

/// Rearranges a spectrum so the zero-frequency bin is at the centre along every axis (like
/// NumPy's `fftshift`).
pub fn fftshift(data: &ArrayD<C64>) -> ArrayD<C64> {
    let shape = data.shape().to_vec();
    ArrayD::from_shape_fn(data.raw_dim(), |idx| {
        let src: Vec<usize> = (0..shape.len()).map(|d| (idx[d] + shape[d] - shape[d] / 2) % shape[d]).collect();
        data[ndarray::IxDyn(&src)]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn dft2(x: &Array2<C64>) -> Array2<C64> {
        let (r, c) = x.dim();
        Array2::from_shape_fn((r, c), |(k, l)| {
            let mut s = C64::new(0.0, 0.0);
            for m in 0..r {
                for n in 0..c {
                    s += x[[m, n]] * C64::from_polar(1.0, -TAU * ((k * m) as f64 / r as f64 + (l * n) as f64 / c as f64));
                }
            }
            s
        })
    }

    #[test]
    fn two_dimensional_transform_matches_the_definition_for_any_shape() {
        for (r, c) in [(4, 4), (3, 5), (8, 6), (1, 7)] {
            let x = Array2::from_shape_fn((r, c), |(i, j)| C64::new((i as f64 * 1.3 + j as f64).sin(), (i * j) as f64 * 0.1));
            let got = fft2(&x).unwrap();
            let want = dft2(&x);
            assert!(got.iter().zip(want.iter()).all(|(a, b)| (a - b).norm() < 1e-9), "{r}x{c}");
            let back = ifft2(&got).unwrap();
            assert!(back.iter().zip(x.iter()).all(|(a, b)| (a - b).norm() < 1e-12));
        }
    }

    #[test]
    fn three_dimensional_plane_wave_lands_in_one_bin_and_shift_centres_dc() {
        let (a, b, c) = (4, 6, 5);
        let x = Array3::from_shape_fn((a, b, c), |(i, j, k)| C64::from_polar(1.0, TAU * (i as f64 / a as f64 + 2.0 * j as f64 / b as f64 + 3.0 * k as f64 / c as f64)));
        let s = fft3(&x).unwrap();
        let total = (a * b * c) as f64;
        assert!((s[[1, 2, 3]].re - total).abs() < 1e-9);
        assert!(s.indexed_iter().filter(|(ix, _)| *ix != (1, 2, 3)).all(|(_, v)| v.norm() < 1e-9));
        assert!(ifft3(&s).unwrap().iter().zip(x.iter()).all(|(p, q)| (p - q).norm() < 1e-12));
        // fftshift moves the DC bin to the centre.
        let mut dc = ArrayD::<C64>::zeros(ndarray::IxDyn(&[4, 5]));
        dc[ndarray::IxDyn(&[0, 0])] = C64::new(1.0, 0.0);
        let shifted = fftshift(&dc);
        assert_eq!(shifted[ndarray::IxDyn(&[2, 2])], C64::new(1.0, 0.0));
        let mut d = x.into_dyn();
        assert!(fft_axis(&mut d, 3, FftDirection::Forward).is_err());
    }
}
