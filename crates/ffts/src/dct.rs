//! Discrete cosine and sine transforms of types I-IV, computed through complex FFTs.
//!
//! Unnormalized definitions follow the common (SciPy / FFTW REDFT-RODFT) convention, for a
//! length-`N` input `x`:
//!
//! | Type | Cosine transform `y_k` | Sine transform `y_k` |
//! |------|------------------------|----------------------|
//! | I    | `x₀ + (−1)ᵏx_{N−1} + 2Σ_{n=1}^{N−2} x_n cos(πkn/(N−1))` | `2Σ x_n sin(π(k+1)(n+1)/(N+1))` |
//! | II   | `2Σ x_n cos(πk(2n+1)/(2N))` | `2Σ x_n sin(π(k+1)(2n+1)/(2N))` |
//! | III  | `x₀ + 2Σ_{n≥1} x_n cos(πn(2k+1)/(2N))` | `(−1)ᵏx_{N−1} + 2Σ_{n<N−1} x_n sin(π(n+1)(2k+1)/(2N))` |
//! | IV   | `2Σ x_n cos(π(2n+1)(2k+1)/(4N))` | `2Σ x_n sin(π(2n+1)(2k+1)/(4N))` |
//!
//! Types II and III are mutual inverses up to `2N`, types I and IV are their own inverses up to
//! `2(N∓1)` and `2N`. [`TrigNormalization::Ortho`] scales each transform to an orthogonal matrix,
//! so the inverse is the transpose (type II ↔ III, types I and IV self-inverse). Every transform
//! is an O(N log N) FFT of a symmetric extension or twiddled zero-padding of length about 2N.

use crate::fft::Fft;
use crate::fft_engine_trait::{FftDirection, FftEngine1D, FftOrdering, FftScaleFactor};
use lib_jsl_core::ErrorsJSL;
use num_complex::Complex;
use std::f64::consts::PI;

type C64 = Complex<f64>;

/// The transform type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrigTransformType {
    I,
    II,
    III,
    IV,
}

/// Scaling of the forward transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrigNormalization {
    /// The unnormalized forward transform; the inverse carries the whole scale factor.
    Backward,
    /// Orthonormal scaling: forward and inverse are transposes of each other.
    Ortho,
}

use TrigNormalization::*;
use TrigTransformType::*;

fn unscaled_fft(v: &[C64], direction: FftDirection) -> Result<Vec<C64>, ErrorsJSL> {
    Fft::with_plan(v.len(), FftScaleFactor::None, direction, FftOrdering::Standard)?.execute(v)
}

fn real(v: &[f64]) -> Vec<C64> {
    v.iter().map(|&a| C64::new(a, 0.0)).collect()
}

fn check(x: &[f64], kind: TrigTransformType, cosine: bool) -> Result<usize, ErrorsJSL> {
    let n = x.len();
    if n == 0 {
        return Err(ErrorsJSL::InvalidInputRange("The input must not be empty."));
    }
    if cosine && kind == I && n < 2 {
        return Err(ErrorsJSL::InvalidInputRange("The type I cosine transform needs at least two samples."));
    }
    Ok(n)
}

/// `V_k = Σ x_n e^{-iπn/(2N)} e^{-2πi nk/(2N)}`, the shared core of the type IV transforms, with
/// the output twiddle `e^{-iπ(2k+1)/(4N)}` already applied.
fn type_four_core(x: &[f64]) -> Result<Vec<C64>, ErrorsJSL> {
    let n = x.len();
    let mut v = vec![C64::new(0.0, 0.0); 2 * n];
    for (i, &a) in x.iter().enumerate() {
        v[i] = C64::from_polar(a, -PI * i as f64 / (2 * n) as f64);
    }
    let big = unscaled_fft(&v, FftDirection::Forward)?;
    Ok((0..n).map(|k| C64::from_polar(2.0, -PI * (2 * k + 1) as f64 / (4 * n) as f64) * big[k]).collect())
}

fn dct_raw(x: &[f64], kind: TrigTransformType) -> Result<Vec<f64>, ErrorsJSL> {
    let n = x.len();
    Ok(match kind {
        I => {
            let ext: Vec<f64> = x.iter().copied().chain(x[1..n - 1].iter().rev().copied()).collect();
            unscaled_fft(&real(&ext), FftDirection::Forward)?[..n].iter().map(|v| v.re).collect()
        }
        II => {
            let ext: Vec<f64> = x.iter().copied().chain(x.iter().rev().copied()).collect();
            let big = unscaled_fft(&real(&ext), FftDirection::Forward)?;
            (0..n).map(|k| (C64::from_polar(1.0, -PI * k as f64 / (2 * n) as f64) * big[k]).re).collect()
        }
        III => {
            let mut w = vec![C64::new(0.0, 0.0); 2 * n];
            for (k, &a) in x.iter().enumerate() {
                let c = if k == 0 { 0.5 } else { 1.0 };
                w[k] = C64::from_polar(c * a, PI * k as f64 / (2 * n) as f64);
            }
            unscaled_fft(&w, FftDirection::Inverse)?[..n].iter().map(|v| 2.0 * v.re).collect()
        }
        IV => type_four_core(x)?.iter().map(|v| v.re).collect(),
    })
}

fn dst_raw(x: &[f64], kind: TrigTransformType) -> Result<Vec<f64>, ErrorsJSL> {
    let n = x.len();
    Ok(match kind {
        I => {
            // Odd extension [0, x, 0, −rev x] of length 2(N+1): its DFT is −2i times the sine sum.
            let mut u = vec![0.0; 2 * (n + 1)];
            for (i, &a) in x.iter().enumerate() {
                u[i + 1] = a;
                u[2 * n + 1 - i] = -a;
            }
            let big = unscaled_fft(&real(&u), FftDirection::Forward)?;
            (0..n).map(|k| -big[k + 1].im).collect()
        }
        // DST-II(x)_k = DCT-II((−1)ⁿx)_{N−1−k}.
        II => {
            let alt: Vec<f64> = x.iter().enumerate().map(|(i, &a)| if i.is_multiple_of(2) { a } else { -a }).collect();
            let d = dct_raw(&alt, II)?;
            (0..n).map(|k| d[n - 1 - k]).collect()
        }
        // DST-III(x)_k = (−1)ᵏ DCT-III(rev x)_k.
        III => {
            let rev: Vec<f64> = x.iter().rev().copied().collect();
            dct_raw(&rev, III)?.iter().enumerate().map(|(k, &a)| if k.is_multiple_of(2) { a } else { -a }).collect()
        }
        IV => type_four_core(x)?.iter().map(|v| -v.im).collect(),
    })
}

/// Orthonormal scaling of a type II/III pair: entry `special` (0 for cosine, N−1 for sine) of the
/// type II output has norm `√(4N)`, the others `√(2N)`.
fn pair_scale(n: usize, i: usize, special: usize) -> f64 {
    if i == special { (4 * n) as f64 } else { (2 * n) as f64 }.sqrt()
}

fn transform(x: &[f64], kind: TrigTransformType, norm: TrigNormalization, cosine: bool) -> Result<Vec<f64>, ErrorsJSL> {
    let n = check(x, kind, cosine)?;
    let raw = |v: &[f64], k| if cosine { dct_raw(v, k) } else { dst_raw(v, k) };
    if norm == Backward {
        return raw(x, kind);
    }
    let special = if cosine { 0 } else { n - 1 };
    Ok(match kind {
        I if cosine => {
            // Endpoint weights ½ make the raw matrix non-symmetric; rescale them by √2 on both sides.
            let mut v = x.to_vec();
            v[0] *= 2f64.sqrt();
            v[n - 1] *= 2f64.sqrt();
            let mut y = raw(&v, I)?;
            y[0] /= 2f64.sqrt();
            y[n - 1] /= 2f64.sqrt();
            let s = (2.0 * (n - 1) as f64).sqrt();
            y.iter().map(|a| a / s).collect()
        }
        I => raw(x, I)?.iter().map(|a| a / (2.0 * (n + 1) as f64).sqrt()).collect(),
        II => raw(x, II)?.iter().enumerate().map(|(i, a)| a / pair_scale(n, i, special)).collect(),
        // The orthonormal type III is the inverse (= transpose) of the orthonormal type II.
        III => {
            let v: Vec<f64> = x.iter().enumerate().map(|(i, a)| a * pair_scale(n, i, special)).collect();
            raw(&v, III)?.iter().map(|a| a / (2 * n) as f64).collect()
        }
        IV => raw(x, IV)?.iter().map(|a| a / (2.0 * n as f64).sqrt()).collect(),
    })
}

fn inverse(y: &[f64], kind: TrigTransformType, norm: TrigNormalization, cosine: bool) -> Result<Vec<f64>, ErrorsJSL> {
    let n = check(y, kind, cosine)?;
    let partner = match kind {
        II => III,
        III => II,
        k => k,
    };
    let out = transform(y, partner, norm, cosine)?;
    if norm == Ortho {
        return Ok(out);
    }
    let scale = match kind {
        I if cosine => 2 * (n - 1),
        I => 2 * (n + 1),
        _ => 2 * n,
    } as f64;
    Ok(out.iter().map(|a| a / scale).collect())
}

/// The discrete cosine transform of the given type.
pub fn dct(x: &[f64], kind: TrigTransformType, norm: TrigNormalization) -> Result<Vec<f64>, ErrorsJSL> {
    transform(x, kind, norm, true)
}

/// The inverse of [`dct`] with the same type and normalization.
pub fn idct(y: &[f64], kind: TrigTransformType, norm: TrigNormalization) -> Result<Vec<f64>, ErrorsJSL> {
    inverse(y, kind, norm, true)
}

/// The discrete sine transform of the given type.
pub fn dst(x: &[f64], kind: TrigTransformType, norm: TrigNormalization) -> Result<Vec<f64>, ErrorsJSL> {
    transform(x, kind, norm, false)
}

/// The inverse of [`dst`] with the same type and normalization.
pub fn idst(y: &[f64], kind: TrigTransformType, norm: TrigNormalization) -> Result<Vec<f64>, ErrorsJSL> {
    inverse(y, kind, norm, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    const KINDS: [TrigTransformType; 4] = [I, II, III, IV];

    fn alt(k: usize) -> f64 {
        if k.is_multiple_of(2) { 1.0 } else { -1.0 }
    }

    fn direct(x: &[f64], kind: TrigTransformType, cosine: bool) -> Vec<f64> {
        let n = x.len();
        let nf = n as f64;
        (0..n)
            .map(|k| {
                let kf = k as f64;
                let sum = |f: &dyn Fn(f64) -> f64| x.iter().enumerate().map(|(i, &a)| a * f(i as f64)).sum::<f64>();
                match (kind, cosine) {
                    (I, true) => x[0] + alt(k) * x[n - 1] + 2.0 * (1..n - 1).map(|i| x[i] * (PI * kf * i as f64 / (nf - 1.0)).cos()).sum::<f64>(),
                    (II, true) => 2.0 * sum(&|i| (PI * kf * (2.0 * i + 1.0) / (2.0 * nf)).cos()),
                    (III, true) => x[0] + 2.0 * (1..n).map(|i| x[i] * (PI * i as f64 * (2.0 * kf + 1.0) / (2.0 * nf)).cos()).sum::<f64>(),
                    (IV, true) => 2.0 * sum(&|i| (PI * (2.0 * i + 1.0) * (2.0 * kf + 1.0) / (4.0 * nf)).cos()),
                    (I, false) => 2.0 * sum(&|i| (PI * (kf + 1.0) * (i + 1.0) / (nf + 1.0)).sin()),
                    (II, false) => 2.0 * sum(&|i| (PI * (kf + 1.0) * (2.0 * i + 1.0) / (2.0 * nf)).sin()),
                    (III, false) => alt(k) * x[n - 1] + 2.0 * (0..n - 1).map(|i| x[i] * (PI * (i as f64 + 1.0) * (2.0 * kf + 1.0) / (2.0 * nf)).sin()).sum::<f64>(),
                    (IV, false) => 2.0 * sum(&|i| (PI * (2.0 * i + 1.0) * (2.0 * kf + 1.0) / (4.0 * nf)).sin()),
                }
            })
            .collect()
    }

    fn signal(n: usize) -> Vec<f64> {
        (0..n).map(|i| (0.7 * i as f64).sin() + 0.3 * ((i * i) % 7) as f64 - 0.4).collect()
    }

    fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(p, q)| (p - q).abs() < tol)
    }

    #[test]
    fn every_type_matches_its_definition_and_inverts() {
        for cosine in [true, false] {
            for kind in KINDS {
                for n in [2, 3, 4, 5, 8, 13, 16, 30] {
                    let x = signal(n);
                    let f = if cosine { dct(&x, kind, Backward) } else { dst(&x, kind, Backward) }.unwrap();
                    assert!(close(&f, &direct(&x, kind, cosine), 1e-9 * n as f64), "{kind:?} cosine={cosine} n={n}");
                    for norm in [Backward, Ortho] {
                        let y = if cosine { dct(&x, kind, norm) } else { dst(&x, kind, norm) }.unwrap();
                        let back = if cosine { idct(&y, kind, norm) } else { idst(&y, kind, norm) }.unwrap();
                        assert!(close(&back, &x, 1e-11), "inverse {kind:?} {norm:?} cosine={cosine} n={n}");
                    }
                }
            }
        }
    }

    #[test]
    fn orthonormal_scaling_gives_orthogonal_matrices() {
        let n = 7;
        for cosine in [true, false] {
            for kind in KINDS {
                let cols: Vec<Vec<f64>> = (0..n)
                    .map(|j| {
                        let e: Vec<f64> = (0..n).map(|i| if i == j { 1.0 } else { 0.0 }).collect();
                        if cosine { dct(&e, kind, Ortho) } else { dst(&e, kind, Ortho) }.unwrap()
                    })
                    .collect();
                for a in 0..n {
                    for b in 0..n {
                        let dot: f64 = cols[a].iter().zip(&cols[b]).map(|(p, q)| p * q).sum();
                        assert!((dot - if a == b { 1.0 } else { 0.0 }).abs() < 1e-12, "{kind:?} cosine={cosine}");
                    }
                }
            }
        }
    }

    #[test]
    fn energy_compaction_and_validation() {
        // A smooth ramp has almost all of its DCT-II energy in the first few coefficients.
        let x: Vec<f64> = (0..64).map(|i| i as f64 / 63.0).collect();
        let y = dct(&x, II, Ortho).unwrap();
        let total: f64 = y.iter().map(|v| v * v).sum();
        let head: f64 = y[..4].iter().map(|v| v * v).sum();
        assert!(head / total > 0.999);
        assert!((total - x.iter().map(|v| v * v).sum::<f64>()).abs() < 1e-10); // Parseval
        assert!(dct(&[], II, Backward).is_err() && dct(&[1.0], I, Backward).is_err());
        assert_eq!(dct(&[3.0], II, Backward).unwrap(), vec![6.0]);
        assert!(dst(&[2.0], I, Backward).is_ok());
    }
}
