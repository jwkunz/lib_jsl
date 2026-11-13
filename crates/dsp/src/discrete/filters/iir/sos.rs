//! Second-order sections: IIR filters as a cascade of biquads.
//!
//! A high-order transfer function `B(z)/A(z)` is numerically fragile: its polynomial
//! coefficients determine the poles very sensitively, so rounding them (or the arithmetic that
//! uses them) can move poles far, even outside the unit circle. Factoring the filter into
//! second-order sections, each with two poles and two zeros, keeps every pole tied to two
//! well-conditioned coefficients. SciPy's convention is followed: a filter is a list of rows
//! `[b0, b1, b2, a0, a1, a2]`, applied in order.
//!
//! See the book entry [Second-order sections and zero-phase filtering](https://jwkunz.github.io/lib_jsl/book/filters/sos-zero-phase.html).

use num_complex::Complex;

use lib_jsl_linalg::EigVals;
use ndarray::Array2;

use crate::prelude::ErrorsJSL;

use super::zero_phase::lfilter_zi;

/// One second-order section, `[b0, b1, b2, a0, a1, a2]`.
pub type Section = [f64; 6];

type C = Complex<f64>;

/// Splits roots into complex-conjugate representatives (positive imaginary part) and real roots,
/// as SciPy's `_cplxreal`: a root is real when `|Im z| ≤ 100 ε |z|`, and each pair is averaged
/// with its conjugate partner. Both lists are sorted by real part, then by `|Im|`.
fn cplxreal(z: &[C]) -> Result<(Vec<C>, Vec<f64>), ErrorsJSL> {
    let tol = 100.0 * f64::EPSILON;
    let mut z: Vec<C> = z.to_vec();
    z.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.abs().total_cmp(&b.im.abs())));
    let mut real = Vec::new();
    let mut positive = Vec::new();
    let mut negative = Vec::new();
    for &v in &z {
        if v.im.abs() <= tol * v.norm() {
            real.push(v.re);
        } else if v.im > 0.0 {
            positive.push(v);
        } else {
            negative.push(v);
        }
    }
    if positive.len() != negative.len() {
        return Err(ErrorsJSL::InvalidInputRange("complex roots must come in conjugate pairs"));
    }
    let mut pairs = Vec::with_capacity(positive.len());
    for zp in positive {
        let (idx, _) = negative
            .iter()
            .enumerate()
            .map(|(i, zn)| (i, (zp - zn.conj()).norm()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .ok_or(ErrorsJSL::RuntimeError("unmatched complex root"))?;
        let zn = negative.swap_remove(idx);
        if (zp - zn.conj()).norm() > 1e-6 * zp.norm().max(1e-300) {
            return Err(ErrorsJSL::InvalidInputRange("complex roots must come in conjugate pairs"));
        }
        pairs.push((zp + zn.conj()) / 2.0);
    }
    Ok((pairs, real))
}

/// The coefficients of `∏ (x − r)`, highest power first, for real or conjugate-paired roots.
fn poly_real(roots: &[C]) -> Vec<f64> {
    let mut c = vec![C::new(1.0, 0.0)];
    for &r in roots {
        let mut next = vec![C::new(0.0, 0.0); c.len() + 1];
        for (i, &ci) in c.iter().enumerate() {
            next[i] += ci;
            next[i + 1] -= ci * r;
        }
        c = next;
    }
    c.iter().map(|v| v.re).collect()
}

/// A section from up to two zeros and two poles, right-aligned as in SciPy's `_single_zpksos`.
fn single_section(z: &[C], p: &[C]) -> Section {
    let b = poly_real(z);
    let a = poly_real(p);
    let mut s = [0.0; 6];
    s[3 - b.len()..3].copy_from_slice(&b);
    s[6 - a.len()..6].copy_from_slice(&a);
    s
}

/// The index of the entry of `from` nearest to `to`, restricted to real or complex entries
/// (`Some(true)` / `Some(false)`) or any (`None`).
fn nearest(from: &[C], to: C, real: Option<bool>) -> Option<usize> {
    let mut order: Vec<usize> = (0..from.len()).collect();
    order.sort_by(|&i, &j| (from[i] - to).norm().total_cmp(&(from[j] - to).norm()));
    order.into_iter().find(|&i| real.is_none_or(|want| (from[i].im == 0.0) == want))
}

/// The index of the pole closest to the unit circle.
fn worst(p: &[C]) -> usize {
    (0..p.len()).fold(0, |best, i| if (1.0 - p[i].norm()).abs() < (1.0 - p[best].norm()).abs() { i } else { best })
}

// ANCHOR: zpk2sos
/// Second-order sections from the zeros, poles and gain of a digital filter (SciPy's `zpk2sos`
/// with the default `pairing='nearest'`).
///
/// Poles are taken in order of closeness to the unit circle, each with its conjugate (or with the
/// next real pole), and paired with the nearest zeros. The sections are stored with the poles
/// nearest the unit circle *last*, where their large resonant gain has the least effect on the
/// rest of the cascade, and the overall gain is put in the first section.
pub fn zpk2sos(z: &[C], p: &[C], k: f64) -> Result<Vec<Section>, ErrorsJSL> {
    if z.is_empty() && p.is_empty() {
        return Ok(vec![[k, 0.0, 0.0, 1.0, 0.0, 0.0]]);
    }
    // Equal numbers of poles and zeros (extra ones at the origin), and an even count.
    let mut z: Vec<C> = z.to_vec();
    let mut p: Vec<C> = p.to_vec();
    while p.len() < z.len() {
        p.push(C::new(0.0, 0.0));
    }
    while z.len() < p.len() {
        z.push(C::new(0.0, 0.0));
    }
    let n_sections = p.len().div_ceil(2);
    if p.len() % 2 == 1 {
        p.push(C::new(0.0, 0.0));
        z.push(C::new(0.0, 0.0));
    }
    let (zc, zr) = cplxreal(&z)?;
    let (pc, pr) = cplxreal(&p)?;
    let mut z: Vec<C> = zc.into_iter().chain(zr.into_iter().map(|r| C::new(r, 0.0))).collect();
    let mut p: Vec<C> = pc.into_iter().chain(pr.into_iter().map(|r| C::new(r, 0.0))).collect();
    let is_real = |v: &C| v.im == 0.0;

    let mut sos = vec![[0.0; 6]; n_sections];
    for si in (0..n_sections).rev() {
        let p1 = p.remove(worst(&p));
        let real_poles_left = p.iter().filter(|v| is_real(v)).count();
        if is_real(&p1) && real_poles_left == 0 {
            // The last real pole: pair it with the nearest real zero (and a pole and zero at 0).
            let i = nearest(&z, p1, Some(true)).ok_or(ErrorsJSL::RuntimeError("no real zero to pair"))?;
            let z1 = z.remove(i);
            sos[si] = single_section(&[z1, C::new(0.0, 0.0)], &[p1, C::new(0.0, 0.0)]);
        } else if p.len() + 1 == z.len()
            && !is_real(&p1)
            && real_poles_left == 1
            && z.iter().filter(|v| is_real(v)).count() == 1
        {
            // One real pole and one real zero remain: this complex pole must take a complex zero.
            let i = nearest(&z, p1, Some(false)).ok_or(ErrorsJSL::RuntimeError("no complex zero to pair"))?;
            let z1 = z.remove(i);
            sos[si] = single_section(&[z1, z1.conj()], &[p1, p1.conj()]);
        } else {
            let p2 = if is_real(&p1) {
                let reals: Vec<usize> = (0..p.len()).filter(|&i| is_real(&p[i])).collect();
                let candidates: Vec<C> = reals.iter().map(|&i| p[i]).collect();
                p.remove(reals[worst(&candidates)])
            } else {
                p1.conj()
            };
            if z.is_empty() {
                sos[si] = single_section(&[], &[p1, p2]);
            } else {
                let z1 = z.remove(nearest(&z, p1, None).expect("non-empty"));
                if !is_real(&z1) {
                    sos[si] = single_section(&[z1, z1.conj()], &[p1, p2]);
                } else if !z.is_empty() {
                    let i = nearest(&z, p1, Some(true)).ok_or(ErrorsJSL::RuntimeError("no real zero to pair"))?;
                    let z2 = z.remove(i);
                    sos[si] = single_section(&[z1, z2], &[p1, p2]);
                } else {
                    sos[si] = single_section(&[z1], &[p1, p2]);
                }
            }
        }
    }
    for c in &mut sos[0][..3] {
        *c *= k;
    }
    Ok(sos)
}
// ANCHOR_END: zpk2sos

/// Second-order sections from transfer-function coefficients (SciPy's `tf2sos`): the roots of the
/// numerator and denominator, then [`zpk2sos`].
pub fn tf2sos(b: &[f64], a: &[f64]) -> Result<Vec<Section>, ErrorsJSL> {
    let b: Vec<f64> = b.iter().copied().skip_while(|&v| v == 0.0).collect();
    let a: Vec<f64> = a.iter().copied().skip_while(|&v| v == 0.0).collect();
    if b.is_empty() || a.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("numerator and denominator must not be zero"));
    }
    let k = b[0] / a[0];
    // Roots as the eigenvalues of the companion matrix (as NumPy's `roots`): the real Schur form
    // returns complex roots in exact conjugate pairs, which the pairing relies on, even for
    // repeated roots that rounding scatters into small clusters.
    let roots = |c: &[f64]| -> Result<Vec<C>, ErrorsJSL> {
        let n = c.len() - 1;
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut companion = Array2::<f64>::zeros((n, n));
        for j in 0..n {
            companion[[0, j]] = -c[j + 1] / c[0];
        }
        for i in 1..n {
            companion[[i, i - 1]] = 1.0;
        }
        let values = companion.eigvals().map_err(|_| ErrorsJSL::RuntimeError("polynomial roots did not converge"))?;
        Ok(values.to_vec())
    };
    zpk2sos(&roots(&b)?, &roots(&a)?, k)
}

/// The transfer function `(b, a)` of a cascade of sections: the product of their polynomials.
pub fn sos2tf(sos: &[Section]) -> (Vec<f64>, Vec<f64>) {
    let mul = |x: &[f64], y: &[f64]| {
        let mut out = vec![0.0; x.len() + y.len() - 1];
        for (i, &xi) in x.iter().enumerate() {
            for (j, &yj) in y.iter().enumerate() {
                out[i + j] += xi * yj;
            }
        }
        out
    };
    let mut b = vec![1.0];
    let mut a = vec![1.0];
    for s in sos {
        b = mul(&b, &s[..3]);
        a = mul(&a, &s[3..]);
    }
    (b, a)
}

fn validate(sos: &[Section]) -> Result<(), ErrorsJSL> {
    if sos.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("at least one section is required"));
    }
    if sos.iter().any(|s| s[3] == 0.0 || s.iter().any(|v| !v.is_finite())) {
        return Err(ErrorsJSL::InvalidInputRange("each section needs finite coefficients and a0 ≠ 0"));
    }
    Ok(())
}

// ANCHOR: sosfilt
/// Filters `x` through the cascade of sections (SciPy's `sosfilt`), each in transposed direct
/// form II, and returns the output with the final state of each section.
///
/// `zi` gives the initial state of each section (two values per section); `None` starts from
/// rest. Coefficients are normalised by each section's `a0`.
pub fn sosfilt(sos: &[Section], x: &[f64], zi: Option<&[[f64; 2]]>) -> Result<(Vec<f64>, Vec<[f64; 2]>), ErrorsJSL> {
    validate(sos)?;
    let mut state: Vec<[f64; 2]> = match zi {
        Some(z) if z.len() != sos.len() => return Err(ErrorsJSL::IncompatibleArraySizes((z.len(), sos.len()))),
        Some(z) => z.to_vec(),
        None => vec![[0.0; 2]; sos.len()],
    };
    let coefficients: Vec<[f64; 5]> = sos
        .iter()
        .map(|s| [s[0] / s[3], s[1] / s[3], s[2] / s[3], s[4] / s[3], s[5] / s[3]])
        .collect();
    let mut y = x.to_vec();
    for (c, z) in coefficients.iter().zip(state.iter_mut()) {
        let [b0, b1, b2, a1, a2] = *c;
        for v in y.iter_mut() {
            let input = *v;
            let out = b0 * input + z[0];
            z[0] = b1 * input - a1 * out + z[1];
            z[1] = b2 * input - a2 * out;
            *v = out;
        }
    }
    Ok((y, state))
}
// ANCHOR_END: sosfilt

/// Initial states for [`sosfilt`] that correspond to the steady state of a unit step (SciPy's
/// `sosfilt_zi`): multiplied by `x[0]`, they start the filter as if the input had always been
/// `x[0]`, so there is no start-up transient.
///
/// Each section's state is its own step steady state ([`lfilter_zi`]) scaled by the DC gain of
/// the sections before it, which is the size of the step that reaches it.
pub fn sosfilt_zi(sos: &[Section]) -> Result<Vec<[f64; 2]>, ErrorsJSL> {
    validate(sos)?;
    let mut scale = 1.0;
    let mut zi = Vec::with_capacity(sos.len());
    for s in sos {
        let z = lfilter_zi(&s[..3], &s[3..])?;
        let mut pair = [0.0; 2];
        for (d, v) in pair.iter_mut().zip(&z) {
            *d = scale * v;
        }
        zi.push(pair);
        let (bs, as_): (f64, f64) = (s[..3].iter().sum(), s[3..].iter().sum());
        scale *= bs / as_;
    }
    Ok(zi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::continuous::filter::design::IirFilterBandType;
    use crate::dsp::discrete::filters::iir::butterworth::{butter, ButterOutput, ButterOutputType};

    fn butter_zpk(order: usize, wn: &[f64], band: IirFilterBandType) -> (Vec<C>, Vec<C>, f64) {
        match butter(order, wn, band, Some(ButterOutputType::Zpk), None).unwrap() {
            ButterOutput::Zpk(z) => (z.z.to_vec(), z.p.to_vec(), z.k.re),
            _ => unreachable!(),
        }
    }

    #[test]
    fn sections_multiply_back_to_the_transfer_function() {
        for (order, wn, band) in [
            (4, vec![0.2], IirFilterBandType::Lowpass),
            (5, vec![0.3], IirFilterBandType::Highpass),
            (3, vec![0.2, 0.5], IirFilterBandType::Bandpass),
        ] {
            let (z, p, k) = butter_zpk(order, &wn, band);
            let sos = zpk2sos(&z, &p, k).unwrap();
            assert_eq!(sos.len(), (p.len() + 1) / 2);
            let (b, a) = sos2tf(&sos);
            let reference = match butter(order, &wn, band, None, None).unwrap() {
                ButterOutput::Ba(r) => r,
                _ => unreachable!(),
            };
            let bref: Vec<f64> = reference.beta.iter().map(|v| v.re).collect();
            let aref: Vec<f64> = reference.alpha.iter().map(|v| v.re).collect();
            for (x, y) in b.iter().zip(&bref) {
                assert!((x - y).abs() < 1e-10, "{b:?} vs {bref:?}");
            }
            for (x, y) in a.iter().zip(&aref) {
                assert!((x - y).abs() < 1e-10, "{a:?} vs {aref:?}");
            }
        }
    }

    #[test]
    fn sosfilt_matches_the_direct_form_on_a_moderate_filter() {
        let (z, p, k) = butter_zpk(4, &[0.2], IirFilterBandType::Lowpass);
        let sos = zpk2sos(&z, &p, k).unwrap();
        let (b, a) = sos2tf(&sos);
        let x: Vec<f64> = (0..200).map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0).collect();
        let (y, _) = sosfilt(&sos, &x, None).unwrap();
        let (y2, _) = super::super::zero_phase::lfilter(&b, &a, &x, None).unwrap();
        for (u, v) in y.iter().zip(&y2) {
            assert!((u - v).abs() < 1e-12);
        }
    }

    #[test]
    fn steady_state_initial_conditions_remove_the_transient() {
        let (z, p, k) = butter_zpk(6, &[0.1], IirFilterBandType::Lowpass);
        let sos = zpk2sos(&z, &p, k).unwrap();
        let zi = sosfilt_zi(&sos).unwrap();
        let x = vec![3.0; 50];
        let scaled: Vec<[f64; 2]> = zi.iter().map(|s| [3.0 * s[0], 3.0 * s[1]]).collect();
        let (y, _) = sosfilt(&sos, &x, Some(&scaled)).unwrap();
        for v in y {
            assert!((v - 3.0).abs() < 1e-12, "{v}");
        }
    }

    #[test]
    fn a_filter_without_poles_or_zeros_is_a_gain() {
        assert_eq!(zpk2sos(&[], &[], 2.5).unwrap(), vec![[2.5, 0.0, 0.0, 1.0, 0.0, 0.0]]);
    }
}
