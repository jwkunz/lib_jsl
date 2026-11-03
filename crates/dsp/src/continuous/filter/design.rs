use ndarray::Array1;
use num_complex::Complex;

pub use super::bessel::{BesselNorm, besselap};
pub use super::butterworth::buttap;
pub use super::chebyshev::{cheb1ap, cheb2ap};
pub(crate) use super::elliptic::{complete_elliptic_k, complete_elliptic_km1};
pub use super::elliptic::ellipap;

use crate::{
    number_theory::polynomial::polynomial_from_roots,
    prelude::{C1D, ErrorsJSL},
};

/// High-level analog filter shape used by the continuous-time prototype
/// transforms and by discrete IIR designers built on top of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IirFilterBandType {
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop,
}

/// Zero-pole-gain representation of a transfer function.
///
/// This form is convenient for analog prototype generation and frequency
/// transformations because poles and zeros can often be mapped directly before
/// the final polynomial coefficients are formed.
#[derive(Clone, Debug, PartialEq)]
pub struct Zpk {
    pub z: C1D,
    pub p: C1D,
    pub k: Complex<f64>,
}

/// Transfer-function polynomial form.
///
/// Coefficients are stored in descending powers, matching the polynomial helper
/// routines and the continuous-time `freqs` implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct Tf {
    pub b: C1D,
    pub a: C1D,
}

fn validate_positive(name: &'static str, value: f64) -> Result<(), ErrorsJSL> {
    if !value.is_finite() || value <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange(name));
    }
    Ok(())
}

fn relative_degree(zpk: &Zpk) -> Result<usize, ErrorsJSL> {
    if zpk.z.len() > zpk.p.len() {
        return Err(ErrorsJSL::InvalidInputRange(
            "number of zeros must be <= number of poles",
        ));
    }
    Ok(zpk.p.len() - zpk.z.len())
}

/// Convert a zero-pole-gain representation into numerator/denominator
/// polynomials.
///
/// This is the last step after prototype and frequency transformations. It is
/// intentionally kept separate because many design steps are cleaner and more
/// stable in ZPK form than in polynomial form.
/// The minimum of `f` on `[a, b]` by golden-section search, assuming `f` is unimodal there.
fn golden_section_minimum(mut a: f64, mut b: f64, f: impl Fn(f64) -> f64) -> f64 {
    let r = 0.5 * (5.0_f64.sqrt() - 1.0);
    // Interior points c < d; each step keeps the bracket around the smaller value.
    let mut c = b - r * (b - a);
    let mut d = a + r * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..100 {
        if fc <= fd {
            b = d;
            d = c;
            fd = fc;
            c = b - r * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + r * (b - a);
            fd = f(d);
        }
    }
    if fc <= fd { c } else { d }
}

/// SciPy's passband refinement for band-stop order selection (`buttord`, `cheb1ord`, `cheb2ord`,
/// `ellipord`): each passband edge is moved, between its given value and the neighbouring stopband
/// edge, to where the fractional order `order_of_nat(nat)` is smallest, first the lower edge and then,
/// with it fixed, the upper one. Edges are prewarped analog frequencies; `nat` is the band-stop
/// natural-frequency ratio `min |ws (p0 - p1) / (ws² - p0 p1)|` over both stopband edges.
pub(crate) fn optimize_bandstop_passband(passb: &[f64], stopb: &[f64], order_of_nat: impl Fn(f64) -> f64) -> Vec<f64> {
    let order = |p0: f64, p1: f64| {
        let nat = stopb.iter().map(|&s| (s * (p0 - p1) / (s * s - p0 * p1)).abs()).fold(f64::INFINITY, f64::min);
        if nat.is_finite() && nat > 1.0 { order_of_nat(nat) } else { f64::INFINITY }
    };
    let low = golden_section_minimum(passb[0], stopb[0] - 1e-12, |w| order(w, passb[1]));
    let high = golden_section_minimum(stopb[1] + 1e-12, passb[1], |w| order(low, w));
    vec![low, high]
}

pub fn zpk_to_tf(zpk: &Zpk) -> Result<Tf, ErrorsJSL> {
    let b = polynomial_from_roots(zpk.z.as_slice().unwrap_or(&[])).mapv(|x| x * zpk.k);
    let a = polynomial_from_roots(zpk.p.as_slice().unwrap_or(&[]));
    if a.is_empty() || a[0].norm_sqr() <= 1e-24 {
        return Err(ErrorsJSL::RuntimeError(
            "invalid denominator generated from poles",
        ));
    }
    Ok(Tf { b, a })
}

/// Analog lowpass-to-lowpass transformation.
///
/// Replaces `s` with `s / wo`, moving the prototype cutoff from `1 rad/s` to
/// `wo rad/s`.
pub fn lp2lp_zpk(zpk: &Zpk, wo: f64) -> Result<Zpk, ErrorsJSL> {
    validate_positive("wo must be finite and > 0", wo)?;
    let degree = relative_degree(zpk)?;
    Ok(Zpk {
        z: zpk.z.mapv(|x| x * wo),
        p: zpk.p.mapv(|x| x * wo),
        k: zpk.k * wo.powi(degree as i32),
    })
}

/// Analog lowpass-to-highpass transformation.
///
/// Replaces `s` with `wo / s`. Finite prototype zeros map by inversion, and
/// zeros at infinity become zeros at the origin.
pub fn lp2hp_zpk(zpk: &Zpk, wo: f64) -> Result<Zpk, ErrorsJSL> {
    validate_positive("wo must be finite and > 0", wo)?;
    let degree = relative_degree(zpk)?;

    if zpk.z.iter().any(|z| z.norm_sqr() <= 1e-24) || zpk.p.iter().any(|p| p.norm_sqr() <= 1e-24) {
        return Err(ErrorsJSL::InvalidInputRange(
            "lp2hp_zpk requires non-zero prototype roots",
        ));
    }

    let mut z = zpk.z.iter().map(|&x| Complex::new(wo, 0.0) / x).collect::<Vec<_>>();
    z.extend((0..degree).map(|_| Complex::new(0.0, 0.0)));
    let p = zpk.p.iter().map(|&x| Complex::new(wo, 0.0) / x).collect::<Vec<_>>();
    let num = zpk
        .z
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &x| acc * (-x));
    let den = zpk
        .p
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &x| acc * (-x));

    Ok(Zpk {
        z: Array1::from_vec(z),
        p: Array1::from_vec(p),
        k: zpk.k * num / den,
    })
}

/// Analog lowpass-to-bandpass transformation.
///
/// Replaces `s` with `(s^2 + wo^2) / (bw s)`. Each prototype pole or zero
/// splits into a conjugate pair, doubling the order. Zeros at infinity become
/// zeros at the origin.
pub fn lp2bp_zpk(zpk: &Zpk, wo: f64, bw: f64) -> Result<Zpk, ErrorsJSL> {
    validate_positive("wo must be finite and > 0", wo)?;
    validate_positive("bw must be finite and > 0", bw)?;
    let degree = relative_degree(zpk)?;
    let half_bw = bw * 0.5;

    let mut z = Vec::with_capacity(zpk.z.len() * 2 + degree);
    for &root in zpk.z.iter() {
        let term = root * half_bw;
        let rad = (term * term - Complex::new(wo * wo, 0.0)).sqrt();
        z.push(term + rad);
        z.push(term - rad);
    }
    z.extend((0..degree).map(|_| Complex::new(0.0, 0.0)));

    let mut p = Vec::with_capacity(zpk.p.len() * 2);
    for &root in zpk.p.iter() {
        let term = root * half_bw;
        let rad = (term * term - Complex::new(wo * wo, 0.0)).sqrt();
        p.push(term + rad);
        p.push(term - rad);
    }

    Ok(Zpk {
        z: Array1::from_vec(z),
        p: Array1::from_vec(p),
        k: zpk.k * bw.powi(degree as i32),
    })
}

/// Analog lowpass-to-bandstop transformation.
///
/// Replaces `s` with `(bw s) / (s^2 + wo^2)`. As with the bandpass mapping,
/// each prototype pole or zero splits into a pair. Zeros at infinity become
/// zeros on the imaginary axis at `+-j*wo`.
pub fn lp2bs_zpk(zpk: &Zpk, wo: f64, bw: f64) -> Result<Zpk, ErrorsJSL> {
    validate_positive("wo must be finite and > 0", wo)?;
    validate_positive("bw must be finite and > 0", bw)?;
    let degree = relative_degree(zpk)?;
    let half_bw = bw * 0.5;

    if zpk.z.iter().any(|z| z.norm_sqr() <= 1e-24) || zpk.p.iter().any(|p| p.norm_sqr() <= 1e-24) {
        return Err(ErrorsJSL::InvalidInputRange(
            "lp2bs_zpk requires non-zero prototype roots",
        ));
    }

    let mut z = Vec::with_capacity(zpk.z.len() * 2 + 2 * degree);
    for &root in zpk.z.iter() {
        let term = Complex::new(half_bw, 0.0) / root;
        let rad = (term * term - Complex::new(wo * wo, 0.0)).sqrt();
        z.push(term + rad);
        z.push(term - rad);
    }
    for _ in 0..degree {
        z.push(Complex::new(0.0, wo));
        z.push(Complex::new(0.0, -wo));
    }

    let mut p = Vec::with_capacity(zpk.p.len() * 2);
    for &root in zpk.p.iter() {
        let term = Complex::new(half_bw, 0.0) / root;
        let rad = (term * term - Complex::new(wo * wo, 0.0)).sqrt();
        p.push(term + rad);
        p.push(term - rad);
    }

    let num = zpk
        .z
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &x| acc * (-x));
    let den = zpk
        .p
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &x| acc * (-x));

    Ok(Zpk {
        z: Array1::from_vec(z),
        p: Array1::from_vec(p),
        k: zpk.k * num / den,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close_complex(actual: Complex<f64>, expected: Complex<f64>, tol: f64) {
        assert!(
            (actual - expected).norm() < tol,
            "actual={actual}, expected={expected}, tol={tol}"
        );
    }

    fn sort_by_real_then_imag(values: &mut [Complex<f64>]) {
        values.sort_by(|a, b| {
            a.re
                .partial_cmp(&b.re)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    a.im
                        .partial_cmp(&b.im)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
    }

    #[test]
    fn test_buttap_order_three() {
        let zpk = buttap(3).unwrap();
        assert!(zpk.z.is_empty());
        assert_eq!(zpk.p.len(), 3);
        let mut poles = zpk.p.to_vec();
        sort_by_real_then_imag(&mut poles);
        assert_close_complex(poles[0], Complex::new(-1.0, 0.0), 1e-12);
        assert!((poles[1].norm() - 1.0).abs() < 1e-12);
        assert!((poles[2].norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_zpk_to_tf_first_order() {
        let zpk = Zpk {
            z: Array1::from_vec(vec![]),
            p: Array1::from_vec(vec![Complex::new(-1.0, 0.0)]),
            k: Complex::new(1.0, 0.0),
        };
        let tf = zpk_to_tf(&zpk).unwrap();
        assert_eq!(tf.b.to_vec(), vec![Complex::new(1.0, 0.0)]);
        assert_eq!(tf.a.to_vec(), vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)]);
    }

    #[test]
    fn test_lp2lp_scales_poles_and_gain() {
        let zpk = buttap(2).unwrap();
        let scaled = lp2lp_zpk(&zpk, 3.0).unwrap();
        assert_eq!(scaled.p.len(), 2);
        assert_close_complex(scaled.k, Complex::new(9.0, 0.0), 1e-12);
        for p in &scaled.p {
            assert!((p.norm() - 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_lp2hp_adds_zeros_at_origin() {
        let zpk = buttap(1).unwrap();
        let hp = lp2hp_zpk(&zpk, 2.0).unwrap();
        assert_eq!(hp.z.len(), 1);
        assert_close_complex(hp.z[0], Complex::new(0.0, 0.0), 1e-12);
        assert_close_complex(hp.p[0], Complex::new(-2.0, 0.0), 1e-12);
        assert_close_complex(hp.k, Complex::new(1.0, 0.0), 1e-12);
    }

    #[test]
    fn test_lp2bp_doubles_order() {
        let zpk = buttap(2).unwrap();
        let bp = lp2bp_zpk(&zpk, 5.0, 2.0).unwrap();
        assert_eq!(bp.p.len(), 4);
        assert_eq!(bp.z.len(), 2);
        assert!(bp.z.iter().all(|z| z.norm() < 1e-12));
    }

    #[test]
    fn test_lp2bs_adds_imaginary_axis_zeros() {
        let zpk = buttap(1).unwrap();
        let bs = lp2bs_zpk(&zpk, 4.0, 2.0).unwrap();
        assert_eq!(bs.z.len(), 2);
        let mut zeros = bs.z.to_vec();
        sort_by_real_then_imag(&mut zeros);
        assert_close_complex(zeros[0], Complex::new(0.0, -4.0), 1e-12);
        assert_close_complex(zeros[1], Complex::new(0.0, 4.0), 1e-12);
    }

}
