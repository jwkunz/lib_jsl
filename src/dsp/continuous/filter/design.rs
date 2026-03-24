use ndarray::Array1;
use num::Complex;

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

/// Butterworth analog lowpass prototype (`buttap`) with cutoff at 1 rad/s.
pub fn buttap(order: usize) -> Result<Zpk, ErrorsJSL> {
    if order == 0 {
        return Err(ErrorsJSL::InvalidInputRange("order must be > 0"));
    }

    // Butterworth poles lie uniformly on the left-half unit semicircle. The
    // resulting prototype has maximally flat magnitude at DC and a -3 dB
    // cutoff at 1 rad/s.
    let poles = (0..order)
        .map(|k| {
            let theta = std::f64::consts::PI * (2.0 * k as f64 + order as f64 + 1.0)
                / (2.0 * order as f64);
            Complex::new(theta.cos(), theta.sin())
        })
        .collect::<Vec<_>>();

    Ok(Zpk {
        z: Array1::from_vec(vec![]),
        p: Array1::from_vec(poles),
        k: Complex::new(1.0, 0.0),
    })
}

/// Convert a zero-pole-gain representation into numerator/denominator
/// polynomials.
///
/// This is the last step after prototype and frequency transformations. It is
/// intentionally kept separate because many design steps are cleaner and more
/// stable in ZPK form than in polynomial form.
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
