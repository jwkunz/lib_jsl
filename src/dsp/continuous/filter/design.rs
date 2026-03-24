use ndarray::Array1;
use num::Complex;

use crate::{
    number_theory::polynomial::{polynomial_from_roots, polynomial_roots},
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

/// Normalization choices for Bessel/Thomson prototypes.
///
/// These mirror SciPy's options:
/// `Phase` is the MATLAB-compatible default,
/// `Delay` is the natural delay-normalized prototype,
/// and `Mag` shifts the prototype so the `-3 dB` point lands at `1 rad/s`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BesselNorm {
    Phase,
    Delay,
    Mag,
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

fn pow10m1(x: f64) -> f64 {
    (std::f64::consts::LN_10 * x).exp_m1()
}

fn falling_factorial(x: usize, n: usize) -> usize {
    let mut value = 1usize;
    for k in (x - n + 1)..=x {
        value *= k;
    }
    value
}

fn bessel_poly(order: usize, reverse: bool) -> Vec<usize> {
    let mut coeffs = Vec::with_capacity(order + 1);
    for k in 0..=order {
        let num = falling_factorial(2 * order - k, order);
        let den = (1usize << (order - k)) * (1..=k).product::<usize>().max(1);
        coeffs.push(num / den);
    }
    if reverse {
        coeffs.reverse();
    }
    coeffs
}

fn bessel_mag_norm_factor(poles: &[Complex<f64>], gain: f64) -> Result<f64, ErrorsJSL> {
    // The magnitude-normalized Bessel prototype is obtained by starting from
    // the delay-normalized version and then finding the frequency scale that
    // places the -3 dB point at 1 rad/s.
    let gain_mag = |w: f64| -> f64 {
        let s = Complex::new(0.0, w);
        let den = poles
            .iter()
            .fold(Complex::new(1.0, 0.0), |acc, &p| acc * (s - p));
        gain / den.norm()
    };

    let target = std::f64::consts::FRAC_1_SQRT_2;
    let mut lo = 1e-6;
    let mut hi = 1.5;
    while gain_mag(hi) > target && hi < 1e6 {
        hi *= 2.0;
    }
    if hi >= 1e6 {
        return Err(ErrorsJSL::RuntimeError(
            "failed to bracket Bessel magnitude normalization factor",
        ));
    }

    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if gain_mag(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Complete elliptic integral of the first kind `K(m)` for real parameter `m`.
///
/// This uses the arithmetic-geometric mean iteration, which converges rapidly
/// and is the standard numerically stable route for real `m in [0, 1]`.
pub(crate) fn complete_elliptic_k(m: f64) -> Result<f64, ErrorsJSL> {
    if !m.is_finite() || !(0.0..=1.0).contains(&m) {
        return Err(ErrorsJSL::InvalidInputRange(
            "elliptic parameter must satisfy 0 <= m <= 1",
        ));
    }
    if m == 1.0 {
        return Ok(f64::INFINITY);
    }
    if m == 0.0 {
        return Ok(std::f64::consts::PI * 0.5);
    }

    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();
    for _ in 0..64 {
        let next_a = 0.5 * (a + b);
        let next_b = (a * b).sqrt();
        if (next_a - next_b).abs() <= 1e-15 * next_a.abs().max(1.0) {
            return Ok(std::f64::consts::PI / (2.0 * next_a));
        }
        a = next_a;
        b = next_b;
    }

    Ok(std::f64::consts::PI / (2.0 * a))
}

pub(crate) fn complete_elliptic_km1(m: f64) -> Result<f64, ErrorsJSL> {
    complete_elliptic_k(1.0 - m)
}

const ELLIPDEG_MMAX: usize = 7;
const ARC_JAC_SN_MAXITER: usize = 10;
const ELLIPJ_EPS: f64 = 1e-12;
const ELLIPJ_MAXITER: usize = 16;
const ELLIP_COMPLEX_EPS: f64 = 2e-16;

/// Compute the elliptic degree relation used by elliptic filter order
/// selection.
///
/// SciPy evaluates this through the nome `q`; this helper mirrors that route so
/// `ellipord` can stay close to the standard closed-form expression.
pub(crate) fn ellipdeg(order: usize, m1: f64) -> Result<f64, ErrorsJSL> {
    if order == 0 {
        return Err(ErrorsJSL::InvalidInputRange("order must be > 0"));
    }
    if !m1.is_finite() || !(0.0..1.0).contains(&m1) {
        return Err(ErrorsJSL::InvalidInputRange(
            "elliptic parameter must satisfy 0 <= m1 < 1",
        ));
    }

    let k1 = complete_elliptic_k(m1)?;
    let k1p = complete_elliptic_km1(m1)?;
    let q1 = (-std::f64::consts::PI * k1p / k1).exp();
    let q = q1.powf(1.0 / order as f64);

    let num = (0..=ELLIPDEG_MMAX)
        .map(|i| q.powi((i * (i + 1)) as i32))
        .sum::<f64>();
    let den = 1.0
        + 2.0
            * (1..=ELLIPDEG_MMAX + 1)
                .map(|i| q.powi((i * i) as i32))
                .sum::<f64>();
    Ok(16.0 * q * (num / den).powi(4))
}

fn jacobi_ellipj_real(u: f64, m: f64) -> Result<(f64, f64, f64), ErrorsJSL> {
    if !u.is_finite() || !m.is_finite() || !(0.0..=1.0).contains(&m) {
        return Err(ErrorsJSL::InvalidInputRange(
            "jacobi_ellipj_real requires finite u and 0 <= m <= 1",
        ));
    }

    if m.abs() < 1e-12 {
        return Ok((u.sin(), u.cos(), 1.0));
    }
    if (1.0 - m).abs() < 1e-12 {
        let sn = u.tanh();
        let cn = 1.0 / u.cosh();
        return Ok((sn, cn, cn));
    }

    // AGM-based real Jacobi evaluator. The forward sweep builds the descending
    // Landen sequence and the backward sweep recovers the amplitude `phi`, from
    // which `sn`, `cn`, and `dn` follow directly.
    let mut a = [0.0_f64; ELLIPJ_MAXITER];
    let mut c = [0.0_f64; ELLIPJ_MAXITER];
    a[0] = 1.0;
    let mut b = (1.0 - m).sqrt();
    c[0] = m.sqrt();
    let mut twon = 1.0;
    let mut n = 0usize;

    while n + 1 < ELLIPJ_MAXITER {
        if c[n].abs() <= ELLIPJ_EPS * a[n].abs().max(1.0) {
            break;
        }
        let ai = a[n];
        c[n + 1] = 0.5 * (ai - b);
        a[n + 1] = 0.5 * (ai + b);
        b = (ai * b).sqrt();
        twon *= 2.0;
        n += 1;
    }

    let mut phi = twon * a[n] * u;
    while n > 0 {
        let t = c[n] * phi.sin() / a[n];
        phi = 0.5 * (phi + t.asin());
        n -= 1;
    }

    let sn = phi.sin();
    let cn = phi.cos();
    let dn = (1.0 - m * sn * sn).sqrt();
    Ok((sn, cn, dn))
}

fn complex_complement(kx: Complex<f64>) -> Complex<f64> {
    ((Complex::new(1.0, 0.0) - kx) * (Complex::new(1.0, 0.0) + kx)).sqrt()
}

/// Inverse Jacobi `sn` for the elliptic-prototype helper path.
///
/// The elliptic prototype formulas need an inverse Jacobi function at purely
/// imaginary arguments. This implementation follows a Landen-transform
/// reduction so we can avoid bringing in a larger special-functions dependency.
fn arc_jac_sn(w: Complex<f64>, m: f64) -> Result<Complex<f64>, ErrorsJSL> {
    if !m.is_finite() || !(0.0..=1.0).contains(&m) {
        return Err(ErrorsJSL::InvalidInputRange(
            "elliptic parameter must satisfy 0 <= m <= 1",
        ));
    }

    let k = m.sqrt();
    if k > 1.0 {
        return Err(ErrorsJSL::InvalidInputRange(
            "elliptic modulus must satisfy 0 <= k <= 1",
        ));
    }
    if (k - 1.0).abs() <= ELLIP_COMPLEX_EPS {
        return Ok(w.atanh());
    }

    // Repeatedly reduce the modulus until the inverse is close to the
    // elementary `asin` limit, then rebuild the original argument scale.
    let mut ks = vec![Complex::new(k, 0.0)];
    while ks
        .last()
        .copied()
        .unwrap_or(Complex::new(0.0, 0.0))
        .norm()
        > ELLIP_COMPLEX_EPS
    {
        let k_curr = *ks.last().unwrap();
        let k_p = complex_complement(k_curr);
        ks.push((Complex::new(1.0, 0.0) - k_p) / (Complex::new(1.0, 0.0) + k_p));
        if ks.len() > ARC_JAC_SN_MAXITER + 1 {
            return Err(ErrorsJSL::RuntimeError(
                "Landen transformation not converging",
            ));
        }
    }

    let k_factor = ks
        .iter()
        .skip(1)
        .fold(Complex::new(1.0, 0.0), |acc, &x| {
            acc * (Complex::new(1.0, 0.0) + x)
        });
    let quarter_period = k_factor * Complex::new(std::f64::consts::PI * 0.5, 0.0);

    let mut wns = vec![w];
    for pair in ks.windows(2) {
        let kn = pair[0];
        let knext = pair[1];
        let wn = *wns.last().unwrap();
        let denom =
            (Complex::new(1.0, 0.0) + knext) * (Complex::new(1.0, 0.0) + complex_complement(kn * wn));
        wns.push(Complex::new(2.0, 0.0) * wn / denom);
    }

    let u = Complex::new(2.0 / std::f64::consts::PI, 0.0) * wns.last().unwrap().asin();
    Ok(quarter_period * u)
}

fn arc_jac_sc1(w: f64, m: f64) -> Result<f64, ErrorsJSL> {
    // `sc(u) = sn(u) / cn(u)`. For the prototype derivation we only need the
    // inverse on the imaginary axis, so convert through the inverse-`sn`
    // implementation and keep the imaginary component.
    let z = arc_jac_sn(Complex::new(0.0, w), m)?;
    if z.re.abs() > 1e-12 {
        return Err(ErrorsJSL::RuntimeError(
            "inverse Jacobi sc returned an unexpected real component",
        ));
    }
    Ok(z.im)
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

/// Chebyshev type I analog lowpass prototype (`cheb1ap`) with ripple `rp` in
/// the passband and cutoff at 1 rad/s.
///
/// The prototype has an equiripple passband with peak ripple `rp` dB and no
/// finite zeros. As with `buttap`, this is the normalized analog starting point
/// used before lowpass/highpass/bandpass/bandstop transformations and any
/// eventual bilinear mapping into the digital domain.
pub fn cheb1ap(order: usize, rp: f64) -> Result<Zpk, ErrorsJSL> {
    if order == 0 {
        return Err(ErrorsJSL::InvalidInputRange("order must be > 0"));
    }
    if !rp.is_finite() || rp <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("rp must be finite and > 0"));
    }

    // `epsilon` is the usual passband ripple parameter from the Chebyshev
    // magnitude expression. `mu` is the hyperbolic scaling that moves the
    // Butterworth unit-circle pole pattern into the Chebyshev ellipse.
    let epsilon = (10.0_f64.powf(0.1 * rp) - 1.0).sqrt();
    let mu = (1.0 / epsilon).asinh() / order as f64;

    let poles = (1..=order)
        .map(|k| {
            let theta = std::f64::consts::PI * (2.0 * k as f64 - 1.0) / (2.0 * order as f64);
            Complex::new(-mu.sinh() * theta.sin(), mu.cosh() * theta.cos())
        })
        .collect::<Vec<_>>();

    // The prototype is normalized so that the gain at DC is unity for odd
    // orders and the ripple-floor value for even orders, matching SciPy's
    // convention for `cheb1ap`.
    let mut gain = poles
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &p| acc * (-p));
    if order % 2 == 0 {
        gain /= (1.0 + epsilon * epsilon).sqrt();
    }

    Ok(Zpk {
        z: Array1::from_vec(vec![]),
        p: Array1::from_vec(poles),
        k: gain,
    })
}

/// Chebyshev type II analog lowpass prototype (`cheb2ap`) with stopband
/// attenuation `rs` and stopband edge at 1 rad/s.
///
/// Unlike the type I prototype, the type II prototype is monotonic in the
/// passband and equiripple in the stopband. It introduces finite zeros on the
/// imaginary axis, which is why ZPK form is especially convenient here.
pub fn cheb2ap(order: usize, rs: f64) -> Result<Zpk, ErrorsJSL> {
    if order == 0 {
        return Err(ErrorsJSL::InvalidInputRange("order must be > 0"));
    }
    if !rs.is_finite() || rs <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("rs must be finite and > 0"));
    }

    // Type II starts from the inverse-Chebyshev form, so the attenuation spec
    // first becomes the stopband ripple factor and is then converted into the
    // same hyperbolic pole-warp parameter used by the prototype formulas.
    let stop_ripple = 1.0 / (10.0_f64.powf(0.1 * rs) - 1.0).sqrt();
    let mu = (1.0 / stop_ripple).asinh() / order as f64;

    // The finite transmission zeros lie on the imaginary axis at the inverse
    // Chebyshev node locations. Odd-order prototypes omit the singular center
    // term because it would correspond to a zero at infinity instead.
    let zero_indices = if order % 2 == 1 {
        (-((order as isize) - 1)..order as isize)
            .step_by(2)
            .filter(|m| *m != 0)
            .collect::<Vec<_>>()
    } else {
        (-((order as isize) - 1)..order as isize)
            .step_by(2)
            .collect::<Vec<_>>()
    };
    let zeros = zero_indices
        .into_iter()
        .map(|m| {
            let theta = std::f64::consts::PI * m as f64 / (2.0 * order as f64);
            Complex::new(0.0, 1.0 / theta.sin())
        })
        .collect::<Vec<_>>();

    // The poles are obtained by taking the Butterworth pole angles, applying
    // the Chebyshev ellipse warp, then inverting them for the type II form.
    let butter_angles = (0..order)
        .map(|k| {
            std::f64::consts::PI * (2.0 * k as f64 - order as f64 + 1.0)
                / (2.0 * order as f64)
        })
        .collect::<Vec<_>>();
    let poles = butter_angles
        .into_iter()
        .map(|theta| {
            let unit = Complex::new(-theta.cos(), -theta.sin());
            let warped = Complex::new(unit.re * mu.sinh(), unit.im * mu.cosh());
            Complex::new(1.0, 0.0) / warped
        })
        .collect::<Vec<_>>();

    // Normalize the prototype so that the passband gain is unity at DC.
    let zero_prod = if zeros.is_empty() {
        Complex::new(1.0, 0.0)
    } else {
        zeros
            .iter()
            .fold(Complex::new(1.0, 0.0), |acc, &z| acc * (-z))
    };
    let pole_prod = poles
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &p| acc * (-p));

    Ok(Zpk {
        z: Array1::from_vec(zeros),
        p: Array1::from_vec(poles),
        k: pole_prod / zero_prod,
    })
}

/// Elliptic (Cauer) analog lowpass prototype (`ellipap`) with passband ripple
/// `rp`, stopband attenuation `rs`, and cutoff at 1 rad/s.
///
/// This is the most selective of the classical IIR prototype families. It is
/// equiripple in both the passband and stopband, introduces finite
/// transmission zeros, and reduces to simpler families as the ripple or
/// attenuation constraints are relaxed.
pub fn ellipap(order: usize, rp: f64, rs: f64) -> Result<Zpk, ErrorsJSL> {
    if order == 0 {
        return Err(ErrorsJSL::InvalidInputRange("order must be > 0"));
    }
    if !rp.is_finite() || rp <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("rp must be finite and > 0"));
    }
    if !rs.is_finite() || rs <= rp {
        return Err(ErrorsJSL::InvalidInputRange("rs must be finite and > rp"));
    }

    if order == 1 {
        let p = -((1.0 / pow10m1(0.1 * rp)).sqrt());
        return Ok(Zpk {
            z: Array1::from_vec(vec![]),
            p: Array1::from_vec(vec![Complex::new(p, 0.0)]),
            k: Complex::new(-p, 0.0),
        });
    }

    // `eps_sq` is the passband ripple parameter, while `ck1_sq` captures the
    // stopband-to-passband ratio that defines the elliptic modulus of the
    // prototype.
    let eps_sq = pow10m1(0.1 * rp);
    let eps = eps_sq.sqrt();
    let ck1_sq = eps_sq / pow10m1(0.1 * rs);
    if ck1_sq == 0.0 || !ck1_sq.is_finite() || !(0.0..1.0).contains(&ck1_sq) {
        return Err(ErrorsJSL::InvalidInputRange(
            "cannot design an elliptic prototype with the given rp and rs",
        ));
    }

    let val0 = complete_elliptic_k(ck1_sq)?;
    let m = ellipdeg(order, ck1_sq)?;
    let capk = complete_elliptic_k(m)?;

    // The zero locations are set by the Jacobi `sn` samples on the prototype's
    // quarter period grid, and they appear in conjugate pairs on the imaginary
    // axis.
    let start = 1 - (order % 2);
    let j_values = (start..order).step_by(2).collect::<Vec<_>>();

    let mut s = Vec::with_capacity(j_values.len());
    let mut c = Vec::with_capacity(j_values.len());
    let mut d = Vec::with_capacity(j_values.len());
    for &j in &j_values {
        let u = j as f64 * capk / order as f64;
        let (sn, cn, dn) = jacobi_ellipj_real(u, m)?;
        s.push(sn);
        c.push(cn);
        d.push(dn);
    }

    let zeros_pos = s
        .iter()
        .copied()
        .filter(|sn| sn.abs() > ELLIP_COMPLEX_EPS)
        .map(|sn| Complex::new(0.0, 1.0 / (m.sqrt() * sn)))
        .collect::<Vec<_>>();
    let mut zeros = zeros_pos.clone();
    zeros.extend(zeros_pos.iter().map(|z| z.conj()));

    // `v0` is the pole-shift parameter in the standard elliptic prototype
    // formulas. It is derived through the inverse Jacobi `sc` term.
    let r = arc_jac_sc1(1.0 / eps, ck1_sq)?;
    let v0 = capk * r / (order as f64 * val0);
    let (sv, cv, dv) = jacobi_ellipj_real(v0, 1.0 - m)?;

    let poles_seed = s
        .iter()
        .zip(c.iter())
        .zip(d.iter())
        .map(|((&sn, &cn), &dn_i)| {
            let num = Complex::new(cn * dn_i * sv * cv, sn * dv);
            let den = 1.0 - (dn_i * sv).powi(2);
            -num / den
        })
        .collect::<Vec<_>>();

    let mut poles = poles_seed.clone();
    if order % 2 == 1 {
        let energy = poles_seed
            .iter()
            .map(|p| p.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1.0);
        poles.extend(
            poles_seed
                .iter()
                .copied()
                .filter(|p| p.im.abs() > ELLIP_COMPLEX_EPS * energy)
                .map(|p| p.conj()),
        );
    } else {
        poles.extend(poles_seed.iter().map(|p| p.conj()));
    }

    let zero_prod = if zeros.is_empty() {
        Complex::new(1.0, 0.0)
    } else {
        zeros
            .iter()
            .fold(Complex::new(1.0, 0.0), |acc, &z| acc * (-z))
    };
    let pole_prod = poles
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &p| acc * (-p));
    // Normalize the prototype to the same DC-gain convention SciPy uses.
    let mut gain = pole_prod / zero_prod;
    if order % 2 == 0 {
        gain /= (1.0 + eps_sq).sqrt();
    }

    Ok(Zpk {
        z: Array1::from_vec(zeros),
        p: Array1::from_vec(poles),
        k: Complex::new(gain.re, 0.0),
    })
}

/// Bessel/Thomson analog lowpass prototype (`besselap`) with selectable
/// normalization.
///
/// The Bessel family is all-pole and is valued for nearly linear phase and
/// maximally flat group delay in the passband. The underlying prototype comes
/// from the reverse Bessel polynomial. Different normalizations simply rescale
/// the same pole pattern for different cutoff conventions.
pub fn besselap(order: usize, norm: Option<BesselNorm>) -> Result<Zpk, ErrorsJSL> {
    let norm = norm.unwrap_or(BesselNorm::Phase);
    if order == 0 {
        return Ok(Zpk {
            z: Array1::from_vec(vec![]),
            p: Array1::from_vec(vec![]),
            k: Complex::new(1.0, 0.0),
        });
    }

    let reverse_coeffs = bessel_poly(order, true)
        .into_iter()
        .map(|x| Complex::new(x as f64, 0.0))
        .collect::<Vec<_>>();
    let mut poles = polynomial_roots(&reverse_coeffs)?;
    let a_last = (falling_factorial(2 * order, order) / (1usize << order)) as f64;

    // `Delay` uses the natural reverse-Bessel scaling directly. `Phase` rescales
    // the poles so the phase midpoint sits at 1 rad/s, matching SciPy/MATLAB.
    // `Mag` starts from the delay-normalized prototype and applies an extra
    // frequency shift so the magnitude is -3 dB at 1 rad/s.
    let gain = match norm {
        BesselNorm::Delay => Complex::new(a_last, 0.0),
        BesselNorm::Mag => {
            let norm_factor = bessel_mag_norm_factor(poles.as_slice().unwrap_or(&[]), a_last)?;
            poles = poles.mapv(|p| p / norm_factor);
            Complex::new(a_last / norm_factor.powi(order as i32), 0.0)
        }
        BesselNorm::Phase => {
            let phase_scale = a_last.powf(-1.0 / order as f64);
            poles = poles.mapv(|p| p * phase_scale);
            Complex::new(1.0, 0.0)
        }
    };

    Ok(Zpk {
        z: Array1::from_vec(vec![]),
        p: poles,
        k: gain,
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
    use crate::number_theory::polynomial::polynomial_eval;

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

    #[test]
    fn test_cheb1ap_order_three_has_unity_dc_gain() {
        let zpk = cheb1ap(3, 1.0).unwrap();
        assert!(zpk.z.is_empty());
        assert_eq!(zpk.p.len(), 3);
        let tf = zpk_to_tf(&zpk).unwrap();
        let dc = polynomial_eval(tf.b.as_slice().unwrap_or(&[]), Complex::new(0.0, 0.0))
            / polynomial_eval(tf.a.as_slice().unwrap_or(&[]), Complex::new(0.0, 0.0));
        assert!((dc.norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cheb1ap_even_order_has_ripple_at_dc() {
        let rp = 1.0;
        let zpk = cheb1ap(4, rp).unwrap();
        let tf = zpk_to_tf(&zpk).unwrap();
        let dc = polynomial_eval(tf.b.as_slice().unwrap_or(&[]), Complex::new(0.0, 0.0))
            / polynomial_eval(tf.a.as_slice().unwrap_or(&[]), Complex::new(0.0, 0.0));
        let expected = 10.0_f64.powf(-rp / 20.0);
        assert!((dc.norm() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_cheb2ap_order_three_has_two_imaginary_zeros() {
        let zpk = cheb2ap(3, 40.0).unwrap();
        assert_eq!(zpk.z.len(), 2);
        assert_eq!(zpk.p.len(), 3);
        assert!(zpk.z.iter().all(|z| z.re.abs() < 1e-12));
    }

    #[test]
    fn test_cheb2ap_hits_stopband_at_unity_frequency() {
        let rs = 40.0;
        let zpk = cheb2ap(4, rs).unwrap();
        let tf = zpk_to_tf(&zpk).unwrap();
        let s = Complex::new(0.0, 1.0);
        let h = polynomial_eval(tf.b.as_slice().unwrap_or(&[]), s)
            / polynomial_eval(tf.a.as_slice().unwrap_or(&[]), s);
        let attenuation_db = -20.0 * h.norm().log10();
        assert!((attenuation_db - rs).abs() < 1e-9);
    }

    #[test]
    fn test_ellipap_order_three_has_two_finite_zeros() {
        let zpk = ellipap(3, 1.0, 40.0).unwrap();
        assert_eq!(zpk.z.len(), 2);
        assert_eq!(zpk.p.len(), 3);
        assert!(zpk.z.iter().all(|z| z.re.abs() < 1e-9));
    }

    #[test]
    fn test_ellipap_odd_order_has_unity_dc_gain() {
        let zpk = ellipap(3, 1.0, 40.0).unwrap();
        let tf = zpk_to_tf(&zpk).unwrap();
        let dc = polynomial_eval(tf.b.as_slice().unwrap_or(&[]), Complex::new(0.0, 0.0))
            / polynomial_eval(tf.a.as_slice().unwrap_or(&[]), Complex::new(0.0, 0.0));
        assert!((dc.norm() - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_besselap_phase_norm_has_unity_gain() {
        let zpk = besselap(3, Some(BesselNorm::Phase)).unwrap();
        assert!((zpk.k - Complex::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_besselap_delay_norm_has_expected_gain_constant() {
        let zpk = besselap(3, Some(BesselNorm::Delay)).unwrap();
        assert!((zpk.k - Complex::new(15.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_besselap_mag_norm_hits_minus_three_db_at_unity_frequency() {
        let zpk = besselap(4, Some(BesselNorm::Mag)).unwrap();
        let tf = zpk_to_tf(&zpk).unwrap();
        let s = Complex::new(0.0, 1.0);
        let h = polynomial_eval(tf.b.as_slice().unwrap_or(&[]), s)
            / polynomial_eval(tf.a.as_slice().unwrap_or(&[]), s);
        let attenuation_db = -20.0 * h.norm().log10();
        let expected_db = 10.0 * 2.0_f64.log10();
        assert!((attenuation_db - expected_db).abs() < 1e-6);
    }
}
