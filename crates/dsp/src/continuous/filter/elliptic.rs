use ndarray::Array1;
use num_complex::Complex;

use crate::prelude::ErrorsJSL;

use super::design::Zpk;

const ELLIPDEG_MMAX: usize = 7;
const ARC_JAC_SN_MAXITER: usize = 10;
const ELLIPJ_EPS: f64 = 1e-12;
const ELLIPJ_MAXITER: usize = 16;
const ELLIP_COMPLEX_EPS: f64 = 2e-16;

fn pow10m1(x: f64) -> f64 {
    (std::f64::consts::LN_10 * x).exp_m1()
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
    if order.is_multiple_of(2) {
        gain /= (1.0 + eps_sq).sqrt();
    }

    Ok(Zpk {
        z: Array1::from_vec(zeros),
        p: Array1::from_vec(poles),
        k: Complex::new(gain.re, 0.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continuous::filter::design::zpk_to_tf;
    use lib_jsl_number_theory::polynomial::polynomial_eval;

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
}
