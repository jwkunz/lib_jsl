use ndarray::Array1;
use num_complex::Complex;

use crate::prelude::ErrorsJSL;

use super::design::Zpk;

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
    if order.is_multiple_of(2) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continuous::filter::design::zpk_to_tf;
    use lib_jsl_number_theory::polynomial::polynomial_eval;

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
}
