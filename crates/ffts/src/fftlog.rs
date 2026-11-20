//! The fast Hankel transform by Hamilton's FFTLog (SciPy's `fft.fht`, `fft.ifht` and
//! `fft.fhtoffset`), and FFTs of Hermitian-symmetric signals (`fft.hfft`, `fft.ihfft`).
//!
//! The Hankel transform `A(k) = ∫₀^∞ a(r) J_μ(kr) k dr` of a function sampled at logarithmically
//! spaced points becomes a convolution in `ln r`, which FFTLog evaluates with two real FFTs and a
//! multiplication by the Mellin transform of the Bessel kernel, a ratio of gamma functions.
//!
//! See the book entry [Real QZ, reordering and Hankel transforms](https://jwkunz.github.io/lib_jsl/book/linear_algebra/qz-reordering.html#the-fast-hankel-transform).

use crate::real_fft::{irfft, rfft};
use lib_jsl_core::ErrorsJSL;
use num_complex::Complex;

type C64 = Complex<f64>;

const LN_2: f64 = std::f64::consts::LN_2;

/// `ln Γ(z)` for complex `z` away from the poles: the recurrence `Γ(z) = Γ(z + N)/(z(z+1)…)`
/// moves the argument to `|z| ≥ 15`, where Stirling's series with eight terms is accurate to
/// rounding. (The imaginary part is determined only modulo 2π, which is all FFTLog needs.)
pub fn ln_gamma_complex(z: C64) -> C64 {
    let mut z = z;
    let mut shift = C64::new(0.0, 0.0);
    while z.re < 15.0 || z.norm() < 15.0 {
        shift += z.ln();
        z += 1.0;
        if z.re > 1e6 {
            break;
        }
    }
    const B: [f64; 8] = [1.0 / 6.0, -1.0 / 30.0, 1.0 / 42.0, -1.0 / 30.0, 5.0 / 66.0, -691.0 / 2730.0, 7.0 / 6.0, -3617.0 / 510.0];
    let mut series = C64::new(0.0, 0.0);
    let z2 = z * z;
    let mut zp = z;
    for (k, b) in B.iter().enumerate() {
        let k2 = 2.0 * (k + 1) as f64;
        series += b / (k2 * (k2 - 1.0)) / zp;
        zp *= z2;
    }
    (z - 0.5) * z.ln() - z + 0.5 * (2.0 * std::f64::consts::PI).ln() + series - shift
}

/// `Γ(a + m)/Γ(a)` at real arguments, with the limits at the poles.
fn poch(a: f64, m: f64) -> f64 {
    let pole = |x: f64| x <= 0.0 && x == x.floor();
    let b = a + m;
    match (pole(b), pole(a)) {
        (true, true) => {
            // Ratio of residues (−1)ⁿ/n!.
            let (nb, na) = (-b, -a);
            let sign = if ((nb - na) as i64).rem_euclid(2) == 0 { 1.0 } else { -1.0 };
            sign * (ln_gamma_complex(C64::new(na + 1.0, 0.0)).re - ln_gamma_complex(C64::new(nb + 1.0, 0.0)).re).exp()
        }
        (true, false) => f64::INFINITY,
        (false, true) => 0.0,
        _ => {
            let g = |x: f64| -> (f64, f64) {
                let l = ln_gamma_complex(C64::new(x, 0.0));
                let sign = if x > 0.0 || (x.floor() as i64).rem_euclid(2) == 0 { 1.0 } else { -1.0 };
                (l.re, sign)
            };
            let (lb, sb) = g(b);
            let (la, sa) = g(a);
            sb * sa * (lb - la).exp()
        }
    }
}

/// The FFTLog coefficients `u_m` (SciPy's `fhtcoeff`).
fn fhtcoeff(n: usize, dln: f64, mu: f64, offset: f64, bias: f64, inverse: bool) -> Vec<C64> {
    let (lnkr, q) = (offset, bias);
    let xp = (mu + 1.0 + q) / 2.0;
    let xm = (mu + 1.0 - q) / 2.0;
    let m = n / 2 + 1;
    let mut u: Vec<C64> = (0..m)
        .map(|i| {
            let y = if m > 1 { std::f64::consts::PI * (n / 2) as f64 / (n as f64 * dln) * i as f64 / (m - 1) as f64 } else { 0.0 };
            let v = ln_gamma_complex(C64::new(xm, y));
            let w = ln_gamma_complex(C64::new(xp, y));
            let re = w.re - v.re + LN_2 * q;
            let im = w.im + v.im + 2.0 * (LN_2 - lnkr) * y;
            C64::new(re, im).exp()
        })
        .collect();
    if n.is_multiple_of(2) {
        let last = u.len() - 1;
        u[last].im = 0.0;
    }
    if !(u[0].re.is_finite() && u[0].im.is_finite()) {
        u[0] = C64::new(2f64.powf(q) * poch(xm, xp - xm), 0.0);
    }
    if !inverse && u[0].re.is_infinite() {
        u[0] = C64::new(0.0, 0.0);
    } else if inverse && u[0] == C64::new(0.0, 0.0) {
        u[0] = C64::new(f64::INFINITY, 0.0);
    }
    u
}

fn fhtq(a: &[f64], u: &[C64], inverse: bool) -> Result<Vec<f64>, ErrorsJSL> {
    let n = a.len();
    let mut spec = rfft(a)?;
    for (s, &c) in spec.iter_mut().zip(u) {
        *s = if inverse { *s / c.conj() } else { *s * c };
    }
    let mut out = irfft(&spec, n)?;
    out.reverse();
    Ok(out)
}

// ANCHOR: fht
/// The fast Hankel transform (SciPy's `fht`) of `a`, sampled at `r_j = r_c exp((j − j_c) dln)`,
/// `j_c = (n − 1)/2`: approximately `A(k_j) = ∫₀^∞ a(r) J_μ(k_j r) k_j dr` at
/// `k_j = exp(offset)/r_c · exp((j − j_c) dln)`. With `bias = q` the input is multiplied by
/// `(r/r_c)^{−q}` and the output by `(k/k_c)^{−q}`, which can reduce ringing. FFTLog computes it
/// exactly for the periodic, log-spaced interpolant of `a`: an rFFT, multiplication by
/// `u_m = 2^{q} Γ((μ+1+q+iy_m)/2)/Γ((μ+1−q−iy_m)/2) · e^{i y_m (ln 2 − offset)}`, an inverse rFFT
/// and a reversal.
pub fn fht(a: &[f64], dln: f64, mu: f64, offset: f64, bias: f64) -> Result<Vec<f64>, ErrorsJSL> {
    let n = a.len();
    if n == 0 || dln.is_nan() || dln <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("fht: need a non-empty input and dln > 0"));
    }
    let jc = (n as f64 - 1.0) / 2.0;
    let biased: Vec<f64> = if bias != 0.0 { a.iter().enumerate().map(|(j, &v)| v * (-bias * (j as f64 - jc) * dln).exp()).collect() } else { a.to_vec() };
    let u = fhtcoeff(n, dln, mu, offset, bias, false);
    let mut out = fhtq(&biased, &u, false)?;
    if bias != 0.0 {
        for (j, v) in out.iter_mut().enumerate() {
            *v *= (-bias * ((j as f64 - jc) * dln + offset)).exp();
        }
    }
    Ok(out)
}
// ANCHOR_END: fht

/// The inverse fast Hankel transform (SciPy's `ifht`), undoing [`fht`] with the same parameters.
pub fn ifht(big_a: &[f64], dln: f64, mu: f64, offset: f64, bias: f64) -> Result<Vec<f64>, ErrorsJSL> {
    let n = big_a.len();
    if n == 0 || dln.is_nan() || dln <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("ifht: need a non-empty input and dln > 0"));
    }
    let jc = (n as f64 - 1.0) / 2.0;
    let biased: Vec<f64> = if bias != 0.0 { big_a.iter().enumerate().map(|(j, &v)| v * (bias * ((j as f64 - jc) * dln + offset)).exp()).collect() } else { big_a.to_vec() };
    let u = fhtcoeff(n, dln, mu, offset, bias, true);
    let mut out = fhtq(&biased, &u, true)?;
    if bias != 0.0 {
        for (j, v) in out.iter_mut().enumerate() {
            *v /= (-bias * (j as f64 - jc) * dln).exp();
        }
    }
    Ok(out)
}

/// The offset closest to `initial` that satisfies Hamilton's low-ringing condition for the given
/// spacing, order and bias (SciPy's `fhtoffset`).
pub fn fhtoffset(dln: f64, mu: f64, initial: f64, bias: f64) -> f64 {
    let (lnkr, q) = (initial, bias);
    let xp = (mu + 1.0 + q) / 2.0;
    let xm = (mu + 1.0 - q) / 2.0;
    let y = std::f64::consts::PI / (2.0 * dln);
    let zp = ln_gamma_complex(C64::new(xp, y));
    let zm = ln_gamma_complex(C64::new(xm, y));
    let arg = (LN_2 - lnkr) / dln + (zp.im + zm.im) / std::f64::consts::PI;
    lnkr + (arg - arg.round_ties_even()) * dln
}

// ANCHOR: hfft
/// The FFT of a signal with Hermitian symmetry, given by its first half (SciPy's `hfft`): a real
/// spectrum of length `n` (default `2(m − 1)` for `m` inputs), `n · irfft(conj(x), n)`.
pub fn hfft(x: &[C64], n: Option<usize>) -> Result<Vec<f64>, ErrorsJSL> {
    let n = n.unwrap_or(2 * x.len().saturating_sub(1));
    if n == 0 {
        return Err(ErrorsJSL::InvalidInputRange("hfft: the output length must be positive"));
    }
    let m = n / 2 + 1;
    let mut c: Vec<C64> = x.iter().take(m).map(|v| v.conj()).collect();
    c.resize(m, C64::new(0.0, 0.0));
    Ok(irfft(&c, n)?.into_iter().map(|v| v * n as f64).collect())
}

/// The inverse of [`hfft`] (SciPy's `ihfft`): the first `n/2 + 1` bins of the Hermitian signal
/// whose FFT is the real `x`, `conj(rfft(x))/n`.
pub fn ihfft(x: &[f64]) -> Result<Vec<C64>, ErrorsJSL> {
    let n = x.len();
    if n == 0 {
        return Err(ErrorsJSL::InvalidInputRange("ihfft: empty input"));
    }
    Ok(rfft(x)?.into_iter().map(|v| v.conj() / n as f64).collect())
}
// ANCHOR_END: hfft

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hankel_transform_of_a_gaussian_and_round_trips() {
        // ∫ r e^{−r²/2} J₀(kr) dr = e^{−k²/2}: with a(r) = r e^{−r²/2}, A(k) = k e^{−k²/2}.
        let n = 128;
        let dln = 0.1;
        let jc = (n as f64 - 1.0) / 2.0;
        let r: Vec<f64> = (0..n).map(|j| ((j as f64 - jc) * dln).exp()).collect();
        let a: Vec<f64> = r.iter().map(|&x| x * (-x * x / 2.0).exp()).collect();
        let offset = fhtoffset(dln, 0.0, 0.0, 0.0);
        let big = fht(&a, dln, 0.0, offset, 0.0).unwrap();
        // SciPy's values (the low-k entries show FFTLog's aliasing, not k e^{−k²/2}).
        assert!((offset - 0.047483887457456575).abs() < 1e-15);
        for (j, want) in [(0, 0.001048823960070111), (10, 0.00441039992322376), (30, 0.03656836034489455), (64, 0.600405791157053), (100, 5.5817333906249544e-05)] {
            assert!((big[j] - want).abs() < 1e-12 * want.abs().max(1e-3), "{j}: {} vs {want}", big[j]);
        }
        let k64 = offset.exp() * ((64.0 - jc) * dln).exp();
        assert!((big[64] - k64 * (-k64 * k64 / 2.0).exp()).abs() < 1e-3);
        let back = ifht(&big, dln, 0.0, offset, 0.0).unwrap();
        assert!(back.iter().zip(&a).all(|(x, y)| (x - y).abs() < 1e-10));
        // Γ(1/2) = √π, Γ(5) = 24, Γ(1 + i) known value.
        assert!((ln_gamma_complex(C64::new(0.5, 0.0)).re - std::f64::consts::PI.sqrt().ln()).abs() < 1e-14);
        assert!((ln_gamma_complex(C64::new(5.0, 0.0)).re - 24f64.ln()).abs() < 1e-14);
        let g = ln_gamma_complex(C64::new(1.0, 1.0)).exp();
        assert!((g - C64::new(0.498015668118356, -0.154949828301811)).norm() < 1e-13);
        // hfft and ihfft invert each other.
        let x: Vec<f64> = (0..10).map(|i| (i as f64 * 0.7).sin() + 0.1 * i as f64).collect();
        let h = ihfft(&x).unwrap();
        let y = hfft(&h, Some(10)).unwrap();
        assert!(x.iter().zip(&y).all(|(p, q)| (p - q).abs() < 1e-13));
    }
}
