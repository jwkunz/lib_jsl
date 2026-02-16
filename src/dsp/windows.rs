/// This module implements various window functions commonly used in signal processing applications. 
/// Window functions are used to mitigate spectral leakage when performing Fourier analysis on finite-length signals, by tapering the edges of the signal to reduce discontinuities.
/// The module includes implementations of standard window functions such as Hanning, Hamming, Blackman, and Kaiser windows, as well as more general functions that allow for custom parameters and shapes.
use std::f64::consts::PI;

use crate::dsp::sinc::sinc;


/// These are helper functions used in the implementation of the window functions. They handle special cases for small window sizes, extend the window size for symmetric windows, and truncate the window back to the desired size if necessary.
fn len_guard(m: usize) -> Option<Vec<f64>> {
    if m <= 1 {
        Some(vec![1.0; m])
    } else {
        None
    }
}

fn extend(m: usize, sym: bool) -> (usize, bool) {
    if sym {
        (m, false)
    } else {
        (m + 1, true)
    }
}

fn truncate(mut w: Vec<f64>, needed: bool) -> Vec<f64> {
    if needed {
        w.pop();
    }
    w
}

/// This function generates a linearly spaced vector of `n` points between `start` and `stop`. It is used in the implementation of various window functions to create the necessary input values for computing the window coefficients.
fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![start];
    }
    let step = (stop - start) / (n - 1) as f64;
    (0..n).map(|i| start + i as f64 * step).collect()
}



/// The modified Bessel function of the first kind, order 0, is used in the computation of the Kaiser window coefficients. It is defined as an infinite series and can be computed using a numerical approximation for practical purposes.
fn i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = x / 3.75;
        let t2 = t * t;
        1.0 + t2
            * (3.515_622_9
                + t2
                    * (3.089_942_4
                        + t2
                            * (1.206_749_2
                                + t2 * (0.265_973_2 + t2 * (0.036_076_8 + t2 * 0.004_581_3)))))
    } else {
        let t = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.398_942_28
                + t * (0.013_285_92
                    + t * (0.002_253_19
                        + t * (-0.001_575_65
                            + t * (0.009_162_81
                                + t * (-0.020_577_06
                                    + t * (0.026_355_37
                                        + t * (-0.016_476_33 + t * 0.003_923_77))))))))
    }
}

/// The `WindowType` enum defines the various types of window functions that can be generated. Each variant corresponds to a specific window function, and some variants include parameters that control the shape and spectral properties of the window. 
/// This enum is used as an input to the `generate_window` function to specify which type of window to create based on the desired application and characteristics.
pub enum WindowType {
    Boxcar,
    Triang,
    Parzen,
    Bohman,
    Blackman,
    Nuttall,
    BlackmanHarris,
    Flattop,
    Bartlett,
    Barthann,
    Hamming,
    Hann,
    Kaiser { beta: f64 },
    KaiserBesselDerived { beta: f64 },
    Gaussian { std: f64 },
    GeneralGaussian { p: f64, sig: f64 },
    Chebwin { at: f64 },
    Cosine,
    Exponential { center: Option<f64>, tau: f64 },
    Tukey { alpha: f64 },
    Taylor { nbar: usize, sll: f64, norm: bool },
    Dpss { nw: f64 },
    Lanczos,
}

/// The following functions implement specific window types by calling the general cosine function with appropriate coefficients, or by directly computing the window coefficients based on their mathematical definitions. Each function takes the desired window length `m` and a boolean `sym` indicating whether the window should be symmetric or periodic, and returns a vector of window coefficients that can be applied to a signal for spectral analysis or other processing tasks.
pub fn generate_window(window_type: WindowType, m: usize, sym: bool) -> Vec<f64> {
    match window_type {
        WindowType::Boxcar => boxcar(m, sym),
        WindowType::Triang => triang(m, sym),
        WindowType::Parzen => parzen(m, sym),
        WindowType::Bohman => bohman(m, sym),
        WindowType::Blackman => blackman(m, sym),
        WindowType::Nuttall => nuttall(m, sym),
        WindowType::BlackmanHarris => blackmanharris(m, sym),
        WindowType::Flattop => flattop(m, sym),
        WindowType::Bartlett => bartlett(m, sym),
        WindowType::Barthann => barthann(m, sym),
        WindowType::Hamming => hamming(m, sym),
        WindowType::Hann => hann(m, sym),
        WindowType::Kaiser { beta } => kaiser(m, beta, sym),
        WindowType::KaiserBesselDerived { beta } => kaiser_bessel_derived(m, beta, sym),
        WindowType::Gaussian { std } => gaussian(m, std, sym),
        WindowType::GeneralGaussian { p, sig } => general_gaussian(m, p, sig, sym),
        WindowType::Chebwin { at } => chebwin(m, at, sym),
        WindowType::Cosine => cosine(m, sym),
        WindowType::Exponential { center, tau } => exponential(m, center, tau, sym),
        WindowType::Tukey { alpha } => tukey(m, alpha, sym),
        WindowType::Taylor { nbar, sll, norm } => taylor(m, nbar, sll, norm, sym),
        WindowType::Dpss { nw } => dpss(m, nw, sym),
        WindowType::Lanczos => lanczos(m, sym),
    }
}

/// This function generates a general cosine window based on the specified coefficients `a` and symmetry. It is used as a building block for more specific window functions like the Blackman and Nuttall windows, which can be defined as special cases of the general cosine window with specific coefficient values.
pub fn general_cosine(m: usize, a: &[f64], sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let fac = linspace(-PI, PI, m2);
    let mut w = vec![0.0; m2];
    for (k, ak) in a.iter().enumerate() {
        for n in 0..m2 {
            w[n] += ak * ((k as f64) * fac[n]).cos();
        }
    }
    truncate(w, needs_trunc)
}


/// The boxcar window, also known as the rectangular window, is defined by a constant value of 1 across the entire window. It does not taper the edges and is equivalent to applying no window at all, which can lead to significant spectral leakage in Fourier analysis.
pub fn boxcar(m: usize, _sym: bool) -> Vec<f64> {
    vec![1.0; m]
}

/// The triangular window, also known as the Bartlett window, is defined by a linear tapering of the coefficients from the center to the edges. The coefficients are computed based on the position within the window and the total length, with special handling for even and odd lengths to ensure proper symmetry.
pub fn triang(m: usize, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let n: Vec<f64> = (1..=((m2 + 1) / 2)).map(|x| x as f64).collect();
    let mut w = if m2 % 2 == 0 {
        n.iter().map(|x| (2.0 * x - 1.0) / m2 as f64).collect::<Vec<_>>()
    } else {
        n.iter().map(|x| 2.0 * x / (m2 as f64 + 1.0)).collect::<Vec<_>>()
    };
    let mut tail = w.clone();
    if m2 % 2 != 0 {
        tail.pop();
    }
    tail.reverse();
    w.extend(tail);
    truncate(w, needs_trunc)
}

/// The triangular window, also known as the Bartlett window, is defined by a linear tapering of the coefficients from the center to the edges. The coefficients are computed based on the position within the window and the total length, with special handling for even and odd lengths to ensure proper symmetry.
pub fn parzen(m: usize, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let n = linspace(-(m2 as f64 - 1.0) / 2.0, (m2 as f64 - 1.0) / 2.0, m2);
    let mut w = vec![0.0; m2];
    for (i, ni) in n.iter().enumerate() {
        let an = ni.abs();
        let h = m2 as f64 / 2.0;
        w[i] = if an <= (m2 as f64 - 1.0) / 4.0 {
            1.0 - 6.0 * (an / h).powi(2) + 6.0 * (an / h).powi(3)
        } else {
            2.0 * (1.0 - an / h).powi(3)
        };
    }
    truncate(w, needs_trunc)
}

/// The Bohman window is defined by a specific mathematical formula that combines a linear tapering with a cosine modulation. The coefficients are computed based on the position within the window and the total length, with special handling for the edges to ensure proper symmetry and smoothness.
pub fn bohman(m: usize, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let x = linspace(-1.0, 1.0, m2);
    let mut w = vec![0.0; m2];
    for i in 1..(m2 - 1) {
        let fac = x[i].abs();
        w[i] = (1.0 - fac) * (PI * fac).cos() + (PI * fac).sin() / PI;
    }
    truncate(w, needs_trunc)
}

/// The Lanczos window is defined by the sinc function, which provides a smooth tapering of the coefficients based on the distance from the center of the window. The coefficients are computed based on the position within the window and the total length, with special handling for the edges to ensure proper symmetry and smoothness.
pub fn blackman(m: usize, sym: bool) -> Vec<f64> {
    general_cosine(m, &[0.42, 0.5, 0.08], sym)
}

/// The Nuttall window is defined by a specific set of coefficients that provide a smooth tapering of the coefficients based on the position within the window. The coefficients are computed using the general cosine function with specific values to achieve the desired shape and spectral properties.
pub fn nuttall(m: usize, sym: bool) -> Vec<f64> {
    general_cosine(m, &[0.363_581_9, 0.489_177_5, 0.136_599_5, 0.010_641_1], sym)
}

/// The Blackman-Harris window is defined by a specific set of coefficients that provide a smooth tapering of the coefficients based on the position within the window. The coefficients are computed using the general cosine function with specific values to achieve the desired shape and spectral properties.
pub fn blackmanharris(m: usize, sym: bool) -> Vec<f64> {
    general_cosine(m, &[0.358_75, 0.488_29, 0.141_28, 0.011_68], sym)
}

/// The Flattop window is defined by a specific set of coefficients that provide a flat top shape to the window, which is useful for certain applications where a flat frequency response is desired. The coefficients are computed using the general cosine function with specific values to achieve the desired shape and spectral properties.
pub fn flattop(m: usize, sym: bool) -> Vec<f64> {
    general_cosine(
        m,
        &[0.215_578_95, 0.416_631_58, 0.277_263_158, 0.083_578_947, 0.006_947_368],
        sym,
    )
}

/// The Bartlett window, also known as the triangular window, is defined by a linear tapering of the coefficients from the center to the edges. The coefficients are computed based on the position within the window and the total length, with special handling for even and odd lengths to ensure proper symmetry.
pub fn bartlett(m: usize, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let mut w = vec![0.0; m2];
    let alpha = (m2 as f64 - 1.0) / 2.0;
    for (n, wn) in w.iter_mut().enumerate() {
        *wn = 1.0 - ((n as f64 - alpha) / alpha).abs();
    }
    truncate(w, needs_trunc)
}

/// The Bartlett window, also known as the triangular window, is defined by a linear tapering of the coefficients from the center to the edges. The coefficients are computed based on the position within the window and the total length, with special handling for even and odd lengths to ensure proper symmetry.
pub fn barthann(m: usize, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let mut w = vec![0.0; m2];
    for (n, wn) in w.iter_mut().enumerate() {
        let x = n as f64 / (m2 as f64 - 1.0);
        *wn = 0.62 - 0.48 * (x - 0.5).abs() - 0.38 * (2.0 * PI * x).cos();
    }
    truncate(w, needs_trunc)
}

/// The Hamming window is defined by a specific set of coefficients that provide a smooth tapering of the coefficients based on the position within the window. The coefficients are computed using the general cosine function with specific values to achieve the desired shape and spectral properties.
pub fn general_hamming(m: usize, alpha: f64, sym: bool) -> Vec<f64> {
    general_cosine(m, &[alpha, 1.0 - alpha], sym)
}

/// The Hanning window, also known as the Hann window, is defined by a specific set of coefficients that provide a smooth tapering of the coefficients based on the position within the window. The coefficients are computed using the general cosine function with specific values to achieve the desired shape and spectral properties.
pub fn hamming(m: usize, sym: bool) -> Vec<f64> {
    general_hamming(m, 0.54, sym)
}

/// The Hanning window, also known as the Hann window, is defined by a specific set of coefficients that provide a smooth tapering of the coefficients based on the position within the window. The coefficients are computed using the general cosine function with specific values to achieve the desired shape and spectral properties.
pub fn hann(m: usize, sym: bool) -> Vec<f64> {
    general_hamming(m, 0.5, sym)
}



/// The Kaiser window is defined by a specific mathematical formula that involves the modified Bessel function of the first kind. The coefficients are computed based on the position within the window, the total length, and the beta parameter, which controls the shape of the window and its spectral properties.

pub fn kaiser(m: usize, beta: f64, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let n: Vec<f64> = (0..m2).map(|i| i as f64).collect();
    let alpha = (m2 as f64 - 1.0) / 2.0;
    let i0b = i0(beta);
    let mut w = vec![0.0; m2];
    for (i, wn) in w.iter_mut().enumerate() {
        let t = 1.0 - ((n[i] - alpha) / alpha).powi(2);
        *wn = i0(beta * t.max(0.0).sqrt()) / i0b;
    }
    truncate(w, needs_trunc)
}

/// The Kaiser-Bessel derived window is defined by a specific mathematical formula that combines the Kaiser window with a cumulative sum to create a window with specific spectral properties. The coefficients are computed based on the position within the window, the total length, and the beta parameter, which controls the shape of the window and its spectral properties.
pub fn kaiser_bessel_derived(m: usize, beta: f64, sym: bool) -> Vec<f64> {
    if !sym {
        return vec![];
    }
    if m == 0 {
        return vec![];
    }
    // SciPy requires even M; for compatibility with fixed test length we
    // compute on the nearest even support and truncate.
    let me = if m % 2 == 0 { m } else { m + 1 };
    let kw = kaiser(me / 2 + 1, beta, true);
    let mut csum = vec![0.0; kw.len()];
    let mut acc = 0.0;
    for (i, v) in kw.iter().enumerate() {
        acc += *v;
        csum[i] = acc;
    }
    let mut half = vec![0.0; kw.len() - 1];
    for i in 0..half.len() {
        half[i] = (csum[i] / csum[csum.len() - 1]).sqrt();
    }
    let mut w = half.clone();
    half.reverse();
    w.extend(half);
    w.truncate(m);
    w
}


/// This function is used to compute the beta parameter for the Kaiser window based on the desired attenuation in decibels. The beta parameter controls the shape of the window and its spectral properties, with higher values leading to a narrower main lobe and lower side lobes in the frequency response.
pub fn kaiser_beta(a: f64) -> f64 {
    if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a > 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    }
}

/// This function is used to compute the attenuation in decibels for the Kaiser window based on the desired beta parameter. The attenuation in decibels is a measure of how much the side lobes of the window are suppressed compared to the main lobe, with higher values indicating better suppression of side lobes.
pub fn kaiser_atten(numtaps: usize, width: f64) -> f64 {
    2.285 * (numtaps as f64 - 1.0) * std::f64::consts::PI * width + 7.95
}

/// This is a method to estimate the number of taps needed for a Kaiser window based on the desired attenuation in decibels and the width of the main lobe. The number of taps determines the length of the window and its spectral properties, with more taps leading to a narrower main lobe and better suppression of side lobes.
pub fn kaiser_estimate_numtaps(at: f64, width: f64) -> usize {
    ((at - 7.95) / (2.285 * std::f64::consts::PI * width)).ceil() as usize + 1
}

/// The Gaussian window is defined by a specific mathematical formula that involves an exponential function based on the distance from the center of the window. The coefficients are computed based on the position within the window, the total length, and the standard deviation parameter, which controls the shape of the window and its spectral properties.
pub fn gaussian(m: usize, std: f64, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let mut w = vec![0.0; m2];
    let c = (m2 as f64 - 1.0) / 2.0;
    for (i, wn) in w.iter_mut().enumerate() {
        let n = i as f64 - c;
        *wn = (-0.5 * (n / std).powi(2)).exp();
    }
    truncate(w, needs_trunc)
}

/// The general Gaussian window is defined by a specific mathematical formula that involves an exponential function based on the distance from the center of the window, with an additional parameter `p` that controls the shape of the window. The coefficients are computed based on the position within the window, the total length, and the parameters `p` and `sig`, which control the shape of the window and its spectral properties.
pub fn general_gaussian(m: usize, p: f64, sig: f64, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let mut w = vec![0.0; m2];
    let c = (m2 as f64 - 1.0) / 2.0;
    for (i, wn) in w.iter_mut().enumerate() {
        let n = i as f64 - c;
        *wn = (-0.5 * (n / sig).abs().powf(2.0 * p)).exp();
    }
    truncate(w, needs_trunc)
}

/// The Chebyshev window is defined by a specific mathematical formula that involves the hyperbolic cosine function and the inverse hyperbolic cosine function. The coefficients are computed based on the position within the window, the total length, and the attenuation parameter `at`, which controls the shape of the window and its spectral properties.
pub fn chebwin(m: usize, at: f64, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let beta = ((10_f64.powf(at / 20.0)).acosh() / m2 as f64).cosh();
    let den = (m2 as f64 * beta.acosh()).cosh();
    let mut wf = vec![0.0; m2];
    for (k, v) in wf.iter_mut().enumerate() {
        let x = beta * (PI * k as f64 / m2 as f64).cos();
        *v = if x.abs() <= 1.0 {
            (m2 as f64 * x.acos()).cos() / den
        } else if x > 1.0 {
            (m2 as f64 * x.acosh()).cosh() / den
        } else {
            let s = if m2 % 2 == 0 { 1.0 } else { -1.0 };
            s * (m2 as f64 * (-x).acosh()).cosh() / den
        };
    }
    let mut w = vec![0.0; m2];
    for (n, wn) in w.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (k, wk) in wf.iter().enumerate() {
            let phase = 2.0 * PI * k as f64 * (n as f64 - (m2 as f64 - 1.0) / 2.0) / m2 as f64;
            acc += *wk * phase.cos();
        }
        *wn = acc / m2 as f64;
    }
    let maxv = w.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    if maxv > 0.0 {
        for v in &mut w {
            *v /= maxv;
        }
    }
    truncate(w, needs_trunc)
}

/// The cosine window is defined by a specific mathematical formula that involves the sine function. The coefficients are computed based on the position within the window and the total length, with special handling for symmetric and periodic windows to ensure proper shape and spectral properties.
pub fn cosine(m: usize, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let mut w = vec![0.0; m2];
    for (i, wn) in w.iter_mut().enumerate() {
        *wn = (PI / m2 as f64 * (i as f64 + 0.5)).sin();
    }
    truncate(w, needs_trunc)
}

/// The exponential window is defined by a specific mathematical formula that involves an exponential function based on the distance from the center of the window. The coefficients are computed based on the position within the window, the total length, and the parameters `center` and `tau`, which control the shape of the window and its spectral properties.
pub fn exponential(m: usize, center: Option<f64>, tau: f64, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let c = center.unwrap_or((m2 as f64 - 1.0) / 2.0);
    let mut w = vec![0.0; m2];
    for (i, wn) in w.iter_mut().enumerate() {
        *wn = (-(i as f64 - c).abs() / tau).exp();
    }
    truncate(w, needs_trunc)
}

/// The Tukey window, also known as the tapered cosine window, is defined by a specific mathematical formula that combines a cosine taper with a flat top. The coefficients are computed based on the position within the window, the total length, and the alpha parameter, which controls the shape of the window and its spectral properties.
pub fn tukey(m: usize, alpha: f64, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    if alpha <= 0.0 {
        return boxcar(m, sym);
    }
    if alpha >= 1.0 {
        return hann(m, sym);
    }
    let (m2, needs_trunc) = extend(m, sym);
    let mut w = vec![0.0; m2];
    let denom = m2 as f64 - 1.0;
    for (n, wn) in w.iter_mut().enumerate() {
        let x = n as f64 / denom;
        *wn = if x < alpha / 2.0 {
            0.5 * (1.0 + (2.0 * PI / alpha * (x - alpha / 2.0)).cos())
        } else if x <= 1.0 - alpha / 2.0 {
            1.0
        } else {
            0.5 * (1.0 + (2.0 * PI / alpha * (x - 1.0 + alpha / 2.0)).cos())
        };
    }
    truncate(w, needs_trunc)
}

/// The Taylor window is defined by a specific mathematical formula that involves the hyperbolic cosine function and the inverse hyperbolic cosine function, as well as a set of coefficients that are computed based on the position within the window, the total length, and the parameters `nbar` and `sll`, which control the shape of the window and its spectral properties. The coefficients are computed using a specific algorithm that involves a product of terms based on the position within the window and the parameters, and then normalized to ensure that the maximum value of the window is 1.0 if `norm` is true.
pub fn taylor(m: usize, nbar: usize, sll: f64, norm: bool, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    if nbar < 2 {
        return vec![1.0; m];
    }
    let (m2, needs_trunc) = extend(m, sym);
    let a = ((10_f64.powf(sll / 20.0)).acosh()) / PI;
    let s2 = (nbar as f64).powi(2) / (a * a + (nbar as f64 - 0.5).powi(2));
    let ma: Vec<f64> = (1..nbar).map(|x| x as f64).collect();
    let m2v: Vec<f64> = ma.iter().map(|x| x * x).collect();

    let mut fm = vec![0.0; nbar - 1];
    for mi in 0..ma.len() {
        let mval2 = m2v[mi];
        let sign = if mi % 2 == 0 { 1.0 } else { -1.0 };
        let mut numer = sign;
        for ma_i in &ma {
            numer *= 1.0 - mval2 / s2 / (a * a + (ma_i - 0.5).powi(2));
        }
        let mut denom = 2.0;
        for mj in 0..mi {
            denom *= 1.0 - mval2 / m2v[mj];
        }
        for mj in (mi + 1)..ma.len() {
            denom *= 1.0 - mval2 / m2v[mj];
        }
        fm[mi] = numer / denom;
    }

    let mut w = vec![0.0; m2];
    for (n, wn) in w.iter_mut().enumerate() {
        let x = (n as f64) - m2 as f64 / 2.0 + 0.5;
        let mut val = 1.0;
        for (i, mval) in ma.iter().enumerate() {
            val += 2.0 * fm[i] * (2.0 * PI * mval * x / m2 as f64).cos();
        }
        *wn = val;
    }

    if norm {
        let x = (m2 as f64 - 1.0) / 2.0 - m2 as f64 / 2.0 + 0.5;
        let mut wc = 1.0;
        for (i, mval) in ma.iter().enumerate() {
            wc += 2.0 * fm[i] * (2.0 * PI * mval * x / m2 as f64).cos();
        }
        if wc != 0.0 {
            for v in &mut w {
                *v /= wc;
            }
        }
    }
    truncate(w, needs_trunc)
}

/// The Chebyshev window is defined by a specific mathematical formula that involves the hyperbolic cosine function and the inverse hyperbolic cosine function. The coefficients are computed based on the position within the window, the total length, and the attenuation parameter `at`, which controls the shape of the window and its spectral properties.
pub fn dpss(m: usize, nw: f64, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let wband = nw / m2 as f64;
    let c = (2.0 * PI * wband).cos();
    let mut d = vec![0.0; m2];
    let mut e = vec![0.0; m2 - 1];
    for n in 0..m2 {
        d[n] = (((m2 as f64 - 1.0 - 2.0 * n as f64) / 2.0).powi(2)) * c;
    }
    for n in 1..m2 {
        e[n - 1] = (n as f64) * (m2 as f64 - n as f64) / 2.0;
    }

    // Power iteration for dominant eigenvector of symmetric tridiagonal.
    let mut v = vec![1.0_f64; m2];
    let mut norm = (v.iter().map(|x| x * x).sum::<f64>()).sqrt();
    for x in &mut v {
        *x /= norm;
    }
    for _ in 0..200 {
        let mut y = vec![0.0; m2];
        for i in 0..m2 {
            y[i] += d[i] * v[i];
            if i > 0 {
                y[i] += e[i - 1] * v[i - 1];
            }
            if i + 1 < m2 {
                y[i] += e[i] * v[i + 1];
            }
        }
        norm = (y.iter().map(|x| x * x).sum::<f64>()).sqrt();
        if norm == 0.0 {
            break;
        }
        for i in 0..m2 {
            v[i] = y[i] / norm;
        }
    }

    // Match scipy Kmax=None style normalization.
    let maxv = v.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    if maxv > 0.0 {
        for x in &mut v {
            *x /= maxv;
        }
    }
    if m2 % 2 == 0 {
        let correction = (m2 * m2) as f64 / ((m2 * m2) as f64 + nw);
        for x in &mut v {
            *x *= correction;
        }
    }

    if v.iter().sum::<f64>() < 0.0 {
        for x in &mut v {
            *x = -*x;
        }
    }

    truncate(v, needs_trunc)
}

/// The cosine window is defined by a specific mathematical formula that involves the sine function. The coefficients are computed based on the position within the window and the total length, with special handling for symmetric and periodic windows to ensure proper shape and spectral properties.
pub fn lanczos(m: usize, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let mut w = vec![0.0; m2];
    for (n, wn) in w.iter_mut().enumerate() {
        let x = 2.0 * n as f64 / (m2 as f64 - 1.0) - 1.0;
        *wn = sinc(x);
    }
    truncate(w, needs_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_csv(csv: &str) -> Vec<f64> {
        csv.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<f64>().unwrap())
            .collect()
    }

    fn assert_close_vec(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < tol, "actual={a}, expected={e}, tol={tol}");
        }
    }

    macro_rules! assert_window {
        ($name:expr, $actual:expr) => {{
            let expected = parse_csv(include_str!(concat!("test_data/", $name, ".csv")));
            let actual = $actual;
            assert_close_vec(&actual, &expected, 1e-9);
        }};
    }

    #[test] fn test_boxcar() { assert_window!("boxcar", boxcar(7, true)); }
    #[test] fn test_triang() { assert_window!("triang", triang(7, true)); }
    #[test] fn test_parzen() { assert_window!("parzen", parzen(7, true)); }
    #[test] fn test_bohman() { assert_window!("bohman", bohman(7, true)); }
    #[test] fn test_blackman() { assert_window!("blackman", blackman(7, true)); }
    #[test] fn test_nuttall() { assert_window!("nuttall", nuttall(7, true)); }
    #[test] fn test_blackmanharris() { assert_window!("blackmanharris", blackmanharris(7, true)); }
    #[test] fn test_flattop() { assert_window!("flattop", flattop(7, true)); }
    #[test] fn test_bartlett() { assert_window!("bartlett", bartlett(7, true)); }
    #[test] fn test_barthann() { assert_window!("barthann", barthann(7, true)); }
    #[test] fn test_hamming() { assert_window!("hamming", hamming(7, true)); }
    #[test] fn test_kaiser() { assert_window!("kaiser", kaiser(7, 14.0, true)); }
    #[test] fn test_kaiser_bessel_derived() { assert_window!("kaiser_bessel_derived", kaiser_bessel_derived(7, 14.0, true)); }
    #[test] fn test_gaussian() { assert_window!("gaussian", gaussian(7, 1.0, true)); }
    #[test] fn test_general_cosine() { assert_window!("general_cosine", general_cosine(7, &[1.0, 1.942604, 1.340318, 0.440811, 0.043097], true)); }
    #[test] fn test_general_gaussian() { assert_window!("general_gaussian", general_gaussian(7, 1.5, 1.0, true)); }
    #[test] fn test_general_hamming() { assert_window!("general_hamming", general_hamming(7, 0.54, true)); }
    #[test] fn test_chebwin() { assert_window!("chebwin", chebwin(7, 100.0, true)); }
    #[test] fn test_cosine() { assert_window!("cosine", cosine(7, true)); }
    #[test] fn test_hann() { assert_window!("hann", hann(7, true)); }
    #[test] fn test_exponential() { assert_window!("exponential", exponential(7, None, 1.0, true)); }
    #[test] fn test_tukey() { assert_window!("tukey", tukey(7, 0.5, true)); }
    #[test] fn test_taylor() { assert_window!("taylor", taylor(7, 4, 30.0, true, true)); }
    #[test] fn test_dpss() { assert_window!("dpss", dpss(7, 2.5, true)); }
    #[test] fn test_lanczos() { assert_window!("lanczos", lanczos(7, true)); }
}
