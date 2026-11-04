//! This module implements various window functions commonly used in signal processing applications. 
//! Window functions are used to mitigate spectral leakage when performing Fourier analysis on finite-length signals, by tapering the edges of the signal to reduce discontinuities.
//! The module includes implementations of standard window functions such as Hanning, Hamming, Blackman, and Kaiser windows, as well as more general functions that allow for custom parameters and shapes.
//!
//! See the book entry [Window functions](https://jwkunz.github.io/lib_jsl/book/spectral/windows.html).

pub mod boxcar;
pub mod triang;
pub mod parzen;
pub mod bohman;
pub mod general_cosine;
pub mod blackman;
pub mod nuttall;
pub mod blackmanharris;
pub mod flattop;
pub mod bartlett;
pub mod barthann;
pub mod hamming;
pub mod hann;
pub mod kaiser;
pub mod kaiser_bessel_derived;
pub mod gaussian;
pub mod general_gaussian;
pub mod chebwin;
pub mod cosine;
pub mod exponential;
pub mod tukey;
pub mod taylor;
pub mod dpss;
pub mod lanczos;

pub use boxcar::{boxcar};
pub use triang::{triang};
pub use parzen::{parzen};
pub use bohman::{bohman};
pub use general_cosine::{general_cosine};
pub use blackman::{blackman};
pub use nuttall::{nuttall};
pub use blackmanharris::{blackmanharris};
pub use flattop::{flattop};
pub use bartlett::{bartlett};
pub use barthann::{barthann};
pub use hamming::{general_hamming, hamming};
pub use hann::{hann};
pub use kaiser::{kaiser, kaiser_beta, kaiser_atten, kaiser_estimate_numtaps};
pub use kaiser_bessel_derived::{kaiser_bessel_derived};
pub use gaussian::{gaussian};
pub use general_gaussian::{general_gaussian};
pub use chebwin::{chebwin};
pub use cosine::{cosine};
pub use exponential::{exponential};
pub use tukey::{tukey};
pub use taylor::{taylor};
pub use dpss::{dpss};
pub use lanczos::{lanczos};

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

#[cfg(test)]
pub(crate) mod test_support {
    fn parse_csv(csv: &str) -> Vec<f64> {
        csv.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<f64>().unwrap())
            .collect()
    }

    /// Compares a generated window against golden CSV contents to 1e-9.
    pub(crate) fn assert_window(golden_csv: &str, actual: Vec<f64>) {
        let expected = parse_csv(golden_csv);
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-9, "actual={a}, expected={e}, tol=1e-9");
        }
    }
}
