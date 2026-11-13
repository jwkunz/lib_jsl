//! Zero-phase (forward-backward) filtering and the steady-state initial conditions it needs.
//!
//! Filtering a signal once with an IIR filter delays its features by a frequency-dependent
//! amount. Filtering it again *backwards* applies the same delay in reverse, so the phase shifts
//! cancel: the result has zero phase and the magnitude response `|H|²`. Offline processing (where
//! the whole record is available) uses this to smooth or band-limit a signal without moving its
//! peaks and edges. The ends of the record need care, since each pass starts with a transient;
//! the record is extended by reflection, and each pass starts from the filter's steady state.
//!
//! See the book entry [Second-order sections and zero-phase filtering](https://jwkunz.github.io/lib_jsl/book/filters/sos-zero-phase.html).

use crate::prelude::ErrorsJSL;

use super::sos::{sosfilt, sosfilt_zi, Section};

/// How [`filtfilt`] and [`sosfiltfilt`] extend the signal at both ends before filtering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Padding {
    /// Point reflection about the end values, `2 x[0] − x[k]` (SciPy's default, `'odd'`): keeps
    /// the signal's value and slope continuous at the ends.
    Odd,
    /// Mirror reflection `x[k]` (`'even'`): keeps the value continuous.
    Even,
    /// Repeat the end values (`'constant'`).
    Constant,
    /// No extension (`padtype=None`).
    None,
}

/// Normalises `(b, a)` by `a[0]` and pads both to the same length.
fn normalise(b: &[f64], a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), ErrorsJSL> {
    if b.is_empty() || a.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("filter coefficients must be non-empty"));
    }
    if a[0] == 0.0 || b.iter().chain(a).any(|v| !v.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("coefficients must be finite with a[0] ≠ 0"));
    }
    let n = b.len().max(a.len());
    let mut bn = vec![0.0; n];
    let mut an = vec![0.0; n];
    for (d, v) in bn.iter_mut().zip(b) {
        *d = v / a[0];
    }
    for (d, v) in an.iter_mut().zip(a) {
        *d = v / a[0];
    }
    Ok((bn, an))
}

// ANCHOR: lfilter
/// Filters `x` with the transfer function `B(z)/A(z)` in transposed direct form II (SciPy's
/// `lfilter`), returning the output and the final state. `zi` is the initial state, of length
/// `max(len(a), len(b)) − 1`; `None` starts from rest.
pub fn lfilter(b: &[f64], a: &[f64], x: &[f64], zi: Option<&[f64]>) -> Result<(Vec<f64>, Vec<f64>), ErrorsJSL> {
    let (b, a) = normalise(b, a)?;
    let order = b.len() - 1;
    let mut z = match zi {
        Some(z) if z.len() != order => return Err(ErrorsJSL::IncompatibleArraySizes((z.len(), order))),
        Some(z) => z.to_vec(),
        None => vec![0.0; order],
    };
    let mut y = Vec::with_capacity(x.len());
    for &input in x {
        let out = b[0] * input + z.first().copied().unwrap_or(0.0);
        for i in 0..order {
            let next = if i + 1 < order { z[i + 1] } else { 0.0 };
            z[i] = b[i + 1] * input - a[i + 1] * out + next;
        }
        y.push(out);
    }
    Ok((y, z))
}
// ANCHOR_END: lfilter

// ANCHOR: lfilter_zi
/// The initial state of [`lfilter`] for the steady state of a unit step (SciPy's `lfilter_zi`).
///
/// In steady state the input and output are constant, `x = 1` and `y = H(1)`, and the state
/// stops changing: `z = Aᵀ_c z + (b[1..] − a[1..] b[0])`, where `A_c` is the companion matrix of
/// the denominator. So `z` solves `(I − A_cᵀ) z = b[1..] − a[1..] b[0]`.
pub fn lfilter_zi(b: &[f64], a: &[f64]) -> Result<Vec<f64>, ErrorsJSL> {
    let (b, a) = normalise(b, a)?;
    let m = b.len() - 1;
    if m == 0 {
        return Ok(Vec::new());
    }
    // I − companion(a)ᵀ: the first column holds 1 + a[1], a[2], …; the superdiagonal holds −1.
    let mut mat = vec![0.0; m * m];
    for i in 0..m {
        mat[i * m + i] = 1.0;
        mat[i * m] += a[i + 1];
        if i + 1 < m {
            mat[i * m + i + 1] = -1.0;
        }
    }
    let mut rhs: Vec<f64> = (0..m).map(|i| b[i + 1] - a[i + 1] * b[0]).collect();
    // Gaussian elimination with partial pivoting.
    for col in 0..m {
        let piv = (col..m).max_by(|&i, &j| mat[i * m + col].abs().total_cmp(&mat[j * m + col].abs())).unwrap();
        if mat[piv * m + col].abs() < 1e-300 {
            return Err(ErrorsJSL::RuntimeError("the filter has a pole at z = 1, so no steady state exists"));
        }
        if piv != col {
            for k in 0..m {
                mat.swap(col * m + k, piv * m + k);
            }
            rhs.swap(col, piv);
        }
        for row in col + 1..m {
            let f = mat[row * m + col] / mat[col * m + col];
            if f != 0.0 {
                for k in col..m {
                    mat[row * m + k] -= f * mat[col * m + k];
                }
                rhs[row] -= f * rhs[col];
            }
        }
    }
    for row in (0..m).rev() {
        let s: f64 = (row + 1..m).map(|k| mat[row * m + k] * rhs[k]).sum();
        rhs[row] = (rhs[row] - s) / mat[row * m + row];
    }
    Ok(rhs)
}
// ANCHOR_END: lfilter_zi

/// Extends `x` by `edge` samples at each end.
fn extend(x: &[f64], edge: usize, padding: Padding) -> Vec<f64> {
    let n = x.len();
    let (first, last) = (x[0], x[n - 1]);
    let mut out = Vec::with_capacity(n + 2 * edge);
    for k in (1..=edge).rev() {
        out.push(match padding {
            Padding::Odd => 2.0 * first - x[k],
            Padding::Even => x[k],
            Padding::Constant | Padding::None => first,
        });
    }
    out.extend_from_slice(x);
    for k in 1..=edge {
        out.push(match padding {
            Padding::Odd => 2.0 * last - x[n - 1 - k],
            Padding::Even => x[n - 1 - k],
            Padding::Constant | Padding::None => last,
        });
    }
    out
}

/// The number of padding samples: `padlen`, or `3 × ntaps` by default (SciPy's rule).
fn edge_length(x: &[f64], padding: Padding, padlen: Option<usize>, ntaps: usize) -> Result<usize, ErrorsJSL> {
    if x.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("the signal is empty"));
    }
    let edge = if padding == Padding::None { 0 } else { padlen.unwrap_or(3 * ntaps) };
    if edge > 0 && x.len() <= edge {
        return Err(ErrorsJSL::InvalidInputRange("the signal must be longer than the padding"));
    }
    Ok(edge)
}

// ANCHOR: filtfilt
/// Zero-phase filtering of `x` by `B(z)/A(z)`, forwards and then backwards (SciPy's `filtfilt`
/// with `method='pad'`).
///
/// The signal is extended by `padlen` samples at each end (default `3 max(len(a), len(b))`), and
/// each pass starts from the steady state for its first input value ([`lfilter_zi`]), so the
/// start-up transients are small. The result has zero phase and magnitude response `|H|²`.
pub fn filtfilt(b: &[f64], a: &[f64], x: &[f64], padding: Padding, padlen: Option<usize>) -> Result<Vec<f64>, ErrorsJSL> {
    let edge = edge_length(x, padding, padlen, b.len().max(a.len()))?;
    let ext = extend(x, edge, padding);
    let zi = lfilter_zi(b, a)?;
    let scaled = |s: f64| zi.iter().map(|v| v * s).collect::<Vec<f64>>();
    let (forward, _) = lfilter(b, a, &ext, Some(&scaled(ext[0])))?;
    let reversed: Vec<f64> = forward.iter().rev().copied().collect();
    let (backward, _) = lfilter(b, a, &reversed, Some(&scaled(reversed[0])))?;
    let mut y: Vec<f64> = backward.into_iter().rev().collect();
    if edge > 0 {
        y = y[edge..y.len() - edge].to_vec();
    }
    Ok(y)
}
// ANCHOR_END: filtfilt

// ANCHOR: sosfiltfilt
/// Zero-phase filtering through second-order sections (SciPy's `sosfiltfilt`), the robust
/// choice for high-order filters. The default padding is `3 (2 n_sections + 1 − t)` samples,
/// where `t` counts trailing zero coefficients shared by all sections, as in SciPy.
pub fn sosfiltfilt(sos: &[Section], x: &[f64], padding: Padding, padlen: Option<usize>) -> Result<Vec<f64>, ErrorsJSL> {
    let zero_b2 = sos.iter().filter(|s| s[2] == 0.0).count();
    let zero_a2 = sos.iter().filter(|s| s[5] == 0.0).count();
    let ntaps = 2 * sos.len() + 1 - zero_b2.min(zero_a2);
    let edge = edge_length(x, padding, padlen, ntaps)?;
    let ext = extend(x, edge, padding);
    let zi = sosfilt_zi(sos)?;
    let scaled = |s: f64| zi.iter().map(|v| [v[0] * s, v[1] * s]).collect::<Vec<[f64; 2]>>();
    let (forward, _) = sosfilt(sos, &ext, Some(&scaled(ext[0])))?;
    let reversed: Vec<f64> = forward.iter().rev().copied().collect();
    let (backward, _) = sosfilt(sos, &reversed, Some(&scaled(reversed[0])))?;
    let mut y: Vec<f64> = backward.into_iter().rev().collect();
    if edge > 0 {
        y = y[edge..y.len() - edge].to_vec();
    }
    Ok(y)
}
// ANCHOR_END: sosfiltfilt

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steady_state_of_a_first_order_filter() {
        // y[n] = 0.5 x[n] + 0.5 y[n-1]: DC gain 1, so with x = 1 the state is 0.5.
        let zi = lfilter_zi(&[0.5], &[1.0, -0.5]).unwrap();
        assert!(zi.is_empty() || (zi[0] - 0.5).abs() < 1e-15, "{zi:?}");
        let zi = lfilter_zi(&[0.5, 0.0], &[1.0, -0.5]).unwrap();
        assert!((zi[0] - 0.5).abs() < 1e-15, "{zi:?}");
    }

    #[test]
    fn zero_phase_filtering_keeps_a_sinusoid_in_place() {
        // A low-pass well above the signal frequency: filtfilt leaves the sine unshifted.
        let b = [0.2, 0.2, 0.2, 0.2, 0.2];
        let x: Vec<f64> = (0..400).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 100.0).sin()).collect();
        let y = filtfilt(&b, &[1.0], &x, Padding::Odd, None).unwrap();
        assert_eq!(y.len(), x.len());
        // |H(ω)|² at ω = 2π/100 for a 5-tap boxcar.
        let w = 2.0 * std::f64::consts::PI / 100.0;
        let h = (5.0 * w / 2.0).sin() / (5.0 * (w / 2.0).sin());
        for i in 50..350 {
            assert!((y[i] - h * h * x[i]).abs() < 1e-3, "{i}: {} vs {}", y[i], h * h * x[i]);
        }
    }

    #[test]
    fn the_padding_rules() {
        let x = [1.0, 2.0, 4.0, 7.0];
        assert_eq!(extend(&x, 2, Padding::Odd), vec![-2.0, 0.0, 1.0, 2.0, 4.0, 7.0, 10.0, 12.0]);
        assert_eq!(extend(&x, 2, Padding::Even), vec![4.0, 2.0, 1.0, 2.0, 4.0, 7.0, 4.0, 2.0]);
        assert_eq!(extend(&x, 1, Padding::Constant), vec![1.0, 1.0, 2.0, 4.0, 7.0, 7.0]);
        assert!(filtfilt(&[1.0, 1.0], &[1.0], &x, Padding::Odd, None).is_err());
    }
}
