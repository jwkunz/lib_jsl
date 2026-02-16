use crate::{
    dsp::windows::{self, WindowType},
    ffts::{
        best_fft::BestFft,
        fft_engine_trait::{FfftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
    },
    prelude::ErrorsJSL,
};
use num::Complex;

fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[xp.len() - 1] {
        return fp[fp.len() - 1];
    }
    let mut lo = 0usize;
    let mut hi = xp.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xp[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (x - xp[lo]) / (xp[hi] - xp[lo]);
    fp[lo] * (1.0 - t) + fp[hi] * t
}

/// FIR filter design using the window method, following scipy.signal.firwin2.
pub fn firwin2(
    numtaps: usize,
    freq: &[f64],
    gain: &[f64],
    nfreqs: Option<usize>,
    window: Option<WindowType>,
    antisymmetric: bool,
    fs: f64,
) -> Result<Vec<f64>, ErrorsJSL> {
    if fs <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("fs must be greater than 0"));
    }
    if numtaps == 0 {
        return Err(ErrorsJSL::InvalidInputRange("numtaps must be greater than 0"));
    }
    if freq.len() != gain.len() {
        return Err(ErrorsJSL::InvalidInputRange(
            "freq and gain must be the same length",
        ));
    }
    if freq.len() < 2 {
        return Err(ErrorsJSL::InvalidInputRange(
            "freq and gain must contain at least two points",
        ));
    }

    let nyq = fs / 2.0;
    if (freq[0] - 0.0).abs() > 0.0 || (freq[freq.len() - 1] - nyq).abs() > 0.0 {
        return Err(ErrorsJSL::InvalidInputRange(
            "freq must start with 0 and end with fs/2",
        ));
    }
    let diffs: Vec<f64> = freq.windows(2).map(|w| w[1] - w[0]).collect();
    if diffs.iter().any(|d| *d < 0.0) {
        return Err(ErrorsJSL::InvalidInputRange(
            "The values in freq must be nondecreasing",
        ));
    }
    if diffs
        .windows(2)
        .map(|w| w[0] + w[1])
        .any(|s| s.abs() <= f64::EPSILON)
    {
        return Err(ErrorsJSL::InvalidInputRange(
            "A value in freq must not occur more than twice",
        ));
    }
    if freq.len() > 2 && (freq[1] - 0.0).abs() <= f64::EPSILON {
        return Err(ErrorsJSL::InvalidInputRange("Value 0 must not be repeated in freq"));
    }
    if freq.len() > 2 && (freq[freq.len() - 2] - nyq).abs() <= f64::EPSILON {
        return Err(ErrorsJSL::InvalidInputRange(
            "Value fs/2 must not be repeated in freq",
        ));
    }

    let ftype = if antisymmetric {
        if numtaps % 2 == 0 { 4 } else { 3 }
    } else if numtaps % 2 == 0 {
        2
    } else {
        1
    };

    if ftype == 2 && gain[gain.len() - 1] != 0.0 {
        return Err(ErrorsJSL::InvalidInputRange(
            "Type II filter must have zero gain at Nyquist",
        ));
    }
    if ftype == 3 && (gain[0] != 0.0 || gain[gain.len() - 1] != 0.0) {
        return Err(ErrorsJSL::InvalidInputRange(
            "Type III filter must have zero gain at zero and Nyquist",
        ));
    }
    if ftype == 4 && gain[0] != 0.0 {
        return Err(ErrorsJSL::InvalidInputRange(
            "Type IV filter must have zero gain at zero",
        ));
    }

    let nf = nfreqs.unwrap_or_else(|| 1 + (1usize << ((numtaps as f64).log2().ceil() as usize)));
    if numtaps >= nf {
        return Err(ErrorsJSL::InvalidInputRange("numtaps must be less than nfreqs"));
    }

    let mut f = freq.to_vec();
    if diffs.iter().any(|d| d.abs() <= f64::EPSILON) {
        let eps = f64::EPSILON * nyq;
        for k in 0..(f.len() - 1) {
            if (f[k] - f[k + 1]).abs() <= f64::EPSILON {
                f[k] -= eps;
                f[k + 1] += eps;
            }
        }
        if f.windows(2).any(|w| w[1] <= w[0]) {
            return Err(ErrorsJSL::InvalidInputRange(
                "freq cannot contain values too close to repeated values",
            ));
        }
    }

    let x: Vec<f64> = (0..nf)
        .map(|i| nyq * i as f64 / (nf as f64 - 1.0))
        .collect();
    let fx: Vec<f64> = x.iter().map(|xi| interp(*xi, &f, gain)).collect();

    let mut fx2 = vec![Complex::new(0.0, 0.0); nf];
    for (i, fxv) in fx.iter().enumerate() {
        let phase = -((numtaps as f64 - 1.0) / 2.0) * std::f64::consts::PI * x[i] / nyq;
        let mut z = Complex::new(phase.cos(), phase.sin()) * *fxv;
        if ftype > 2 {
            z *= Complex::new(0.0, 1.0);
        }
        fx2[i] = z;
    }

    let nfft = 2 * (nf - 1);
    let mut spec = vec![Complex::new(0.0, 0.0); nfft];
    for i in 0..nf {
        spec[i] = fx2[i];
    }
    for k in 1..(nf - 1) {
        spec[nfft - k] = fx2[k].conj();
    }

    let mut ifft = BestFft::new();
    ifft.plan(
        nfft,
        FftScaleFactor::N,
        FftDirection::Inverse,
        FftOrdering::Standard,
    )?;
    let out_full = ifft.execute(&spec)?;

    let wind = if let Some(w) = window {
        windows::generate_window(w, numtaps, true)
    } else {
        vec![1.0; numtaps]
    };

    let mut out = vec![0.0; numtaps];
    for i in 0..numtaps {
        out[i] = out_full[i].re * wind[i];
    }
    if ftype == 3 {
        let mid = out.len() / 2;
        out[mid] = 0.0;
    }
    Ok(out)
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

    #[test]
    fn test_firwin2_doc_example_golden_slice() {
        let taps = firwin2(
            150,
            &[0.0, 0.5, 1.0],
            &[1.0, 1.0, 0.0],
            None,
            Some(WindowType::Hamming),
            false,
            2.0,
        )
        .unwrap();

        let golden = parse_csv(include_str!("../test_data/firwin2_doc_example_taps72_77.csv"));
        assert_eq!(golden.len(), 6);
        for (a, g) in taps[72..78].iter().zip(golden.iter()) {
            assert!((a - g).abs() < 1e-7, "actual={a}, golden={g}");
        }
    }
}
