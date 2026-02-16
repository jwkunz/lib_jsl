use crate::{
    dsp::{sinc::sinc, windows::{self, WindowType, kaiser_atten}},
    prelude::ErrorsJSL,
};

/// Pass-zero behavior compatible with scipy.signal.firwin.
#[derive(Clone, Copy, Debug)]
pub enum FirwinPassZero {
    True,
    False,
}


/// FIR filter design using the window method, following scipy.signal.firwin.
pub fn firwin(
    numtaps: usize,
    cutoff: &[f64],
    width: Option<f64>,
    window: WindowType,
    pass_zero: FirwinPassZero,
    scale: bool,
    fs: f64,
) -> Result<Vec<f64>, ErrorsJSL> {
    if numtaps == 0 {
        return Err(ErrorsJSL::InvalidInputRange("numtaps must be greater than 0"));
    }
    if cutoff.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange(
            "At least one cutoff frequency must be given",
        ));
    }
    if fs <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("fs must be greater than 0"));
    }

    let nyq = fs / 2.0;
    let mut c: Vec<f64> = cutoff.iter().map(|x| *x / nyq).collect();

    if c.iter().any(|x| *x <= 0.0 || *x >= 1.0) {
        return Err(ErrorsJSL::InvalidInputRange(
            "Cutoff frequencies must be between 0 and fs/2",
        ));
    }
    if c.windows(2).any(|w| w[1] <= w[0]) {
        return Err(ErrorsJSL::InvalidInputRange(
            "Cutoff frequencies must be strictly increasing",
        ));
    }

    let pass_zero_b = matches!(pass_zero, FirwinPassZero::True);
    let pass_nyquist = (c.len() % 2 == 1) ^ pass_zero_b;
    if pass_nyquist && numtaps % 2 == 0 {
        return Err(ErrorsJSL::InvalidInputRange(
            "Even numtaps cannot include Nyquist in a passband",
        ));
    }

    if pass_zero_b {
        c.insert(0, 0.0);
    }
    if pass_nyquist {
        c.push(1.0);
    }

    let mut h = vec![0.0; numtaps];
    let alpha = 0.5 * (numtaps as f64 - 1.0);
    for band in c.chunks_exact(2) {
        let left = band[0];
        let right = band[1];
        for (n, hn) in h.iter_mut().enumerate() {
            let m = n as f64 - alpha;
            *hn += right * sinc(right * m) - left * sinc(left * m);
        }
    }

    let win = if let Some(w) = width {
        let atten = kaiser_atten(numtaps, w / nyq);
        windows::kaiser(numtaps, windows::kaiser_beta(atten), true)
    } else {
        windows::generate_window(window, numtaps, true)
    };
    for (hn, wn) in h.iter_mut().zip(win.iter()) {
        *hn *= *wn;
    }

    if scale {
        let first_left = c[0];
        let first_right = c[1];
        let scale_frequency = if first_left == 0.0 {
            0.0
        } else if first_right == 1.0 {
            1.0
        } else {
            0.5 * (first_left + first_right)
        };
        let mut s = 0.0;
        for (n, hn) in h.iter().enumerate() {
            let m = n as f64 - alpha;
            s += *hn * (std::f64::consts::PI * m * scale_frequency).cos();
        }
        if s != 0.0 {
            for hn in &mut h {
                *hn /= s;
            }
        }
    }

    Ok(h)
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
    fn test_firwin_lowpass_scipy_golden() {
        // Matches scipy.signal.firwin(numtaps=3, cutoff=0.1) with defaults.
        let h = firwin(
            3,
            &[0.1],
            None,
            WindowType::Hamming,
            FirwinPassZero::True,
            true,
            2.0,
        )
        .unwrap();

        let golden = parse_csv(include_str!("../test_data/firwin_lowpass_numtaps3_cutoff0p1.csv"));
        assert_eq!(h.len(), golden.len());
        for (a, g) in h.iter().zip(golden.iter()) {
            assert!((a - g).abs() < 1e-8, "actual={a}, golden={g}");
        }
    }
}
