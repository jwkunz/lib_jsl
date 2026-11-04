//! Cascaded integrator-comb (CIC) decimators and interpolators (Hogenauer).
//!
//! A CIC filter is `N` cascaded moving sums of length `R·M` built from `N` integrators and `N`
//! combs (differential delay `M`) with the rate change between them, so it needs no multipliers:
//! `H(z) = ((1 − z^{−RM}) / (1 − z^{−1}))^N`, magnitude `|sin(πRMf) / sin(πf)|^N` and DC gain
//! `(RM)^N`. Integer (two's complement) arithmetic is required: the integrators overflow by
//! design, and the wrap-around cancels exactly in the combs as long as the output itself fits.
//!
//! See the book entry [Resampling: polyphase, half-band and CIC filters](https://jwkunz.github.io/lib_jsl/book/spectral/resampling.html#cic-filters).

use lib_jsl_core::ErrorsJSL;
use std::collections::VecDeque;

fn check(rate: usize, stages: usize, delay: usize) -> Result<(), ErrorsJSL> {
    if rate == 0 || stages == 0 || delay == 0 {
        return Err(ErrorsJSL::InvalidInputRange("Rate, stage count and differential delay must be positive."));
    }
    Ok(())
}

/// A comb `y[n] = x[n] − x[n − M]` with wrapping arithmetic.
#[derive(Debug, Clone)]
struct Comb {
    history: VecDeque<i64>,
}

impl Comb {
    fn new(delay: usize) -> Self {
        Comb { history: VecDeque::from(vec![0; delay]) }
    }
    fn step(&mut self, x: i64) -> i64 {
        let old = self.history.pop_front().expect("delay >= 1");
        self.history.push_back(x);
        x.wrapping_sub(old)
    }
}

/// A CIC decimator by `R`: integrators at the input rate, then combs at the output rate.
#[derive(Debug, Clone)]
pub struct CicDecimator {
    rate: usize,
    integrators: Vec<i64>,
    combs: Vec<Comb>,
    phase: usize,
    gain: f64,
}

impl CicDecimator {
    pub fn new(rate: usize, stages: usize, differential_delay: usize) -> Result<Self, ErrorsJSL> {
        check(rate, stages, differential_delay)?;
        Ok(CicDecimator {
            rate,
            integrators: vec![0; stages],
            combs: (0..stages).map(|_| Comb::new(differential_delay)).collect(),
            phase: 0,
            gain: ((rate * differential_delay) as f64).powi(stages as i32),
        })
    }

    /// The DC gain `(RM)^N`; divide the output by it (or shift by `N·log₂(RM)` bits) to normalise.
    pub fn gain(&self) -> f64 {
        self.gain
    }

    /// Consumes input samples and returns the decimated outputs produced (one per `R` inputs).
    // ANCHOR: cic
    pub fn process(&mut self, x: &[i64]) -> Vec<i64> {
        let mut out = Vec::with_capacity(x.len() / self.rate + 1);
        for &v in x {
            let mut acc = v;
            for integrator in self.integrators.iter_mut() {
                *integrator = integrator.wrapping_add(acc);
                acc = *integrator;
            }
            self.phase += 1;
            if self.phase == self.rate {
                self.phase = 0;
                out.push(self.combs.iter_mut().fold(acc, |a, comb| comb.step(a)));
            }
        }
        out
    }
    // ANCHOR_END: cic
}

/// A CIC interpolator by `R`: combs at the input rate, zero-stuffing, integrators at the output
/// rate.
#[derive(Debug, Clone)]
pub struct CicInterpolator {
    rate: usize,
    integrators: Vec<i64>,
    combs: Vec<Comb>,
    gain: f64,
}

impl CicInterpolator {
    pub fn new(rate: usize, stages: usize, differential_delay: usize) -> Result<Self, ErrorsJSL> {
        check(rate, stages, differential_delay)?;
        Ok(CicInterpolator {
            rate,
            integrators: vec![0; stages],
            combs: (0..stages).map(|_| Comb::new(differential_delay)).collect(),
            gain: ((rate * differential_delay) as f64).powi(stages as i32) / rate as f64,
        })
    }

    /// The DC gain `(RM)^N / R` of the interpolated output.
    pub fn gain(&self) -> f64 {
        self.gain
    }

    /// Returns `R` output samples per input sample.
    pub fn process(&mut self, x: &[i64]) -> Vec<i64> {
        let mut out = Vec::with_capacity(x.len() * self.rate);
        for &v in x {
            let combed = self.combs.iter_mut().fold(v, |a, comb| comb.step(a));
            for k in 0..self.rate {
                let mut acc = if k == 0 { combed } else { 0 };
                for integrator in self.integrators.iter_mut() {
                    *integrator = integrator.wrapping_add(acc);
                    acc = *integrator;
                }
                out.push(acc);
            }
        }
        out
    }
}

/// The magnitude response of a CIC decimator at frequency `f` in cycles per *input* sample,
/// normalised to unit DC gain: `|sin(πRMf) / (RM sin(πf))|^N`.
pub fn cic_magnitude_response(rate: usize, stages: usize, differential_delay: usize, f: f64) -> f64 {
    let rm = (rate * differential_delay) as f64;
    let s = (std::f64::consts::PI * f).sin();
    if s.abs() < 1e-15 {
        return 1.0;
    }
    ((std::f64::consts::PI * rm * f).sin() / (rm * s)).abs().powi(stages as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// N boxcars of length L (exact integer convolution).
    fn boxcars(x: &[i64], length: usize, stages: usize) -> Vec<i64> {
        let mut y = x.to_vec();
        for _ in 0..stages {
            y = (0..y.len()).map(|n| (0..length).filter(|&k| k <= n).map(|k| y[n - k]).sum()).collect();
        }
        y
    }

    fn signal(n: usize) -> Vec<i64> {
        (0..n).map(|i| ((i as i64 * 7919) % 2001) - 1000).collect()
    }

    #[test]
    fn decimator_equals_cascaded_moving_sums_then_downsampling() {
        for (r, n, m) in [(4, 3, 1), (8, 4, 2), (5, 1, 1)] {
            let x = signal(400);
            let mut cic = CicDecimator::new(r, n, m).unwrap();
            let y = cic.process(&x);
            let reference: Vec<i64> = boxcars(&x, r * m, n).into_iter().skip(r - 1).step_by(r).collect();
            assert_eq!(y, reference, "R={r} N={n} M={m}");
        }
    }

    #[test]
    fn interpolator_equals_zero_stuffing_then_moving_sums() {
        let (r, n, m) = (4, 3, 1);
        let x = signal(60);
        let y = CicInterpolator::new(r, n, m).unwrap().process(&x);
        let stuffed: Vec<i64> = x.iter().flat_map(|&v| std::iter::once(v).chain(std::iter::repeat_n(0, r - 1))).collect();
        assert_eq!(y, boxcars(&stuffed, r * m, n));
    }

    #[test]
    fn integrator_wrap_around_cancels_and_response_has_nulls() {
        // DC input of 2³⁰ with gain 256⁴ = 2³²: the integrators overflow i64 many times over, yet
        // every settled output is exactly 2⁶².
        let mut cic = CicDecimator::new(256, 4, 1).unwrap();
        let y = cic.process(&vec![1i64 << 30; 256 * 40]);
        assert!(y[4..].iter().all(|&v| v == 1i64 << 62), "{:?}", &y[..6]);
        assert_eq!(cic.gain(), 2f64.powi(32));
        // Nulls at multiples of the output rate, unit gain at DC, sinc-like droop at the band edge.
        assert!(cic_magnitude_response(8, 3, 1, 1.0 / 8.0) < 1e-12);
        assert_eq!(cic_magnitude_response(8, 3, 1, 0.0), 1.0);
        let droop = cic_magnitude_response(8, 3, 1, 1.0 / 32.0);
        assert!(droop > 0.6 && droop < 0.75, "{droop}");
        assert!(CicDecimator::new(0, 1, 1).is_err() && CicInterpolator::new(2, 0, 1).is_err());
    }
}
