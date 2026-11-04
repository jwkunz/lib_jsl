//! Pulse correlation tracker: locating and following a known pulse by matched filtering.
//!
//! Radar, sonar, ranging and burst receivers send a known pulse and need to know *when* it comes
//! back, often to a fraction of a sample, and how that delay evolves from one pulse repetition
//! interval (PRI) to the next. The tracker works on one *frame* of samples per PRI:
//!
//! 1. **Detection.** Correlating the frame with the pulse template (a matched filter) concentrates
//!    the pulse energy into a peak. The peak position gives the delay to the nearest sample; a
//!    parabola through the peak and its two neighbours refines it to a fraction of a sample. The
//!    *normalised* correlation `|c| / (‖template‖ ‖segment‖)` lies in `[0, 1]` and is the detection
//!    statistic: it is 1 for a perfect copy and small for noise, independent of amplitude.
//! 2. **Prediction.** An α-β filter carries the delay and its rate of change per frame. Each new
//!    frame is searched only inside a small *gate* around the predicted delay (cheap, and immune to
//!    distant interference).
//! 3. **Update.** If the gated peak is detected, the innovation (measured minus predicted delay)
//!    corrects the delay (`α`) and rate (`β`). If not, the tracker *coasts* on its prediction and
//!    counts a miss; after too many consecutive misses it declares the track lost and must
//!    [`acquire`](PulseCorrelationTracker::acquire) again with a full search.

use crate::discrete::fft_cross_correlation::fft_cross_correlate;
use crate::discrete::cross_correlation::CorrelationMode;
use lib_jsl_core::ErrorsJSL;
use num_complex::Complex;

/// A located pulse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PulsePeak {
    /// Delay of the pulse start within the frame, in samples (fractional).
    pub delay: f64,
    /// Magnitude of the (unnormalised) correlation at the peak.
    pub magnitude: f64,
    /// Normalised correlation `|c| / (‖template‖ ‖segment‖)` at the peak, in `[0, 1]`.
    pub normalized: f64,
}

/// Offset of the vertex of the parabola through `(-1, a), (0, b), (1, c)`, in `[-0.5, 0.5]`.
fn parabolic_offset(a: f64, b: f64, c: f64) -> f64 {
    let denom = a - 2.0 * b + c;
    if denom.abs() < 1e-300 { 0.0 } else { (0.5 * (a - c) / denom).clamp(-0.5, 0.5) }
}

/// Correlation of `template` with `frame` starting at integer delay `d`.
fn correlation_at(frame: &[Complex<f64>], template: &[Complex<f64>], d: usize) -> Complex<f64> {
    template.iter().enumerate().map(|(k, t)| frame[d + k] * t.conj()).sum()
}

fn segment_energy(frame: &[Complex<f64>], len: usize, d: usize) -> f64 {
    frame[d..d + len].iter().map(|v| v.norm_sqr()).sum()
}

fn normalized_at(frame: &[Complex<f64>], template: &[Complex<f64>], template_energy: f64, d: usize) -> f64 {
    let e = segment_energy(frame, template.len(), d);
    if e <= 0.0 || template_energy <= 0.0 { 0.0 } else { correlation_at(frame, template, d).norm() / (template_energy * e).sqrt() }
}

fn check_lengths(frame: &[Complex<f64>], template: &[Complex<f64>]) -> Result<(), ErrorsJSL> {
    if template.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("The pulse template must be non-empty"));
    }
    if frame.len() < template.len() {
        return Err(ErrorsJSL::InvalidInputRange("A frame must be at least as long as the pulse template"));
    }
    Ok(())
}

/// Finds the strongest occurrence of `template` anywhere in `frame` (a full FFT search), with
/// sub-sample delay interpolation.
pub fn locate_pulse(frame: &[Complex<f64>], template: &[Complex<f64>]) -> Result<PulsePeak, ErrorsJSL> {
    check_lengths(frame, template)?;
    let n_delays = frame.len() - template.len() + 1;
    // Full correlation index (template.len() - 1 + d) holds the correlation at delay d.
    let full = fft_cross_correlate(frame, template, CorrelationMode::Full)?;
    let mags: Vec<f64> = (0..n_delays).map(|d| full[template.len() - 1 + d].norm()).collect();
    let best = (0..n_delays).max_by(|&a, &b| mags[a].partial_cmp(&mags[b]).unwrap()).unwrap();
    let offset = if best > 0 && best + 1 < n_delays { parabolic_offset(mags[best - 1], mags[best], mags[best + 1]) } else { 0.0 };
    let template_energy = template.iter().map(|v| v.norm_sqr()).sum::<f64>();
    Ok(PulsePeak { delay: best as f64 + offset, magnitude: mags[best], normalized: normalized_at(frame, template, template_energy, best) })
}

/// The tracker's output for a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackState {
    /// Filtered delay estimate in samples.
    pub delay: f64,
    /// Filtered delay rate in samples per frame.
    pub rate: f64,
    /// Correlation magnitude at the gated peak.
    pub magnitude: f64,
    /// Normalised correlation at the gated peak.
    pub normalized: f64,
    /// True if the pulse was detected in this frame (false when coasting).
    pub detected: bool,
    /// True while the track is held.
    pub locked: bool,
}

/// An α-β pulse tracker.
#[derive(Clone, Debug)]
pub struct PulseCorrelationTracker {
    template: Vec<Complex<f64>>,
    template_energy: f64,
    gate_half_width: usize,
    alpha: f64,
    beta: f64,
    detection_threshold: f64,
    max_misses: usize,
    delay: f64,
    rate: f64,
    locked: bool,
    misses: usize,
}

impl PulseCorrelationTracker {
    /// A tracker for `template`, searching `±gate_half_width` samples around the prediction, with
    /// α-β gains `0 < alpha <= 1` and `0 <= beta <= 2 - alpha`-ish (try 0.5 and 0.1). Detection
    /// needs a normalised correlation of at least 0.4, and 10 consecutive misses lose the track;
    /// both can be changed with the `with_` methods.
    pub fn new(template: &[Complex<f64>], gate_half_width: usize, alpha: f64, beta: f64) -> Result<Self, ErrorsJSL> {
        if template.is_empty() {
            return Err(ErrorsJSL::InvalidInputRange("The pulse template must be non-empty"));
        }
        if !(alpha > 0.0 && alpha <= 1.0) || !(0.0..=2.0).contains(&beta) {
            return Err(ErrorsJSL::InvalidInputRange("Need 0 < alpha <= 1 and 0 <= beta <= 2"));
        }
        Ok(PulseCorrelationTracker {
            template: template.to_vec(),
            template_energy: template.iter().map(|v| v.norm_sqr()).sum(),
            gate_half_width,
            alpha,
            beta,
            detection_threshold: 0.4,
            max_misses: 10,
            delay: 0.0,
            rate: 0.0,
            locked: false,
            misses: 0,
        })
    }

    /// Sets the normalised-correlation threshold for declaring a detection (`0 < t <= 1`).
    pub fn with_detection_threshold(mut self, threshold: f64) -> Result<Self, ErrorsJSL> {
        if !(threshold > 0.0 && threshold <= 1.0) {
            return Err(ErrorsJSL::InvalidInputRange("Detection threshold must be in (0, 1]"));
        }
        self.detection_threshold = threshold;
        Ok(self)
    }

    /// Sets how many consecutive missed detections are tolerated before the track is dropped.
    pub fn with_max_misses(mut self, max_misses: usize) -> Self {
        self.max_misses = max_misses;
        self
    }

    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Searches the whole frame for the pulse and, if its normalised correlation passes the
    /// threshold, starts tracking from it (rate zero).
    pub fn acquire(&mut self, frame: &[Complex<f64>]) -> Result<TrackState, ErrorsJSL> {
        let peak = locate_pulse(frame, &self.template)?;
        let detected = peak.normalized >= self.detection_threshold;
        if detected {
            self.delay = peak.delay;
            self.rate = 0.0;
            self.misses = 0;
        }
        self.locked = detected;
        Ok(TrackState { delay: self.delay, rate: self.rate, magnitude: peak.magnitude, normalized: peak.normalized, detected, locked: self.locked })
    }

    /// Processes the next frame: predicts, searches the gate, and corrects (or coasts). If the
    /// track is not locked this performs a full [`acquire`](Self::acquire).
    pub fn update(&mut self, frame: &[Complex<f64>]) -> Result<TrackState, ErrorsJSL> {
        check_lengths(frame, &self.template)?;
        if !self.locked {
            return self.acquire(frame);
        }
        let max_delay = frame.len() - self.template.len();
        let predicted = self.delay + self.rate;
        let centre = predicted.round().clamp(0.0, max_delay as f64) as usize;
        let lo = centre.saturating_sub(self.gate_half_width);
        let hi = (centre + self.gate_half_width).min(max_delay);
        let mags: Vec<f64> = (lo..=hi).map(|d| correlation_at(frame, &self.template, d).norm()).collect();
        let best_i = (0..mags.len()).max_by(|&a, &b| mags[a].partial_cmp(&mags[b]).unwrap()).unwrap();
        let best = lo + best_i;
        let normalized = normalized_at(frame, &self.template, self.template_energy, best);
        let detected = normalized >= self.detection_threshold;
        if detected {
            let offset = if best > 0 && best < max_delay {
                let left = correlation_at(frame, &self.template, best - 1).norm();
                let right = correlation_at(frame, &self.template, best + 1).norm();
                parabolic_offset(left, mags[best_i], right)
            } else {
                0.0
            };
            // ANCHOR: alpha_beta
            let measured = best as f64 + offset;
            let innovation = measured - predicted;
            self.delay = predicted + self.alpha * innovation;
            self.rate += self.beta * innovation;
            self.misses = 0;
            // ANCHOR_END: alpha_beta
        } else {
            // Coast on the prediction.
            self.delay = predicted;
            self.misses += 1;
            if self.misses > self.max_misses {
                self.locked = false;
            }
        }
        Ok(TrackState { delay: self.delay, rate: self.rate, magnitude: mags[best_i], normalized, detected, locked: self.locked })
    }

    /// Drops the track; the next update will search the whole frame.
    pub fn reset(&mut self) {
        self.locked = false;
        self.misses = 0;
        self.delay = 0.0;
        self.rate = 0.0;
    }
}

/// Convenience: runs a fresh tracker over a sequence of frames, acquiring on the first and
/// tracking after, and returns the state for every frame.
pub fn pulse_correlation_tracker(
    frames: &[Vec<Complex<f64>>],
    template: &[Complex<f64>],
    gate_half_width: usize,
    alpha: f64,
    beta: f64,
) -> Result<Vec<TrackState>, ErrorsJSL> {
    let mut tracker = PulseCorrelationTracker::new(template, gate_half_width, alpha, beta)?;
    frames.iter().map(|f| tracker.update(f)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const L: usize = 64;
    const CHIRP_RATE: f64 = 0.006; // sweeps -0.19 to +0.19 cycles per sample (zero-centred)

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }

    fn gauss(seed: &mut u64) -> f64 {
        (0..12).map(|_| lcg(seed)).sum::<f64>()
    }

    /// Linear-FM pulse value at (fractional) time `t` samples after its start. A wide-band pulse
    /// gives a narrow correlation peak, which is what makes sub-sample delay estimation possible.
    fn chirp(t: f64) -> Complex<f64> {
        if (0.0..L as f64).contains(&t) {
            let c = t - L as f64 / 2.0;
            Complex::from_polar(1.0, PI * CHIRP_RATE * c * c)
        } else {
            Complex::new(0.0, 0.0)
        }
    }

    fn template() -> Vec<Complex<f64>> {
        (0..L).map(|n| chirp(n as f64)).collect()
    }

    /// A frame of `len` samples with the pulse at fractional `delay` (None: noise only).
    fn frame(len: usize, delay: Option<f64>, amplitude: f64, noise: f64, seed: &mut u64) -> Vec<Complex<f64>> {
        (0..len)
            .map(|n| {
                let s = delay.map_or(Complex::new(0.0, 0.0), |d| chirp(n as f64 - d) * amplitude);
                s + Complex::new(gauss(seed), gauss(seed)) * noise
            })
            .collect()
    }

    #[test]
    fn locates_a_pulse_to_a_fraction_of_a_sample() {
        let mut seed = 1;
        for d in [50.0, 50.25, 50.5, 77.8, 120.33] {
            let f = frame(400, Some(d), 1.0, 0.0, &mut seed);
            let p = locate_pulse(&f, &template()).unwrap();
            assert!((p.delay - d).abs() < 0.12, "true {d}, estimated {}", p.delay);
            assert!(p.normalized > 0.9, "{}", p.normalized);
        }
    }

    #[test]
    fn sub_sample_accuracy_over_the_whole_fractional_range() {
        let mut seed = 9;
        let mut worst = 0.0f64;
        for i in 0..40 {
            let d = 100.0 + i as f64 / 40.0;
            let p = locate_pulse(&frame(400, Some(d), 1.0, 0.0, &mut seed), &template()).unwrap();
            worst = worst.max((p.delay - d).abs());
        }
        // Parabolic interpolation of a narrow peak is biased toward the nearest sample; it still
        // resolves the delay to a small fraction of a sample.
        assert!(worst < 0.1, "worst error {worst} samples");
    }

    #[test]
    fn finds_the_pulse_in_heavy_noise_and_rejects_noise_alone() {
        let mut seed = 2;
        // Per-sample SNR of about -3 dB: the matched filter's processing gain (64 samples) makes it easy.
        let f = frame(600, Some(211.4), 1.0, 1.0, &mut seed);
        let p = locate_pulse(&f, &template()).unwrap();
        assert!((p.delay - 211.4).abs() < 0.6, "{}", p.delay);
        let mut tracker = PulseCorrelationTracker::new(&template(), 8, 0.5, 0.1).unwrap();
        let noise_only = frame(600, None, 1.0, 1.0, &mut seed);
        let s = tracker.acquire(&noise_only).unwrap();
        assert!(!s.detected && !s.locked, "noise normalised correlation {}", s.normalized);
        assert!(tracker.acquire(&f).unwrap().locked);
    }

    #[test]
    fn follows_a_moving_pulse_and_estimates_its_rate() {
        let mut seed = 3;
        let rate = 0.35; // samples per frame
        let mut tracker = PulseCorrelationTracker::new(&template(), 6, 0.5, 0.15).unwrap();
        let mut worst_late_error = 0.0f64;
        let mut last = None;
        for k in 0..200 {
            let truth = 100.0 + rate * k as f64;
            let f = frame(500, Some(truth), 1.0, 0.3, &mut seed);
            let s = tracker.update(&f).unwrap();
            assert!(s.locked, "frame {k}");
            if k >= 25 {
                worst_late_error = worst_late_error.max((s.delay - truth).abs());
            }
            last = Some(s);
        }
        assert!(worst_late_error < 0.2, "tracking error {worst_late_error}");
        assert!((last.unwrap().rate - rate).abs() < 0.04, "rate {}", last.unwrap().rate);
    }

    #[test]
    fn coasts_through_a_dropout_and_recovers() {
        let mut seed = 4;
        let rate = 0.4;
        let mut tracker = PulseCorrelationTracker::new(&template(), 8, 0.5, 0.15).unwrap();
        for k in 0..160 {
            let truth = 80.0 + rate * k as f64;
            let present = !(100..105).contains(&k);
            let f = frame(500, if present { Some(truth) } else { None }, 1.0, 0.2, &mut seed);
            let s = tracker.update(&f).unwrap();
            if (100..105).contains(&k) {
                assert!(!s.detected && s.locked, "frame {k}: coasting");
                assert!((s.delay - truth).abs() < 1.0, "frame {k}: coasted to {} vs {truth}", s.delay);
            }
            if k >= 115 {
                assert!(s.detected && (s.delay - truth).abs() < 0.25, "frame {k}: {} vs {truth}", s.delay);
            }
        }
    }

    #[test]
    fn the_track_is_lost_after_too_many_misses_and_reacquired() {
        let mut seed = 5;
        let mut tracker = PulseCorrelationTracker::new(&template(), 8, 0.5, 0.1).unwrap().with_max_misses(3);
        let present = frame(400, Some(150.0), 1.0, 0.2, &mut seed);
        assert!(tracker.update(&present).unwrap().locked);
        let absent = frame(400, None, 1.0, 0.2, &mut seed);
        let mut states = Vec::new();
        for _ in 0..6 {
            states.push(tracker.update(&absent).unwrap());
        }
        assert!(states[..3].iter().all(|s| s.locked), "still coasting for the first {} misses", 3);
        assert!(!states[3].locked && !tracker.is_locked(), "dropped after exceeding max_misses");
        let again = tracker.update(&frame(400, Some(222.0), 1.0, 0.2, &mut seed)).unwrap();
        assert!(again.locked && (again.delay - 222.0).abs() < 0.3, "reacquired at {}", again.delay);
        tracker.reset();
        assert!(!tracker.is_locked());
    }

    #[test]
    fn convenience_function_tracks_a_sequence() {
        let mut seed = 6;
        let frames: Vec<Vec<Complex<f64>>> = (0..40).map(|k| frame(400, Some(90.0 + 0.5 * k as f64), 1.0, 0.1, &mut seed)).collect();
        let states = pulse_correlation_tracker(&frames, &template(), 6, 0.6, 0.2).unwrap();
        assert_eq!(states.len(), 40);
        assert!((states[39].delay - (90.0 + 0.5 * 39.0)).abs() < 0.3);
        assert!(states.iter().all(|s| s.locked));
    }

    #[test]
    fn invalid_arguments() {
        let t = template();
        assert!(PulseCorrelationTracker::new(&[], 4, 0.5, 0.1).is_err());
        assert!(PulseCorrelationTracker::new(&t, 4, 0.0, 0.1).is_err());
        assert!(PulseCorrelationTracker::new(&t, 4, 1.5, 0.1).is_err());
        assert!(PulseCorrelationTracker::new(&t, 4, 0.5, -0.1).is_err());
        assert!(PulseCorrelationTracker::new(&t, 4, 0.5, 0.1).unwrap().with_detection_threshold(0.0).is_err());
        assert!(locate_pulse(&t[..10], &t).is_err(), "frame shorter than the template");
        assert!(locate_pulse(&t, &[]).is_err());
        let mut tr = PulseCorrelationTracker::new(&t, 4, 0.5, 0.1).unwrap();
        assert!(tr.update(&t[..5]).is_err());
    }
}
