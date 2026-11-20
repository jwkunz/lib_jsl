//! Linear time-invariant systems held as zeros, poles and gain (SciPy's `ZerosPolesGain`), in
//! continuous or discrete time.
//!
//! The factored form `H(s) = k ∏(s − zᵢ) / ∏(s − pⱼ)` is the natural output of filter design
//! and the numerically safe one: the frequency response is evaluated factor by factor, without
//! forming polynomial coefficients whose roots are ill-conditioned for high orders.
//!
//! See the book entry [Continuous-time systems](https://jwkunz.github.io/lib_jsl/book/comms/continuous-time.html#zeros-poles-and-gain-as-a-system).

use super::lti::{ContinuousStateSpace, Discretization, InputHold, ss2zpk, tf2ss, unwrap};
use crate::discrete::filters::iir::zero_phase::lfilter;
use crate::discrete::partial_fractions::{tf2zpk, zpk2tf};
use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;
use num_complex::Complex;

type C = Complex<f64>;

// ANCHOR: zpk
/// A single-input, single-output system `H = k ∏(x − zᵢ) / ∏(x − pⱼ)` with `x = s` in
/// continuous time (`dt = None`) or `x = z` in discrete time with sample time `dt` (SciPy's
/// `ZerosPolesGain` and `lti(z, p, k)` / `dlti(z, p, k, dt=dt)`). Complex zeros and poles must
/// come in conjugate pairs for the real-coefficient conversions.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ZerosPolesGain {
    pub zeros: Vec<C>,
    pub poles: Vec<C>,
    pub gain: f64,
    /// The sample time of a discrete-time system; `None` in continuous time.
    pub dt: Option<f64>,
}

impl ZerosPolesGain {
    /// A continuous-time system.
    pub fn new(zeros: Vec<C>, poles: Vec<C>, gain: f64) -> Self {
        ZerosPolesGain { zeros, poles, gain, dt: None }
    }

    /// A discrete-time system with sample time `dt > 0`.
    pub fn new_discrete(zeros: Vec<C>, poles: Vec<C>, gain: f64, dt: f64) -> Result<Self, ErrorsJSL> {
        if !(dt > 0.0 && dt.is_finite()) {
            return Err(ErrorsJSL::InvalidInputRange("the sample time must be positive"));
        }
        Ok(ZerosPolesGain { zeros, poles, gain, dt: Some(dt) })
    }

    /// From a transfer function `num/den` (highest power first), through [`tf2zpk`].
    pub fn from_tf(num: &[f64], den: &[f64], dt: Option<f64>) -> Result<Self, ErrorsJSL> {
        let (z, p, k) = tf2zpk(num, den)?;
        Ok(ZerosPolesGain { zeros: z, poles: p, gain: k, dt })
    }

    /// From a single-input, single-output state-space model.
    pub fn from_ss(sys: &ContinuousStateSpace, dt: Option<f64>) -> Result<Self, ErrorsJSL> {
        if sys.b.ncols() != 1 || sys.c.nrows() != 1 {
            return Err(ErrorsJSL::InvalidInputRange("a ZerosPolesGain system has one input and one output"));
        }
        let (z, p, k) = ss2zpk(sys, 0)?.remove(0);
        Ok(ZerosPolesGain { zeros: z, poles: p, gain: k, dt })
    }

    /// The transfer function `(num, den)`, highest power first (SciPy's `to_tf`).
    pub fn to_tf(&self) -> (Vec<f64>, Vec<f64>) {
        zpk2tf(&self.zeros, &self.poles, self.gain)
    }

    /// A state-space realisation in controllable canonical form (SciPy's `to_ss`). For a
    /// discrete-time system the matrices are those of `x[k+1] = A x[k] + B u[k]`.
    pub fn to_ss(&self) -> Result<ContinuousStateSpace, ErrorsJSL> {
        let (b, a) = self.to_tf();
        tf2ss(&b, &a)
    }

    fn eval(&self, x: C) -> C {
        let num = self.zeros.iter().fold(C::new(self.gain, 0.0), |acc, z| acc * (x - z));
        self.poles.iter().fold(num, |acc, p| acc / (x - p))
    }

    /// The frequency response at the angular frequencies `w`: `H(iω)` in continuous time
    /// (rad/s), `H(e^{iω})` in discrete time (rad/sample, as SciPy's `dfreqresp`), evaluated
    /// factor by factor.
    pub fn freqresp(&self, w: &[f64]) -> Vec<C> {
        match self.dt {
            None => w.iter().map(|&w| self.eval(C::new(0.0, w))).collect(),
            Some(_) => w.iter().map(|&w| self.eval(C::from_polar(1.0, w))).collect(),
        }
    }

    /// The Bode diagram: `20 log₁₀|H|` in dB and the unwrapped phase in degrees (SciPy's `bode`
    /// and `dbode`; in discrete time `w` is in rad/sample).
    pub fn bode(&self, w: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let h = self.freqresp(w);
        let mag = h.iter().map(|v| 20.0 * v.norm().log10()).collect();
        let phase = unwrap(&h.iter().map(|v| v.arg()).collect::<Vec<_>>()).iter().map(|p| p.to_degrees()).collect();
        (mag, phase)
    }

    fn continuous(&self) -> Result<ContinuousStateSpace, ErrorsJSL> {
        if self.dt.is_some() {
            return Err(ErrorsJSL::InvalidInputRange("this is a discrete-time system: use dlsim, dstep or dimpulse"));
        }
        self.to_ss()
    }

    /// The response to the input samples `u` at equally spaced times `t` from rest, with the
    /// input held or interpolated between samples (SciPy's `lsim`). Continuous time only.
    pub fn lsim(&self, u: &[f64], t: &[f64], hold: InputHold) -> Result<Vec<f64>, ErrorsJSL> {
        let ss = self.continuous()?;
        let u = Array2::from_shape_vec((u.len(), 1), u.to_vec()).map_err(|_| ErrorsJSL::InvalidInputRange("bad input"))?;
        Ok(ss.lsim(&u, t, None, hold)?.y.column(0).to_vec())
    }

    /// The step response at times `t` (SciPy's `step`). Continuous time only.
    pub fn step(&self, t: &[f64]) -> Result<Vec<f64>, ErrorsJSL> {
        Ok(self.continuous()?.step(t, 0)?.column(0).to_vec())
    }

    /// The impulse response at times `t` (SciPy's `impulse`). Continuous time only.
    pub fn impulse(&self, t: &[f64]) -> Result<Vec<f64>, ErrorsJSL> {
        Ok(self.continuous()?.impulse(t, 0)?.column(0).to_vec())
    }

    fn discrete_tf(&self) -> Result<(Vec<f64>, Vec<f64>), ErrorsJSL> {
        if self.dt.is_none() {
            return Err(ErrorsJSL::InvalidInputRange("this is a continuous-time system: use lsim, step or impulse"));
        }
        if self.zeros.len() > self.poles.len() {
            return Err(ErrorsJSL::InvalidInputRange("a causal discrete-time system has no more zeros than poles"));
        }
        let (b, a) = self.to_tf();
        // Positive powers of z to the delay form b(z⁻¹)/a(z⁻¹): pad the numerator in front.
        let mut bp = vec![0.0; a.len() - b.len()];
        bp.extend(b);
        Ok((bp, a))
    }

    /// The response to the samples `u` from rest (SciPy's `dlsim`). Discrete time only.
    pub fn dlsim(&self, u: &[f64]) -> Result<Vec<f64>, ErrorsJSL> {
        let (b, a) = self.discrete_tf()?;
        Ok(lfilter(&b, &a, u, None)?.0)
    }

    /// The first `n` samples of the step response (SciPy's `dstep`). Discrete time only.
    pub fn dstep(&self, n: usize) -> Result<Vec<f64>, ErrorsJSL> {
        self.dlsim(&vec![1.0; n])
    }

    /// The first `n` samples of the response to a unit pulse at `k = 0` (SciPy's `dimpulse`).
    /// Discrete time only.
    pub fn dimpulse(&self, n: usize) -> Result<Vec<f64>, ErrorsJSL> {
        let mut u = vec![0.0; n];
        if n > 0 {
            u[0] = 1.0;
        }
        self.dlsim(&u)
    }

    /// The discrete-time system for sample time `dt` (SciPy's `to_discrete`, i.e.
    /// `cont2discrete` on the state-space form and back).
    pub fn to_discrete(&self, dt: f64, method: Discretization) -> Result<ZerosPolesGain, ErrorsJSL> {
        let d = self.continuous()?.discretize(dt, method)?;
        let ss = ContinuousStateSpace::new(d.a, d.b, d.c, d.d)?;
        ZerosPolesGain::from_ss(&ss, Some(dt))
    }
}
// ANCHOR_END: zpk

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_order_lag() {
        // H(s) = 2/(s + 1).
        let h = ZerosPolesGain::new(vec![], vec![C::new(-1.0, 0.0)], 2.0);
        let r = h.freqresp(&[0.0, 1.0]);
        assert!((r[0] - C::new(2.0, 0.0)).norm() < 1e-15);
        assert!((r[1] - C::new(1.0, -1.0)).norm() < 1e-15);
        let (mag, ph) = h.bode(&[1.0]);
        assert!((mag[0] - 20.0 * 2f64.sqrt().log10()).abs() < 1e-12 && (ph[0] + 45.0).abs() < 1e-12);
        let t: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let s = h.step(&t).unwrap();
        assert!(t.iter().zip(&s).all(|(t, y)| (y - 2.0 * (1.0 - (-t).exp())).abs() < 1e-10));
        assert_eq!(h.to_tf(), (vec![2.0], vec![1.0, 1.0]));
        let back = ZerosPolesGain::from_tf(&[2.0], &[1.0, 1.0], None).unwrap();
        assert_eq!(back, h);
        assert!(h.dstep(3).is_err());
        // Zero-order hold: pole e^{−dt}, gain 2(1 − e^{−dt}).
        let d = h.to_discrete(0.5, Discretization::ZeroOrderHold).unwrap();
        assert!((d.poles[0].re - (-0.5f64).exp()).abs() < 1e-14);
        let y = d.dstep(20).unwrap();
        assert!(y.iter().enumerate().all(|(k, y)| (y - 2.0 * (1.0 - (-0.5 * k as f64).exp())).abs() < 1e-12));
        assert!(d.step(&t).is_err());
    }
}
