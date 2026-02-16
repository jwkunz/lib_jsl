use crate::{
    dsp::{
        filters::biquad::BiquadFilter,
        stream_operator::{StreamOperator, StreamOperatorManagement},
    },
    prelude::ErrorsJSL,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PidOutput {
    pub control: f64,
    pub error: f64,
}

/// Runtime-programmable PID controller for scalar (`f64`) streams.
///
/// - Setpoint and coefficients can be updated at runtime.
/// - `kp`, `ki`, `kd` are optional; `None` means that term is skipped.
/// - Derivative is taken on a low-pass-filtered measurement using a
///   biquad low-pass filter with normalized cutoff 0.1.\
///   This reduces noise sensitivity and avoids derivative kick from setpoint changes.
/// - Optional output limits with anti-windup that prevents integral growth when the controller is saturated and pushing further in the same direction.
/// - `dt` is the time step for integration and differentiation, and can be set at initialization. It does not have to match the actual sample rate, but should be chosen based on the expected dynamics of the system being controlled.
/// - The controller maintains internal state for the integral term and the previous filtered measurement for the derivative term, and provides a `reset_integral` method to reset the integral state without affecting the derivative state.
/// - The `process` method takes a slice of measurements and returns a vector of `PidOutput` containing the control output and error for each input measurement.
/// - The `flush` method does nothing and returns `None`, as the PID controller does not have any buffered output that needs to be flushed.
/// - The `reset` method resets the integral state and derivative state, and also resets the internal state of the biquad filter used for the derivative term.
/// - The `finalize` method does nothing, as there are no resources that need to be cleaned up when the controller is finalized.
pub struct PidController {
    setpoint: f64,
    kp: Option<f64>,
    ki: Option<f64>,
    kd: Option<f64>,
    // High and low limits for the control output. If None, output is not limited.
    output_limits: Option<(f64, f64)>,
    anti_windup: Option<bool>,
    dt: f64,
    integral_state: f64,
    prev_filtered_measurement: f64,
    deriv_lpf: BiquadFilter,
}

impl PidController {
    pub fn new(
        setpoint: Option<f64>,
        kp: Option<f64>,
        ki: Option<f64>,
        kd: Option<f64>,
        output_limits: Option<(f64, f64)>,
        anti_windup: Option<bool>,
    ) -> Result<Self, ErrorsJSL> {
        Self::new_with_dt(setpoint, kp, ki, kd, output_limits, anti_windup, 1.0)
    }

    pub fn new_with_dt(
        setpoint: Option<f64>,
        kp: Option<f64>,
        ki: Option<f64>,
        kd: Option<f64>,
        output_limits: Option<(f64, f64)>,
        anti_windup: Option<bool>,
        dt: f64,
    ) -> Result<Self, ErrorsJSL> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("dt must be finite and > 0"));
        }
        if let Some((lo, hi)) = output_limits {
            if !lo.is_finite() || !hi.is_finite() || lo > hi {
                return Err(ErrorsJSL::InvalidInputRange(
                    "output_limits must be finite and satisfy min <= max",
                ));
            }
        }
        Ok(Self {
            setpoint: setpoint.unwrap_or(0.0),
            kp,
            ki,
            kd,
            output_limits,
            anti_windup,
            dt,
            integral_state: 0.0,
            prev_filtered_measurement: 0.0,
            deriv_lpf: BiquadFilter::lowpass(0.1)?,
        })
    }

    pub fn set_setpoint(&mut self, setpoint: f64) {
        self.setpoint = setpoint;
    }

    pub fn set_coefficients(&mut self, kp: Option<f64>, ki: Option<f64>, kd: Option<f64>) {
        self.kp = kp;
        self.ki = ki;
        self.kd = kd;
    }

    pub fn set_output_limits(&mut self, output_limits: Option<(f64, f64)>) -> Result<(), ErrorsJSL> {
        if let Some((lo, hi)) = output_limits
            && (!lo.is_finite() || !hi.is_finite() || lo > hi)
        {
            return Err(ErrorsJSL::InvalidInputRange(
                "output_limits must be finite and satisfy min <= max",
            ));
        }
        self.output_limits = output_limits;
        Ok(())
    }

    pub fn set_anti_windup(&mut self, anti_windup: Option<bool>) {
        self.anti_windup = anti_windup;
    }

    pub fn set_kp(&mut self, kp: Option<f64>) {
        self.kp = kp;
    }

    pub fn set_ki(&mut self, ki: Option<f64>) {
        self.ki = ki;
    }

    pub fn set_kd(&mut self, kd: Option<f64>) {
        self.kd = kd;
    }

    pub fn reset_integral(&mut self) {
        self.integral_state = 0.0;
    }

    fn step(&mut self, measurement: f64) -> PidOutput {
        let error = self.setpoint - measurement;

        let p_term = self.kp.unwrap_or(0.0) * error;

        let i_prev = self.integral_state;
        self.integral_state += error * self.dt;
        let i_term = self.ki.unwrap_or(0.0) * self.integral_state;

        // Derivative on filtered measurement to reduce noise sensitivity and
        // avoid setpoint-derivative kick.
        let filtered = self.deriv_lpf.step(measurement);
        let d_meas = (filtered - self.prev_filtered_measurement) / self.dt;
        self.prev_filtered_measurement = filtered;
        let d_term = self.kd.unwrap_or(0.0) * (-d_meas);

        let mut control = p_term + i_term + d_term;
        let mut saturated = false;
        if let Some((lo, hi)) = self.output_limits {
            if control < lo {
                control = lo;
                saturated = true;
            } else if control > hi {
                control = hi;
                saturated = true;
            }
        }

        if self.anti_windup.unwrap_or(false) && saturated {
            let pushing_further = if let Some((lo, hi)) = self.output_limits {
                (control >= hi && error > 0.0) || (control <= lo && error < 0.0)
            } else {
                false
            };
            if pushing_further {
                self.integral_state = i_prev;
            }
        }

        PidOutput {
            control,
            error,
        }
    }
}

impl StreamOperatorManagement for PidController {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.integral_state = 0.0;
        self.prev_filtered_measurement = 0.0;
        self.deriv_lpf.reset()?;
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl StreamOperator<f64, PidOutput> for PidController {
    fn process(&mut self, data_in: &[f64]) -> Result<Option<Vec<PidOutput>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        Ok(Some(data_in.iter().map(|&x| self.step(x)).collect()))
    }

    fn flush(&mut self) -> Result<Option<Vec<PidOutput>>, ErrorsJSL> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_p_only() {
        let mut pid = PidController::new(Some(1.0), Some(2.0), None, None, None, None).unwrap();
        let y = pid.process(&[0.0, 0.5, 1.0]).unwrap().unwrap();
        assert!((y[0].control - 2.0).abs() < 1e-12);
        assert!((y[1].control - 1.0).abs() < 1e-12);
        assert!((y[2].control - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_pid_integral_accumulates() {
        let mut pid = PidController::new(Some(1.0), None, Some(1.0), None, None, None).unwrap();
        let y = pid.process(&[0.0, 0.0, 0.0]).unwrap().unwrap();
        assert!(y[1].control > y[0].control);
        assert!(y[2].control > y[1].control);
    }

    #[test]
    fn test_pid_runtime_programming() {
        let mut pid = PidController::new(Some(0.0), Some(1.0), None, None, None, None).unwrap();
        let a = pid.process(&[1.0]).unwrap().unwrap()[0];
        pid.set_setpoint(2.0);
        pid.set_coefficients(Some(0.5), None, None);
        let b = pid.process(&[1.0]).unwrap().unwrap()[0];
        assert!((a.error + 1.0).abs() < 1e-12);
        assert!((b.error - 1.0).abs() < 1e-12);
        assert!((b.control - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_pid_output_limiting() {
        let mut pid =
            PidController::new(Some(10.0), Some(10.0), None, None, Some((-1.0, 1.0)), None)
                .unwrap();
        let y = pid.process(&[0.0]).unwrap().unwrap();
        assert!((y[0].control - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_pid_anti_windup_prevents_integral_growth_on_saturation() {
        let mut pid = PidController::new(
            Some(10.0),
            None,
            Some(1.0),
            None,
            Some((-0.5, 0.5)),
            Some(true),
        )
        .unwrap();
        let y = pid.process(&[0.0, 0.0, 0.0, 0.0]).unwrap().unwrap();
        for yi in y {
            assert!((yi.control - 0.5).abs() < 1e-12);
        }
    }
}
