//! State-space models, PID control and a Kalman filter.

use lib_jsl_dsp::discrete::controls::kalman::KalmanFilter;
use lib_jsl_dsp::discrete::controls::pid::PidController;
use lib_jsl_dsp::discrete::controls::state_space::StateSpace;
use lib_jsl_dsp::discrete::stream_operator::StreamOperator;
use ndarray::{Array1, arr2};
use num_complex::Complex;

fn c(m: ndarray::Array2<f64>) -> ndarray::Array2<Complex<f64>> {
    m.mapv(|v| Complex::new(v, 0.0))
}

fn main() {
    // A discretised mass-spring-damper.
    let a = c(arr2(&[[1.0, 0.1], [-0.2, 0.95]]));
    let b = c(arr2(&[[0.0], [0.1]]));
    let cm = c(arr2(&[[1.0, 0.0]]));
    let d = c(arr2(&[[0.0]]));
    let sys = StateSpace::new(a, b, cm, d, Array1::from(vec![Complex::new(0.0, 0.0); 2])).unwrap();
    println!("poles {:.4}, stable {}, controllable {}, observable {}", sys.get_poles().unwrap(), sys.is_stable().unwrap(), sys.is_controllable().unwrap(), sys.is_observable().unwrap());
    // PI control of a first-order plant y' = −y + u.
    let mut pid = PidController::new_with_dt(Some(1.0), Some(2.0), Some(1.0), None, None, None, 0.05).unwrap();
    let mut y = 0.0;
    for _ in 0..400 {
        let u = pid.process(&[y]).unwrap().unwrap()[0].control;
        y += 0.05 * (-y + u);
    }
    println!("PI loop output after 20 s: {y:.4} (setpoint 1)");
    // A Kalman filter estimating a constant from noisy readings is a running average.
    let one = |v: f64| arr2(&[[v]]).mapv(|x| Complex::new(x, 0.0));
    let mut kf = KalmanFilter::new(one(1.0), one(1.0), one(0.0), one(0.25), one(100.0), Array1::from(vec![Complex::new(0.0, 0.0)]), None, None).unwrap();
    let readings = [5.3, 4.6, 5.1, 4.8, 5.4, 4.9, 5.0, 5.2];
    let measurements: Vec<Array1<Complex<f64>>> = readings.iter().map(|&r| Array1::from(vec![Complex::new(r, 0.0)])).collect();
    let out = kf.process(&measurements).unwrap().unwrap();
    println!("Kalman estimate {:.4}, sample mean {:.4}, variance {:.4} (0.25 / 8 = 0.03125)", out.last().unwrap().next_state[0].re, readings.iter().sum::<f64>() / 8.0, kf.estimate_covariance()[[0, 0]].re);
}
