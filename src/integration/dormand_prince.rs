// The Dorman Prince method integrates y = f(t) from t0 to t1, but adapts the size as it progresses
// f : Function pointer
// t0 : start t
// y0 : starting y
// t1 : end t
// tolerance : the error bound to accept
pub fn dormand_prince<F>(f: &F, t0: f64, y0: f64, t1: f64, tolerance : f64) -> Vec<(f64,f64,f64)>
where
    F: Fn(f64) -> f64,
{
    let mut t = t0;
    let mut y = y0;
    let mut h = tolerance;
    let mut result = Vec::<(f64,f64,f64)>::new();
    result.push((t,y0,h));
    while t <  t1{
        let step = dormand_prince_adaptive_step(f, t, y, h, tolerance);
        result.push(step);
        t = step.0;
        y = step.1;
        h = step.2.min(t1-t);
    }
    result
}

// The Runge-Kutta 4 5 iteration
pub fn dormand_prince_step<F>(
    f: &F,
    t: f64,
    y: f64,
    h: f64,
) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let k1 = h * f(t);
    //let k2 = h * f(t + h * 1.0 / 5.0);
    let k3 = h * f(t + h * 3.0 / 10.0);
    let k4 = h * f(t + h * 4.0 / 5.0);
    let k5 = h * f(t + h * 8.0 / 9.0);
    let k6 = h * f(t + h);
    let k7 = h * f(t + h);

    // 5th order
    let y5 = y
        + (35.0 / 384.0) * k1
        + (500.0 / 1113.0) * k3
        + (125.0 / 192.0) * k4
        - (2187.0 / 6784.0) * k5
        + (11.0 / 84.0) * k6;

    // 4th order (embedded)
    let y4 = y
        + (5179.0 / 57600.0) * k1
        + (7571.0 / 16695.0) * k3
        + (393.0 / 640.0) * k4
        - (92097.0 / 339200.0) * k5
        + (187.0 / 2100.0) * k6
        + (1.0 / 40.0) * k7;

    let err = (y5 - y4).abs();
    (y5, err)
}


/// Perform one adaptive Dormand–Prince step.
///
/// Attempts to advance from (t, y) using initial step h.
/// On success, returns (t_next, y_next, h_next).
pub fn dormand_prince_adaptive_step<F>(
    f: &F,
    t: f64,
    y: f64,
    h: f64,
    tol: f64,
) -> (f64, f64, f64)
where
    F: Fn(f64) -> f64,
{
    // Controller parameters
    const SAFETY: f64 = 0.9;
    const MIN_SCALE: f64 = 0.2;
    const MAX_SCALE: f64 = 5.0;
    const ORDER: f64 = 4.0; // lower order of the embedded pair

    let mut h_try = h;

    loop {
        let (y_next, err) = dormand_prince_step(f, t, y, h_try);

        // Scale error relative to tolerance
        let err_norm = err.max(1e-14); // prevent division by zero

        if err_norm <= tol {
            // Accept step
            let scale = (tol / err_norm).powf(1.0 / (ORDER + 1.0));
            let scale = scale.clamp(MIN_SCALE, MAX_SCALE);

            let h_next = SAFETY * h_try * scale;
            return (t + h_try, y_next, h_next);
        } else {
            // Reject step, reduce h
            let scale = (tol / err_norm).powf(1.0 / (ORDER + 1.0));
            let scale = scale.clamp(MIN_SCALE, 1.0);

            h_try *= SAFETY * scale;
        }
    }
}



#[cfg(test)]
mod test{  
    use std::f64::consts::PI;
    fn f(x : f64)->f64{
        x.sin()
    }
    use super::*;
    #[test]
    fn test_dormand_prince(){
        let error_limit : f64 = 1E-6;
        let upper_bound : f64 = PI/2.0;
        let lower_bound : f64 = 0.0;
        let result = dormand_prince(&f, lower_bound,0.0,upper_bound,error_limit);
        let error = 1.0-result.last().unwrap().1;
        dbg!(&error);
        assert!(error.abs() < error_limit);
    }
}