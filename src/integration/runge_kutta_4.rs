// Runge Kutta 4 method integrates y = f(t) from t0 to t1 with a fixed number of steps
// f : Function pointer
// t0 : start t
// y0 : starting y
// t1 : end t
// tolerance : the error bound to accept
pub fn runge_kutta_4<F>(f: F, t0: f64, y0: f64, t1: f64, n_steps: usize) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    let dt = (t1 - t0) / n_steps as f64;
    let mut y = y0;
    let mut t = t0;
    let mut result = vec![y0];

    for _ in 0..n_steps {
        let k1 = f(t);
        let k2 = f(t + 0.5 * dt);
        let k3 = f(t + 0.5 * dt);
        let k4 = f(t + dt);

        y += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        t += dt;
        result.push(y);
    }

    result
}

#[cfg(test)]
mod test{  
    use std::f64::consts::PI;

    use super::*;
    #[test]
    fn test_runge_kutta(){
        let error_limit = 1E-3;
        let upper_bound = PI/2.0;
        let lower_bound = 0.0;
        let number_of_points = 100;
        let result : Vec<f64> = runge_kutta_4(|x| x.sin(), lower_bound,0.0,upper_bound,number_of_points);
        let error = 1.0-result.last().unwrap();
        dbg!(&error);
        assert!(error.abs() < error_limit);

    }
}

