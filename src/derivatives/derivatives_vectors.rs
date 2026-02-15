/// This file contains derivative routines for vector valued functions. 

use crate::prelude::ErrorsJSL;  

/// Multivariate scalar valued functions can be differentiated using a simple 2 point stencil, which is a numerical method for approximating the gradient of a function. 

/// The gradient of a scalar valued function can be computed using a simple 2 point stencil, which is a numerical method for approximating the derivative of a function. The gradient is a vector that contains the partial derivatives of the scalar valued function with respect to each variable. The 2 point stencil method works by evaluating the function at two points that are close to each other, and then using the difference between these two evaluations to approximate the derivative. The step size can be optionally provided, and if not provided, it defaults to the machine epsilon. This method is simple and efficient, but it may not be accurate for functions that have high curvature or for functions that are not smooth.
pub fn gradient_2_point<T>(f: &T, x: &Vec<f64>, step_size: Option<f64>) -> Result<Vec<f64>, ErrorsJSL> 
where T: Fn(&Vec<f64>) -> f64 {
    let step = step_size.unwrap_or(10.0*f64::EPSILON);
    let n = x.len();
    let mut gradient = vec![0.0; n];
    for i in 0..n {
        let mut x_forward = x.clone();
        let mut x_backward = x.clone();
        x_forward[i] += step;
        x_backward[i] -= step;
        let f_forward = f(&x_forward);
        let f_backward = f(&x_backward);
        gradient[i] = (f_forward - f_backward) / (2.0 * step);
    }
    Ok(gradient)
}

/// The hessian of a scalar valued function can be computed using a simple 2 point stencil, which is a numerical method for approximating the second derivative of a function. The hessian is a matrix that contains the second partial derivatives of the scalar valued function with respect to each variable. The 2 point stencil method works by evaluating the function at two points that are close to each other, and then using the difference between these two evaluations to approximate the second derivative. The step size can be optionally provided, and if not provided, it defaults to the machine epsilon. This method is simple and efficient, but it may not be accurate for functions that have high curvature or for functions that are not smooth.
pub fn hessian_2_point<T>(f: &T, x: &Vec<f64>, step_size: Option<f64>) -> Result<Vec<Vec<f64>>, ErrorsJSL> 
where T: Fn(&Vec<f64>) -> f64 {
    let step = step_size.unwrap_or(10.0*f64::EPSILON);
    let n = x.len();
    let mut hessian = vec![vec![0.0; n]; n];
    let fx = f(x);
    for i in 0..n {
        // Diagonal second derivatives
        let mut x_forward = x.clone();
        let mut x_backward = x.clone();

        x_forward[i] += step;
        x_backward[i] -= step;

        let f_forward = f(&x_forward);
        let f_backward = f(&x_backward);

        hessian[i][i] = (f_forward - 2.0 * fx + f_backward) / (step * step);

        // Mixed partial derivatives
        for j in (i + 1)..n {
            let mut x_pp = x.clone(); // +i +j
            let mut x_pm = x.clone(); // +i -j
            let mut x_mp = x.clone(); // -i +j
            let mut x_mm = x.clone(); // -i -j

            x_pp[i] += step;
            x_pp[j] += step;

            x_pm[i] += step;
            x_pm[j] -= step;

            x_mp[i] -= step;
            x_mp[j] += step;

            x_mm[i] -= step;
            x_mm[j] -= step;

            let value = (
                f(&x_pp)
                - f(&x_pm)
                - f(&x_mp)
                + f(&x_mm)
            ) / (4.0 * step * step);

            hessian[i][j] = value;
            hessian[j][i] = value; // symmetry
        }
    }
    Ok(hessian)
}

/// Vector valued functions can be differentiated using a simple 2 point stencil, which is a numerical method for approximating the jacobian of a function. 

/// The jacobian of a vector valued function can be computed using a simple 2 point stencil, which is a numerical method for approximating the derivative of a function. The jacobian is a matrix that contains the partial derivatives of each component of the vector valued function with respect to each variable. The 2 point stencil method works by evaluating the function at two points that are close to each other, and then using the difference between these two evaluations to approximate the derivative. The step size can be optionally provided, and if not provided, it defaults to the machine epsilon. This method is simple and efficient, but it may not be accurate for functions that have high curvature or for functions that are not smooth.
pub fn jacobian_2_point<T>(f: &T, x: &Vec<f64>, step_size: Option<f64>) -> Result<Vec<Vec<f64>>, ErrorsJSL> 
where T: Fn(&Vec<f64>) -> Vec<f64> {
    let step = step_size.unwrap_or(10.0*f64::EPSILON);
    let n = x.len();
    let m = f(x).len();
    let mut jacobian = vec![vec![0.0; n]; m];
    for i in 0..n {
        let mut x_forward = x.clone();
        let mut x_backward = x.clone();
        x_forward[i] += step;
        x_backward[i] -= step;
        let f_forward = f(&x_forward);
        let f_backward = f(&x_backward);
        for j in 0..m {
            jacobian[j][i] = (f_forward[j] - f_backward[j]) / (2.0 * step);
        }
    }
    Ok(jacobian)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_gradient_2_point() {
        // This is a simple test case for the gradient_2_point function. It uses a simple quadratic function with a minimum at (2.0, 3.0) and a minimum value of 0.0. The gradient should be zero at the minimum, and the function should return a vector of zeros.
        let f = |x: &Vec<f64>| x[0] * x[0] + x[1] * x[1];
        let x = vec![0.0, 0.0];
        let result = gradient_2_point(&f, &x, None).unwrap();
        assert_eq!(result, vec![0.0, 0.0]);
    }
    #[test]
    fn test_hessian_2_point() {
        // This is a simple test case for the hessian_2_point function. It uses a simple quadratic function with a minimum at (2.0, 3.0) and a minimum value of 0.0. The hessian should be a matrix of zeros at the minimum, and the function should return a matrix of zeros.
        let f = |x: &Vec<f64>| x[0] * x[0] + x[1] * x[1];
        let x = vec![0.0, 0.0];
        let result = hessian_2_point(&f, &x, None).unwrap();
        assert_eq!(result, vec![vec![2.0, 0.0], vec![0.0, 2.0]]);
    }
    #[test]
    fn test_jacobian_2_point() {
        // This is a simple test case for the jacobian_2_point function. It uses a simple vector valued function with a minimum at (2.0, 3.0) and a minimum value of (4.0, 9.0). The jacobian should be a matrix of zeros at the minimum, and the function should return a matrix of zeros.
        let f = |x: &Vec<f64>| vec![x[0] * x[0], x[1] * x[1]];
        let x = vec![0.0, 0.0];
        let result = jacobian_2_point(&f, &x, None).unwrap();
        assert_eq!(result, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
    }
}