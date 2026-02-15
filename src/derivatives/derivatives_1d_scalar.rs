/// This file contains routines that will numerically comute the derivatives on scalar valued functions.
use crate::prelude::ErrorsJSL;

/// This function compute the simple 2 point derivative of a scalar valued function at a given point. The step size can be optionally provided, and if not provided, it defaults to the machine epsilon. The function uses central differences to compute the derivative, which provides a good balance between accuracy and computational efficiency.
pub fn derivative_2_point<T>(f: &T, x: f64, step_size: Option<f64>) -> Result<f64, ErrorsJSL> 
where T: Fn(f64) -> f64 {
    let step = step_size.unwrap_or(1E-6);
    Ok((f(x + step) - f(x - step)) / (2.0 * step))
}   

/// This function uses a 5 point stencil to compute the numerical derivative of a scalar valued function at a given point. The order of the derivative can be specified as an argument, and the step size can also be optionally provided. The function returns a vector of the computed derivatives, where the length of the vector corresponds to the order of the derivative. For example, if the order is set to FirstOrder, the function will return a vector containing only the first derivative. If the order is set to SecondOrder, the function will return a vector containing both the first and second derivatives, and so on. The function uses central differences to compute the derivatives, which provides a good balance between accuracy and computational efficiency.
pub enum DerivativeStencilOrder {
    FirstOrder, // f'
    SecondOrder, // f''
    ThirdOrder, // f'''
    FourthOrder, // f''''
}
pub fn derivative_stencil<T>(
    f: &T,
    x: f64,
    step_size: Option<f64>,
    order: DerivativeStencilOrder
) -> Result<Vec<f64>, ErrorsJSL> 
where T: Fn(f64) -> f64 {
    let step = step_size.unwrap_or(1E-6);
    match order {
        DerivativeStencilOrder::FirstOrder => Ok(vec![
            (-f(x + 2.0 * step) + 8.0 * f(x + step) - 8.0 * f(x - step) + f(x - 2.0 * step)) / (12.0 * step)
        ]),
        DerivativeStencilOrder::SecondOrder => Ok(vec![
            (-f(x + 2.0 * step) + 16.0 * f(x + step) - 30.0 * f(x) + 16.0 * f(x - step) - f(x - 2.0 * step)) / (12.0 * step.powi(2))
        ]),
        DerivativeStencilOrder::ThirdOrder => Ok(vec![
            (f(x + 2.0 * step) - 2.0 * f(x + step) + 2.0 * f(x - step) - f(x - 2.0 * step)) / (2.0 * step.powi(3))
        ]),
        DerivativeStencilOrder::FourthOrder => Ok(vec![
            (f(x + 2.0 * step) - 4.0 * f(x + step) + 6.0 * f(x) - 4.0 * f(x - step) + f(x - 2.0 * step)) / (step.powi(4))
        ]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivative_2_point() {
        let f = |x| x * x;
        let result = derivative_2_point(&f, 2.0, None).unwrap();
        assert!((result - 4.0).abs() < 1E-4);
    }

    #[test]
    fn test_derivative_stencil_first_order() {
        let f = |x| x * x;
        let result = derivative_stencil(&f, 2.0, None, DerivativeStencilOrder::FirstOrder).unwrap();
        assert!((result[0] - 4.0).abs() < 1E-4);
    }
}