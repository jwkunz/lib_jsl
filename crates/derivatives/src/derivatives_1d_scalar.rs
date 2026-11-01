//! Finite difference derivatives of scalar functions of one variable.
//!
//! See the book entry [Finite difference derivatives](https://jwkunz.github.io/lib_jsl/book/derivatives/finite-differences.html).

use lib_jsl_core::ErrorsJSL;

/// A step near `h · max(|x|, 1)`, adjusted so that `x + step` is exactly representable and
/// `(x + step) − x == step`: this removes the rounding error in the step itself.
pub(crate) fn representable_step(x: f64, h: f64) -> f64 {
    let step = h * x.abs().max(1.0);
    (x + step) - x
}

/// The default relative step for a formula whose truncation error is `O(hᵖ)` and whose rounding
/// error is `O(ε / hᵐ)` (an `m`-th derivative): the `h ≈ ε^{1/(p+m)}` that balances the two.
pub(crate) fn optimal_relative_step(truncation_order: i32, derivative_order: i32) -> f64 {
    f64::EPSILON.powf(1.0 / (truncation_order + derivative_order) as f64)
}

fn validate(x: f64, step_size: Option<f64>) -> Result<(), ErrorsJSL> {
    if !x.is_finite() {
        return Err(ErrorsJSL::InvalidInputRange("The point must be finite."));
    }
    if let Some(h) = step_size
        && !(h > 0.0 && h.is_finite())
    {
        return Err(ErrorsJSL::InvalidInputRange("The step size must be positive and finite."));
    }
    Ok(())
}

/// The central difference `(f(x + h) − f(x − h)) / 2h`, a second-order approximation to `f'(x)`.
///
/// `step_size` is relative: the step is `h · max(|x|, 1)`, which suits functions that vary on the
/// scale of `x` (for one that oscillates on a scale of 1 at large `x`, such as `sin`, pass a smaller
/// `step_size`). The default `h = ε^{1/3} ≈ 6·10⁻⁶`
/// balances the truncation error `h² f⁽³⁾ / 6` against the rounding error `ε |f| / h`, giving
/// about ten correct digits for a well-scaled function.
pub fn derivative_2_point<T>(f: &T, x: f64, step_size: Option<f64>) -> Result<f64, ErrorsJSL>
where
    T: Fn(f64) -> f64,
{
    validate(x, step_size)?;
    let h = representable_step(x, step_size.unwrap_or(optimal_relative_step(2, 1)));
    // ANCHOR: central_difference
    Ok((f(x + h) - f(x - h)) / (2.0 * h))
    // ANCHOR_END: central_difference
}

/// Which derivative [`derivative_stencil`] approximates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativeStencilOrder {
    /// `f'`, fourth-order accurate.
    FirstOrder,
    /// `f''`, fourth-order accurate.
    SecondOrder,
    /// `f'''`, second-order accurate.
    ThirdOrder,
    /// `f''''`, second-order accurate.
    FourthOrder,
}

/// A derivative of `f` at `x` from the five-point stencil `f(x − 2h), …, f(x + 2h)`. Returns a
/// one-element vector holding the requested derivative.
///
/// `step_size` is relative, as for [`derivative_2_point`]. The default balances truncation against
/// rounding for each formula: `ε^{1/5}` for the first and third derivatives and `ε^{1/6}` for the
/// second and fourth (around `10⁻³`), since higher derivatives divide rounding errors by higher
/// powers of `h`.
pub fn derivative_stencil<T>(f: &T, x: f64, step_size: Option<f64>, order: DerivativeStencilOrder) -> Result<Vec<f64>, ErrorsJSL>
where
    T: Fn(f64) -> f64,
{
    validate(x, step_size)?;
    let (truncation, derivative) = match order {
        DerivativeStencilOrder::FirstOrder => (4, 1),
        DerivativeStencilOrder::SecondOrder => (4, 2),
        DerivativeStencilOrder::ThirdOrder => (2, 3),
        DerivativeStencilOrder::FourthOrder => (2, 4),
    };
    let h = representable_step(x, step_size.unwrap_or(optimal_relative_step(truncation, derivative)));
    let (m2, m1, z, p1, p2) = (f(x - 2.0 * h), f(x - h), f(x), f(x + h), f(x + 2.0 * h));
    // ANCHOR: five_point
    let value = match order {
        DerivativeStencilOrder::FirstOrder => (-p2 + 8.0 * p1 - 8.0 * m1 + m2) / (12.0 * h),
        DerivativeStencilOrder::SecondOrder => (-p2 + 16.0 * p1 - 30.0 * z + 16.0 * m1 - m2) / (12.0 * h * h),
        DerivativeStencilOrder::ThirdOrder => (p2 - 2.0 * p1 + 2.0 * m1 - m2) / (2.0 * h.powi(3)),
        DerivativeStencilOrder::FourthOrder => (p2 - 4.0 * p1 + 6.0 * z - 4.0 * m1 + m2) / h.powi(4),
    };
    // ANCHOR_END: five_point
    Ok(vec![value])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivative_2_point() {
        let f = |x| x * x;
        let result = derivative_2_point(&f, 2.0, None).unwrap();
        assert!((result - 4.0).abs() < 1E-8);
    }

    #[test]
    fn test_derivative_stencil_first_order() {
        let f = |x| x * x;
        let result = derivative_stencil(&f, 2.0, None, DerivativeStencilOrder::FirstOrder).unwrap();
        assert!((result[0] - 4.0).abs() < 1E-10);
    }

    #[test]
    fn default_steps_are_accurate_for_every_order() {
        // Derivatives of e^x at x = 1 are all e. With the old fixed step 1e-6 the third and fourth
        // derivatives were dominated by rounding (errors of about 10² and 10⁸).
        let e = std::f64::consts::E;
        let cases = [
            (DerivativeStencilOrder::FirstOrder, 1e-11),
            (DerivativeStencilOrder::SecondOrder, 1e-9),
            (DerivativeStencilOrder::ThirdOrder, 1e-5),
            (DerivativeStencilOrder::FourthOrder, 1e-4),
        ];
        for (order, tol) in cases {
            let d = derivative_stencil(&f64::exp, 1.0, None, order).unwrap()[0];
            assert!((d - e).abs() < tol, "{order:?}: error {:e}", (d - e).abs());
        }
        assert!((derivative_2_point(&f64::exp, 1.0, None).unwrap() - e).abs() < 1e-9);
        // The step scales with |x|, suiting functions that vary on the scale of x: d/dx ln x at
        // 10⁸ is 10⁻⁸, to about ten digits.
        let x = 1e8;
        assert!((derivative_2_point(&f64::ln, x, None).unwrap() * x - 1.0).abs() < 1e-9);
        assert!(derivative_2_point(&f64::sin, f64::NAN, None).is_err());
        assert!(derivative_2_point(&f64::sin, 1.0, Some(0.0)).is_err());
    }
}
