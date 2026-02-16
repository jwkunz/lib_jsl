/// This module defines common types, traits, and error handling for the library. 
use std::{
    fmt::Debug,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign, Add, Div, Mul, Neg, Sub},
};

use ndarray::prelude::*;
use num::{Complex, One, Zero};

/// These are 1D vector types for real and complex data, along with their view types. They provide convenient aliases for working with 1D arrays of real and complex numbers throughout the library, allowing us to write more concise and readable code when dealing with these common data structures.
pub type R1D = Array1<f64>;
pub type C1D = Array1<Complex<f64>>;
pub type VR1D<'a> = ArrayView1<'a, f64>;
pub type VC1D<'a> = ArrayView1<'a, num::Complex<f64>>;

/// These are 2D array types for real and complex data, along with their view types. They provide convenient aliases for working with 2D arrays of real and complex numbers throughout the library, allowing us to write more concise and readable code when dealing with these common data structures.
pub type R2D = Array2<f64>;
pub type C2D = Array2<Complex<f64>>;
pub type VR2D<'a> = ArrayView2<'a, f64>;
pub type VC2D<'a> = ArrayView2<'a, num::Complex<f64>>;

/// This module defines common error types and traits for the library. The `ErrorsJSL` enum provides a standardized way to represent various error conditions that may arise during the execution of the library's functions, such as invalid input, runtime errors, missing dependencies, and misconfigurations.
#[derive(Debug)]
pub enum ErrorsJSL {
    NotImplementedYet,
    IncompatibleArraySizes((usize, usize)),
    InvalidInputRange(&'static str),
    RuntimeError(&'static str),
    MissingDependency(&'static str),
    Other(&'static str),
    Misconfiguration(&'static str),
}

/// The `IsLinearOperatable` trait defines a set of operations that must be supported by any type that can be used in linear operations, such as addition, subtraction, multiplication, division, and negation. This trait is implemented for common numeric types such as f64, f32, Complex<f64>, Complex<f32>, and various integer types, allowing them to be used in linear operations throughout the library.
pub trait IsLinearOperatable:
    Debug
    + Copy
    + Clone
    + Sized
    + PartialEq
    + Zero
    + One
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + SubAssign<Self>
    + AddAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
{
}

impl IsLinearOperatable for f64 {}
impl IsLinearOperatable for f32 {}
impl IsLinearOperatable for Complex<f64> {}
impl IsLinearOperatable for Complex<f32> {}
impl IsLinearOperatable for i64 {}
impl IsLinearOperatable for i32 {}
impl IsLinearOperatable for i16 {}
impl IsLinearOperatable for i8 {}

/// The `IsAnalytic` trait extends the `IsLinearOperatable` trait with additional operations that are specific to analytic functions, such as complex conjugation, exponentiation, logarithm, power functions, and magnitude/phase calculations. This trait is implemented for both real and complex types, allowing us to perform a wide range of mathematical operations on these types in a consistent manner throughout the library.
pub trait IsAnalytic: IsLinearOperatable {
    fn f_conj(&self) -> Self;
    fn f_exp(&self) -> Self;
    fn f_ln(&self) -> Self;
    fn f_powf(&self, n: f64) -> Self;
    fn f_powi(&self, n: i32) -> Self;
    fn f_abs(&self) -> f64;
    fn f_arg(&self) -> f64;
    fn f_abs2(&self) -> f64;
    fn f_scale(&self, factor: f64) -> Self;
    fn to_complex(&self) -> Complex<f64>;
    fn from_complex(value: Complex<f64>) -> Self where Self: Sized;
}

/// Trait for reconstructing a value from a `Complex<f64>` representation.
/// This complements `IsAnalytic::to_complex()` for generic FFT-based routines.
pub trait FromComplex64: Sized {
    fn from_complex(value: Complex<f64>) -> Self;
}

impl IsAnalytic for f64 {
    fn f_conj(&self) -> Self {
        *self
    }
    fn f_exp(&self) -> Self {
        self.exp()
    }
    fn f_ln(&self) -> Self {
        self.ln()
    }
    fn f_powf(&self, n: f64) -> Self {
        self.powf(n)
    }
    fn f_powi(&self, n: i32) -> Self {
        self.powi(n)
    }
    fn f_abs(&self) -> f64 {
        self.abs()
    }
    fn f_abs2(&self) -> f64 {
        self*self
    }
    fn f_arg(&self) -> f64 {
        if *self >= 0.0 {
            0.0
        } else {
            std::f64::consts::PI
        }   
    }
    fn f_scale(&self, factor: f64) -> Self {
        self * factor
    }
    fn to_complex(&self) -> Complex<f64> {
        Complex::new(*self, 0.0)
    }
    fn from_complex(value: Complex<f64>) -> Self where Self: Sized {
        // This is a lossy conversion, but we take the real part for real types. The user should ensure that the imaginary part is negligible when using this conversion for real types.
        value.re
    }
}

impl FromComplex64 for f64 {
    fn from_complex(value: Complex<f64>) -> Self {
        value.re
    }
}

impl IsAnalytic for Complex<f64> {
    fn f_conj(&self) -> Self {
        self.conj()
    }
    fn f_exp(&self) -> Self {
        self.exp()
    }
    fn f_ln(&self) -> Self {
        self.ln()
    }
    fn f_powf(&self, n: f64) -> Self {
        self.powf(n)
    }
    fn f_powi(&self, n: i32) -> Self {
        self.powi(n)
    }
    fn f_abs(&self) -> f64 {
        self.norm()
    }
    fn f_arg(&self) -> f64 {
        self.arg()  
    }
    fn f_abs2(&self) -> f64 {
        self.norm_sqr()
    }
    fn f_scale(&self, factor: f64) -> Self {
        self * factor
    }
    fn to_complex(&self) -> Complex<f64> {
        *self
    }
    fn from_complex(value: Complex<f64>) -> Self where Self: Sized {
        value
    }
}

impl FromComplex64 for Complex<f64> {
    fn from_complex(value: Complex<f64>) -> Self {
        value
    }
}
