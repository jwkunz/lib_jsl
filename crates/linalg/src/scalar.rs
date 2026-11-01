//! The scalar types the decompositions work over: `f64` and `Complex<f64>`.

use num_complex::Complex;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A real or complex double-precision scalar.
pub trait Scalar:
    Copy
    + Debug
    + PartialEq
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// `true` for complex scalars.
    const IS_COMPLEX: bool;
    fn zero() -> Self;
    fn one() -> Self;
    fn from_real(re: f64) -> Self;
    /// The complex conjugate (the identity for reals).
    fn conj(self) -> Self;
    /// The real part.
    fn re(self) -> f64;
    /// The imaginary part (zero for reals).
    fn im(self) -> f64;
    /// The modulus `|z|`.
    fn abs(self) -> f64;
    /// `|z|²`.
    fn abs_sqr(self) -> f64;
    /// `z · r` for a real `r`.
    fn scale(self, r: f64) -> Self;
    fn is_finite(self) -> bool;
    /// As a complex number.
    fn to_complex(self) -> Complex<f64>;
    /// From a complex number (the real part, for reals).
    fn from_complex(z: Complex<f64>) -> Self;
}

impl Scalar for f64 {
    const IS_COMPLEX: bool = false;
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_real(re: f64) -> Self {
        re
    }
    fn conj(self) -> Self {
        self
    }
    fn re(self) -> f64 {
        self
    }
    fn im(self) -> f64 {
        0.0
    }
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    fn abs_sqr(self) -> f64 {
        self * self
    }
    fn scale(self, r: f64) -> Self {
        self * r
    }
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    fn to_complex(self) -> Complex<f64> {
        Complex::new(self, 0.0)
    }
    fn from_complex(z: Complex<f64>) -> Self {
        z.re
    }
}

impl Scalar for Complex<f64> {
    const IS_COMPLEX: bool = true;
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }
    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }
    fn from_real(re: f64) -> Self {
        Complex::new(re, 0.0)
    }
    fn conj(self) -> Self {
        Complex::conj(&self)
    }
    fn re(self) -> f64 {
        self.re
    }
    fn im(self) -> f64 {
        self.im
    }
    fn abs(self) -> f64 {
        self.norm()
    }
    fn abs_sqr(self) -> f64 {
        self.norm_sqr()
    }
    fn scale(self, r: f64) -> Self {
        self * r
    }
    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    fn to_complex(self) -> Complex<f64> {
        self
    }
    fn from_complex(z: Complex<f64>) -> Self {
        z
    }
}
