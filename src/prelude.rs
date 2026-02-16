use std::{
    fmt::Debug,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign, Add, Div, Mul, Neg, Sub},
};

use ndarray::prelude::*;
use num::{Complex, One, Zero};

pub type R1D = Array1<f64>;
pub type C1D = Array1<Complex<f64>>;
pub type VR1D<'a> = ArrayView1<'a, f64>;
pub type VC1D<'a> = ArrayView1<'a, num::Complex<f64>>;

pub type R2D = Array2<f64>;
pub type C2D = Array2<Complex<f64>>;
pub type VR2D<'a> = ArrayView2<'a, f64>;
pub type VC2D<'a> = ArrayView2<'a, num::Complex<f64>>;

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

pub trait IsAnalytic: IsLinearOperatable {
    fn f_conj(&self) -> Self;
    fn f_exp(&self) -> Self;
    fn f_ln(&self) -> Self;
    fn f_powf(&self, n: f64) -> Self;
    fn f_powi(&self, n: i32) -> Self;
    fn f_abs(&self) -> f64;
    fn f_arg(&self) -> f64;
    fn f_abs2(&self) -> f64;
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
}
