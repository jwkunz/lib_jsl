use ndarray::prelude::*;
use num::{Complex};

pub type R1D = Array1<f64>;
pub type C1D = Array1<Complex<f64>>;
pub type VR1D<'a> = ArrayView1<'a,f64>;
pub type VC1D<'a> = ArrayView1<'a,num::Complex<f64>>;

pub type R2D = Array2<f64>;
pub type C2D = Array2<Complex<f64>>;
pub type VR2D<'a> = ArrayView2<'a,f64>;
pub type VC2D<'a> = ArrayView2<'a,num::Complex<f64>>;

#[derive(Debug)]
pub enum ErrorsJSL{
    NotImplementedYet,
    IncompatibleArraySizes((usize,usize)), 
    InvalidInputRange(&'static str),
    RuntimeError(&'static str),
    MissingDependency(&'static str),
    Other(&'static str),
    Misconfiguration(&'static str)
}