use crate::random::xoshiro256plusplus::Xoshiro256PlusPlus;

pub trait UniformGenerator{
    fn next_u64(&mut self) -> u64;
    fn next_f64(&mut self) -> f64;
}

pub type UniformRNG = Xoshiro256PlusPlus;