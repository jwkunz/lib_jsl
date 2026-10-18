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
