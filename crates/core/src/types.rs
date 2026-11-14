/// The error type of the library: every fallible function in the `lib_jsl` crates returns
/// `Result<T, ErrorsJSL>` (the linear-algebra crate's `LinalgError` converts into it with `?`).
///
/// It implements [`std::error::Error`] and [`std::fmt::Display`], so it works with `?` into
/// `Box<dyn Error>` and with error-handling crates. New variants may be added in minor releases
/// (`#[non_exhaustive]`), so a `match` on it needs a wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
// ANCHOR: errors
pub enum ErrorsJSL {
    /// The requested case is not implemented.
    NotImplementedYet,
    /// Two sizes that must agree do not.
    IncompatibleArraySizes((usize, usize)),
    /// An argument is outside its valid range or otherwise invalid.
    InvalidInputRange(&'static str),
    /// A computation failed: no convergence, a singular system, a non-finite value.
    RuntimeError(&'static str),
    /// An optional dependency or input that the operation needs is missing.
    MissingDependency(&'static str),
    /// Anything else.
    Other(&'static str),
    /// An object was configured inconsistently before use.
    Misconfiguration(&'static str),
    /// A linear-algebra failure, with the message of the underlying `LinalgError`.
    LinearAlgebra(String),
    /// A failure described by a message built at run time.
    Message(String),
}
// ANCHOR_END: errors

impl std::fmt::Display for ErrorsJSL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorsJSL::NotImplementedYet => write!(f, "not implemented yet"),
            ErrorsJSL::IncompatibleArraySizes((a, b)) => write!(f, "incompatible sizes: {a} and {b}"),
            ErrorsJSL::InvalidInputRange(m) => write!(f, "invalid input: {m}"),
            ErrorsJSL::RuntimeError(m) => write!(f, "computation failed: {m}"),
            ErrorsJSL::MissingDependency(m) => write!(f, "missing dependency: {m}"),
            ErrorsJSL::Other(m) => write!(f, "{m}"),
            ErrorsJSL::Misconfiguration(m) => write!(f, "misconfiguration: {m}"),
            ErrorsJSL::LinearAlgebra(m) => write!(f, "linear algebra: {m}"),
            ErrorsJSL::Message(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ErrorsJSL {}

impl From<String> for ErrorsJSL {
    /// A run-time message becomes [`ErrorsJSL::Message`].
    fn from(message: String) -> Self {
        ErrorsJSL::Message(message)
    }
}
