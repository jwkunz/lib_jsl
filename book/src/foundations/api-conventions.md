# API conventions and the 2.0 migration guide

Version 2.0.0 is a stabilisation release: no new algorithms, only changes that make the public API consistent and
leave room to grow without further breaking changes. This page records the conventions the crates follow from 2.0
on, and how to update code written against 1.x.

## Conventions

### Errors

- Every fallible function returns `Result<T, ErrorsJSL>` (from `lib_jsl_core`, re-exported as `lib_jsl::prelude::ErrorsJSL`).
  The linear-algebra crate keeps its richer `LinalgError`, which converts into `ErrorsJSL` with `?`.
- `ErrorsJSL` implements `Display` and `std::error::Error`, so it works with `?` into `Box<dyn Error>` and with error-handling crates.
- The variants say what kind of failure occurred:

  | Variant | Meaning |
  |---|---|
  | `InvalidInputRange` | A bad argument |
  | `IncompatibleArraySizes` | Mismatched sizes |
  | `RuntimeError` | A computation failed: no convergence, a singular system |
  | `Misconfiguration` | An object was set up inconsistently |
  | `NotImplementedYet` | The case is not implemented |
  | `LinearAlgebra` | A converted linear-algebra error |
  | `Message` | A message built at run time |

  A `String` converts into `Message`.
- Error enums are `#[non_exhaustive]`: new variants can arrive in minor releases, so a `match` needs a wildcard arm.

### Options and results

- Tunable parameters come in an options struct named `…Options`, with public fields and a `Default` that matches
  SciPy's defaults where there is a SciPy counterpart. Override what you need:
  `LbfgsbOptions { gtol: 1e-8, ..Default::default() }`.
- Outputs with more than one part come in a struct named `…Result` (or a descriptive name such as `OdeSolution`).
  Result structs are `#[non_exhaustive]`: read their fields freely, but do not construct them in your own code (use a
  constructor such as `TestResult::new` where one is provided). New fields can then be added without a major release.
- Counters and flags use the same names everywhere:

  | Field | Meaning |
  |---|---|
  | `nfev`, `njev`, `nhev` | Evaluations of the function, its Jacobian or gradient, and its Hessian |
  | `iterations` | Iterations of the main loop |
  | `converged` | Whether the convergence test was met |
  | `status` | A numeric code, where a method distinguishes several kinds of failure |

### Data

- Vectors are passed as slices `&[f64]` and returned as `Vec<f64>`. Matrices and N-D arrays use `ndarray` (`Array2<f64>`,
  `ArrayD<f64>`), and complex values `num_complex::Complex<f64>`.
- Polynomials are stored highest power first, as in NumPy and SciPy. Angles are in radians unless a parameter's name
  or documentation says degrees.
- Functions that mirror SciPy keep SciPy's name (`linear_sum_assignment`, `zoom_fft`, `gammaincinv`), except where a
  Rust name would clash (`square_wave` for SciPy's `signal.square`).

### Stability

From 2.0.0 the crates follow [semantic versioning](https://semver.org) strictly. A minor release may add functions,
types, enum variants marked `#[non_exhaustive]`, and fields to `#[non_exhaustive]` structs. Removing or renaming any
of these needs a major release. Options structs are not `#[non_exhaustive]`, so that they can be written as literals
with `..Default::default()`. Adding an options field is therefore a breaking change for code that names every field.
That is the documented exception: always end an options literal with `..Default::default()`.

Some older modules still follow earlier styles: trait-based "engine" objects in parts of optimisation and
integration, and positional tolerances in the scalar root finders. They keep working and are covered by the same
stability promise. New code in those areas uses the function-and-options style above.

## Migrating from 1.x

| 1.x | 2.0 |
|---|---|
| Root finders (`bisection`, `brent_method`, …, `newton_system`, `broyden_system`) return `Result<_, String>` | They return `Result<_, ErrorsJSL>`: `InvalidInputRange` for bad input (no sign change), `RuntimeError` for no convergence |
| Geometry constructors and editors (`PolygonFace2D::from_triangles`, `VolumeMesh3D::from_tetrahedra`, …) return `Result<_, String>` | They return `Result<_, ErrorsJSL>`, with the message in `ErrorsJSL::Message` |
| `BoundedResult::function_evaluations`, `OdeSolution::function_evaluations`, `SystemRootResult::function_evaluations` | `nfev` |
| `MonteCarloResult::evaluations`, `QuasiMonteCarloResult::evaluations`, `CubatureResult::evaluations`, `PathIntegral::evaluations` | `nfev` |
| `DerivativeResult::success` | `converged` |
| `calculate_midpoints` (deprecated since 1.x) | `grid_points` (or `panel_midpoints` for panel midpoints) |
| `runge_kutta_4`, `dormand_prince`, `dormand_prince_step`, `dormand_prince_adaptive_step` (deprecated; they solved only `y' = f(t)`) | `rk4_solve`, `dormand_prince_solve` for `y' = f(t, y)`; for `∫ f(t) dt` use the quadrature routines |
| `match` on `ErrorsJSL` or `LinalgError` without a wildcard | Add a `_ =>` arm |
| Struct literals of `…Result` types, such as `TestResult { … }` | Use the provided constructor, `TestResult::new(statistic, p_value, dof)` |
| `MetropolisOptions` had no `Default` | `MetropolisOptions { samples: 50_000, ..Default::default() }`; an empty `proposal_scale` means 1 for every coordinate |

Most code needs only the field renames and, where errors were handled as strings, `err.to_string()` in place of the
`String`. For example:

```rust,ignore
// 1.x
let root = brent_method(f, 0.0, 2.0, 1e-12).map_err(|e: String| e)?;
let n = result.function_evaluations;

// 2.0
let root = brent_method(f, 0.0, 2.0, 1e-12)?;   // ErrorsJSL, works with `?` into Box<dyn Error>
let n = result.nfev;
```
