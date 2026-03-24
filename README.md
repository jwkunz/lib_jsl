# lib_jsl
"Jake's Scientific Library" is a Rust scientific computing workspace. It collects numerical, signal-processing, geometry, optimization, random, and interpolation routines behind one umbrella crate while also exposing focused subcrates for direct use.

## Workspace Layout

The repository is organized as a Cargo workspace.

- `lib_jsl`
  Thin facade crate that re-exports the domain crates under the existing top-level names.
- `lib_jsl_core`
  Shared foundation types, aliases, and traits used across the workspace.
- `lib_jsl_geometry`
  Geometry traits, keyed geometry tables, and concrete 1D/2D/3D geometry types.
- `lib_jsl_dsp`
  DSP routines including filters, transforms, resampling, windows, and control helpers.
- `lib_jsl_ffts`
  FFT engines and related spectral transforms.
- `lib_jsl_integration`
  Numerical integration engines.
- `lib_jsl_interpolation`
  Interpolation utilities.
- `lib_jsl_number_theory`
  Number theory and polynomial routines.
- `lib_jsl_optimization`
  Optimization algorithms.
- `lib_jsl_random`
  Random number generators, distributions, and histogram helpers.
- `lib_jsl_derivatives`
  Numerical differentiation utilities.

## Using The Library

If you want the original broad import surface, depend on the umbrella crate:

```toml
[dependencies]
lib_jsl = { path = "." }
```

Then import through the facade:

```rust
use lib_jsl::geometry::concrete::CoordinateVector3D;
use lib_jsl::ffts::best_fft::BestFFT;
```

If you only need one domain, depend on the focused subcrate directly:

```toml
[dependencies]
lib_jsl_geometry = { path = "crates/geometry" }
lib_jsl_ffts = { path = "crates/ffts" }
```

## Development Notes

- The workspace keeps `mod.rs` as the preferred library entry style, including crate roots.
- Domain-specific examples and benches now live with their owning crates:
  - DSP examples: `crates/dsp/examples/`
  - FFT benches: `crates/ffts/benches/`
- Shared dependencies are centralized in the workspace manifest.

## Dependencies

The workspace intentionally keeps its dependency set fairly small. Core building blocks include `num`, `ndarray`, `ndarray-linalg`, `serde`, and `serde_json`, plus `criterion` and `rustfft` for benchmarking and comparison work.
