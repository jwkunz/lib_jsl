# Changelog

All crates in the workspace share one version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org).

## [1.0.0] - 2026-11-01

The first stable release: every item of the [roadmap](ROADMAP.md) (457 algorithms across 23
library crates) is implemented and tested.

### Added

- `lib_jsl_linalg`, a pure-Rust dense linear algebra crate (LU, Cholesky, Hermitian and general
  eigenproblems, SVD, least squares) whose traits mirror `ndarray-linalg`. It replaces the
  LAPACK bindings, so the library needs no system libraries. Recursive and blocked
  factorisations, compact WY reflector blocks, Francis double-shift QR and a Golub-Kahan SVD put
  it within a small factor of single-threaded OpenBLAS (faster for the SVD and the symmetric
  eigenproblem); one-sided Jacobi remains available through `SvdJacobi`.
- `lib_jsl_linalg_reference` and `lib_jsl_ffts_reference`: readable textbook implementations for
  study, with benchmarks against the production crates.
- An optional `threading` feature for multithreaded matrix multiplication in the
  factorisations, and an optional `serde` feature for geometry types.
- The documentation book (mdBook with KaTeX), with the numerical integration chapter as its
  pilot, published to GitHub Pages with the API reference.

### Changed

- Dependencies are pinned once in the workspace manifest; `num` was replaced by `num-complex`
  and `num-traits`, and `ndarray-linalg` by `lib_jsl_linalg`.

### Fixed

- The Simpson's 1/3 engine had its 4 and 2 weights swapped; it now also uses a 3/8 tail for an
  even number of points.

[1.0.0]: https://github.com/jwkunz/lib_jsl/releases/tag/v1.0.0
