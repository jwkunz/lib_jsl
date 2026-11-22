# lib_jsl

**Jake's Scientific Library** is a pure-Rust workspace of algorithms for scientific computing:
numerical integration and differential equations, linear algebra, optimisation, special
functions, signal processing, electromagnetics, statistics, machine learning, geometry, graphs
and more. Everything is implemented from first principles, tested against analytic results and
independent references, and documented for people who want to understand how the algorithms
work as well as use them.

- **Home page:** <https://jwkunz.github.io/lib_jsl/>, which leads to the API reference and the
  technical documentation below.
- **Technical documentation (the book):** <https://jwkunz.github.io/lib_jsl/book/>, an
  encyclopedia of the algorithms (what they do, who devised them, why they work, how they are
  implemented), written for advanced undergraduates.
- **API reference:** <https://jwkunz.github.io/lib_jsl/api/>
- **Catalogue:** the book's [catalogue of algorithms](https://jwkunz.github.io/lib_jsl/book/catalogue.html) lists everything the library
  implements (655 items), grouped by subject, and what it deliberately does not cover.

## Status

| | |
|---|---|
| Version | 2.21.1 (all crates share the workspace version); see the [changelog](CHANGELOG.md) |
| Algorithms | 655 algorithms and components across 23 library crates |
| Tests | 1,510 unit tests, doctests and example builds; `cargo clippy` clean |
| Dependencies | Pure Rust; no system libraries. Runtime dependencies: `ndarray`, `num-complex`, `num-traits` (and `matrixmultiply`, which ndarray already uses) |
| Documentation | API reference for every crate; a book of 188 entries covering every subject of the catalogue, each with its idea, algorithm, measured results and exercises |
| Stability | From 2.0.0 the API follows semantic versioning strictly: one error type, `#[non_exhaustive]` results and errors; see [API conventions](https://jwkunz.github.io/lib_jsl/book/foundations/api-conventions.html) |
| Publishing | 24 crates on [crates.io](https://crates.io/search?q=lib_jsl): the umbrella [`lib_jsl`](https://crates.io/crates/lib_jsl) and one `lib_jsl_*` crate per domain (listed below); each has its API reference on [docs.rs](https://docs.rs/lib_jsl) and the project [home page](https://jwkunz.github.io/lib_jsl/) as its homepage |

## Crates

The umbrella crate [`lib_jsl`](https://crates.io/crates/lib_jsl) re-exports the domain crates (`lib_jsl::integration`,
`lib_jsl::linalg`, ...). Each can also be used on its own.

| Crate | Contents |
|---|---|
| [`lib_jsl_core`](https://crates.io/crates/lib_jsl_core) | Shared types, array aliases, traits, errors and physical constants |
| [`lib_jsl_linalg`](https://crates.io/crates/lib_jsl_linalg) | Dense linear algebra for real and complex matrices: LU, Cholesky, eigensolvers, SVD, least squares, the matrix exponential, Sylvester, Lyapunov and Riccati equations; sparse matrices with minimum-degree sparse LU, Krylov solvers with ILU preconditioning, Lanczos eigensolvers and expm_multiply |
| [`lib_jsl_linalg_reference`](https://crates.io/crates/lib_jsl_linalg_reference) | Readable textbook linear algebra (Gaussian elimination, Gram-Schmidt, power iteration, Jacobi, the QR algorithm) for study, each paired with a test showing why production code does something else |
| [`lib_jsl_integration`](https://crates.io/crates/lib_jsl_integration) | Quadrature (Gauss, Romberg, tanh-sinh, adaptive, cubature, quasi-Monte Carlo), ODE solvers (Runge-Kutta, Rosenbrock, BDF), boundary value problems, PDEs (finite difference, finite element, spectral) |
| [`lib_jsl_optimization`](https://crates.io/crates/lib_jsl_optimization) | Line searches, BFGS and L-BFGS(-B), conjugate gradient, trust-region Newton (trust-ncg, dogleg, trust-exact), Levenberg-Marquardt, SQP and augmented Lagrangian, Nelder-Mead, Powell, simulated and dual annealing, basin hopping, DIRECT, differential evolution, particle swarm, linear sum assignment, linear and dynamic programming |
| [`lib_jsl_root_finding`](https://crates.io/crates/lib_jsl_root_finding) | Bracketing methods, Brent, Newton, polynomial roots, nonlinear systems (Newton, Broyden) |
| [`lib_jsl_special_functions`](https://crates.io/crates/lib_jsl_special_functions) | Gamma, beta, error, Bessel, Airy, elliptic, hypergeometric and zeta functions, orthogonal polynomials, spherical harmonics, Lambert W, polylogarithm, statistical distributions, log-domain functions, polygamma, inverse incomplete gamma and beta |
| [`lib_jsl_interpolation`](https://crates.io/crates/lib_jsl_interpolation) | 1D, 2D, 3D and N-D interpolation: splines, piecewise polynomials, Hermite and Krogh, rational (AAA, Floater-Hormann, Bulirsch-Stoer), Chebyshev, Fourier, scattered data (Delaunay linear, Clough-Tocher, N-D radial basis functions) |
| [`lib_jsl_derivatives`](https://crates.io/crates/lib_jsl_derivatives) | Numerical differentiation |
| [`lib_jsl_ffts`](https://crates.io/crates/lib_jsl_ffts) | FFT engines (radix, Bluestein, SIMD), real and multidimensional FFTs, DCT/DST (1-D and N-D), fractional FFT, frequency grids and fast lengths |
| [`lib_jsl_ffts_reference`](https://crates.io/crates/lib_jsl_ffts_reference) | Readable reference FFTs for study |
| [`lib_jsl_dsp`](https://crates.io/crates/lib_jsl_dsp) | Filters (FIR, IIR, adaptive), spectral estimation (Welch, STFT, MUSIC), resampling (polyphase, CIC, half-band), synchronisation (PLL, Costas), communications, radar pulse compression, controls, partial fractions, zoom FFT, test waveforms |
| [`lib_jsl_electromagnetics`](https://crates.io/crates/lib_jsl_electromagnetics) | Field containers on structured domains, vector calculus, antenna models, sensor arrays, beamformers, direction finding, transmission lines and networks, propagation, spherical waves, near-to-far-field transformation, method of moments, FDTD |
| [`lib_jsl_statistics`](https://crates.io/crates/lib_jsl_statistics) | Descriptive and circular statistics, correlation, robust regression, distances, distributions, hypothesis tests, kernel density estimation, bootstrap |
| [`lib_jsl_random`](https://crates.io/crates/lib_jsl_random) | Generators, quasi-random sequences (Sobol, Halton), distributions, MCMC |
| [`lib_jsl_machine_learning`](https://crates.io/crates/lib_jsl_machine_learning) | Regression, classification, clustering, dimensionality reduction, neural networks, gradient boosting, model selection, hidden Markov models |
| [`lib_jsl_graph`](https://crates.io/crates/lib_jsl_graph) | Graph representations, shortest paths, spanning trees, maximum flow, orderings |
| [`lib_jsl_geometry`](https://crates.io/crates/lib_jsl_geometry) | 1D to 4D geometry types, Delaunay and Voronoi, convex hulls, spatial trees, intersections, mesh processing, polygon booleans |
| [`lib_jsl_number_theory`](https://crates.io/crates/lib_jsl_number_theory) | Modular arithmetic, primes and primality, factorisation, Chinese remainder theorem, polynomials |
| [`lib_jsl_information`](https://crates.io/crates/lib_jsl_information) | Source coding (Huffman, arithmetic, LZ family), LFSRs, CRC and SHA-256, error-correcting codes (LDPC, polar, Reed-Solomon, convolutional) |
| [`lib_jsl_file_io`](https://crates.io/crates/lib_jsl_file_io) | WAV, JPEG, PNG, Netpbm, CSV, NumPy `.npy`, Matrix Market, MATLAB, Harwell-Boeing, NetCDF-3, Fortran unformatted and ARFF files |
| [`lib_jsl_sorting`](https://crates.io/crates/lib_jsl_sorting) | Sorting and selection |
| [`lib_jsl_utils`](https://crates.io/crates/lib_jsl_utils) | Terminal plots, unit conversion, numerical helpers |

## Using the library

```toml
[dependencies]
lib_jsl = "2.21"
# or a single domain:
lib_jsl_integration = "2.21"
```

Every crate is listed on crates.io (the table above links each one; [search for `lib_jsl`](https://crates.io/search?q=lib_jsl)
to see them all) and documented on [docs.rs](https://docs.rs/lib_jsl). Each listing's homepage link leads to the
[project home page](https://jwkunz.github.io/lib_jsl/), which routes to the API reference and the book.

```rust
use lib_jsl::integration::gaussian_quadrature::GaussLegendreEngine;
use lib_jsl::integration::can_integrate::CanIntegrate1D;

let engine = GaussLegendreEngine::new(16, 0.0, std::f64::consts::PI).unwrap();
let value = engine.integrate(f64::sin).unwrap(); // 2 to machine precision
```

Optional features: `serde` enables serialisation of geometry types; `threading` enables
multithreaded matrix multiplication inside the linear algebra factorisations.

## Linear algebra performance

`lib_jsl_linalg` replaces the LAPACK bindings the library used to depend on, with no system
libraries. Its traits mirror `ndarray-linalg`'s names, so code ports by changing an import. LU
and Cholesky are recursive or blocked around a matrix-multiply kernel (`matrixmultiply`, already
a dependency of ndarray), Householder reflectors are accumulated in compact WY blocks, real
eigenproblems use Francis double-shift QR (with aggressive early deflation on large matrices since 2.21.0), and the SVD uses Golub-Kahan bidiagonalisation (with
one-sided Jacobi available as `SvdJacobi` for high relative accuracy). Mean times from the
criterion benchmarks against `ndarray-linalg` with OpenBLAS, both single-threaded (Intel Core
i7-7600U, real matrices; the last figure in each cell is our time relative to LAPACK, below 1
meaning faster):

| Operation | n = 16 | n = 64 | n = 256 |
|---|---:|---:|---:|
| Solve (LU) | 3.8 µs vs 3.4 µs (1.1×) | 55.4 µs vs 39.4 µs (1.4×) | 1.11 ms vs 0.70 ms (1.6×) |
| Inverse | 6.2 µs vs 6.5 µs (1.0×) | 109 µs vs 96.8 µs (1.1×) | 2.76 ms vs 2.57 ms (1.1×) |
| Cholesky | 2.3 µs vs 1.7 µs (1.4×) | 38.5 µs vs 21.1 µs (1.8×) | 0.89 ms vs 0.79 ms (1.1×) |
| Symmetric eigenproblem | 31.2 µs vs 30.5 µs (1.0×) | 623 µs vs 721 µs (0.9×) | 23.5 ms vs 26.9 ms (0.9×) |
| General eigenproblem | 71.9 µs vs 110 µs (0.7×) | 2.35 ms vs 2.37 ms (1.0×) | 128 ms vs 51.7 ms (2.5×) |
| SVD with vectors | 44.1 µs vs 56.2 µs (0.8×) | 996 µs vs 1.20 ms (0.8×) | 37.1 ms vs 93.9 ms (0.4×) |
| Least squares (2n × n) | 35.7 µs vs 62.9 µs (0.6×) | 668 µs vs 1.00 ms (0.7×) | 23.5 ms vs 16.5 ms (1.4×) |

Most operations are within a factor of 1.6 of OpenBLAS, and several are faster. The general
eigenproblem is the exception at larger sizes. The table above was measured before 2.21.0, which
added aggressive early deflation: on a later test machine (single thread, minimum of 15 runs) that
brought the general eigenproblem with eigenvectors from 2.3 to about 1.5 times LAPACK's time at
n = 256 and from 3.3 to 2.3 times at n = 512. LAPACK's blocked multishift sweeps are not
implemented here ([#9](https://github.com/jwkunz/lib_jsl/issues/9)). OpenBLAS's default multithreading is slower than its
single-threaded mode at these sizes; the optional `threading` feature enables
`matrixmultiply`'s threads, which pay off from a few hundred rows (see the `large_sizes`
example in `crates/linalg_bench`). Reproduce with

```console
OPENBLAS_NUM_THREADS=1 cargo bench --manifest-path crates/linalg_bench/Cargo.toml --bench linalg_vs_lapack
```

(this standalone package needs the system OpenBLAS library; the library itself does not).

The textbook algorithms in `lib_jsl_linalg_reference` race the production crate with
`cargo bench -p lib_jsl_linalg_reference`.

## Development

```console
cargo test --workspace --release        # all tests, doctests and examples
cargo clippy --workspace --all-targets  # kept free of warnings
cargo run --release -p lib_jsl_electromagnetics --example beamforming_demo
```

Examples live in each crate's `examples/` directory. Demonstrations include
`antenna_patterns`, `beamforming_demo` and `fields_demo` (electromagnetics); `classification_demo`,
`clustering_demo`, `embedding_demo` and `regression_demo` (machine learning); and the quadrature
examples used by the book (integration).

### Documentation

The book lives in `book/` (mdBook with KaTeX). Its figures and tables are generated by running the
library (`book/generator`), and its code excerpts are included directly from the sources, so it
cannot drift from the code.

```console
cargo install mdbook mdbook-katex
book/build-site.sh                      # book + API reference into target/site/
python3 -m http.server -d target/site 8000
```

Pushes to `main` publish the site to GitHub Pages through `.github/workflows/docs.yml`. See
[Writing an entry](book/src/writing-an-entry.md) for the conventions used by book chapters.

### Releasing

The crates are published to crates.io together, at the workspace version. Check every package, then
publish in dependency order (`lib_jsl_core` first):

```console
cargo package --workspace --exclude lib_jsl_book_generator   # builds each package from its tarball
cargo publish --workspace --exclude lib_jsl_book_generator
```

Published versions cannot be deleted, only yanked, and the metadata of an uploaded version (such as its
homepage) cannot change, so a fix to it needs a new version. crates.io limits how fast *new* crate names are
created (one every ten minutes), but not new versions of existing crates. The release checklist for new
features is at the end of [#12](https://github.com/jwkunz/lib_jsl/issues/12).

### Conventions

- Crate roots use `mod.rs`.
- Dependency versions are pinned once, in the workspace manifest.
- One commit per feature, each with tests; a new algorithm is added to the book's catalogue
  (`book/src/catalogue.md`) in the same commit.
- [Semantic versioning](https://semver.org): breaking API changes need a new major version.
  All crates are versioned together; internal dependencies are declared once in the workspace
  manifest with both a path and a version, so the crates can be published.

## Roadmap

2.0.0 stabilised the API after twenty minor releases that closed most of the gaps against SciPy, and releases
2.1 to 2.21 closed most of the rest. Planned work is tracked in [GitHub issues](https://github.com/jwkunz/lib_jsl/issues),
with the plan in [#12](https://github.com/jwkunz/lib_jsl/issues/12):

- **Next releases:** sparse direct solvers (COLAMD, nested dissection, incomplete Cholesky;
  [#32](https://github.com/jwkunz/lib_jsl/issues/32)), the remaining file formats (IDL `.sav`, MATLAB v4;
  [#33](https://github.com/jwkunz/lib_jsl/issues/33)) and NumPy `.npz` archives
  ([#34](https://github.com/jwkunz/lib_jsl/issues/34)).
- **Deferred:** HDF5-based formats ([#35](https://github.com/jwkunz/lib_jsl/issues/35)), complex sparse matrices
  ([#36](https://github.com/jwkunz/lib_jsl/issues/36)) and the `stats.sampling` generators
  ([#37](https://github.com/jwkunz/lib_jsl/issues/37)).
- **Known limitations** of recent features ([#38](https://github.com/jwkunz/lib_jsl/issues/38)) and
  **linear algebra performance**: blocked multishift QR for large general eigenproblems
  ([#9](https://github.com/jwkunz/lib_jsl/issues/9)) and LU and small Cholesky
  ([#10](https://github.com/jwkunz/lib_jsl/issues/10)).
- The SciPy gap issues ([#13](https://github.com/jwkunz/lib_jsl/issues/13) to
  [#18](https://github.com/jwkunz/lib_jsl/issues/18)) record, item by item, what has been implemented.

## License

MIT; see [LICENSE](LICENSE).
