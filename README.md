# lib_jsl

**Jake's Scientific Library** is a pure-Rust workspace of algorithms for scientific computing:
numerical integration and differential equations, linear algebra, optimisation, special
functions, signal processing, electromagnetics, statistics, machine learning, geometry, graphs
and more. Everything is implemented from first principles, tested against analytic results and
independent references, and documented for people who want to understand how the algorithms
work as well as use them.

- **Book:** <https://jwkunz.github.io/lib_jsl/book/>, an encyclopedia of the algorithms (what
  they do, who devised them, why they work, how they are implemented), written for advanced
  undergraduates.
- **API reference:** <https://jwkunz.github.io/lib_jsl/api/>
- **Catalogue:** the book's [catalogue of algorithms](https://jwkunz.github.io/lib_jsl/book/catalogue.html) lists everything the library
  implements (481 items), grouped by subject, and what it deliberately does not cover.

## Status

| | |
|---|---|
| Version | 1.7.0 (all crates share the workspace version); see the [changelog](CHANGELOG.md) |
| Algorithms | 481 algorithms and components across 23 library crates |
| Tests | 1,257 unit tests, doctests and example builds; `cargo clippy` clean |
| Dependencies | Pure Rust; no system libraries. Runtime dependencies: `ndarray`, `num-complex`, `num-traits` (and `matrixmultiply`, which ndarray already uses) |
| Documentation | API reference for every crate; a book of 149 entries covering every subject of the catalogue, each with its idea, algorithm, measured results and exercises |
| Publishing | Every crate packages and verifies for crates.io (`cargo package --workspace`); not yet published, so use a git dependency for now |

## Crates

The umbrella crate `lib_jsl` re-exports the domain crates (`lib_jsl::integration`,
`lib_jsl::linalg`, ...). Each can also be used on its own.

| Crate | Contents |
|---|---|
| `lib_jsl_core` | Shared types, array aliases, traits, errors and physical constants |
| `lib_jsl_linalg` | Dense linear algebra for real and complex matrices: LU, Cholesky, eigensolvers, SVD, least squares, the matrix exponential, Sylvester, Lyapunov and Riccati equations |
| `lib_jsl_linalg_reference` | Readable textbook linear algebra (Gaussian elimination, Gram-Schmidt, power iteration, Jacobi, the QR algorithm) for study, each paired with a test showing why production code does something else |
| `lib_jsl_integration` | Quadrature (Gauss, Romberg, tanh-sinh, adaptive, cubature, quasi-Monte Carlo), ODE solvers (Runge-Kutta, Rosenbrock, BDF), boundary value problems, PDEs (finite difference, finite element, spectral) |
| `lib_jsl_optimization` | Line searches, BFGS and L-BFGS, conjugate gradient, Levenberg-Marquardt, SQP and augmented Lagrangian, Nelder-Mead, Powell, simulated annealing, differential evolution, particle swarm, linear and dynamic programming |
| `lib_jsl_root_finding` | Bracketing methods, Brent, Newton, polynomial roots, nonlinear systems (Newton, Broyden) |
| `lib_jsl_special_functions` | Gamma, beta, error, Bessel, Airy, elliptic, hypergeometric and zeta functions, orthogonal polynomials, spherical harmonics, Lambert W, polylogarithm, statistical distributions |
| `lib_jsl_interpolation` | 1D, 2D, 3D and N-D interpolation: splines, rational (AAA, Floater-Hormann, Bulirsch-Stoer), Chebyshev, Fourier, radial basis functions |
| `lib_jsl_derivatives` | Numerical differentiation |
| `lib_jsl_ffts` | FFT engines (radix, Bluestein, SIMD), real and multidimensional FFTs, DCT/DST, fractional FFT |
| `lib_jsl_ffts_reference` | Readable reference FFTs for study |
| `lib_jsl_dsp` | Filters (FIR, IIR, adaptive), spectral estimation (Welch, STFT, MUSIC), resampling (polyphase, CIC, half-band), synchronisation (PLL, Costas), communications, radar pulse compression, controls |
| `lib_jsl_electromagnetics` | Field containers on structured domains, vector calculus, antenna models, sensor arrays, beamformers, direction finding, transmission lines and networks, propagation, spherical waves, near-to-far-field transformation, method of moments, FDTD |
| `lib_jsl_statistics` | Descriptive statistics, correlation, hypothesis tests, kernel density estimation, bootstrap |
| `lib_jsl_random` | Generators, quasi-random sequences (Sobol, Halton), distributions, MCMC |
| `lib_jsl_machine_learning` | Regression, classification, clustering, dimensionality reduction, neural networks, gradient boosting, model selection, hidden Markov models |
| `lib_jsl_graph` | Graph representations, shortest paths, spanning trees, maximum flow, orderings |
| `lib_jsl_geometry` | 1D to 4D geometry types, Delaunay and Voronoi, convex hulls, spatial trees, intersections, mesh processing, polygon booleans |
| `lib_jsl_number_theory` | Modular arithmetic, primes and primality, factorisation, Chinese remainder theorem, polynomials |
| `lib_jsl_information` | Source coding (Huffman, arithmetic, LZ family), LFSRs, CRC and SHA-256, error-correcting codes (LDPC, polar, Reed-Solomon, convolutional) |
| `lib_jsl_file_io` | WAV, JPEG, PNG, Netpbm, CSV and NumPy `.npy` files |
| `lib_jsl_sorting` | Sorting and selection |
| `lib_jsl_utils` | Terminal plots, unit conversion, numerical helpers |

## Using the library

```toml
[dependencies]
lib_jsl = { git = "https://github.com/jwkunz/lib_jsl", tag = "v1.7.0" }
# or a single domain:
lib_jsl_integration = { git = "https://github.com/jwkunz/lib_jsl", tag = "v1.7.0" }
```

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
eigenproblems use Francis double-shift QR, and the SVD uses Golub-Kahan bidiagonalisation (with
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
eigenproblem is the exception at larger sizes: LAPACK's multishift QR with aggressive early
deflation is not implemented here. OpenBLAS's default multithreading is slower than its
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

### Conventions

- Crate roots use `mod.rs`.
- Dependency versions are pinned once, in the workspace manifest.
- One commit per feature, each with tests; a new algorithm is added to the book's catalogue
  (`book/src/catalogue.md`) in the same commit.
- [Semantic versioning](https://semver.org): breaking API changes need a new major version.
  All crates are versioned together; internal dependencies are declared once in the workspace
  manifest with both a path and a version, so the crates can be published.

## Next phase

1.0.0 completes the original plan of algorithms. Planned work is tracked in
[GitHub issues](https://github.com/jwkunz/lib_jsl/issues):

- **The book**: the chapters are written; remaining work is to close the gaps found in review
  ([#21](https://github.com/jwkunz/lib_jsl/issues/21) to [#30](https://github.com/jwkunz/lib_jsl/issues/30)).
- **Publishing** the crates to crates.io ([#8](https://github.com/jwkunz/lib_jsl/issues/8)).
- **Linear algebra performance**: multishift QR for large general eigenproblems
  ([#9](https://github.com/jwkunz/lib_jsl/issues/9)), LU and small Cholesky
  ([#10](https://github.com/jwkunz/lib_jsl/issues/10)) and blocked pivoted QR
  ([#11](https://github.com/jwkunz/lib_jsl/issues/11)).

## License

MIT; see [LICENSE](LICENSE).
