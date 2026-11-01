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
- **Roadmap:** [ROADMAP.md](ROADMAP.md) tracks every algorithm (457 implemented).

## Status

| | |
|---|---|
| Algorithms | 457 roadmap items across 23 library crates, all implemented |
| Tests | 1,256 unit tests, doctests and example builds; `cargo clippy` clean |
| Dependencies | Pure Rust; no system libraries. Runtime dependencies: `ndarray`, `num-complex`, `num-traits` (and `matrixmultiply`, which ndarray already uses) |
| Documentation | API reference for every crate; the book's integration chapter is the pilot, with the other chapters in progress |
| Publishing | Not yet on crates.io; use a path or git dependency |

## Crates

The umbrella crate `lib_jsl` re-exports the domain crates (`lib_jsl::integration`,
`lib_jsl::linalg`, ...). Each can also be used on its own.

| Crate | Contents |
|---|---|
| `lib_jsl_core` | Shared types, array aliases, traits, errors and physical constants |
| `lib_jsl_linalg` | Dense linear algebra for real and complex matrices: LU, Cholesky, eigensolvers, SVD, least squares |
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
| `lib_jsl_file_io` | WAV, MP3, JPEG, PNG, Netpbm, CSV and NumPy `.npy` files |
| `lib_jsl_sorting` | Sorting and selection |
| `lib_jsl_utils` | Terminal plots, unit conversion, numerical helpers |

## Using the library

```toml
[dependencies]
lib_jsl = { git = "https://github.com/jwkunz/lib_jsl" }
# or a single domain:
lib_jsl_integration = { git = "https://github.com/jwkunz/lib_jsl" }
```

```rust
use lib_jsl::integration::gaussian_quadrature::GaussLegendreEngine;
use lib_jsl::integration::can_integrate::CanIntegrate1D;

let engine = GaussLegendreEngine::new(16, 0.0, std::f64::consts::PI).unwrap();
let value = engine.integrate(f64::sin).unwrap(); // 2 to machine precision
```

Optional features: `serde` enables serialisation of geometry types.

## Linear algebra performance

`lib_jsl_linalg` replaces the LAPACK bindings the library used to depend on, with no system
libraries. Its traits mirror `ndarray-linalg`'s names, so code ports by changing an import. The
factorisations are blocked around a matrix-multiply kernel (`matrixmultiply`, already a
dependency of ndarray), real eigenproblems use Francis double-shift QR, and the SVD is a
QR-preconditioned one-sided Jacobi method. Mean times from the criterion benchmarks against
`ndarray-linalg` with OpenBLAS (Intel Core i7-7600U, real matrices, single-threaded; the last
figure in each cell is our time relative to LAPACK, below 1 meaning faster):

| Operation | n = 16 | n = 64 | n = 256 |
|---|---:|---:|---:|
| Solve (LU) | 2.6 µs vs 3.6 µs (0.7×) | 52.2 µs vs 39.0 µs (1.3×) | 1.28 ms vs 1.07 ms (1.2×) |
| Inverse | 4.7 µs vs 6.8 µs (0.7×) | 99.6 µs vs 90.5 µs (1.1×) | 3.01 ms vs 2.52 ms (1.2×) |
| Cholesky | 2.3 µs vs 2.0 µs (1.2×) | 39.1 µs vs 23.7 µs (1.7×) | 0.86 ms vs 1.69 ms (0.5×) |
| Symmetric eigenproblem | 28.4 µs vs 32.0 µs (0.9×) | 643 µs vs 1.17 ms (0.5×) | 25.5 ms vs 53.0 ms (0.5×) |
| General eigenproblem | 73.4 µs vs 120 µs (0.6×) | 2.69 ms vs 2.37 ms (1.1×) | 141 ms vs 84.5 ms (1.7×) |
| SVD with vectors | 50.8 µs vs 58.0 µs (0.9×) | 2.26 ms vs 1.20 ms (1.9×) | 132 ms vs 122 ms (1.1×) |
| Least squares (2n × n) | 35.6 µs vs 63.7 µs (0.6×) | 680 µs vs 977 µs (0.7×) | 24.0 ms vs 27.4 ms (0.9×) |

Every operation is within a factor of 2 of OpenBLAS, and several are faster. Reproduce with

```console
cargo bench --manifest-path crates/linalg_bench/Cargo.toml --bench linalg_vs_lapack
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
- One commit per roadmap feature, each with tests.

## License

MIT; see [LICENSE](LICENSE).
