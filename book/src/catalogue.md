# Catalogue of algorithms

This catalogue lists everything the library implements, grouped by subject. Use it to find
where an algorithm lives, to see the neighbours of the one you are studying, or to plan a path
through the book. It describes the current release and is updated whenever an algorithm is
added; the [changelog](https://github.com/jwkunz/lib_jsl/blob/main/CHANGELOG.md) records what
changed between releases.

Book chapters follow the same subjects. Where a chapter has been written, its crate heading
links to it; the integration chapter is the first. For the precise types, functions and
parameters, follow the API reference link under each crate.

<div class="catalogue">

The library implements **457 algorithms and components in 23 crates**:

| Subject | Crates | Items |
|---|---|---:|
| [Calculus and analysis](#calculus-and-analysis) | Derivatives, Integration, Interpolation, Root finding, Optimization, Special functions | 107 |
| [Linear algebra and transforms](#linear-algebra-and-transforms) | Linear algebra, Linear algebra reference, FFTs, FFT reference | 31 |
| [Probability, statistics and learning](#probability-statistics-and-learning) | Random, Statistics, Machine learning | 64 |
| [Signals, information and fields](#signals-information-and-fields) | DSP, Information, Electromagnetics | 174 |
| [Discrete mathematics and geometry](#discrete-mathematics-and-geometry) | Number theory, Graph, Geometry | 64 |
| [Foundations and input/output](#foundations-and-inputoutput) | Core, Utils, Sorting and selection, File IO | 17 |

## Calculus and analysis

Derivatives, integrals, interpolation, equations and optimisation of functions, and the special functions of mathematical physics.

### Derivatives

Numerical differentiation routines for scalar and vector-valued functions.

Crate `lib_jsl_derivatives`: [API reference](../api/lib_jsl_derivatives/index.html) · 2 items

- Scalar derivatives (1D)
- Vector derivatives

### Integration

Numerical integration: quadrature, cubature, quasi-Monte Carlo, ODE solvers, boundary value problems and PDEs.

Crate `lib_jsl_integration`: [API reference](../api/lib_jsl_integration/index.html) · [book chapter](integration/index.md) · 21 items

- Integrator trait (1D)
- Adaptive Gauss-Kronrod (G7-K15)
- Adaptive trapezoid
- Dormand-Prince
- Midpoint
- Midpoint calculation helper
- Monte Carlo
- Runge-Kutta
- Simpson's 1/3 rule
- Trapezoidal
- Complex path integrals
- Gauss-Legendre, Gauss-Hermite and Gauss-Laguerre quadrature
- Tanh-sinh quadrature
- Romberg integration
- Multidimensional cubature
- Quasi-Monte Carlo integration
- Stiff ODE solvers (BDF, Rosenbrock)
- Boundary value problems (shooting, collocation)

#### Partial differential equations

- Finite difference methods
- Finite element methods
- Spectral methods

### Interpolation

Interpolation in one to N dimensions: splines, rational, Chebyshev, Fourier and radial basis functions.

Crate `lib_jsl_interpolation`: [API reference](../api/lib_jsl_interpolation/index.html) · 33 items

- Interpolation trait

#### 1D

- Linear
- Natural cubic spline
- Polynomial
- PCHIP
- Akima
- Cubic basis
- AAA
- Prony's method
- Padé approximants
- Floater-Hormann
- Bulirsch-Stoer rational interpolation
- Fourier 1D
- Chebyshev approximations
- Lagrange interpolant
- Savitzky-Golay

#### 2D

- Nearest
- Bilinear
- Bicubic
- Lanczos
- Cubic basis
- All-phase
- Radial basis functions
- Fourier 2D

#### 3D

- Nearest
- Trilinear
- B-spline
- Cubic Lagrangian
- Fourier 3D
- Shepard's inverse distance weighting

#### ND

- Grid data
- Nearest
- Cubic basis

### Root finding

Root finding: bracketing methods, Brent, Newton, polynomial roots and nonlinear systems.

Crate `lib_jsl_root_finding`: [API reference](../api/lib_jsl_root_finding/index.html) · 9 items

- Bracketing
- Bisection
- Secant method
- False position method
- Ridders' method
- Van Wijngaarden-Dekker-Brent
- Newton-Raphson
- Polynomial roots
- Nonlinear systems (Newton, Broyden)

### Optimization

Optimization algorithms including gradient-based, gradient-free, and stochastic methods.

Crate `lib_jsl_optimization`: [API reference](../api/lib_jsl_optimization/index.html) · 15 items

- Optimization traits
- Golden section search
- Gradient descent
- Nelder-Mead
- Newton-Raphson
- Simulated annealing
- Powell's method
- Linear programming
- Dynamic programming
- Nonlinear least squares and curve fitting (Levenberg-Marquardt)
- Quasi-Newton methods (BFGS, L-BFGS)
- Nonlinear conjugate gradient
- Constrained optimization (SQP, augmented Lagrangian)
- Differential evolution
- Particle swarm optimization

### Special functions

Special functions: gamma, beta, error, Bessel, Airy, elliptic, hypergeometric and zeta functions, orthogonal polynomials.

Crate `lib_jsl_special_functions`: [API reference](../api/lib_jsl_special_functions/index.html) · 27 items

- Gamma
- Beta
- Factorial
- Binomial coefficients
- Incomplete gamma
- Error function
- Exponential integrals
- Incomplete beta
- Bessel functions (integer order)
- Bessel functions (fractional order)
- Bessel functions (spherical)
- Airy functions
- Fresnel integrals
- Cosine integrals
- Sine integrals
- Dawson's integral
- Fermi-Dirac integrals
- Inverse of x log x
- Elliptic integrals
- Jacobian elliptic functions
- Hypergeometric functions
- Statistical functions
- Orthogonal polynomials (Legendre, Hermite, Laguerre, Chebyshev, Jacobi)
- Associated Legendre functions and spherical harmonics
- Riemann zeta
- Lambert W
- Polylogarithm

## Linear algebra and transforms

Dense matrix factorisations and eigenproblems, Fourier transforms, and readable reference implementations of both.

### Linear algebra

Dense linear algebra in pure Rust: LU, Cholesky, eigensolvers, the SVD and least squares for real and complex matrices.

Crate `lib_jsl_linalg`: [API reference](../api/lib_jsl_linalg/index.html) · 7 items

- LU factorisation: solves, inverse, determinant
- Cholesky factorisation and triangular solves
- Hermitian eigenproblem (Householder tridiagonalisation, implicit QL)
- General eigenproblem (Hessenberg reduction, shifted QR, complex Schur form)
- Singular value decomposition (one-sided Jacobi)
- Least squares (pivoted QR, SVD for rank-deficient systems)
- Benchmarks against LAPACK (`ndarray-linalg`)

### Linear algebra reference

Readable textbook linear algebra algorithms for study, plus a benchmark comparing them with lib_jsl_linalg.

Crate `lib_jsl_linalg_reference`: [API reference](../api/lib_jsl_linalg_reference/index.html) · 10 items

- Gaussian elimination without and with partial pivoting
- Doolittle LU: solves, determinant, inverse
- Cholesky-Banachiewicz
- Classical and modified Gram-Schmidt, Householder QR
- Power, inverse and Rayleigh quotient iteration
- Cyclic Jacobi eigenvalue method
- Unshifted and Wilkinson-shifted QR algorithm with deflation
- SVD from the eigenproblem of AᵀA
- Least squares by normal equations and by QR
- Linear algebra race benchmark

### FFTs

Fast Fourier transform engines and related spectral transforms.

Crate `lib_jsl_ffts`: [API reference](../api/lib_jsl_ffts/index.html) · 10 items

- FFT (default engine: picks the algorithm by size)
- Best FFT
- Bluestein FFT
- Fractional FFT
- SIMD FFT (generic, x86, ARM, RISC-V)
- FFT engine trait
- Real-input FFT
- Multidimensional FFT (2D, 3D)
- Discrete cosine and sine transforms
- Size-sweep benchmark

### FFT reference

Readable reference FFT implementations for study, plus a benchmark comparing them with the production engines.

Crate `lib_jsl_ffts_reference`: [API reference](../api/lib_jsl_ffts_reference/index.html) · 4 items

- Simple Cooley-Tukey
- Optimized radix-2
- Optimized split radix
- FFT race benchmark

## Probability, statistics and learning

Random numbers, statistical inference and machine learning.

### Random

Random number generators, quasi-random sequences, distributions and MCMC.

Crate `lib_jsl_random`: [API reference](../api/lib_jsl_random/index.html) · 22 items

- Histogram
- Uniform RNG
- xoshiro256++
- SplitMix64
- Quasi-random sequences (Sobol, Halton)
- Markov chain Monte Carlo (Metropolis-Hastings)

#### Distributions

- Exponential
- Uniform (float)
- Bernoulli
- Geometric
- Weibull
- Poisson
- Gaussian
- Chi-squared
- Student's t
- Gamma
- Beta
- Log-normal
- Cauchy
- Binomial
- Multivariate normal
- Dirichlet

### Statistics

Descriptive statistics, correlation, hypothesis tests, density estimation and resampling.

Crate `lib_jsl_statistics`: [API reference](../api/lib_jsl_statistics/index.html) · 5 items

- Descriptive statistics
- Correlation and covariance
- Hypothesis tests
- Kernel density estimation
- Bootstrap

### Machine learning

Machine learning: regression, classification, clustering, dimensionality reduction, neural networks and boosting.

Crate `lib_jsl_machine_learning`: [API reference](../api/lib_jsl_machine_learning/index.html) · 37 items

#### Foundations

- Model traits (regressor, classifier, clusterer, transformer)
- Metrics
- Synthetic datasets
- Preprocessing
- Kernels

#### Supervised

##### Regression

- Least squares
- General least squares
- Ridge
- Elastic net
- Support vector regression

##### Classification

- k-nearest neighbors
- Phylogenetic trees
- Viterbi algorithm
- Decision trees and random forests
- Support vector machine
- Logistic regression
- Naive Bayes

#### Unsupervised

##### Clustering

- k-means
- k-medoids
- Hierarchical clustering
- DBSCAN
- HDBSCAN
- Gaussian mixture models
- Spectral clustering

##### Reduction

- Principal component analysis
- Kernel PCA
- t-SNE
- UMAP
- Linear discriminant analysis

#### Neural networks

- Multilayer perceptron with backpropagation

#### Ensembles and model selection

- Gradient boosting
- Cross-validation and grid search

#### Sequence models

- HMM training (Baum-Welch)

#### Demonstrations

- Clustering
- Regression
- Classification
- Embedding

## Signals, information and fields

Signal processing, information theory and coding, and electromagnetics.

### DSP

Digital signal processing routines including filters, transforms, resampling, and controls.

Crate `lib_jsl_dsp`: [API reference](../api/lib_jsl_dsp/index.html) · 116 items

#### Continuous

##### Filters

- Bilinear transform
- Bilinear transform (zero-pole-gain form)
- Filter design types and band transforms (Zpk, Tf, lp2lp, lp2hp, lp2bp, lp2bs)
- Bessel
- Butterworth
- Chebyshev type I and II
- Elliptic

##### Spectral

- freqs
- Fourier integrals

#### Discrete

- Stream operators
- Sinc
- Cross-correlation
- FFT cross-correlation
- Convolution
- FFT convolution

##### Windows

- Boxcar
- Triangular
- Parzen
- Bohman
- Blackman
- Nuttall
- Blackman-Harris
- Flat top
- Bartlett
- Bartlett-Hann
- Hamming
- Hann
- Kaiser (including beta and tap-count estimation helpers)
- Kaiser-Bessel derived
- Gaussian
- General Gaussian
- Dolph-Chebyshev
- Cosine
- Exponential
- Tukey
- Taylor
- DPSS (Slepian)
- Lanczos
- General cosine
- General Hamming

##### Controls

- Kalman filter
- PID loop
- State-space model

##### Sources

- Pulse train
- Ramp
- Triangle
- CORDIC (fixed point)
- Sinusoid and quadrature
- Square
- Noise

##### Digital communications

- Constellations
- Barker codes
- Decision feedback equalizer
- Farrow filter
- Mueller timing detector
- OFDM
- FM
- Pulse correlation tracker
- Addressable shift register
- Ping-pong buffer

##### Image

###### Filters

- Sobel
- Gaussian blur
- Histogram equalization
- Sharpening
- Wiener filtering

###### Segmentation

- Otsu's method
- Canny edge detection
- Watershed

###### Feature extraction

- SIFT (scale-invariant feature transform)
- SURF (speeded-up robust features)
- Hough transform
- Blob detection

##### Filters

###### FIR

- Boxcar integrator
- Derivative filters
- firwin
- firwin2
- Overlap-add FIR
- Remez exchange algorithm
- FFT FIR

###### IIR

- Bessel
- Biquad
- Butterworth
- Chebyshev type I
- Chebyshev type II
- Discrete linear filter (direct forms)
- Elliptic
- Comb
- Notch
- Peak

##### Resampling

- Polyphase arbitrary resampling
- Polyphase integer resampling

##### Spectral

- Chirp-z transform
- freqz
- Critically sampled polyphase filter bank
- Oversampled polyphase filter bank
- Non-uniformly sampled spectra
- MUSIC
- Maximum entropy
- Linear predictive coding
- Wavelet transform
- Welch power spectral density
- Short-time Fourier transform and spectrogram
- Hilbert transform and analytic signal
- Goertzel algorithm

##### Transformations

- Channel impairment
- Channel receiver
- Channel transmitter
- Frequency mixer
- Non-linearity transformer

##### Adaptive filters

- LMS and normalized LMS
- RLS

##### Synchronization

- Phase-locked loop
- Costas loop

##### Multirate and radar

- CIC filters
- Half-band filters
- Pulse compression (matched filtering, chirps)

### Information

Information theory and coding: source coding, LFSRs, CRC and SHA-256, error-correcting codes.

Crate `lib_jsl_information`: [API reference](../api/lib_jsl_information/index.html) · 19 items

#### Source coders

- Source coder trait
- Arithmetic
- Huffman
- LZ77
- LZ78
- LZW
- Shannon-Fano
- Gray codes

#### LFSRs

- LFSR
- Maximum length polynomials
- Gold codes

#### Operators

- SHA-256
- CRC-32
- Entropy

#### Forward error correction

- LDPC
- Polar
- Hamming
- Reed-Solomon
- Convolutional

### Electromagnetics

Electromagnetics: fields, antennas, arrays, beamforming, direction finding, transmission lines, propagation, MoM and FDTD.

Crate `lib_jsl_electromagnetics`: [API reference](../api/lib_jsl_electromagnetics/index.html) · 39 items


Design and conventions: `crates/electromagnetics/DESIGN.md`


#### Foundations

- Conventions (units, configurable time convention, media)
- Coordinate systems
- Polarization bases (linear, circular)
- Rigid transforms (translation, rotation)

#### Fields

- Domains and quadrature (surface, volume)
- Field container (scalar/vector, harmonic/time-varying, quantities)
- Coordinate conversion
- Interpolation (nearest, linear, cubic)
- Vector calculus (volume)
- Surface vector calculus (gradient, divergence, curl, Laplace-Beltrami)
- Physics operations (Poynting, power, potentials)

#### Antenna models

- Hertzian dipole
- Magnetic loop
- Half-wave and finite dipoles
- Dish
- Patch antenna

#### Arrays and signal models

- Sensor arrays and responses
- Point source scenes
- Distributed source and noise temperature scenes
- Covariance models

#### Algorithms

- Overlap integral
- CFM beamformer
- Max-SNR beamformer
- LCMV and MVDR beamformers

#### Transmission lines and networks

- Transmission lines (impedance transformation, standing waves)
- S-parameters and network conversions
- Smith chart

#### Propagation and systems

- Friis transmission and path loss
- Multipath and two-ray models
- Atmospheric attenuation
- Radar equation

#### Advanced algorithms

- Spherical wave expansion
- Near-field to far-field transformation
- Direction-of-arrival estimation on sensor arrays
- Method of moments (wire antennas)
- FDTD solver

#### Demonstrations

- Antenna patterns
- Field containers
- Beamforming

## Discrete mathematics and geometry

Number theory, graphs and computational geometry.

### Number theory

Number theory: modular arithmetic, primes, factorisation, the Chinese remainder theorem and polynomials.

Crate `lib_jsl_number_theory`: [API reference](../api/lib_jsl_number_theory/index.html) · 10 items

- Base exponent in modulus
- Greatest common divisor
- Integer square root
- Multiplicative inverse modulo
- Polynomial
- Text coded integer
- Prime sieves
- Primality testing (Miller-Rabin)
- Integer factorization
- Chinese remainder theorem

### Graph

Graph representations and algorithms: shortest paths, spanning trees, flows and orderings.

Crate `lib_jsl_graph`: [API reference](../api/lib_jsl_graph/index.html) · 6 items

- Graph representation helpers (edge lists, adjacency)
- Shortest paths (Dijkstra, Bellman-Ford, A*)
- All-pairs shortest paths (Floyd-Warshall)
- Minimum spanning trees (Kruskal, Prim)
- Maximum flow (Edmonds-Karp, Dinic)
- Topological sort and strongly connected components

### Geometry

Geometry: 1D to 4D types, Delaunay and Voronoi, convex hulls, spatial trees, intersections and mesh processing.

Crate `lib_jsl_geometry`: [API reference](../api/lib_jsl_geometry/index.html) · 48 items

- Common
- Concrete types
- Coordinate systems
- Geometric traits
- Registry
- Tables
- Transformation traits
- Transform support (2D and 3D helpers)

#### 0D

- Points

#### 1D

- Coordinate vector 1D
- Line 1D
- Plane 1D
- Unit vector 1D

#### 2D

- Conversions
- Coordinate vector 2D
- Free vector 2D
- Line 2D
- Mesh 2D
- Plane 2D
- Polygon face 2D
- Triangle 2D
- Unit vector 2D

#### 3D

- Conversions
- Coordinate vector 3D
- Free vector 3D
- Line 3D
- Plane 3D
- Polygon face 3D
- Surface mesh 3D
- Tetrahedron 3D
- Triangle 3D
- Unit vector 3D
- Volume mesh 3D

#### 4D

- Space-time vector
- Christoffel symbols

#### Algorithms

- Delaunay triangulation
- Convex hull
- Slicing
- Rotations
- Quaternions
- Nearest point finding
- Intersection
- k-d trees
- Octrees
- Quadtrees
- Voronoi diagrams
- Mesh smoothing and decimation
- Polygon boolean operations

## Foundations and input/output

Shared types, utilities, sorting and file formats.

### Core

Shared foundation types, aliases, and traits for the lib_jsl workspace.

Crate `lib_jsl_core`: [API reference](../api/lib_jsl_core/index.html) · 2 items

- Data types and traits
- Constants

### Utils

Terminal plots, unit conversion and numerical helpers.

Crate `lib_jsl_utils`: [API reference](../api/lib_jsl_utils/index.html) · 3 items

- Terminal plots (Cartesian and polar)
- Unit conversion
- Machine epsilon

### Sorting and selection

Sorting and selection algorithms.

Crate `lib_jsl_sorting`: [API reference](../api/lib_jsl_sorting/index.html) · 5 items

- Quicksort
- Heapsort
- Select M largest
- Equivalence classes
- Shell's method

### File IO

Readers and writers for WAV, MP3, JPEG, PNG, Netpbm, CSV and NumPy .npy files.

Crate `lib_jsl_file_io`: [API reference](../api/lib_jsl_file_io/index.html) · 7 items

#### Audio

- WAV files
- MP3

#### Image

- JPEG
- JPEG files
- Lossless images (PNG, PPM)

#### Data

- CSV files
- NumPy .npy files

## What the library does not cover

- Sparse matrix formats and sparse solvers as a public facility (the finite element solver assembles and solves a sparse system internally)
- Calling external BLAS or LAPACK, and distributed computing; parallelism is limited to the optional multithreaded matrix multiply (the `threading` feature)
- Hand-written SIMD intrinsics, except in the FFT engines
- Single precision and generic floating-point types: numerical routines compute in `f64` and `Complex<f64>` (`f32` appears only in data containers and file formats)
- General-purpose containers; data structures appear only where an algorithm needs them (k-d trees, union-find, graph representations, geometry tables)
- Complex number primitives (provided by `num-complex`)
- Arbitrary-precision arithmetic

</div>
