# lib_jsl roadmap

Sections follow the workspace crates (`crates/<name>`, published as `lib_jsl_<name>`), in
alphabetical order. `[x]` marks implemented items, `[ ]` planned ones.

# Core (`core`)
[x] Data types and traits
[x] Constants

# Derivatives (`derivatives`)
[x] Scalar derivatives (1D)
[x] Vector derivatives

# DSP (`dsp`)

## Continuous

### Filters
[x] Bilinear transform
[x] Bilinear transform (zero-pole-gain form)
[x] Filter design types and band transforms (Zpk, Tf, lp2lp, lp2hp, lp2bp, lp2bs)
[x] Bessel
[x] Butterworth
[x] Chebyshev type I and II
[x] Elliptic

### Spectral
[x] freqs
[x] Fourier integrals

## Discrete
[x] Stream operators
[x] Sinc
[x] Cross-correlation
[x] FFT cross-correlation
[x] Convolution
[x] FFT convolution

### Windows
[x] Boxcar
[x] Triangular
[x] Parzen
[x] Bohman
[x] Blackman
[x] Nuttall
[x] Blackman-Harris
[x] Flat top
[x] Bartlett
[x] Bartlett-Hann
[x] Hamming
[x] Hann
[x] Kaiser (including beta and tap-count estimation helpers)
[x] Kaiser-Bessel derived
[x] Gaussian
[x] General Gaussian
[x] Dolph-Chebyshev
[x] Cosine
[x] Exponential
[x] Tukey
[x] Taylor
[x] DPSS (Slepian)
[x] Lanczos
[x] General cosine
[x] General Hamming

### Controls
[x] Kalman filter
[x] PID loop
[x] State-space model

### Sources
[x] Pulse train
[x] Ramp
[x] Triangle
[x] CORDIC (fixed point)
[x] Sinusoid and quadrature
[x] Square
[x] Noise

### Digital communications
[x] Constellations
[x] Barker codes
[x] Decision feedback equalizer
[x] Farrow filter
[x] Mueller timing detector
[x] OFDM
[x] FM
[x] Pulse correlation tracker
[x] Addressable shift register
[x] Ping-pong buffer

### Image

#### Filters
[x] Sobel
[x] Gaussian blur
[x] Histogram equalization
[x] Sharpening
[x] Wiener filtering

#### Segmentation
[x] Otsu's method
[x] Canny edge detection
[x] Watershed

#### Feature extraction
[x] SIFT (scale-invariant feature transform)
[x] SURF (speeded-up robust features)
[x] Hough transform
[x] Blob detection

### Filters

#### FIR
[x] Boxcar integrator
[x] Derivative filters
[x] firwin
[x] firwin2
[x] Overlap-add FIR
[x] Remez exchange algorithm
[x] FFT FIR

#### IIR
[x] Bessel
[x] Biquad
[x] Butterworth
[x] Chebyshev type I
[x] Chebyshev type II
[x] Discrete linear filter (direct forms)
[x] Elliptic
[x] Comb
[x] Notch
[x] Peak

### Resampling
[x] Polyphase arbitrary resampling
[x] Polyphase integer resampling

### Spectral
[x] Chirp-z transform
[x] freqz
[x] Critically sampled polyphase filter bank
[x] Oversampled polyphase filter bank
[x] Non-uniformly sampled spectra
[x] MUSIC
[x] Maximum entropy
[x] Linear predictive coding
[x] Wavelet transform
[ ] Welch power spectral density
[ ] Short-time Fourier transform and spectrogram
[ ] Hilbert transform and analytic signal
[ ] Goertzel algorithm

### Transformations
[x] Channel impairment
[x] Channel receiver
[x] Channel transmitter
[x] Frequency mixer
[x] Non-linearity transformer

### Adaptive filters
[ ] LMS and normalized LMS
[ ] RLS

### Synchronization
[ ] Phase-locked loop
[ ] Costas loop

### Multirate and radar
[ ] CIC filters
[ ] Half-band filters
[ ] Pulse compression (matched filtering, chirps)

# Electromagnetics (`electromagnetics`)
Design and conventions: `crates/electromagnetics/DESIGN.md`

## Foundations
[x] Conventions (units, configurable time convention, media)
[x] Coordinate systems
[x] Polarization bases (linear, circular)
[x] Rigid transforms (translation, rotation)

## Fields
[x] Domains and quadrature (surface, volume)
[x] Field container (scalar/vector, harmonic/time-varying, quantities)
[x] Coordinate conversion
[x] Interpolation (nearest, linear, cubic)
[x] Vector calculus (volume)
[x] Surface vector calculus (gradient, divergence, curl, Laplace-Beltrami)
[x] Physics operations (Poynting, power, potentials)

## Antenna models
[x] Hertzian dipole
[x] Magnetic loop
[x] Half-wave and finite dipoles
[x] Dish
[x] Patch antenna

## Arrays and signal models
[x] Sensor arrays and responses
[x] Point source scenes
[x] Distributed source and noise temperature scenes
[x] Covariance models

## Algorithms
[x] Overlap integral
[x] CFM beamformer
[x] Max-SNR beamformer
[x] LCMV and MVDR beamformers

## Transmission lines and networks
[ ] Transmission lines (impedance transformation, standing waves)
[ ] S-parameters and network conversions
[ ] Smith chart

## Propagation and systems
[ ] Friis transmission and path loss
[ ] Multipath and two-ray models
[ ] Atmospheric attenuation
[ ] Radar equation

## Advanced algorithms
[ ] Spherical wave expansion
[ ] Near-field to far-field transformation
[ ] Direction-of-arrival estimation on sensor arrays
[ ] Method of moments (wire antennas)
[ ] FDTD solver

## Demonstrations
[x] Antenna patterns
[x] Field containers
[x] Beamforming

# FFTs (`ffts`)
[x] FFT (default engine: picks the algorithm by size)
[x] Best FFT
[x] Bluestein FFT
[x] Fractional FFT
[x] SIMD FFT (generic, x86, ARM, RISC-V)
[x] FFT engine trait
[x] Real-input FFT
[x] Multidimensional FFT (2D, 3D)
[x] Discrete cosine and sine transforms
[x] Size-sweep benchmark

# FFT reference (`ffts_reference`)
[x] Simple Cooley-Tukey
[x] Optimized radix-2
[x] Optimized split radix
[x] FFT race benchmark

# File IO (`file_io`)

## Audio
[x] WAV files
[x] MP3

## Image
[x] JPEG
[x] JPEG files
[ ] Lossless images (PNG, PPM)

## Data
[ ] CSV files
[ ] NumPy .npy files

# Geometry (`geometry`)
[x] Common
[x] Concrete types
[x] Coordinate systems
[x] Geometric traits
[x] Registry
[x] Tables
[x] Transformation traits
[x] Transform support (2D and 3D helpers)

## 0D
[x] Points

## 1D
[x] Coordinate vector 1D
[x] Line 1D
[x] Plane 1D
[x] Unit vector 1D

## 2D
[x] Conversions
[x] Coordinate vector 2D
[x] Free vector 2D
[x] Line 2D
[x] Mesh 2D
[x] Plane 2D
[x] Polygon face 2D
[x] Triangle 2D
[x] Unit vector 2D

## 3D
[x] Conversions
[x] Coordinate vector 3D
[x] Free vector 3D
[x] Line 3D
[x] Plane 3D
[x] Polygon face 3D
[x] Surface mesh 3D
[x] Tetrahedron 3D
[x] Triangle 3D
[x] Unit vector 3D
[x] Volume mesh 3D

## 4D
[x] Space-time vector
[x] Christoffel symbols

## Algorithms
[x] Delaunay triangulation
[x] Convex hull
[x] Slicing
[x] Rotations
[x] Quaternions
[x] Nearest point finding
[x] Intersection
[x] k-d trees
[x] Octrees
[x] Quadtrees
[ ] Voronoi diagrams
[ ] Mesh smoothing and decimation
[ ] Polygon boolean operations

# Graph (`graph`, planned crate)
[ ] Graph representation helpers (edge lists, adjacency)
[ ] Shortest paths (Dijkstra, Bellman-Ford, A*)
[ ] All-pairs shortest paths (Floyd-Warshall)
[ ] Minimum spanning trees (Kruskal, Prim)
[ ] Maximum flow (Edmonds-Karp, Dinic)
[ ] Topological sort and strongly connected components

# Information (`information`)

## Source coders
[x] Source coder trait
[x] Arithmetic
[x] Huffman
[x] LZ77
[x] LZ78
[x] LZW
[x] Shannon-Fano
[x] Gray codes

## LFSRs
[x] LFSR
[x] Maximum length polynomials
[x] Gold codes

## Operators
[x] SHA-256
[x] CRC-32
[x] Entropy

## Forward error correction
[x] LDPC
[x] Polar
[x] Hamming
[x] Reed-Solomon
[x] Convolutional

# Integration (`integration`)
[x] Integrator trait (1D)
[x] Adaptive Gauss-Kronrod (G7-K15)
[x] Adaptive trapezoid
[x] Dormand-Prince
[x] Midpoint
[x] Midpoint calculation helper
[x] Monte Carlo
[x] Runge-Kutta
[x] Simpson's 1/3 rule
[x] Trapezoidal
[x] Complex path integrals
[x] Gauss-Legendre, Gauss-Hermite and Gauss-Laguerre quadrature
[x] Tanh-sinh quadrature
[x] Romberg integration
[x] Multidimensional cubature
[x] Quasi-Monte Carlo integration
[x] Stiff ODE solvers (BDF, Rosenbrock)
[x] Boundary value problems (shooting, collocation)

## Partial differential equations
[ ] Finite difference methods
[ ] Finite element methods
[ ] Spectral methods

# Interpolation (`interpolation`)
[x] Interpolation trait

## 1D
[x] Linear
[x] Natural cubic spline
[x] Polynomial
[x] PCHIP
[x] Akima
[x] Cubic basis
[x] AAA
[x] Prony's method
[x] Padé approximants
[x] Floater-Hormann
[x] Bulirsch-Stoer rational interpolation
[x] Fourier 1D
[x] Chebyshev approximations
[x] Lagrange interpolant
[x] Savitzky-Golay

## 2D
[x] Nearest
[x] Bilinear
[x] Bicubic
[x] Lanczos
[x] Cubic basis
[x] All-phase
[x] Radial basis functions
[x] Fourier 2D

## 3D
[x] Nearest
[x] Trilinear
[x] B-spline
[x] Cubic Lagrangian
[x] Fourier 3D
[x] Shepard's inverse distance weighting

## ND
[x] Grid data
[x] Nearest
[x] Cubic basis

# Machine learning (`machine_learning`)

## Foundations
[x] Model traits (regressor, classifier, clusterer, transformer)
[x] Metrics
[x] Synthetic datasets
[x] Preprocessing
[x] Kernels

## Supervised

### Regression
[x] Least squares
[x] General least squares
[x] Ridge
[x] Elastic net
[x] Support vector regression

### Classification
[x] k-nearest neighbors
[x] Phylogenetic trees
[x] Viterbi algorithm
[x] Decision trees and random forests
[x] Support vector machine
[x] Logistic regression
[x] Naive Bayes

## Unsupervised

### Clustering
[x] k-means
[x] k-medoids
[x] Hierarchical clustering
[x] DBSCAN
[x] HDBSCAN
[x] Gaussian mixture models
[x] Spectral clustering

### Reduction
[x] Principal component analysis
[x] Kernel PCA
[x] t-SNE
[x] UMAP
[x] Linear discriminant analysis

## Neural networks
[ ] Multilayer perceptron with backpropagation

## Ensembles and model selection
[ ] Gradient boosting
[ ] Cross-validation and grid search

## Sequence models
[ ] HMM training (Baum-Welch)

## Demonstrations
[x] Clustering
[x] Regression
[x] Classification
[x] Embedding

# Number theory (`number_theory`)
[x] Base exponent in modulus
[x] Greatest common divisor
[x] Integer square root
[x] Multiplicative inverse modulo
[x] Polynomial
[x] Text coded integer
[x] Prime sieves
[ ] Primality testing (Miller-Rabin)
[ ] Integer factorization
[ ] Chinese remainder theorem

# Optimization (`optimization`)
[x] Optimization traits
[x] Golden section search
[x] Gradient descent
[x] Nelder-Mead
[x] Newton-Raphson
[x] Simulated annealing
[x] Powell's method
[x] Linear programming
[x] Dynamic programming
[x] Nonlinear least squares and curve fitting (Levenberg-Marquardt)
[x] Quasi-Newton methods (BFGS, L-BFGS)
[x] Nonlinear conjugate gradient
[x] Constrained optimization (SQP, augmented Lagrangian)
[x] Differential evolution
[x] Particle swarm optimization

# Random (`random`)
[x] Histogram
[x] Uniform RNG
[x] xoshiro256++
[x] SplitMix64
[x] Quasi-random sequences (Sobol, Halton)
[x] Markov chain Monte Carlo (Metropolis-Hastings)

## Distributions
[x] Exponential
[x] Uniform (float)
[x] Bernoulli
[x] Geometric
[x] Weibull
[x] Poisson
[x] Gaussian
[x] Chi-squared
[x] Student's t
[x] Gamma
[x] Beta
[x] Log-normal
[x] Cauchy
[x] Binomial
[x] Multivariate normal
[x] Dirichlet

# Root finding (`root_finding`)
[x] Bracketing
[x] Bisection
[x] Secant method
[x] False position method
[x] Ridders' method
[x] Van Wijngaarden-Dekker-Brent
[x] Newton-Raphson
[x] Polynomial roots
[x] Nonlinear systems (Newton, Broyden)

# Sorting and selection (`sorting`)
[x] Quicksort
[x] Heapsort
[x] Select M largest
[x] Equivalence classes
[x] Shell's method

# Special functions (`special_functions`)
[x] Gamma
[x] Beta
[x] Factorial
[x] Binomial coefficients
[x] Incomplete gamma
[x] Error function
[x] Exponential integrals
[x] Incomplete beta
[x] Bessel functions (integer order)
[x] Bessel functions (fractional order)
[x] Bessel functions (spherical)
[x] Airy functions
[x] Fresnel integrals
[x] Cosine integrals
[x] Sine integrals
[x] Dawson's integral
[x] Fermi-Dirac integrals
[x] Inverse of x log x
[x] Elliptic integrals
[x] Jacobian elliptic functions
[x] Hypergeometric functions
[x] Statistical functions
[x] Orthogonal polynomials (Legendre, Hermite, Laguerre, Chebyshev, Jacobi)
[x] Associated Legendre functions and spherical harmonics
[x] Riemann zeta
[x] Lambert W
[x] Polylogarithm

# Statistics (`statistics`, planned crate)
[ ] Descriptive statistics
[ ] Correlation and covariance
[ ] Hypothesis tests
[ ] Kernel density estimation
[ ] Bootstrap

# Utils (`utils`)
[x] Terminal plots (Cartesian and polar)
[x] Unit conversion
[x] Machine epsilon

# Out of scope
* Linear algebra primitives (dense linear algebra comes from `ndarray-linalg`)
* Intrinsics or SIMD optimizations, except in the FFT engines
* General-purpose containers or data structures
* Complex number primitives (from `num`)
* Arbitrary precision primitives
