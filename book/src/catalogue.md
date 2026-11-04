# Catalogue of algorithms

This catalogue lists everything the library implements, grouped by subject. Use it to find
where an algorithm lives, to see the neighbours of the one you are studying, or to plan a path
through the book. It describes the current release and is updated whenever an algorithm is
added; the [changelog](https://github.com/jwkunz/lib_jsl/blob/main/CHANGELOG.md) records what
changed between releases, and planned work is tracked in
[GitHub issues](https://github.com/jwkunz/lib_jsl/issues).

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

Crate `lib_jsl_derivatives`: [API reference](../api/lib_jsl_derivatives/index.html) · [book chapter](derivatives/index.md) · 2 items

- [Scalar derivatives (1D)](derivatives/finite-differences.md)
- [Vector derivatives](derivatives/finite-differences.md#gradients-hessians-and-jacobians)

### Integration

Numerical integration: quadrature, cubature, quasi-Monte Carlo, ODE solvers, boundary value problems and PDEs.

Crate `lib_jsl_integration`: [API reference](../api/lib_jsl_integration/index.html) · [book chapter](integration/index.md) · 21 items

- [Integrator trait (1D)](integration/index.md#in-the-library)
- [Adaptive Gauss-Kronrod (G7-K15)](integration/gauss-kronrod.md)
- [Adaptive trapezoid](integration/newton-cotes.md#the-adaptive-trapezoid-rule)
- [Dormand-Prince](integration/runge-kutta.md#error-estimation-with-embedded-pairs)
- [Midpoint](integration/newton-cotes.md#the-midpoint-rule)
- [Sample-point helpers (grid points, panel midpoints)](integration/newton-cotes.md#implementation)
- [Monte Carlo](integration/monte-carlo.md)
- [Runge-Kutta](integration/runge-kutta.md)
- [Simpson's 1/3 rule](integration/newton-cotes.md#simpsons-rules)
- [Trapezoidal](integration/newton-cotes.md#the-trapezoid-rule)
- [Complex path integrals](integration/contour-integration.md)
- [Gauss-Legendre, Gauss-Hermite and Gauss-Laguerre quadrature](integration/gaussian-quadrature.md)
- [Tanh-sinh quadrature](integration/tanh-sinh.md)
- [Romberg integration](integration/romberg.md)
- [Multidimensional cubature](integration/cubature.md)
- [Quasi-Monte Carlo integration](integration/monte-carlo.md#randomised-quasi-monte-carlo)
- [Stiff ODE solvers (BDF, Rosenbrock)](integration/stiff-ode.md)
- [Boundary value problems (shooting, collocation)](integration/boundary-value-problems.md)

#### Partial differential equations

- [Finite difference methods](integration/finite-differences.md)
- [Finite element methods](integration/finite-elements.md)
- [Spectral methods](integration/spectral-methods.md)

### Interpolation

Interpolation in one to N dimensions: splines, rational, Chebyshev, Fourier and radial basis functions.

Crate `lib_jsl_interpolation`: [API reference](../api/lib_jsl_interpolation/index.html) · [book chapter](interpolation/index.md) · 33 items

- [Interpolation trait](interpolation/index.md#in-the-library)

#### 1D

- [Linear](interpolation/splines.md#linear-interpolation)
- [Natural cubic spline](interpolation/splines.md#cubic-splines)
- [Polynomial](interpolation/polynomial.md#newtons-divided-differences)
- [PCHIP](interpolation/splines.md#cubic-hermite-interpolation)
- [Akima](interpolation/splines.md#cubic-hermite-interpolation)
- [Cubic basis (Catmull-Rom)](interpolation/splines.md#cubic-hermite-interpolation)
- [AAA](interpolation/rational.md#the-aaa-algorithm)
- [Prony's method](interpolation/trigonometric.md#pronys-method)
- [Padé approximants](interpolation/rational.md#padé-approximants)
- [Floater-Hormann](interpolation/rational.md#floater-hormann-interpolation)
- [Bulirsch-Stoer rational interpolation](interpolation/rational.md#bulirsch-stoer-rational-interpolation)
- [Fourier 1D](interpolation/trigonometric.md#trigonometric-interpolation)
- [Chebyshev approximations](interpolation/polynomial.md#chebyshev-series)
- [Lagrange interpolant](interpolation/polynomial.md#the-barycentric-formula)
- [Savitzky-Golay](interpolation/savitzky-golay.md)

#### 2D

- [Nearest](interpolation/grids.md#tensor-products)
- [Bilinear](interpolation/grids.md#tensor-products)
- [Bicubic](interpolation/grids.md#cubic-methods)
- [Lanczos](interpolation/grids.md#interpolation-as-convolution)
- [Cubic basis](interpolation/grids.md#cubic-methods)
- [All-phase](interpolation/grids.md#interpolation-as-convolution)
- [Radial basis functions](interpolation/scattered.md#radial-basis-functions)
- [Fourier 2D](interpolation/grids.md#accuracy)

#### 3D

- [Nearest](interpolation/grids.md#tensor-products)
- [Trilinear](interpolation/grids.md#tensor-products)
- [B-spline](interpolation/grids.md#cubic-methods)
- [Cubic Lagrangian](interpolation/grids.md#cubic-methods)
- [Fourier 3D](interpolation/grids.md#accuracy)
- [Shepard's inverse distance weighting](interpolation/scattered.md#shepards-inverse-distance-weighting)

#### ND

- [Grid data](interpolation/grids.md#tensor-products)
- [Nearest](interpolation/grids.md#tensor-products)
- [Cubic basis](interpolation/grids.md#cubic-methods)

### Root finding

Root finding: bracketing methods, Brent, Newton, polynomial roots and nonlinear systems.

Crate `lib_jsl_root_finding`: [API reference](../api/lib_jsl_root_finding/index.html) · [book chapter](root_finding/index.md) · 9 items

- [Bracketing](root_finding/bracketing.md#finding-a-bracket)
- [Bisection](root_finding/bracketing.md#bisection)
- [Secant method](root_finding/newton.md#the-secant-method)
- [False position method](root_finding/bracketing.md#false-position-and-the-illinois-modification)
- [Ridders' method](root_finding/bracketing.md#ridders-method)
- [Van Wijngaarden-Dekker-Brent](root_finding/bracketing.md#brents-method)
- [Newton-Raphson](root_finding/newton.md#newtons-method)
- [Polynomial roots](root_finding/polynomial-roots.md)
- [Nonlinear systems (Newton, Broyden)](root_finding/nonlinear-systems.md)

### Optimization

Optimization algorithms including gradient-based, gradient-free, and stochastic methods.

Crate `lib_jsl_optimization`: [API reference](../api/lib_jsl_optimization/index.html) · [book chapter](optimization/index.md) · 15 items

- [Optimization traits](optimization/index.md#in-the-library)
- [Golden section search](optimization/one-dimensional.md#golden-section-search)
- [Gradient descent](optimization/gradient-methods.md#gradient-descent)
- [Nelder-Mead](optimization/direct-search.md#the-nelder-mead-simplex-method)
- [Newton-Raphson](optimization/newton-quasi-newton.md#newtons-method)
- [Simulated annealing](optimization/stochastic.md#simulated-annealing)
- [Powell's method](optimization/direct-search.md#powells-method)
- [Linear programming](optimization/linear-programming.md)
- [Dynamic programming](optimization/dynamic-programming.md)
- [Nonlinear least squares and curve fitting (Levenberg-Marquardt)](optimization/least-squares.md#the-levenberg-marquardt-method)
- [Quasi-Newton methods (BFGS, L-BFGS)](optimization/newton-quasi-newton.md#bfgs)
- [Nonlinear conjugate gradient](optimization/gradient-methods.md#nonlinear-conjugate-gradients)
- [Constrained optimization (SQP, augmented Lagrangian)](optimization/constrained.md)
- [Differential evolution](optimization/stochastic.md#differential-evolution)
- [Particle swarm optimization](optimization/stochastic.md#particle-swarm-optimisation)

### Special functions

Special functions: gamma, beta, error, Bessel, Airy, elliptic, hypergeometric and zeta functions, orthogonal polynomials.

Crate `lib_jsl_special_functions`: [API reference](../api/lib_jsl_special_functions/index.html) · [book chapter](special_functions/index.md) · 27 items

- [Gamma](special_functions/gamma.md#the-gamma-function)
- [Beta](special_functions/gamma.md#the-beta-function)
- [Factorial](special_functions/gamma.md#factorials-and-binomial-coefficients)
- [Binomial coefficients](special_functions/gamma.md#factorials-and-binomial-coefficients)
- [Incomplete gamma](special_functions/incomplete.md#the-incomplete-gamma-function)
- [Error function](special_functions/incomplete.md#the-error-function)
- [Exponential integrals](special_functions/exponential-integrals.md#exponential-integrals)
- [Incomplete beta](special_functions/incomplete.md#the-incomplete-beta-function)
- [Bessel functions (integer order)](special_functions/bessel.md#bessel-functions-of-integer-order)
- [Bessel functions (fractional order)](special_functions/bessel.md#bessel-functions-of-real-order)
- [Bessel functions (spherical)](special_functions/bessel.md#spherical-bessel-functions)
- [Airy functions](special_functions/bessel.md#airy-functions)
- [Fresnel integrals](special_functions/exponential-integrals.md#fresnel-integrals)
- [Cosine integrals](special_functions/exponential-integrals.md#sine-and-cosine-integrals)
- [Sine integrals](special_functions/exponential-integrals.md#sine-and-cosine-integrals)
- [Dawson's integral](special_functions/incomplete.md#dawsons-integral)
- [Fermi-Dirac integrals](special_functions/zeta.md#fermi-dirac-integrals)
- [Inverse of x log x](special_functions/lambert-w.md#the-inverse-of-x-ln-x)
- [Elliptic integrals](special_functions/elliptic.md#elliptic-integrals)
- [Jacobian elliptic functions](special_functions/elliptic.md#jacobian-elliptic-functions)
- [Hypergeometric functions](special_functions/hypergeometric.md)
- [Statistical functions](special_functions/statistical.md)
- [Orthogonal polynomials (Legendre, Hermite, Laguerre, Chebyshev, Jacobi)](special_functions/orthogonal-polynomials.md#classical-orthogonal-polynomials)
- [Associated Legendre functions and spherical harmonics](special_functions/orthogonal-polynomials.md#associated-legendre-functions-and-spherical-harmonics)
- [Riemann zeta](special_functions/zeta.md#the-riemann-zeta-function)
- [Lambert W](special_functions/lambert-w.md#the-lambert-w-function)
- [Polylogarithm](special_functions/zeta.md#the-polylogarithm)

## Linear algebra and transforms

Dense matrix factorisations and eigenproblems, Fourier transforms, and readable reference implementations of both.

### Linear algebra

Dense linear algebra in pure Rust: LU, Cholesky, eigensolvers, the SVD and least squares for real and complex matrices.

Crate `lib_jsl_linalg`: [API reference](../api/lib_jsl_linalg/index.html) · [book chapter](linear_algebra/index.md) · 7 items

- [LU factorisation: solves, inverse, determinant](linear_algebra/lu.md#lu-factorisation)
- [Cholesky factorisation and triangular solves](linear_algebra/cholesky.md)
- [Hermitian eigenproblem (Householder tridiagonalisation, implicit QL)](linear_algebra/eigensolvers.md#hermitian-matrices-tridiagonalisation-and-ql)
- [General eigenproblem (Hessenberg reduction, shifted QR, complex Schur form)](linear_algebra/eigensolvers.md#general-matrices-hessenberg-reduction-and-franciss-algorithm)
- [Singular value decomposition (Golub-Kahan; one-sided Jacobi)](linear_algebra/svd.md)
- [Least squares (pivoted QR, SVD for rank-deficient systems)](linear_algebra/least-squares.md)
- [Benchmarks against LAPACK (`ndarray-linalg`)](linear_algebra/index.md#the-production-crate)

### Linear algebra reference

Readable textbook linear algebra algorithms for study, plus a benchmark comparing them with lib_jsl_linalg.

Crate `lib_jsl_linalg_reference`: [API reference](../api/lib_jsl_linalg_reference/index.html) · [book chapter](linear_algebra/index.md#the-reference-crate) · 10 items

- [Gaussian elimination without and with partial pivoting](linear_algebra/lu.md#gaussian-elimination-and-pivoting)
- [Doolittle LU: solves, determinant, inverse](linear_algebra/lu.md#lu-factorisation)
- [Cholesky-Banachiewicz](linear_algebra/cholesky.md#the-factorisation)
- [Classical and modified Gram-Schmidt, Householder QR](linear_algebra/qr.md)
- [Power, inverse and Rayleigh quotient iteration](linear_algebra/eigen-iterations.md#power-iteration-inverse-iteration-and-the-rayleigh-quotient)
- [Cyclic Jacobi eigenvalue method](linear_algebra/eigen-iterations.md#the-jacobi-method)
- [Unshifted and Wilkinson-shifted QR algorithm with deflation](linear_algebra/eigen-iterations.md#the-qr-algorithm)
- [SVD from the eigenproblem of AᵀA](linear_algebra/svd.md#the-decomposition)
- [Least squares by normal equations and by QR](linear_algebra/least-squares.md)
- [Linear algebra race benchmark](linear_algebra/index.md#the-reference-crate)

### FFTs

Fast Fourier transform engines and related spectral transforms.

Crate `lib_jsl_ffts`: [API reference](../api/lib_jsl_ffts/index.html) · [book chapter](fft/index.md) · 10 items

- [FFT (default engine: picks the algorithm by size)](fft/fft.md#the-production-engines)
- [Best FFT](fft/fft.md#the-production-engines)
- [Bluestein FFT](fft/bluestein-fractional.md#bluesteins-algorithm)
- [Fractional FFT](fft/bluestein-fractional.md#the-fractional-fourier-transform)
- [SIMD FFT (generic, x86, ARM, RISC-V)](fft/fft.md#the-production-engines)
- [FFT engine trait](fft/index.md#two-crates)
- [Real-input FFT](fft/real-multidimensional.md#the-real-fft-by-half-length-packing)
- [Multidimensional FFT (2D, 3D)](fft/real-multidimensional.md#multidimensional-transforms)
- [Discrete cosine and sine transforms](fft/dct.md)
- [Size-sweep benchmark](fft/index.md#two-crates)

### FFT reference

Readable reference FFT implementations for study, plus a benchmark comparing them with the production engines.

Crate `lib_jsl_ffts_reference`: [API reference](../api/lib_jsl_ffts_reference/index.html) · [book chapter](fft/index.md#two-crates) · 4 items

- [Simple Cooley-Tukey](fft/fft.md#the-radix-2-cooley-tukey-algorithm)
- [Optimized radix-2](fft/fft.md#twiddle-factors-and-accuracy)
- [Optimized split radix](fft/fft.md#split-radix)
- [FFT race benchmark](fft/index.md#two-crates)

## Probability, statistics and learning

Random numbers, statistical inference and machine learning.

### Random

Random number generators, quasi-random sequences, distributions and MCMC.

Crate `lib_jsl_random`: [API reference](../api/lib_jsl_random/index.html) · [book chapter](random/index.md) · 22 items

- [Histogram](random/generators.md#histograms)
- [Uniform RNG](random/generators.md#xoshiro256)
- [xoshiro256++](random/generators.md#xoshiro256)
- [SplitMix64](random/generators.md#seeding-with-splitmix64)
- Quasi-random sequences ([Sobol](random/quasi-random.md#the-sobol-sequence), [Halton](random/quasi-random.md#the-radical-inverse-and-the-halton-sequence))
- Markov chain Monte Carlo ([Metropolis-Hastings](random/mcmc.md#the-metropolis-hastings-algorithm))

#### Distributions

- [Exponential](random/distributions.md#inversion)
- [Uniform (float)](random/generators.md#xoshiro256)
- [Bernoulli](random/distributions.md#inversion)
- [Geometric](random/distributions.md#inversion)
- [Weibull](random/distributions.md#inversion)
- [Poisson](random/distributions.md#discrete-distributions-poisson-and-binomial)
- [Gaussian](random/distributions.md#the-normal-distribution-box-muller)
- [Chi-squared](random/distributions.md#the-normal-distribution-box-muller)
- [Student's t](random/distributions.md#the-normal-distribution-box-muller)
- [Gamma](random/distributions.md#rejection-the-gamma-distribution)
- [Beta](random/distributions.md#rejection-the-gamma-distribution)
- [Log-normal](random/distributions.md#the-normal-distribution-box-muller)
- [Cauchy](random/distributions.md#inversion)
- [Binomial](random/distributions.md#discrete-distributions-poisson-and-binomial)
- [Multivariate normal](random/distributions.md#multivariate-normal)
- [Dirichlet](random/distributions.md#rejection-the-gamma-distribution)

### Statistics

Descriptive statistics, correlation, hypothesis tests, density estimation and resampling.

Crate `lib_jsl_statistics`: [API reference](../api/lib_jsl_statistics/index.html) · [book chapter](statistics/index.md) · 5 items

- [Descriptive statistics](statistics/descriptive.md)
- Correlation and covariance ([Pearson](statistics/correlation.md#pearsons-correlation), [Spearman](statistics/correlation.md#spearmans-rank-correlation), [Kendall](statistics/correlation.md#kendalls-tau))
- Hypothesis tests ([t tests](statistics/hypothesis-tests.md#students-and-welchs-t-tests), [chi-square](statistics/hypothesis-tests.md#chi-square-tests-for-counts), [ANOVA](statistics/hypothesis-tests.md#one-way-analysis-of-variance), [Kolmogorov-Smirnov](statistics/hypothesis-tests.md#kolmogorov-smirnov-tests), [Mann-Whitney](statistics/hypothesis-tests.md#the-mann-whitney-u-test))
- [Kernel density estimation](statistics/kernel-density.md)
- Bootstrap ([percentile](statistics/bootstrap.md#percentile-intervals), [BCa](statistics/bootstrap.md#the-bca-interval))

### Machine learning

Machine learning: regression, classification, clustering, dimensionality reduction, neural networks and boosting.

Crate `lib_jsl_machine_learning`: [API reference](../api/lib_jsl_machine_learning/index.html) · book chapters: [regression and classification](machine_learning/index.md), [clustering and dimensionality reduction](unsupervised/index.md), [neural networks, ensembles, model selection and sequence models](learning/index.md) · 37 items

#### Foundations

- [Model traits (regressor, classifier, clusterer, transformer)](machine_learning/foundations.md#the-model-traits)
- [Metrics](machine_learning/foundations.md#metrics)
- [Synthetic datasets](machine_learning/foundations.md#synthetic-data-sets)
- Preprocessing ([scaling](machine_learning/foundations.md#scaling), [train/test split](machine_learning/foundations.md#training-and-test-data))
- [Kernels](machine_learning/support-vector-machines.md#the-dual-problem-and-kernels)

#### Supervised

##### Regression

- [Least squares](machine_learning/linear-regression.md#least-squares)
- [General least squares](machine_learning/linear-regression.md#weighted-and-generalised-least-squares)
- [Ridge](machine_learning/linear-regression.md#ridge-regression)
- [Elastic net](machine_learning/linear-regression.md#the-lasso-and-the-elastic-net) ([coordinate descent](machine_learning/linear-regression.md#coordinate-descent))
- [Support vector regression](machine_learning/support-vector-machines.md#support-vector-regression)

##### Classification

- [k-nearest neighbors](machine_learning/nearest-neighbours.md)
- Phylogenetic trees ([UPGMA](unsupervised/hierarchical.md#upgma-and-the-molecular-clock), [neighbour joining](unsupervised/hierarchical.md#neighbour-joining))
- [Viterbi algorithm](learning/hidden-markov-models.md#the-viterbi-algorithm)
- [Decision trees](machine_learning/decision-trees.md#growing-a-tree-by-gini-impurity) and [random forests](machine_learning/decision-trees.md#bagging-and-random-features)
- [Support vector machine](machine_learning/support-vector-machines.md#the-maximal-margin-hyperplane) ([SMO](machine_learning/support-vector-machines.md#sequential-minimal-optimisation))
- [Logistic regression](machine_learning/logistic-regression.md)
- Naive Bayes ([Gaussian](machine_learning/naive-bayes.md#gaussian-naive-bayes), [multinomial](machine_learning/naive-bayes.md#multinomial-naive-bayes-for-counts))

#### Unsupervised

##### Clustering

- [k-means](unsupervised/k-means.md#lloyds-algorithm) ([k-means++](unsupervised/k-means.md#k-means-seeding-and-restarts))
- [k-medoids](unsupervised/k-means.md#k-medoids)
- [Hierarchical clustering](unsupervised/hierarchical.md#linkage-criteria)
- [DBSCAN](unsupervised/density.md#dbscan-core-border-and-noise-points)
- [HDBSCAN](unsupervised/density.md#hdbscan-every-density-at-once)
- [Gaussian mixture models](unsupervised/gaussian-mixtures.md#expectation-maximisation)
- [Spectral clustering](unsupervised/spectral.md#normalised-cuts-and-the-embedding)

##### Reduction

- [Principal component analysis](unsupervised/pca.md#directions-of-greatest-variance)
- [Kernel PCA](unsupervised/pca.md#kernel-pca)
- [t-SNE](unsupervised/embeddings.md#t-sne)
- [UMAP](unsupervised/embeddings.md#umap)
- [Linear discriminant analysis](unsupervised/lda.md#fishers-criterion)

#### Neural networks

- [Multilayer perceptron with backpropagation](learning/neural-networks.md#backpropagation)

#### Ensembles and model selection

- [Gradient boosting](learning/gradient-boosting.md#functional-gradient-descent)
- [Cross-validation](learning/model-selection.md#k-fold-cross-validation) and [grid search](learning/model-selection.md#grid-search)

#### Sequence models

- [HMM training (Baum-Welch)](learning/hidden-markov-models.md#baum-welch)

#### Demonstrations

- [Clustering](learning/index.md#demonstration-programs)
- [Regression](learning/index.md#demonstration-programs)
- [Classification](learning/index.md#demonstration-programs)
- [Embedding](learning/index.md#demonstration-programs)

## Signals, information and fields

Signal processing, information theory and coding, and electromagnetics.

### DSP

Digital signal processing routines including filters, transforms, resampling, and controls.

Crate `lib_jsl_dsp`: [API reference](../api/lib_jsl_dsp/index.html) · book chapters: [filter design, FIR and IIR filters, adaptive filters](filters/index.md), [spectral estimation, windows, resampling and multirate processing](spectral/index.md), [synchronisation, communications, radar and controls](comms/index.md) · 116 items

#### Continuous

##### Filters

- [Bilinear transform](filters/iir-design.md#the-bilinear-transform)
- [Bilinear transform (zero-pole-gain form)](filters/iir-design.md#the-bilinear-transform)
- [Filter design types and band transforms (Zpk, Tf, lp2lp, lp2hp, lp2bp, lp2bs)](filters/analog-prototypes.md#frequency-transformations)
- [Bessel](filters/analog-prototypes.md#where-the-poles-are)
- [Butterworth](filters/analog-prototypes.md#where-the-poles-are)
- [Chebyshev type I and II](filters/analog-prototypes.md#where-the-poles-are)
- [Elliptic](filters/analog-prototypes.md#where-the-poles-are)

##### Spectral

- [freqs](filters/analog-prototypes.md#the-magnitude-squared-function)
- [Fourier integrals](spectral/chosen-frequencies.md#fourier-integrals)

#### Discrete

- [Stream operators](filters/convolution.md#implementation)
- [Sinc](filters/fir-design.md#the-window-method)
- [Cross-correlation](filters/convolution.md#cross-correlation-and-delay-estimation)
- [FFT cross-correlation](filters/convolution.md#cross-correlation-and-delay-estimation)
- [Convolution](filters/convolution.md#output-lengths-and-modes)
- [FFT convolution](filters/convolution.md#fast-convolution)

##### Windows

- [Boxcar](spectral/windows.md#families)
- [Triangular](spectral/windows.md#families)
- [Parzen](spectral/windows.md#families)
- [Bohman](spectral/windows.md#families)
- [Blackman](spectral/windows.md#families)
- [Nuttall](spectral/windows.md#families)
- [Blackman-Harris](spectral/windows.md#families)
- [Flat top](spectral/windows.md#families)
- [Bartlett](spectral/windows.md#families)
- [Bartlett-Hann](spectral/windows.md#families)
- [Hamming](spectral/windows.md#families)
- [Hann](spectral/windows.md#families)
- [Kaiser (including beta and tap-count estimation helpers)](spectral/windows.md#families)
- [Kaiser-Bessel derived](spectral/windows.md#families)
- [Gaussian](spectral/windows.md#families)
- [General Gaussian](spectral/windows.md#families)
- [Dolph-Chebyshev](spectral/windows.md#the-dolph-chebyshev-and-dpss-windows)
- [Cosine](spectral/windows.md#families)
- [Exponential](spectral/windows.md#families)
- [Tukey](spectral/windows.md#families)
- [Taylor](spectral/windows.md#families)
- [DPSS (Slepian)](spectral/windows.md#the-dolph-chebyshev-and-dpss-windows)
- [Lanczos](spectral/windows.md#families)
- [General cosine](spectral/windows.md#families)
- [General Hamming](spectral/windows.md#families)

##### Controls

- [Kalman filter](comms/controls.md#the-kalman-filter)
- [PID loop](comms/controls.md#pid-control)
- [State-space model](comms/controls.md#state-space-models)

##### Sources

- [Pulse train](comms/sources.md#waveforms-and-noise)
- [Ramp](comms/sources.md#waveforms-and-noise)
- [Triangle](comms/sources.md#waveforms-and-noise)
- [CORDIC (fixed point)](comms/sources.md#cordic)
- [Sinusoid and quadrature](comms/sources.md#phase-accumulation)
- [Square](comms/sources.md#waveforms-and-noise)
- [Noise](comms/sources.md#waveforms-and-noise)

##### Digital communications

- [Constellations](comms/modulation.md#constellations-and-gray-labelling)
- [Barker codes](comms/radar.md#barker-codes)
- [Decision feedback equalizer](comms/equalisation.md#the-decision-feedback-equalizer)
- [Farrow filter](comms/synchronisation.md#fractional-delay-the-farrow-filter)
- [Mueller timing detector](comms/synchronisation.md#timing-recovery)
- [OFDM](comms/modulation.md#ofdm)
- [FM](comms/modulation.md#frequency-modulation)
- [Pulse correlation tracker](comms/radar.md#finding-and-tracking-a-pulse)
- [Addressable shift register](comms/index.md#addressable-shift-register)
- [Ping-pong buffer](comms/index.md#ping-pong-buffer)

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

- [Boxcar integrator](filters/fir-design.md#differentiators-and-integrators)
- [Derivative filters](filters/fir-design.md#differentiators-and-integrators)
- [firwin](filters/fir-design.md#the-window-method)
- [firwin2](filters/fir-design.md#frequency-sampling)
- [Overlap-add FIR](filters/convolution.md#streams-overlap-add-and-overlap-save)
- [Remez exchange algorithm](filters/fir-design.md#parks-mcclellan-equiripple-design)
- [FFT FIR](filters/convolution.md#streams-overlap-add-and-overlap-save)

###### IIR

- [Bessel](filters/iir-design.md#the-design-recipe)
- [Biquad](filters/iir-structures.md#coefficient-sensitivity)
- [Butterworth](filters/iir-design.md#the-design-recipe)
- [Chebyshev type I](filters/iir-design.md#the-design-recipe)
- [Chebyshev type II](filters/iir-design.md#the-design-recipe)
- [Discrete linear filter (direct forms)](filters/iir-structures.md#direct-forms)
- [Elliptic](filters/iir-design.md#the-design-recipe)
- [Comb](filters/iir-structures.md#comb-filters)
- [Notch](filters/iir-structures.md#notch-and-peak-filters)
- [Peak](filters/iir-structures.md#notch-and-peak-filters)

##### Resampling

- [Polyphase arbitrary resampling](spectral/resampling.md#arbitrary-ratios)
- [Polyphase integer resampling](spectral/resampling.md#polyphase-resampling)

##### Spectral

- [Chirp-z transform](spectral/chosen-frequencies.md#the-chirp-z-transform)
- [freqz](filters/iir-design.md#evaluating-the-response)
- [Critically sampled polyphase filter bank](spectral/filter-banks.md#the-polyphase-analysis-bank)
- [Oversampled polyphase filter bank](spectral/filter-banks.md#critical-sampling-and-oversampling)
- [Non-uniformly sampled spectra](spectral/power-spectra.md#unevenly-sampled-data-lomb-scargle)
- [MUSIC](spectral/parametric.md#music)
- [Maximum entropy](spectral/parametric.md#burgs-method)
- [Linear predictive coding](spectral/parametric.md#linear-prediction-and-the-yule-walker-equations)
- [Wavelet transform](spectral/wavelets.md)
- [Welch power spectral density](spectral/power-spectra.md#variance-welch-averaging)
- [Short-time Fourier transform and spectrogram](spectral/power-spectra.md#the-short-time-fourier-transform-and-the-spectrogram)
- [Hilbert transform and analytic signal](spectral/hilbert.md)
- [Goertzel algorithm](spectral/chosen-frequencies.md#goertzels-recursion)

##### Transformations

- [Channel impairment](comms/equalisation.md#a-channel-in-software)
- [Channel receiver](comms/equalisation.md#a-channel-in-software)
- [Channel transmitter](comms/equalisation.md#a-channel-in-software)
- [Frequency mixer](comms/sources.md#frequency-mixing)
- [Non-linearity transformer](comms/equalisation.md#non-linearities)

##### Adaptive filters

- [LMS and normalized LMS](filters/adaptive.md#lms-stochastic-gradient-descent)
- [RLS](filters/adaptive.md#rls-exact-recursive-least-squares)

##### Synchronization

- [Phase-locked loop](comms/synchronisation.md#the-loop)
- [Costas loop](comms/synchronisation.md#costas-loops)

##### Multirate and radar

- [CIC filters](spectral/resampling.md#cic-filters)
- [Half-band filters](spectral/resampling.md#half-band-filters)
- [Pulse compression (matched filtering, chirps)](comms/radar.md#chirps)

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
