# Out-of-scope
*  Linear algebra primitives
*  Intrinsics or SIMD optimizations
*  Containers or data-structures
*  Complex number primitives
*  Arbitrary precision primitives

# Core
[x] Data types and traits
[x] Constants

# Derivatives
[x] Scalar Derivatives 1D
[x] Vector Derivatives


# DSP

## Continuous

### Filters
[x] Bi-linear transform
[x] Bessel
[x] Elliptic
[x] Butterworth
[x] Chebyshev 1 + 2
[x] Bi-linear transform (zero-pole-gain form)
[x] Filter design types and band transforms (Zpk, Tf, lp2lp, lp2hp, lp2bp, lp2bs)

### Spectral
[x] freqs
[x] Fourier integrals

## Discrete
[x] Stream Operators
[x] Sinc
[x] Cross Correlation
[ ] FFT cross correlation
[x] Convolution
[ ] FFT convolution

### Windows
[x] Boxcar
[x] Triang
[x] Parzen
[x] Bohman
[x] Blackman
[x] Nuttall
[x] BlackmanHarris
[x] Flattop
[x] Bartlett
[x] Barthann
[x] Hamming
[x] Hann
[x] Kaiser (including beta and tap-count estimation helpers)
[x] KaiserBesselDerived
[x] Gaussian
[x] GeneralGaussian
[x] Chebwin
[x] Cosine
[x] Exponential
[x] Tukey
[x] Taylor
[x] Dpss 
[x] Lanczos
[x] General cosine
[x] General Hamming

### Controls
[x] Kalman Filter
[x] PID Loop
[x] State Space Model

### Sources
[ ] Pulse train
[ ] Ramp
[ ] Triangle
[ ] CORDIC
[ ] Sinusoid
[ ] Square
[ ] Noise

### Digital Communications
[ ] Constellations 
[ ] Barker codes 
[ ] Decision feedback equalizer 
[ ] Farrow filter
[ ] Mueller timing detector
[ ] OFDM
[ ] FM
[ ] Pulse correlation tracker
[ ] Addressable shift register 
[ ] Ping pong buffer

### Image

#### Filters
[ ] Sobel
[ ] Gaussian blur
[ ] Histogram equalization
[ ] Sharpening
[ ] Wiener filtering

#### Segmentation
[ ] Otsu's method
[ ] Canny Edge Detection
[ ] Watershed

#### Feature Extraction
[ ] Scale-Invariant Feature Transform
[ ] Speeded-Up Robust Features
[ ] Hough Transform
[ ] BLOB detection

### Filters

#### FIR
[x] Boxcar integrator
[x] Derivative filters
[x] firwin
[x] firwin2
[x] Overlap and add fir
[x] Remez exchange algorithm
[ ] FFT fir

#### IIR
[x] Bessel
[x] Biquad
[x] Butterworth
[x] Chebyshev 1
[x] Chebyshev 2
[x] Discrete linear filter (direct forms)
[x] Elliptic
[x] IIR comb
[x] IIR notch
[x] IIR peak

### Resampling
[x] Polyphase arbitrary resampling
[x] Polyphase integer resampling

### Spectral
[x] Chirpz transform
[x] freqz
[x] Critically sampled polyphase filterbank
[x] Oversampled polyphase filterbank
[ ] Non-uniformly sampled
[ ] MUSIC
[ ] Maximum entropy
[ ] Linear predictive coding
[ ] Wavelet transform

### Transformations
[x] Channel impairment
[x] Channel receiver
[x] Channel transmitter
[x] Frequency mixer
[x] Non-linearity transformer

# FFTs
[x] Best FFT
[x] Bluestein FFT
[x] Fractional FFT
[x] Simple Cooley Tukey
[x] Optimized radix-2
[x] Optimized split radix
[x] SIMD FFT (generic, x86, ARM, RISC-V)
[x] FFT engine trait

# Geometry
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

## Algorithms
[ ] Delaunay Triangulation
[ ] Convex hull
[ ] Slicing
[ ] Rotations
[ ] Quaternions
[ ] Nearest point finding
[ ] Intersection
[ ] KD Trees
[ ] Oct Trees
[ ] Quad Trees

## 4D
[ ] Space-time vector
[ ] Christoffel Symbols

# Information

## Source Coders
[x] Arithmetic
[x] Huffman
[x] lz77
[x] lz78
[x] lzw
[x] Shannon Fano
[x] Source coder trait
[ ] Gray Codes
[ ] MP3
[ ] JPEG

## LFSRs
[ ] LFSR
[ ] Maximum length polynomials
[ ] Gold codes

## Operators
[ ] SHA256
[ ] CRC32
[ ] Entropy

## Forward Error Correction
[ ] LDPC
[ ] Polar
[ ] Hamming
[ ] Reed-Solomon
[ ] Convolutional

# Integration
[x] Adaptive gl7k15
[x] Adaptive trapezoid
[x] Dormand Prince
[x] Midpoint
[x] Monte Carlo
[x] Runge Kutta
[x] Simpsons 1 3
[x] Trapezoidal
[x] Integrator trait (1D)
[x] Midpoint calculation helper
[ ] Complex path integrals

# Interpolation
[x] Interpolation trait

## 1D
[x] Linear
[x] Natural cubic spline
[x] Polynomial
[ ] PCHIP
[ ] Akima
[ ] Cubic basis
[ ] AAA
[ ] Prony's method
[ ] Pade approximants
[ ] Floater-Hormann
[ ] REIM
[ ] Fourier 1D
[ ] Chebyshev approximations
[ ] Lagrange interpolant
[ ] Savitzky-Golay

## 2D
[ ] Nearest
[ ] Bilinear
[ ] Bicubic
[ ] Lanczos
[ ] Cubic basis
[ ] All-phase
[ ] Radial basis functions
[ ] Fourier 2D

## 3D
[ ] Nearest
[ ] Trilinear
[ ] B-spline
[ ] Cubic Lagrangian
[ ] Fourier 3D
[ ] Shepards

## ND
[ ] Grid data
[ ] Nearest 
[ ] Cubic basis

# Number Theory
[x] Base exponent in modulus
[x] Greatest common divisor
[x] Integer square root
[x] Multiplicative inverse modulo
[x] Polynomial
[x] Text coded integer

# Optimization
[x] Golden section search
[x] Gradient descent
[x] Nelder Mead
[x] Newton Raphson
[x] Simulated annealing
[x] Optimization traits
[ ] Linear programming
[ ] Dynamic programming
[ ] Powell's method

# Random
[x] Histogram
[x] Uniform RNG
[x] xoshiro256++
[x] Split Mix 64

## Distributions
[x] Exponential
[x] Uniform (float)
[x] Bernoulli
[x] Geometric
[x] Weibull
[x] Poisson
[x] Gaussian
[x] Chi Square
[x] Student's T
[x] Gamma
[x] Beta

# Sorting and Selection
[x] Quicksort
[x] Heapsort
[x] Select M largest
[x] Equivalence classes
[x] Shell's method

# Special Functions 
[x] Gamma
[x] Beta
[x] Factorial
[x] Binomial coefficients
[x] Incomplete Gamma
[x] Error function
[x] Exponential integrals
[x] Incomplete Beta function
[x] Bessel functions integer
[x] Bessel functions fractional
[x] Bessel functions spherical
[x] Airy functions
[x] Fresnel integrals
[x] Cosine integrals
[x] Sine integrals
[x] Dawson's integral
[x] Fermi-Dirac integrals
[x] Inverse x log (x)
[x] Elliptic integrals
[ ] Jacobian elliptic functions
[ ] Hypergeometric functions
[ ] Statistical functions

# Root Finding
[x] Bracketing
[x] Bisection
[x] Secant method
[x] False position method
[x] Ridder's method
[x] Van Wijngaarden-Dekker-Brent
[x] Newton-Raphson
[x] Polynomial roots

# Machine Learning

## Supervised 

### Regression 
[ ] Least squares
[ ] General least squares
[ ] Ridge 
[ ] Elastic Net
[ ] Support vector regression


### Classification 
[ ] k-nearest neighbors
[ ] Phylogenetic trees
[ ] Viterbi algorithm 
[ ] Random Forest 
[ ] Support vector machine 
[ ] Logistic regression 
[ ] Naive Bayes

## Unsupervised 

### Clustering 
[ ] k means
[ ] k medoids 
[ ] Hierarchical clustering
[ ] DBSCAN
[ ] HDBSCAN
[ ] Gaussian mixture models
[ ] Spectral clustering 

### Reduction 
[ ] Principal component analysis 
[ ] Kernel PCA
[ ] t-SNE
[ ] UMAP
[ ] LDA


# Electromagnetics

## Antenna Models
[ ] Half-wave dipole
[ ] Dish
[ ] Hertzian dipole
[ ] Magnetic loop
[ ] Patch antenna 

## Fields
[ ] Field container
[ ] Interpolation
[ ] Coordinate conversion
[ ] Vector calculus

## Algorithms
[ ] Overlap integral
[ ] CFM beamformer
[ ] Hadamard beamformer
[ ] LCMV beamformer

# Utils
[ ] Terminal plot generation
[ ] Unit conversion
[ ] WAV files
[ ] JPEG files
[ ] Machine epsilon