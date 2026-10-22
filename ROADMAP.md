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
[x] Chebychev 1 + 2

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
[x] Kaiser
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
[ ] Farro filter
[ ] Mueller timing detector
[ ] OFDM
[ ] FM
[ ] Pulse correlation tracker
[ ] Addressable shift register 
[ ] Ping pong buffer

### Image

#### Filters
[ ] Sobel
[ ] Guassian blur
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
[ ] Savitzky-Golay

#### IIR
[x] Bessel
[x] Biquad
[x] Butterworth
[x] Chebychev 1
[x] Chebychev 2
[x] Discrete linear filter (direct forms)
[x] Elliptic
[x] IIR comb
[x] IIR notch
[x] III peak

### Resampling
[x] Polyphase arbitrary resampling
[x] Polyphase integeter resampling

### Spectral
[x] Chirpz transform
[x] freqz
[x] Critically sampled polyphase filterbank
[x] Oversampled polyphase filterbank
[ ] Non-uniformly sampled
[ ] MUSIC
[ ] Maximum entropy
[ ] Linear predictive coding
[ ] Wavelett transform

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

# Geometry
[x] Common
[x] Concrete types
[x] Coordinate systems
[x] Geometric traits
[x] Registry
[x] Tables
[x] Transformation traits

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
[x] Triangle 2D
[x] Unit vector 2D
[x] Volume mesh 3D

## Algorithms
[ ] Delanauy Triangulation
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
[ ] Christofel Symbols

# Information

## Source Coders
[x] Arithmetic
[x] Huffman
[x] lz77
[x] lz78
[x] lzw
[x] Shannon Fano
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
[ ] Reed Soloman
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
[ ] Complex path integrals

# Interpolation

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
[ ] Chebychev Approximations
[ ] Lagrange interpolant

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
[ ] Cubic Langrangian
[ ] Fourier 3D
[ ] Shepards

## ND
[ ] Grid data
[ ] Nearest 
[ ] Cubic basis

# Number Theory
[ ] Base exponent in modulus
[ ] Greatest commond divisor
[ ] Integer square root
[ ] Multiplicative inverse modulo
[ ] Polynomial
[ ] Text coded integer

# Optimization
[x] Golden section search
[x] Gradient descent
[x] Nelder Mead
[x] Newton Raphson
[x] Simulated annealing 
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
[x] Bernouli
[x] Geometric
[x] Weibull
[x] Poisson
[x] Guassian
[x] Chi Square
[x] Student's T
[x] Gamma
[x] Beta

# Sorting and Selection
[ ] Quicksort
[ ] Heapsort
[ ] Select M largest
[ ] Equivalence classes
[ ] Shell's method

# Special Functions 
[ ] Gamma
[ ] Beta
[ ] Factorial
[ ] Binomial coeficients
[ ] Incomplete Gamma
[ ] Error function
[ ] Exponential integrals
[ ] Imcomplete Beta function
[ ] Bessel functions integer
[ ] Bessel functions fractional
[ ] Bessel functions spherical
[ ] Airy functions
[ ] Fresnel integrals
[ ] Cosine integrals
[ ] Sine integrals
[ ] Dawson's integral
[ ] Fermi-Dirac integrals
[ ] Inverse x log (x)
[ ] Elliptic integrals
[ ] Jacobian elliptic functions
[ ] Hypergeometric functions
[ ] Statistical functions

# Root Finding
[ ] Bracketing
[ ] Bisection
[ ] Secant method
[ ] False position method
[ ] Ridder's method
[ ] Van Wijngaarden-Dekker-Brent
[ ] Newton-Raphson
[ ] Polynomial roots

# Machine Learning

## Supervised 

### Regression 
[ ] Least squares
[ ] General least squares
[ ] Ridge 
[ ] Elastic Net


### Classification 
[ ] k-nearest neighbors
[ ] Phylogentic trees
[ ] Vitirbi algorithm 
[ ] Random Forest 
[ ] Support vector machine 
[ ] Logistic regression 
[ ] Naive Bayes

## Unsupervised 

### Clustering 
[ ] k means
[ ] k medoids 
[ ] Heirarchacle 
[ ] Support vector regression 
[ ] DBSCAN
[ ] HDBSCAN
[ ] Gaussian mixture models
[ ] Spectral clustering 

### Reduction 
[ ] Principle component analysis 
[ ] Kernel PCA
[ ] t-SNE
[ ] UMAP
[ ] LDA


## Electromagnetics

### Antenna Models
[ ] Half-wave dipole
[ ] Dish
[ ] Hertzian dipole
[ ] Magnetic loop
[ ] Patch antenna 

### Fields
[ ] Field container
[ ] Interpolation
[ ] Coordinate converstion
[ ] Vector calculus

### Algorithms
[ ] Overlap integral
[ ] CFM beamformer
[ ] Hadamrd beamformer
[ ] LCMV beamformer

## Utils
[ ] Terminal plot generation
[ ] Unit conversion
[ ] WAV files
[ ] JPEG files
[ ] Machine epsilon