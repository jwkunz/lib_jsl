# Changelog

All crates in the workspace share one version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org).

## [Unreleased]

### Changed

- The book was revised for clarity and teaching flow after feedback from early readers.
  Derivations now show their intermediate steps, terms are defined where they are first used,
  and most entries gained small worked examples. The DSP chapters share a new background
  section on frequency responses and transfer functions, "How to read an entry" has a study
  guide, and "Writing an entry" records the conventions. Notes about the library's own development
  history were moved out of the chapters (see Fixed below). No code or generated figure changed.

### Performance

- `least_squares` (in `lib_jsl_linalg`) computes only the singular values of `R` for its report,
  instead of a full SVD with vectors that were discarded. On a 512 × 256 problem this halves the
  time (38 ms to 20 ms single-threaded on a 2.1 GHz Xeon; OpenBLAS takes 12 ms).
- Householder QR with column pivoting is blocked for 192 columns and more (LAPACK's `laqps`): the
  trailing update is deferred within blocks of 32 columns and applied by matrix multiplication.
  It is 1.3× faster at 1024 × 512 and 1.65× faster at 2048 × 1024 (#11).

### Fixed

Defects found and corrected while the book's chapters were first written. The notes were moved here from the
chapters' text; the defects themselves were fixed before this entry.

- **coding**: fixed one problem. The **polar code construction** ranked the bit channels in bit-reversed order, so information bits went on the wrong channels.
- **comms**: fixed one documentation problem: the `KalmanFilter`'s description was attached to its result type, so the filter itself appeared undocumented.
- **em-arrays**: corrected two documentation details: a demonstration compared the dish beamwidth with 1.02 λ/D instead of the exact 1.029 λ/D, and the crate's description omitted most of its contents.
- **em-waves**: found no defects in these modules.
- **filters**: fixed six problems.
  - `remez` did not find the equiripple optimum (its worst error was up to 12 times SciPy's), rejected even lengths, differentiators and Hilbert transformers, and was tested against its own output. It is now a full Parks-McClellan implementation.
  - `buttord`, `cheb1ord`, `cheb2ord` and `ellipord` returned band-stop orders one too high: two of them searched for the optimal passband edges in the wrong direction, and two did not search at all.
  - `generate_derivative_taps` returned the first derivative for every order.
  - The Kaiser window's Bessel function was accurate to only seven digits.
  - `DiscreteLinearFilter` silently assumed $a_0 = 1$.
  - Fourteen modules' descriptions were attached to `use` statements, where the API reference does not show them.
- **foundations**: fixed seven problems.
  - The trailing digits of six **constants** were wrong or truncated.
  - **Unit conversion** ignored case, so `mW` was a megawatt and `kb` a kilobyte. It also treated the torr as the mmHg, and rounded the psi to 13 digits.
  - The **Netpbm, PNG and JPEG decoders** allocated for whatever size a header claimed, so a few bytes could abort the process.
  - The **DEFLATE encoder** expanded incompressible data instead of storing it.
  - The **JPEG decoder** replicated subsampled chroma instead of interpolating it as libjpeg does.
- **geometry**: fixed nine problems.
  - The **Delaunay triangulation** produced overlapping triangles on nearly cocircular points and lost the hull's sliver triangles on thin point sets. It is now a sweep with Lawson flips and error-bounded predicates.
  - The **3D convex hull** took 46 s for 10 000 points on a sphere. It is now Quickhull and takes under 0.1 s. Its documentation had also claimed that points on faces are never vertices.
  - **`Quaternion::to_euler`** returned the wrong roll at a pitch of $-90°$.
  - **`EulerSequence::from_matrix`** returned angles outside its documented ranges, and at gimbal lock split the angle between the first and third axes by rounding noise.
  - **Mesh slicing** reported edges lying in the plane twice, breaking the contours.
  - The documentation of **polygon booleans** said unions cannot have holes, and that of **`CanShear`** described a shear the code does not do.
- **graph**: fixed one problem: the topological sort promised to break ties by vertex index but used a first-in-first-out queue, so its order depended on edge insertion. It now returns the lexicographically smallest order.
- **information**: fixed four problems.
  - The **arithmetic coder** could corrupt inputs over 1 GiB: its 32-bit code range allows a model total of at most $2^{30}$, and larger totals can give a rare symbol an empty interval. It now rescales such models.
  - **Huffman** and **Shannon-Fano** coding spent one bit per byte on single-symbol inputs, which their decoders never needed; a constant file now compresses to its header.
  - The **entropy** of a certain outcome was returned as $-0$.
  - The **crate documentation** described only three compressors.
- **learning**: fixed four problems.
  - An MLP with one input feature and two hidden layers could not be trained: a weight gradient came back column-major and the optimiser panicked reading it as a row-major slice.
  - MLP predictions broadcast inputs with the wrong number of features instead of rejecting them.
  - Gradient boosting predictions panicked on rows with too few features.
  - The Baum-Welch likelihood history was documented as describing the model after each iteration; it describes the model before.
- **number-theory**: fixed nine problems.
  - The integer **Newton square root** looped forever on $n = k^2 - 1$.
  - The **modular power** accepted a zero modulus, because its checks used bitwise NOT, and then panicked; it also overflowed for moduli above about $10^{19}$.
  - The **gcd** could be negative.
  - A negative modulus gave a negative **modular inverse**.
  - **`factorize(0)`** never returned.
  - **Pollard's rho** could overflow for numbers near $2^{64}$, and so could the **segmented sieve** near `u64::MAX`.
  - **`TextCodedInteger`** ordered its digit strings rather than its values, accepted malformed text, and had a wrong `abs_sub` and unimplemented checked operations.
  - The documented **Miller–Rabin bound** quoted the thirteen-base value.
- **random**: uncovered three errors, now fixed: the Poisson sampler returned $-1$ for every rate (one uniform reused, a loop condition inverted); `Histogram` ignored its lower bin edge and panicked at its upper one; and a helper that draws vectors of normal samples returned two extra values for even lengths, which made chi-square samples with even degrees of freedom sum two extra squares.
- **sorting**: fixed quicksort, which took quadratic time on runs of equal keys. Its documentation also called its partition Hoare-style when it was Lomuto's. It now uses Hoare's partition, and falls back to heapsort as introsort does, so even an adversary cannot make it quadratic.
- **special functions**: uncovered three errors, which have been fixed: the Jacobian elliptic function dn was computed as $0/0$ at the quarter period, ${}_0F_1$ lost seven digits to cancellation for large negative arguments, and the Student's $t$ quantile lost six digits of small lower-tail probabilities.
- **spectral**: fixed six problems.
  - Both **polyphase resamplers** held samples instead of interpolating: every output between two inputs was equal. They now compute `upfirdn` exactly.
  - Both **polyphase analysis filter banks** reversed the prototype filter within each block, so their channels had 15–20 dB side lobes instead of 60 dB.
  - The **Dolph-Chebyshev window** was wrong by up to 0.99, from a polynomial of the wrong order and a sign error.
  - The **DPSS window** found the wrong eigenvector for bandwidths above a quarter of the sampling rate.
  - The polyphase **synthesis filter banks** filtered each frame with the analysis prototype's polyphase taps, which does not invert the analysis bank: with the default prototypes the output differed from the input by 18–41%. They now overlap-add through a synthesis filter derived from the reconstruction conditions, and the oversampled banks reconstruct to better than −65 dB. A critically sampled DFT bank cannot reconstruct with a selective prototype at all; the filter bank entry explains why.
  - The **window tests** compared each window with golden files generated by the library itself; they now use SciPy's values.
- **unsupervised**: fixed three problems.
  - The documentation of `KMeans` was attached to a private type alias, so the public struct appeared undocumented.
  - `GaussianMixture` reported the log-likelihood (and BIC and AIC) of the previous parameters when it stopped at the iteration limit.
  - A kernel PCA test claimed a scaling difference between kernel and linear PCA scores that does not exist; the scores are identical.
- **machine learning**: fixed multinomial naive Bayes without smoothing, which dropped a class from the vote (a NaN from $0 \times (-\infty)$) whenever a word unseen in that class was absent from the document.
- **statistics**: fixed the analysis-of-variance p-value, which was computed as $1 - \text{CDF}$ and lost every digit below $10^{-16}$, and documented the accuracy of the two-sample Kolmogorov-Smirnov p-value.
- **image**: fixed one documentation mistake: the support of a complete Hough circle was said to be about 1, but depends on how the circle is drawn.
- **fft**: uncovered three errors, now fixed: Bluestein's algorithm computed its chirp angles $\pi k^2/n$ in floating point and lost five digits at large lengths (taking the default `Fft` with it for every non-power-of-two length); the reference split-radix engine returned wrong results, unnoticed because its tests never checked the error; and the fractional Fourier transform returned $1/n$ of the input energy at non-integer orders.

### Removed

- The `mp3` and `mp3_tables` modules of `lib_jsl_file_io` (`mp3_encode`, `mp3_decode`,
  `mp3_encode_with_model`, `Mp3Audio` and the psychoacoustic models). The codec followed the
  structure of MPEG-1 Layer III but could neither read nor write real MP3 files.

## [1.0.0] - 2026-11-01

The first stable release: every planned algorithm (457 items across 23 library crates, listed
in the book's [catalogue](https://jwkunz.github.io/lib_jsl/book/catalogue.html)) is implemented and tested.

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

- The book's catalogue of algorithms replaces `ROADMAP.md` as the list of what the library
  implements and what it does not cover.

- Dependencies are pinned once in the workspace manifest; `num` was replaced by `num-complex`
  and `num-traits`, and `ndarray-linalg` by `lib_jsl_linalg`.

### Fixed

- The Simpson's 1/3 engine had its 4 and 2 weights swapped; it now also uses a 3/8 tail for an
  even number of points.

[1.0.0]: https://github.com/jwkunz/lib_jsl/releases/tag/v1.0.0
