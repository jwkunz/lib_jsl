# Savitzky-Golay smoothing

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Smooth noisy, equally spaced samples, or estimate their derivatives, without flattening peaks |
| **Devised by** | Abraham Savitzky and Marcel Golay (1964), building on the moving polynomial least squares of actuarial graduation (Sheppard, 1913; Whittaker and Robinson, 1924) |
| **Accuracy** | Reproduces polynomials of the filter's order exactly; reduces white-noise variance by roughly the window length |
| **Cost** | One convolution with a fixed set of coefficients: $O(n \cdot \text{window})$ |
| **API** | [`savitzky_golay`](../../api/lib_jsl_interpolation/one_d/savitzky_golay/fn.savitzky_golay.html), [`savitzky_golay_coefficients`](../../api/lib_jsl_interpolation/one_d/savitzky_golay/fn.savitzky_golay_coefficients.html) |

</div>

## Overview

Measured signals are noisy, and the simplest cure, averaging each value with its neighbours, works
against noise but also against the signal: a moving average flattens peaks and shifts their shapes.
A **Savitzky-Golay filter** replaces the average by a local **polynomial least-squares fit**: around
each sample, fit a polynomial of low degree to the samples in a window, and take the value (or a
derivative) of that polynomial at the centre. Because a polynomial can follow a peak, the filter
removes noise while preserving the shape, height and position of features much better than averaging.

The remarkable fact that makes the method practical is that, for equally spaced data, the fitted
value is a **fixed linear combination** of the samples in the window. The whole filter is a
convolution with a short set of coefficients, computed once.

## History

Fitting local polynomials to smooth data was practised by actuaries in the nineteenth and early
twentieth centuries, who "graduated" mortality tables with moving least-squares formulas. Abraham
Savitzky and Marcel Golay, chemists at the Perkin-Elmer Corporation, brought the idea to the
smoothing of spectra in a 1964 paper in *Analytical Chemistry*, with tables of coefficients for many
window sizes and polynomial degrees. It became one of the most cited papers in the history of the
journal. Their tables contained some errors, corrected by Steinier, Termonia and Deltour in 1972.

## The idea

### Local polynomial least squares

Take a window of $2m + 1$ samples centred on sample $i$, with offsets $k = -m, \ldots, m$. Fit the
polynomial $q(k) = \sum_{j=0}^{d} b_j k^j$ by least squares:

$$
\min_{b} \sum_{k=-m}^{m} \big(q(k) - y_{i+k}\big)^2.
$$

With the Vandermonde matrix $A_{kj} = k^j$, the solution is $\mathbf b = A^{+}\mathbf y$, where $A^{+}$ is the
pseudo-inverse. The smoothed value is $q(0) = b_0$, the first row of $A^{+}$ applied to the window,
and the $r$-th derivative is $q^{(r)}(0) = r!\, b_r / \Delta^r$, the row $r$. Since $A$ depends only on $m$
and $d$, not on the data, these rows are the **filter coefficients**, the same for every position.
For $m = 2$ and $d = 2$ they are the classic $(-3, 12, 17, 12, -3)/35$.

The filter reproduces any polynomial of degree $\le d$ exactly, so it does not distort features that
are locally polynomial, and its noise reduction is close to that of a moving average of the same
length. A moving average is the special case $d = 0$ (or $d = 1$, which gives the same weights):

![Smoothing a noisy peak](../generated/interpolation/savitzky_golay_peak.svg)

The moving average cuts the peak to two-thirds of its height and widens it; the order-4 Savitzky-Golay
filter follows it.

### Derivatives of noisy data

Differentiating noisy data by [finite differences](../derivatives/finite-differences.md) amplifies the
noise by $1/\Delta$: the noise in the samples plays the role of rounding error. The derivative of the
local fit averages the noise first. On the noisy peak above:

{{#include ../generated/interpolation/savitzky_golay_derivative.md}}

The best window balances the two sources of error, just as the best step does for finite differences:
too short and noise dominates, too long and the window's polynomial cannot follow the signal.

### Conditioning

The Vandermonde matrix in the raw offsets $k = -m, \ldots, m$ is badly conditioned for wide windows and
high orders (its columns $k^j$ range from 1 to $m^d$), and forming the normal equations $A^{\mathsf T}A$
squares the condition number. The library scales the offsets to $t = k/m \in [-1, 1]$ and computes the
pseudo-inverse from an SVD, then adjusts the derivative factors for the scaling. (An earlier version
formed the normal equations directly: a 51-point, order-8 filter reproduced its own polynomials only to
$10^{-5}$, and 101-point windows failed as singular.)

## Algorithm

1. Build the scaled Vandermonde matrix $A_{kj} = (k/m)^j$ and its pseudo-inverse by SVD.
2. The coefficients of the $r$-th derivative filter are row $r$ of $A^{+}$, times $r!/(m\Delta)^r$.
3. Away from the ends, correlate the coefficients with the data.
4. Near the ends, where the window would run off the data, fit one polynomial to the first (or last)
   full window and evaluate it, or its derivative, at the end points.

## Implementation

```rust,ignore
{{#include ../../../crates/interpolation/src/one_d/savitzky_golay.rs:savitzky_golay}}
```

- The window must be odd and larger than the polynomial order; a derivative of order above the
  polynomial's is identically zero.
- The end treatment matches SciPy's `mode="interp"`, so the output has the same length as the input.

## In practice

```rust,ignore
{{#include ../../../crates/interpolation/examples/interpolation_savitzky_golay.rs}}
```

**Strengths**

- Preserves peak heights and widths far better than a moving average of the same noise reduction.
- Smoothing and differentiation in one cheap convolution.

**Weaknesses**

- Requires equally spaced samples (for uneven spacing, fit local polynomials directly).
- The window and order must be chosen; a too-high order follows the noise, a too-wide window flattens
  the signal.
- Its frequency response has ripples in the stop band, so it is a poorer low-pass filter than a
  designed one ([digital filters](../../api/lib_jsl_dsp/index.html)) when frequency selectivity matters.

## Exercises

1. Derive the 5-point quadratic smoothing coefficients $(-3, 12, 17, 12, -3)/35$ by hand from the
   normal equations.
2. Show that the smoothing coefficients for orders $2s$ and $2s + 1$ are identical. Why?
3. Compute the noise-reduction factor $\sum_k c_k^2$ of a 21-point filter of orders 0, 2 and 4, and
   compare with $1/21$.
4. Compute and plot the frequency response $\sum_k c_k e^{-i\omega k}$ of the 21-point order-4 filter.
5. Estimate the second derivative of the noisy peak with Savitzky-Golay filters of several windows
   and find the best.

## Further reading

- A. Savitzky and M. J. E. Golay, "Smoothing and differentiation of data by simplified least squares
  procedures", *Analytical Chemistry* 36 (1964) 1627–1639.
- J. Steinier, Y. Termonia and J. Deltour, "Comments on smoothing and differentiation of data by
  simplified least square procedure", *Analytical Chemistry* 44 (1972) 1906–1909.
- R. W. Schafer, "What is a Savitzky-Golay filter?", *IEEE Signal Processing Magazine* 28 (2011)
  111–117. The filter from the signal-processing point of view.
