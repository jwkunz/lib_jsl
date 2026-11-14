# DSP: image processing

The earlier DSP chapters worked on one-dimensional signals. Images are two-dimensional signals, and most of the same ideas carry over: convolution, smoothing, differentiation and
optimal linear filtering. Images also raise questions with no one-dimensional counterpart. Which pixels belong to which object? Where are the edges, lines and circles? Which points in one picture are the same as points in another? This chapter follows
that path, from filtering through segmentation to shapes and local features.

## The central questions

- **Which scale?** Every operation here looks at a neighbourhood, and its size decides what counts as noise and what as structure. Gaussian scale space makes the choice explicit, and
  blob and feature detectors make it automatic.
- **Local or global?** A global threshold or equalisation treats every pixel alike; adaptive filters, edge detectors and features look only at a neighbourhood.
- **Pixels or shapes?** Filters and segmentation produce images; the Hough transform and feature detectors produce lists of shapes and keypoints.
- **What should be invariant?** Brightness, contrast, rotation and scale all change between photographs of the same scene. Each method is designed to ignore some of them.

## Conventions

Images are `ndarray::Array2` arrays indexed `(row, column)`, with rows increasing downwards: `f64` for filtering and features, `u8` for histogram methods. Where a method speaks of $x$ and $y$, $x$ is the column and $y$ the row.
Kernels have odd dimensions and are centred on the output pixel. Outside the image, pixels are zero, replicated or reflected, as the [`Border`](../../api/lib_jsl_dsp/discrete/image/convolution/enum.Border.html) mode says.

## Testing against SciPy and scikit-image

The test images (a noisy scene of a disc and a square, drawn lines and circles, and Gaussian spots) and the reference results come from SciPy 1.17.1 and scikit-image 0.26.0, by `book/generator/data/make_image_refs.py`. The
filtered images were stored to 12 significant digits, which limits the differences seen:

<p class="caption" id="tab-17-1"><strong>Table 17.1.</strong> Image routines against SciPy and scikit-image.</p>

{{#include ../generated/image/reference.md}}

Everything agrees except the Gaussian blur at σ = 2.3, where the library's kernel radius ($\lceil 4\sigma\rceil = 10$) is one longer than SciPy's ($\mathrm{round}(4\sigma) = 9$), and the Hough lines' $\rho$, whose bins are placed differently. Canny's edges
match to within a pixel; scikit-image interpolates the gradient in non-maximum suppression, where the library compares with the nearest neighbours.

The morphology, distance-transform, labelling, rank-filter, derivative-filter and geometric-transform routines are compared with
`scipy.ndimage` on a separate $20 \times 24$ test image, for every structuring element, boundary mode and spline order, by
`book/generator/data/make_ndimage_refs.py`:

<p class="caption" id="tab-17-2"><strong>Table 17.2.</strong> Morphology, distance transforms, labelling, rank and derivative filters and geometric transforms against scipy.ndimage.</p>

{{#include ../generated/image/ndimage_reference.md}}

Binary results, labels, chamfer distances and rank filters are identical. Everything else agrees to rounding error, except quintic
splines in `Nearest` mode, which agree to $2\times10^{-10}$ (both libraries pad the image by 12 pixels before prefiltering, which the
slowly decaying poles of order 5 do not quite forget).

## In the library

All of these are in `lib_jsl_dsp` ([API reference](../../api/lib_jsl_dsp/index.html)), under `discrete::image`.

| You need | Use | Entry |
|---|---|---|
| A convolution with border handling; a blur, gradient or sharpening | `correlate2d`, `convolve2d`, `separable_convolve`, `gaussian_blur`, `sobel`, `sharpening` | [Image filtering](filtering.md) |
| More contrast; less noise; less blur | `histogram_equalization`, `wiener_filter`, `wiener_deconvolution` | [Contrast and restoration](restoration.md) |
| A foreground mask, an edge map or a set of regions | `otsu`, `canny_edge_detection`, `watershed`, `label_boundaries` | [Segmentation](segmentation.md) |
| Straight lines or circles | `hough_lines`, `hough_transform`, `hough_line_peaks`, `hough_circles` | [The Hough transform](hough.md) |
| Blobs, keypoints and descriptors; matches between images | `blob_detection`, `sift`, `surf`, `match_descriptors` | [Local features](features.md) |
| To clean a binary mask; to measure depth inside objects; to count and measure objects | `binary_opening`, `binary_closing`, `binary_fill_holes`, `grey_erosion`, `white_tophat`, `distance_transform_edt`, `label`, `regions` | [Morphology, distance transforms and labelling](morphology.md) |
| To remove impulse noise; derivatives at a scale; to shift, rotate, zoom or warp an image | `median_filter`, `rank_filter`, `gaussian_filter`, `gaussian_laplace`, `map_coordinates`, `affine_transform`, `rotate`, `zoom` | [Rank filters, derivative filters and geometric transforms](rank-geometric.md) |
