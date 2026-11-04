| SciPy or scikit-image function and case | compared | result |
|---|---:|---:|
| `ndimage.correlate`, mode constant | largest difference | 4.9e-13 |
| `ndimage.convolve`, mode constant | largest difference | 4.9e-13 |
| `ndimage.correlate`, mode nearest | largest difference | 4.9e-13 |
| `ndimage.convolve`, mode nearest | largest difference | 4.9e-13 |
| `ndimage.correlate`, mode reflect | largest difference | 4.9e-13 |
| `ndimage.convolve`, mode reflect | largest difference | 4.9e-13 |
| `ndimage.gaussian_filter`, σ = 1.5, mode nearest | largest difference | 5.0e-13 |
| `ndimage.gaussian_filter`, σ = 2.3, mode nearest | largest difference | 8.7e-6 |
| `ndimage.gaussian_filter`, σ = 1.5, mode reflect | largest difference | 5.0e-13 |
| `ndimage.sobel`, along columns | largest difference | 4.9e-13 |
| `ndimage.sobel`, along rows | largest difference | 4.9e-13 |
| `signal.wiener`, 5 × 5, given noise | largest difference away from the edges (SciPy pads with zeros) | 5.0e-13 |
| `filters.threshold_otsu` | threshold | 101 (scikit-image 101) |
| `feature.canny`, σ = 2 | edge pixels; fraction of each within one pixel of the other's | 230 and 232; 100% and 100% |
| `feature.canny`, σ = 1 | edge pixels; fraction of each within one pixel of the other's | 229 and 230; 100% and 100% |
| `segmentation.watershed`, 3 markers | pixels labelled the same | 9216 of 9216 (100.00%) |
| `transform.hough_line` and `hough_line_peaks` | the two strongest lines (θ, ρ) | (0°, 70.2), (117°, 7.2); scikit-image (0°, 70.0), (117°, 7.0) |
| `transform.hough_circle` and `hough_circle_peaks` | centres (row, column) and radii | (62, 60, r 20), (30, 30, r 12); scikit-image (62, 60, r 20), (30, 30, r 12) |
| `feature.blob_log`, σ from 1 to 10 in 10 log steps | blobs (row, column, σ) | (20, 20, σ 2.15), (30, 65, σ 3.59), (70, 30, σ 5.99), (72, 75, σ 2.78); scikit-image (20, 20, σ 2.15), (30, 65, σ 3.59), (70, 30, σ 5.99), (72, 75, σ 2.78) |
