"""Reference values for the DSP chapter on image processing, from SciPy and scikit-image.

Run with a Python that has SciPy and scikit-image (the book used SciPy 1.17.1 and
scikit-image 0.26.0):

    python book/generator/data/make_image_refs.py > book/generator/data/image_reference.txt

Each output line is `case|values`, with comma-separated values and `;` between lists. The test
images themselves are written first, so the generator works on exactly the same pixels. Images
are flattened in row-major order; filtered images are given for their first 12 rows only,
which include the top corners and part of every edge.
"""

import numpy as np
from scipy import ndimage, signal
from skimage import feature, filters, segmentation, transform


def fmt(v):
    v = float(v)
    return str(int(v)) if v == int(v) and abs(v) < 1e15 else f"{v:.12g}"


def out(name, *lists):
    print(name + "|" + ";".join(",".join(fmt(v) for v in np.ravel(l)) for l in lists))


def digits(name, a):
    """A binary map or a small label image as one digit per pixel."""
    print(name + "|" + "".join(str(int(v)) for v in np.ravel(a)))


def top(a):
    """The first 12 rows: enough to cover two corners and part of every edge, keeping the file small."""
    return a[:12]


rng = np.random.default_rng(7)
n = 96
r, c = np.mgrid[0:n, 0:n]
scene = np.full((n, n), 50.0)
scene[(r - 30) ** 2 + (c - 34) ** 2 < 14 ** 2] = 200
scene[(r >= 55) & (r < 85) & (c >= 50) & (c < 85)] = 130
scene = np.clip(np.round(scene + rng.normal(0, 12, (n, n))), 0, 255).astype(np.uint8)
out("scene", scene)
f = scene.astype(float) / 255
kernel = np.round(rng.normal(0, 1, (3, 5)), 3)
out("kernel", kernel)

# Convolution and border modes (scipy.ndimage names: constant = zero, nearest = replicate,
# reflect = half-sample symmetric).
for mode in ["constant", "nearest", "reflect"]:
    out(f"correlate {mode}", top(ndimage.correlate(f, kernel, mode=mode, cval=0.0)))
    out(f"convolve {mode}", top(ndimage.convolve(f, kernel, mode=mode, cval=0.0)))
for sigma in [1.5, 2.3]:
    out(f"gaussian nearest {sigma}", top(ndimage.gaussian_filter(f, sigma, mode="nearest", truncate=4.0)))
out("gaussian reflect 1.5", top(ndimage.gaussian_filter(f, 1.5, mode="reflect", truncate=4.0)))
out("sobel x", top(ndimage.sobel(f, axis=1, mode="nearest")))
out("sobel y", top(ndimage.sobel(f, axis=0, mode="nearest")))

# Wiener denoising with a given noise variance (SciPy pads with zeros; compare the interior).
noise = (12 / 255) ** 2
out("wiener 5", top(signal.wiener(f, 5, noise)))

# Otsu's threshold on the 8-bit scene.
out("otsu", [filters.threshold_otsu(scene)])

# Canny edges (scikit-image thresholds the raw Sobel magnitude of the smoothed image).
for sigma, low, high in [(2.0, 0.05, 0.15), (1.0, 0.1, 0.3)]:
    digits(f"canny {sigma} {low} {high}", feature.canny(f, sigma=sigma, low_threshold=low, high_threshold=high))

# Marker-controlled watershed of the smoothed gradient magnitude.
smooth = ndimage.gaussian_filter(f, 2.0, mode="nearest")
# Rounded to integers (in units of 1e-5 per pixel) so that both sides flood exactly the same landscape.
elevation = np.round(1e5 * np.hypot(ndimage.sobel(smooth, axis=0, mode="nearest"), ndimage.sobel(smooth, axis=1, mode="nearest")) / 8)
markers = np.zeros((n, n), dtype=int)
markers[30, 34] = 1
markers[70, 67] = 2
markers[5, 5] = 3
out("watershed elevation", elevation)
out("watershed markers", [30, 34, 1, 70, 67, 2, 5, 5, 3])
digits("watershed", segmentation.watershed(elevation, markers, connectivity=1))

# Hough lines on a drawn edge map: a line of slope 1/2 and a vertical line.
lines = np.zeros((n, n), bool)
for t in range(90):
    lines[10 + t // 2, t + 3] = True
lines[:, 70] = True
digits("hough edges", lines)
h, theta, dist = transform.hough_line(lines, theta=np.arange(180) * np.pi / 180)
votes, peak_theta, peak_rho = transform.hough_line_peaks(h, theta, dist, num_peaks=2)
out("hough peaks", votes, peak_theta, peak_rho)

# Hough circles of radius 12 and 20.
circles = np.zeros((n, n), bool)
for k in range(720):
    t = k * np.pi / 360
    for (cr, cc, rad) in [(30, 30, 12), (62, 60, 20)]:
        circles[int(round(cr + rad * np.sin(t))), int(round(cc + rad * np.cos(t)))] = True
digits("circle edges", circles)
hc = transform.hough_circle(circles, [12, 20])
accums, cx, cy, radii = transform.hough_circle_peaks(hc, [12, 20], total_num_peaks=2)
out("circle peaks", cy, cx, radii)

# Laplacian-of-Gaussian blobs on Gaussian spots of known size (the generator builds the same image).
blobs = np.zeros((n, n))
for (br, bc, s, a) in [(20, 20, 2.0, 1.0), (30, 65, 4.0, 1.0), (70, 30, 6.0, 1.0), (72, 75, 3.0, 0.7)]:
    blobs += a * np.exp(-((r - br) ** 2 + (c - bc) ** 2) / (2 * s * s))
found = feature.blob_log(blobs, min_sigma=1, max_sigma=10, num_sigma=10, threshold=0.1, overlap=0.5, log_scale=True)
found = found[np.lexsort((found[:, 1], found[:, 0]))]
out("blobs", found[:, 0], found[:, 1], found[:, 2])
