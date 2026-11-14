"""Reference values for the image-processing chapter's morphology, distance-transform, labelling,
rank-filter, derivative-filter and geometric-transform routines, from scipy.ndimage.

Run with a Python that has SciPy (the book used SciPy 1.17.1):

    python book/generator/data/make_ndimage_refs.py > book/generator/data/ndimage_reference.txt

Each line is `case|values` (comma-separated, row-major). The two test images come first, so the
generator works on exactly the same pixels; every case is computed on them.
"""

import numpy as np
from scipy import ndimage as ndi


def fmt(v):
    v = float(v)
    if not np.isfinite(v):
        return "inf" if v > 0 else "-inf"
    return str(int(v)) if v == int(v) and abs(v) < 1e15 else f"{v:.15g}"


def out(name, a):
    print(name + "|" + ",".join(fmt(v) for v in np.ravel(a)))


rng = np.random.default_rng(1202)
rows, cols = 20, 24
grey = np.round(ndi.gaussian_filter(rng.normal(size=(rows, cols)), 1.5) * 10, 4)
binary = grey > 0.3
binary[5:9, 6:10] = True
binary[6:8, 7:9] = False  # a hole
out("image grey", grey)
out("image binary", binary)

cross = ndi.generate_binary_structure(2, 1)
square = np.ones((3, 3), bool)
disk = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
disk5 = np.array([[x * x + y * y <= 4 for x in range(-2, 3)] for y in range(-2, 3)])

# Morphology.
for sname, s in [("cross", cross), ("square", square)]:
    out(f"binary_erosion {sname}", ndi.binary_erosion(binary, s))
    out(f"binary_dilation {sname}", ndi.binary_dilation(binary, s))
    out(f"binary_opening {sname}", ndi.binary_opening(binary, s))
    out(f"binary_closing {sname}", ndi.binary_closing(binary, s))
    out(f"binary_fill_holes {sname}", ndi.binary_fill_holes(binary, s))
out("binary_erosion cross 2 iterations", ndi.binary_erosion(binary, cross, iterations=2))
out("binary_dilation disk5", ndi.binary_dilation(binary, disk5))
for mname, mode in [("reflect", "reflect"), ("nearest", "nearest"), ("mirror", "mirror"), ("wrap", "wrap"), ("constant", "constant")]:
    out(f"grey_erosion square {mname}", ndi.grey_erosion(grey, footprint=square, mode=mode))
    out(f"grey_dilation disk5 {mname}", ndi.grey_dilation(grey, footprint=disk5, mode=mode))
out("grey_opening square", ndi.grey_opening(grey, footprint=square))
out("grey_closing square", ndi.grey_closing(grey, footprint=square))
out("white_tophat disk5", ndi.white_tophat(grey, footprint=disk5))
out("black_tophat disk5", ndi.black_tophat(grey, footprint=disk5))
out("morphological_gradient square", ndi.morphological_gradient(grey, footprint=square))

# Distance transforms.
out("distance_transform_edt", ndi.distance_transform_edt(binary))
out("distance_transform_edt sampling 2 0.5", ndi.distance_transform_edt(binary, sampling=(2.0, 0.5)))
out("distance_transform_cdt taxicab", ndi.distance_transform_cdt(binary, metric="taxicab"))
out("distance_transform_cdt chessboard", ndi.distance_transform_cdt(binary, metric="chessboard"))

# Labelling and measurements.
for sname, s in [("cross", cross), ("square", square)]:
    lab, n = ndi.label(binary, s)
    out(f"label {sname}", lab)
    idx = np.arange(1, n + 1)
    out(f"label {sname} sizes", ndi.sum(np.ones_like(grey), lab, idx))
    out(f"label {sname} sum", ndi.sum(grey, lab, idx))
    out(f"label {sname} mean", ndi.mean(grey, lab, idx))
    out(f"label {sname} minimum", ndi.minimum(grey, lab, idx))
    out(f"label {sname} maximum", ndi.maximum(grey, lab, idx))
    out(f"label {sname} center_of_mass", np.array(ndi.center_of_mass(grey, lab, idx)))
    out(f"label {sname} find_objects", np.array([[s0.start, s0.stop, s1.start, s1.stop] for s0, s1 in ndi.find_objects(lab)]))

# Rank filters.
for mname in ["reflect", "nearest", "mirror", "wrap", "constant"]:
    out(f"median_filter 3x3 {mname}", ndi.median_filter(grey, size=3, mode=mname))
out("median_filter 4x4", ndi.median_filter(grey, size=4))
out("median_filter 2x5", ndi.median_filter(grey, size=(2, 5)))
out("percentile_filter 25 5x5", ndi.percentile_filter(grey, 25, size=5))
out("percentile_filter 100 3x3", ndi.percentile_filter(grey, 100, size=3))
out("minimum_filter 3x2", ndi.minimum_filter(grey, size=(3, 2)))
out("maximum_filter 2x3", ndi.maximum_filter(grey, size=(2, 3)))
out("rank_filter -2 disk5", ndi.rank_filter(grey, -2, footprint=disk5))

# Derivative filters.
for mname in ["reflect", "nearest", "mirror", "wrap", "constant"]:
    out(f"gaussian_filter 1.5 {mname}", ndi.gaussian_filter(grey, 1.5, mode=mname))
out("gaussian_filter (1, 2.5) order (0, 1)", ndi.gaussian_filter(grey, (1.0, 2.5), order=(0, 1)))
out("gaussian_filter 2 order (2, 0)", ndi.gaussian_filter(grey, 2.0, order=(2, 0)))
out("gaussian_filter 1.2 order (1, 1)", ndi.gaussian_filter(grey, 1.2, order=(1, 1)))
out("gaussian_laplace 1.5", ndi.gaussian_laplace(grey, 1.5))
out("gaussian_gradient_magnitude 1.5", ndi.gaussian_gradient_magnitude(grey, 1.5))
out("laplace", ndi.laplace(grey))

# Geometric transforms.
pts = np.array([[-0.7, 2.5], [0.0, 0.0], [3.3, 7.8], [10.25, 15.5], [19.0, 23.0], [19.4, 23.2], [20.6, -1.2], [8.5, 23.9], [-2.0, 30.0], [12.0, 0.4]])
for mname, mode in [("constant", "constant"), ("nearest", "nearest"), ("mirror", "mirror"), ("reflect", "reflect"), ("wrap", "grid-wrap")]:
    for order in range(6):
        out(f"map_coordinates order {order} {mname}", ndi.map_coordinates(grey, pts.T, order=order, mode=mode))
out("affine_transform shear order 3", ndi.affine_transform(grey, [[1.0, 0.3], [-0.2, 0.9]], offset=(1.5, -2.0), output_shape=(18, 26), order=3))
out("shift (2.5, -1.25) order 1 nearest", ndi.shift(grey, (2.5, -1.25), order=1, mode="nearest"))
out("shift (2.5, -1.25) order 3", ndi.shift(grey, (2.5, -1.25), order=3))
out("zoom (1.5, 0.75) order 3", ndi.zoom(grey, (1.5, 0.75), order=3))
out("zoom (0.5, 2) order 1", ndi.zoom(grey, (0.5, 2.0), order=1))
out("rotate 30 reshape order 3", ndi.rotate(grey, 30, reshape=True, order=3))
out("rotate -20 order 1 nearest", ndi.rotate(grey, -20, reshape=False, order=1, mode="nearest"))
out("rotate 30 reshape order 3 shape", np.array(ndi.rotate(grey, 30, reshape=True, order=3).shape))

# Fourier-domain filters (1.4.0): full spectra (filtered image after the inverse FFT) and a real
# FFT along the rows (the filtered spectrum, real parts then imaginary parts).
F = np.fft.fft2(grey)
out("fourier_gaussian (1.5, 0.7)", np.fft.ifft2(ndi.fourier_gaussian(F, (1.5, 0.7))).real)
out("fourier_uniform (3, 4)", np.fft.ifft2(ndi.fourier_uniform(F, (3, 4))).real)
out("fourier_ellipsoid (5, 3)", np.fft.ifft2(ndi.fourier_ellipsoid(F, (5, 3))).real)
out("fourier_shift (2.5, -1.25)", np.fft.ifft2(ndi.fourier_shift(F, (2.5, -1.25))).real)
R = np.fft.rfft(grey, axis=-1)
for name, f, p in [("gaussian", ndi.fourier_gaussian, (1.5, 0.7)), ("uniform", ndi.fourier_uniform, (3, 4)), ("ellipsoid", ndi.fourier_ellipsoid, (5, 3)), ("shift", ndi.fourier_shift, (2.5, -1.25))]:
    s = f(R, p, n=24)
    out(f"fourier_{name} real rows", np.concatenate([s.real.ravel(), s.imag.ravel()]))
