"""Writes the window reference files (<name>.csv) from SciPy (1.17.1 was used) for the window tests.

Run from crates/dsp with a Python that has SciPy:  python src/discrete/test_data/make_window_goldens.py
"""

from scipy.signal import windows as w

items = {
    "boxcar": w.boxcar(7, True), "triang": w.triang(7, True), "parzen": w.parzen(7, True), "bohman": w.bohman(7, True),
    "blackman": w.blackman(7, True), "nuttall": w.nuttall(7, True), "blackmanharris": w.blackmanharris(7, True),
    "flattop": w.flattop(7, True), "bartlett": w.bartlett(7, True), "barthann": w.barthann(7, True),
    "hamming": w.hamming(7, True), "kaiser": w.kaiser(7, 14.0, True), "kaiser_bessel_derived": w.kaiser_bessel_derived(8, 14.0),
    "gaussian": w.gaussian(7, 1.0, True), "general_cosine": w.general_cosine(7, [1.0, 1.942604, 1.340318, 0.440811, 0.043097], True),
    "general_gaussian": w.general_gaussian(7, 1.5, 1.0, True), "general_hamming": w.general_hamming(7, 0.54, True),
    "chebwin": w.chebwin(7, 100.0, True), "cosine": w.cosine(7, True), "hann": w.hann(7, True),
    "exponential": w.exponential(7, None, 1.0, True), "tukey": w.tukey(7, 0.5, True), "taylor": w.taylor(7, 4, 30.0, True, True),
    "dpss": w.dpss(7, 2.5, sym=True), "lanczos": w.lanczos(7, sym=True),
}
for name, values in items.items():
    with open(f"src/discrete/test_data/{name}.csv", "w") as f:
        f.write("".join(f"{float(v)!r}\n" for v in values))
