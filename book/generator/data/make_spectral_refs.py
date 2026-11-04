"""Reference values for the DSP chapter on spectral estimation, windows and multirate processing,
from SciPy.

Run with a Python that has SciPy (the book used SciPy 1.17.1):

    python book/generator/data/make_spectral_refs.py > book/generator/data/spectral_reference.txt

Each output line is `case|values`, with comma-separated values and `;` between lists (for
example real and imaginary parts). The generator recomputes each named case with the library.
"""

import numpy as np
from scipy import signal
from scipy.signal import windows as w


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.atleast_1d(l)) for l in lists))


for sym in [True, False]:
    tag = "sym" if sym else "periodic"
    for m in [16, 17]:
        for name, f in [
            ("boxcar", lambda m: w.boxcar(m, sym)), ("triang", lambda m: w.triang(m, sym)), ("parzen", lambda m: w.parzen(m, sym)),
            ("bohman", lambda m: w.bohman(m, sym)), ("blackman", lambda m: w.blackman(m, sym)), ("nuttall", lambda m: w.nuttall(m, sym)),
            ("blackmanharris", lambda m: w.blackmanharris(m, sym)), ("flattop", lambda m: w.flattop(m, sym)), ("bartlett", lambda m: w.bartlett(m, sym)),
            ("barthann", lambda m: w.barthann(m, sym)), ("hamming", lambda m: w.hamming(m, sym)), ("hann", lambda m: w.hann(m, sym)),
            ("kaiser 8", lambda m: w.kaiser(m, 8.0, sym)), ("gaussian 2.5", lambda m: w.gaussian(m, 2.5, sym)),
            ("general_gaussian 1.5 3", lambda m: w.general_gaussian(m, 1.5, 3.0, sym)), ("chebwin 80", lambda m: w.chebwin(m, 80.0, sym)),
            ("cosine", lambda m: w.cosine(m, sym)), ("exponential tau 3", lambda m: w.exponential(m, None, 3.0, sym)),
            ("tukey 0.4", lambda m: w.tukey(m, 0.4, sym)), ("taylor 4 30", lambda m: w.taylor(m, 4, 30, True, sym)),
            ("dpss 3", lambda m: w.dpss(m, 3.0, sym=sym)), ("lanczos", lambda m: w.lanczos(m, sym=sym)),
            ("general_hamming 0.6", lambda m: w.general_hamming(m, 0.6, sym)),
        ]:
            out(f"window {name} {m} {tag}", f(m))
out("window kaiser_bessel_derived 8 16", w.kaiser_bessel_derived(16, 8.0))

rng = np.random.default_rng(20251104)
x = np.round(np.sin(2 * np.pi * 0.11 * np.arange(256)) + rng.normal(0, 0.5, 256), 6)
out("signal", x)
f, p = signal.periodogram(x, fs=10.0, window="hann", detrend="constant", scaling="density")
out("periodogram hann density", p)
f, p = signal.welch(x, fs=10.0, window="hann", nperseg=64, noverlap=32, detrend="linear", scaling="spectrum")
out("welch hann 64/32 linear spectrum", p)
h = signal.hilbert(x[:64])
out("hilbert 64", h.real, h.imag)
c = signal.czt(x[:64], m=20, w=np.exp(-2j * np.pi * 0.01), a=np.exp(2j * np.pi * 0.05))
out("czt 20 points", c.real, c.imag)
t = np.sort(rng.uniform(0, 20, 80))
y = np.sin(2 * np.pi * 0.7 * t) + rng.normal(0, 0.3, 80)
freqs = np.linspace(0.1, 6.0, 30)
out("lomb t", t)
out("lomb y", y)
out("lombscargle unnormalized", signal.lombscargle(t, y, freqs, precenter=True))
out("upfirdn up 3", signal.upfirdn([0.25, 0.5, 1.0, 0.5, 0.25], x[:20], up=3))
h = signal.firwin(31, 0.2)
out("upfirdn up 3 down 2", signal.upfirdn(3 * h, x[:40], up=3, down=2))
from scipy.linalg import solve_toeplitz
r = np.array([np.dot(x[: 256 - k], x[k:]) / 256 for k in range(9)])
out("levinson order 8", solve_toeplitz(r[:8], r[1:9]))
