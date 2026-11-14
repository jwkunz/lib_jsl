"""Reference results for partial fractions, zoom FFT, zero-pole-gain conversions and waveforms
(release 1.20.0), from SciPy 1.17.1.

    python book/generator/data/make_signal2_refs.py > book/generator/data/signal2_reference.txt

Complex results are written as real parts then imaginary parts.
"""

import numpy as np
from scipy import signal

rng = np.random.default_rng(1200)


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


def cout(name, z):
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    out(name, z.real, z.imag)


# residue: simple, improper, repeated, complex with a triple pole.
residue_cases = {
    "simple": ([1, 2], [1, 4, 3]),
    "improper": ([2, 3, 4, 5], [1, 3, 2]),
    "repeated": ([1, 0, 1], np.poly([-1, -1, -2])),
    "complex": ([1, 2, 3], np.poly([-0.5, -0.5, -0.5, -1 + 2j, -1 - 2j]).real),
}
for name, (b, a) in residue_cases.items():
    r, p, k = signal.residue(b, a)
    cout(f"residue {name} r", r)
    cout(f"residue {name} p", p)
    out(f"residue {name} k", np.atleast_1d(k).real if len(k) else [])
    bb, aa = signal.invres(r, p, k)
    cout(f"invres {name} b", bb)
    cout(f"invres {name} a", aa)

residuez_cases = {
    "simple": ([1, -0.5, 0.25], [1, -0.9, 0.2]),
    "repeated": ([1, 2], [1, -1, 0.25]),
    "improper": ([1, 2, 3, 4], [1, -0.5]),
}
for name, (b, a) in residuez_cases.items():
    r, p, k = signal.residuez(b, a)
    cout(f"residuez {name} r", r)
    cout(f"residuez {name} p", p)
    out(f"residuez {name} k", np.atleast_1d(k).real if len(k) else [])
    bb, aa = signal.invresz(r, p, k)
    cout(f"invresz {name} b", bb)
    cout(f"invresz {name} a", aa)

# deconvolve.
h = np.round(rng.normal(size=5), 4)
h[0] = 1.5
x = np.round(rng.normal(size=40), 4)
y = np.convolve(h, x)
y2 = y + np.round(rng.normal(scale=0.01, size=y.size), 6)
out("input h", h)
out("input x", x)
out("input y2", y2)
q, r = signal.deconvolve(y2, h)
out("deconvolve q", q)
out("deconvolve r", r)

# tf2zpk, zpk2tf, ss2zpk.
z, p, k = signal.tf2zpk([2, 3, 1], [1, 0.5, 3, 1])
cout("tf2zpk z", z)
cout("tf2zpk p", p)
out("tf2zpk k", [k])
bb, aa = signal.zpk2tf(z, p, k)
out("zpk2tf b", bb)
out("zpk2tf a", aa)
A, B, C, D = signal.tf2ss([1, 3, 2], [1, 2, 10, 4])
z, p, k = signal.ss2zpk(A, B, C, D)
cout("ss2zpk z", z)
cout("ss2zpk p", p)
out("ss2zpk k", [k])
T, _ = signal.step((A, B, C, D))
out("default times", T)

# zoom_fft.
fs = 1000.0
n = np.arange(256)
xz = np.cos(2 * np.pi * 100.3 * n / fs) + 0.5 * np.cos(2 * np.pi * 101.7 * n / fs + 0.3)
for endpoint in (False, True):
    cout(f"zoom_fft {int(endpoint)}", signal.zoom_fft(xz, [95, 110], m=128, fs=fs, endpoint=endpoint))

# Waveforms.
t = np.linspace(-1, 1, 201)
for w in (1.0, 0.5, 0.25):
    out(f"sawtooth {w:g}", signal.sawtooth(2 * np.pi * 3 * t, w))
out("square 0.3", signal.square(2 * np.pi * 3 * t, 0.3))
yi, yq, ye = signal.gausspulse(t, fc=5, bw=0.5, retquad=True, retenv=True)
out("gausspulse", yi, yq, ye)
out("gausspulse cutoff", [signal.gausspulse("cutoff", fc=5, bw=0.5, tpr=-60)])
tc = np.linspace(0, 2, 201)
for method, vz in (("linear", True), ("quadratic", True), ("quadratic", False), ("logarithmic", True), ("hyperbolic", True)):
    out(f"chirp {method} {int(vz)}", signal.chirp(tc, f0=10, t1=2, f1=40, method=method, phi=30, vertex_zero=vz))
out("sweep_poly", signal.sweep_poly(tc, [0.025, -0.36, 1.25, 2.0], phi=10))
