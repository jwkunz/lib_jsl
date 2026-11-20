"""SciPy references for release 2.15.0: whiten, inconsistent, optimal_leaf_ordering and
leaves_list (scipy.cluster), the ZerosPolesGain system type (scipy.signal) and the CODATA table
(scipy.constants). Writes sys215_reference.txt."""
import numpy as np
from scipy import signal
from scipy.cluster import hierarchy, vq
from scipy.constants import physical_constants, find
from scipy.spatial.distance import pdist, squareform

out = []
def put(key, *lists):
    out.append(key + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))

rng = np.random.default_rng(215)
# Three loose groups in the plane, with features on different scales.
x = np.vstack([rng.normal([0, 0], [1.0, 0.3], (12, 2)), rng.normal([4, 1], [0.8, 0.4], (10, 2)), rng.normal([1, 3], [1.2, 0.2], (14, 2))])
x[:, 1] *= 10.0
put("points", x)
put("whiten", vq.whiten(x))
y = pdist(x)
for method in ["single", "complete", "average", "ward"]:
    z = hierarchy.linkage(y, method)
    put(f"{method} z", z)
    for d in [1, 2, 3, 4]:
        put(f"{method} inconsistent {d}", hierarchy.inconsistent(z, d))
    put(f"{method} leaves", hierarchy.leaves_list(z))
    o = hierarchy.optimal_leaf_ordering(z, y)
    lo = hierarchy.leaves_list(o)
    d = squareform(y)
    put(f"{method} olo", lo, [sum(d[lo[i], lo[i + 1]] for i in range(len(lo) - 1))])

# Zeros, poles and gain.
z, p, k = signal.butter(6, 2.0, analog=True, output="zpk")
z2, p2, k2 = signal.cheby1(5, 1.0, 3.0, btype="high", analog=True, output="zpk")
w = np.logspace(-1, 2, 40)
t = np.linspace(0, 8, 81)
for name, (zz, pp, kk) in [("butter", (z, p, k)), ("cheby", (z2, p2, k2))]:
    sys = signal.ZerosPolesGain(zz, pp, kk)
    put(f"{name} zpk", np.real(zz), np.imag(zz), np.real(pp), np.imag(pp), [kk])
    _, h = signal.freqresp(sys, w)
    _, mag, ph = signal.bode(sys, w)
    put(f"{name} freqresp", h.real, h.imag, mag, ph)
    _, ys = signal.step(sys, T=t)
    _, yi = signal.impulse(sys, T=t)
    put(f"{name} step", ys, yi)
    for method in ["zoh", "bilinear", "foh"]:
        dsys = sys.to_discrete(0.1, method=method)
        _, yd = signal.dstep(dsys, n=60)
        _, yi = signal.dimpulse(dsys, n=60)
        _, hd = signal.dfreqresp(dsys, w=np.linspace(0, np.pi, 30))
        put(f"{name} {method}", np.real(dsys.poles), np.imag(dsys.poles), np.real(dsys.zeros), np.imag(dsys.zeros), [dsys.gain], np.ravel(yd[0]), np.ravel(yi[0]), hd.real, hd.imag)
put("w", w)
put("t", t)

# CODATA: values and uncertainties in SciPy's order, and find() counts.
vals = [v[0] for v in physical_constants.values()]
unc = [v[2] for v in physical_constants.values()]
put("codata", vals, unc)
put("find", [len(find(s)) for s in ["", "radius", "magn", "electron", "Planck"]])

open("data/sys215_reference.txt", "w").write("\n".join(out) + "\n")
