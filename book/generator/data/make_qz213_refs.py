"""SciPy references for release 2.13.0: real QZ, ordqz and sorted Schur forms (scipy.linalg),
and hfft/ihfft and the fast Hankel transform (scipy.fft). Writes qz213_reference.txt.

The generalised Schur forms are not unique, so the comparisons use what is: the eigenvalues,
the number of selected eigenvalues, and the orthogonal projectors onto the leading invariant
or deflating subspaces."""
import numpy as np
from scipy import linalg, fft

out = []
def put(key, *lists):
    out.append(key + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))

rng = np.random.default_rng(213)
cases = {}
for n in [6, 12, 30]:
    cases[f"rand{n}"] = (rng.normal(size=(n, n)), rng.normal(size=(n, n)))
a = rng.normal(size=(8, 8)); b = rng.normal(size=(8, 8)); b[[2, 5]] = 0.0
cases["singular8"] = (a, b)
for name, (a, b) in cases.items():
    put(f"{name} a", a); put(f"{name} b", b)
    aa, bb, q, z = linalg.qz(a, b, output="real")
    w = linalg.eigvals(a, b, homogeneous_eigvals=True)
    put(f"{name} eig", w[0].real, w[0].imag, w[1].real)
    for sort in ["lhp", "rhp", "iuc", "ouc"]:
        r = linalg.ordqz(a, b, sort=sort, output="real")
        aa, bb, al, be, q, z = r
        lam = np.where(be != 0, al / np.where(be != 0, be, 1), np.inf)
        if sort == "lhp": sel = (be != 0) & (lam.real < 0)
        elif sort == "rhp": sel = (be != 0) & (lam.real > 0)
        elif sort == "iuc": sel = (be != 0) & (np.abs(lam) <= 1)
        else: sel = ~((al == 0) & (be == 0)) & ((be == 0) | (np.abs(lam) > 1))
        s = int(sel.sum())
        pz = z[:, :s] @ z[:, :s].T; pq = q[:, :s] @ q[:, :s].T
        put(f"{name} ordqz {sort}", [s], pz, pq)
        if name != "singular8":
            t, zs, sdim = linalg.schur(a, output="real", sort=sort)
            put(f"{name} schur {sort}", [sdim], zs[:, :sdim] @ zs[:, :sdim].T)
            tc, zc, sdc = linalg.schur(a, output="complex", sort=sort)
            pc = zc[:, :sdc] @ zc[:, :sdc].conj().T
            put(f"{name} schurc {sort}", [sdc], pc.real, pc.imag)

x = rng.normal(size=9) + 1j * rng.normal(size=9)
put("hfft x", x.real, x.imag)
put("hfft", fft.hfft(x), fft.hfft(x, 13), fft.hfft(x, 6))
y = rng.normal(size=11)
put("ihfft y", y)
h = fft.ihfft(y)
put("ihfft", h.real, h.imag)

n, dln = 128, 0.05
jc = (n - 1) / 2
r = np.exp((np.arange(n) - jc) * dln)
for name, mu, bias, f in [("gauss0", 0.0, 0.0, lambda r: r * np.exp(-r ** 2 / 2)),
                          ("pow1", 1.0, 0.0, lambda r: r ** 2 * np.exp(-r)),
                          ("bias", 0.5, 0.4, lambda r: np.exp(-(np.log(r)) ** 2)),
                          ("neg", -0.5, -0.3, lambda r: 1 / (1 + r ** 2))]:
    a = f(r)
    off = fft.fhtoffset(dln, mu, initial=0.3, bias=bias)
    A = fft.fht(a, dln, mu, offset=off, bias=bias)
    put(f"fht {name}", [mu, bias, off], a, A, fft.ifht(A, dln, mu, offset=off, bias=bias))
open("qz213_reference.txt", "w").write("\n".join(out) + "\n")
