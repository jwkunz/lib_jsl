"""SciPy references for release 2.20.0: Mathieu, parabolic cylinder and spheroidal wave functions.
Writes wave220_reference.txt; each line holds argument tuples and SciPy's (value, derivative)."""
import numpy as np
import scipy.special as sp

out = []
def put(key, *lists):
    out.append(key + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))

ms = range(0, 7)
qs = [-12.0, -1.5, 0.0, 0.3, 1.5, 5.0, 12.0, 30.0]
xs = [0.0, 17.0, 45.0, 73.0, 120.0, 211.0]
rows = []
for m in ms:
    for q in qs:
        a = sp.mathieu_a(m, q)
        b = sp.mathieu_b(m, q) if m > 0 else np.nan
        rows.append([m, q, a, b])
put("mathieu ab", rows)
for name, f, m0 in [("cem", sp.mathieu_cem, 0), ("sem", sp.mathieu_sem, 1)]:
    rows = []
    for m in range(m0, 7):
        for q in qs:
            for x in xs:
                y, dy = f(m, q, x)
                rows.append([m, q, x, y, dy])
    put(f"mathieu {name}", rows)
zs = [0.1, 0.5, 1.0, 1.8]
for name, f, m0 in [("modcem1", sp.mathieu_modcem1, 0), ("modcem2", sp.mathieu_modcem2, 0), ("modsem1", sp.mathieu_modsem1, 1), ("modsem2", sp.mathieu_modsem2, 1)]:
    rows = []
    for m in range(m0, 6):
        for q in [0.3, 1.5, 5.0, 12.0]:
            for z in zs:
                y, dy = f(m, q, z)
                rows.append([m, q, z, y, dy])
    put(f"mathieu {name}", rows)
# Parabolic cylinder functions: SciPy and mpmath (50 digits).
import mpmath as mp
mp.mp.dps = 50
vs = [-3.3, -1.0, -0.4, 0.0, 0.5, 1.0, 2.5, 4.0, 7.2]
xs = [-6.0, -3.0, -1.2, 0.0, 0.7, 2.0, 4.5, 7.0, 10.0]
for name, f, mpf in [("pbdv", sp.pbdv, lambda v, x: mp.pcfd(v, x)), ("pbvv", sp.pbvv, lambda v, x: mp.pcfv(-v - 0.5, x))]:
    rows = []
    for v in vs:
        for x in xs:
            y, dy = f(v, x)
            try:
                t = mpf(v, x); dt = mp.diff(lambda z: mpf(v, z), x)
            except Exception:
                t, dt = mp.nan, mp.nan
            rows.append([v, 0.0, x, y, dy, float(t), float(dt)])
    put(name, rows)
rows = []
for a in [-4.0, -1.5, 0.0, 0.5, 2.0, 4.5]:
    for x in [-4.5, -2.0, -0.5, 0.0, 1.0, 3.0, 4.9]:
        y, dy = sp.pbwa(a, x)
        t = mp.pcfw(a, x); dt = mp.diff(lambda z: mp.pcfw(a, z), x)
        rows.append([a, 0.0, x, y, dy, float(t), float(dt)])
put("pbwa", rows)
# Spheroidal wave functions.
cs = [0.5, 1.5, 4.0, 8.0]
mns = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (1, 4), (2, 3), (3, 6)]
rows = []
for m, n in mns:
    for c in cs:
        rows.append([m, n, c, sp.pro_cv(m, n, c), sp.obl_cv(m, n, c)])
put("spheroidal cv", rows)
for name, f, args in [("pro_ang1", sp.pro_ang1, [-0.8, -0.3, 0.0, 0.4, 0.9]), ("obl_ang1", sp.obl_ang1, [-0.8, -0.3, 0.0, 0.4, 0.9]),
                      ("pro_rad1", sp.pro_rad1, [1.1, 1.5, 2.5, 5.0]), ("obl_rad1", sp.obl_rad1, [0.1, 0.8, 2.0, 5.0])]:
    rows = []
    for m, n in mns:
        for c in cs:
            for x in args:
                y, dy = f(m, n, c, x)
                rows.append([m, n, c, x, y, dy])
    put(name, rows)
open("data/wave220_reference.txt", "w").write("\n".join(out) + "\n")
