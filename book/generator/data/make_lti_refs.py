"""Reference results for the entry on continuous-time systems, from `scipy.signal` (the book used
SciPy 1.17.1).

    python book/generator/data/make_lti_refs.py > book/generator/data/lti_reference.txt

Inputs are written first (`data <name>|values`, matrices as `rows;cols;values`), then results as
`case|values` (`;` between lists; complex values as real parts then imaginary parts).
"""

import numpy as np
from scipy import signal


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


def matrix(name, a):
    a = np.atleast_2d(np.asarray(a, dtype=float))
    print(f"data {name}|{a.shape[0]};{a.shape[1]};" + ",".join(repr(float(v)) for v in a.ravel()))
    return a


# A transfer function and its state-space form.
num, den = [2.0, 3.0, 5.0], [1.0, 0.4, 4.0, 1.2]
out("data num", num)
out("data den", den)
A, B, C, D = signal.tf2ss(num, den)
out("tf2ss", A, B, C, D)

# A 3-state, 2-input, 2-output system.
A = matrix("A", [[-0.5, 2.0, 0.0], [-2.0, -0.5, 1.0], [0.0, -1.0, -1.5]])
B = matrix("B", [[1.0, 0.0], [0.0, 0.5], [0.3, 1.0]])
C = matrix("C", [[1.0, 0.0, 0.5], [0.0, 1.0, -1.0]])
D = matrix("D", [[0.0, 0.1], [0.2, 0.0]])
for i in range(2):
    n_, d_ = signal.ss2tf(A, B, C, D, input=i)
    out(f"ss2tf {i}", n_, d_)
t = np.linspace(0.0, 10.0, 201)
u = np.c_[np.sin(1.3 * t), np.where(t < 4.0, 1.0, -0.5)]
out("data t", t)
x0 = [0.2, -0.1, 0.4]
for interp in (False, True):
    _, y, x = signal.lsim((A, B, C, D), u, t, X0=x0, interp=interp)
    out(f"lsim {int(interp)}", y, x)
t2 = np.linspace(0.5, 10.5, 101)
_, y, _ = signal.lsim((A, B, C, D), u[:101], t2, X0=x0, interp=True)
out("lsim delayed", y)

# Step, impulse and Bode for the SISO transfer function.
ts = np.linspace(0.0, 20.0, 401)
_, ys = signal.step((num, den), T=ts)
_, yi = signal.impulse((num, den), T=ts)
out("step", ys)
out("impulse", yi)
w = np.logspace(-2, 2, 161)
out("data w", w)
_, mag, phase = signal.bode((num, den), w=w)
out("bode", mag, phase)

# Discretisation.
for method in ("zoh", "foh", "bilinear", "euler", "backward_diff"):
    ad, bd, cd, dd, _ = signal.cont2discrete((A, B, C, D), 0.25, method=method)
    out(f"c2d {method}", ad, bd, cd, dd)
ad, bd, cd, dd, _ = signal.cont2discrete((A, B, C, np.zeros((2, 2))), 0.25, method="impulse")
out("c2d impulse", ad, bd, cd, dd)
ad, bd, cd, dd, _ = signal.cont2discrete((A, B, C, D), 0.25, method="gbt", alpha=0.3)
out("c2d gbt", ad, bd, cd, dd)

# Pole placement: a linearised inverted pendulum on a cart.
Ap = matrix("Ap", [[0, 1, 0, 0], [0, 0, -0.98, 0], [0, 0, 0, 1], [0, 0, 21.56, 0]])
Bp = matrix("Bp", [[0], [1], [0], [-2]])
Bp2 = matrix("Bp2", [[0, 0], [1, 0], [0, 0], [-2, 1]])
P = np.array([-2 + 1j, -2 - 1j, -3, -4])
out("data poles", P.real, P.imag)
r = signal.place_poles(Ap, Bp, P)
out("place single", r.gain_matrix)
r2 = signal.place_poles(Ap, Bp2, P)
out("place multi", r2.gain_matrix)
out("place multi cond", [np.linalg.cond(r2.X)])
