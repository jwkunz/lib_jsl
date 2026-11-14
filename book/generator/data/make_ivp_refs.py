"""Reference results for the entry on Radau, LSODA and event location, from
`scipy.integrate.solve_ivp` (the book used SciPy 1.17.1).

    python book/generator/data/make_ivp_refs.py > book/generator/data/ivp_reference.txt

For each problem: a reference solution at tight tolerance (`exact <problem>`), and for each method
and tolerance SciPy's steps, function evaluations, Jacobian evaluations, LU factorisations and
final state (`<method> <problem> <rtol>|steps,nfev,njev,nlu;y_final`). Event times are written as
`events <problem> <method>|times`.
"""

import numpy as np
from scipy.integrate import solve_ivp


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


def vdp(mu):
    f = lambda t, y: [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]
    j = lambda t, y: [[0.0, 1.0], [-2 * mu * y[0] * y[1] - 1.0, mu * (1 - y[0] ** 2)]]
    return f, j


def robertson():
    f = lambda t, y: [-0.04 * y[0] + 1e4 * y[1] * y[2], 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2, 3e7 * y[1] ** 2]
    j = lambda t, y: [[-0.04, 1e4 * y[2], 1e4 * y[1]], [0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]], [0.0, 6e7 * y[1], 0.0]]
    return f, j


def arenstorf():
    m = 0.012277471
    mp = 1 - m
    def f(t, y):
        d1 = ((y[0] + m) ** 2 + y[1] ** 2) ** 1.5
        d2 = ((y[0] - mp) ** 2 + y[1] ** 2) ** 1.5
        return [y[2], y[3], y[0] + 2 * y[3] - mp * (y[0] + m) / d1 - m * (y[0] - mp) / d2,
                y[1] - 2 * y[2] - mp * y[1] / d1 - m * y[1] / d2]
    return f, None


problems = {
    "vdp": (vdp(100.0), (0.0, 200.0), [2.0, 0.0]),
    "robertson": (robertson(), (0.0, 1e5), [1.0, 0.0, 0.0]),
    "arenstorf": (arenstorf(), (0.0, 17.0652165601579625588917206249), [0.994, 0.0, 0.0, -2.00158510637908252240537862224]),
}

for name, ((f, j), span, y0) in problems.items():
    ref = solve_ivp(f, span, y0, method="Radau", jac=j, rtol=1e-12, atol=1e-14) if name != "arenstorf" else \
        solve_ivp(f, span, y0, method="DOP853", rtol=1e-13, atol=1e-14)
    out(f"exact {name}", ref.y[:, -1])
    for method in ("Radau", "LSODA"):
        for rtol in (1e-4, 1e-7):
            atol = rtol * 1e-3
            kw = {"jac": j} if j is not None else {}
            s = solve_ivp(f, span, y0, method=method, rtol=rtol, atol=atol, **kw)
            out(f"{method} {name} {rtol:g}", [len(s.t) - 1, s.nfev, s.njev, s.nlu], s.y[:, -1])

# Events: the zero crossings of x in the Arenstorf orbit, and a terminal event for Van der Pol.
f, _ = arenstorf()
span, y0 = problems["arenstorf"][1], problems["arenstorf"][2]
cross = lambda t, y: y[1]
ref = solve_ivp(f, span, y0, method="DOP853", rtol=1e-13, atol=1e-14, events=cross)
out("events arenstorf exact", ref.t_events[0])
for method in ("Radau", "LSODA"):
    s = solve_ivp(f, span, y0, method=method, rtol=1e-7, atol=1e-10, events=cross)
    out(f"events arenstorf {method}", s.t_events[0])
(f, j), span, y0 = problems["vdp"]
hit = lambda t, y: y[0] + 1.5
hit.terminal = True
hit.direction = -1
ref = solve_ivp(f, span, y0, method="Radau", jac=j, rtol=1e-12, atol=1e-14, events=hit)
out("events vdp exact", ref.t_events[0])
for method in ("Radau", "LSODA"):
    s = solve_ivp(f, span, y0, method=method, jac=j, rtol=1e-7, atol=1e-10, events=hit)
    out(f"events vdp {method}", s.t_events[0])
