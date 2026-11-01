"""Reference values for the special functions chapter of the lib_jsl book.

Reads special_functions_reference.csv (name, space-separated arguments, reference), recomputes every
reference value with mpmath at 50 significant digits, drops rows whose reference is not a finite,
nonzero real number (relative error is then undefined), and writes the file back. The book
generator evaluates the library at the same arguments and tabulates the relative errors.

Run with:  python3 -I book/generator/data/make_refs.py   (needs mpmath)
"""
import math, os
import mpmath as mp
mp.mp.dps = 50
I = lambda x: int(round(x))

def tcdf(t, nu):
    t = mp.mpf(t); nu = mp.mpf(nu)
    tail = mp.betainc(nu/2, mp.mpf(1)/2, 0, nu/(nu+t*t), regularized=True)/2
    return 1 - tail if t > 0 else tail

def fd(j, x):
    j = mp.mpf(j); x = mp.mpf(x)
    f = lambda t: t**j/(1+mp.exp(t-x))
    pts = [0, max(x, 0) + 0, max(x, 0) + 50 + abs(x)]
    return mp.quad(f, [0, max(mp.mpf(0), x), max(mp.mpf(0), x) + 60, mp.inf]) / mp.gamma(j+1)

def realylm(l, m, t, p):
    if m == 0: return mp.re(mp.spherharm(l, 0, t, p))
    if m > 0: return mp.sqrt(2)*(-1)**m*mp.re(mp.spherharm(l, m, t, p))
    return mp.sqrt(2)*(-1)**m*mp.im(mp.spherharm(l, -m, t, p))

R = {
 'gamma': lambda x: mp.gamma(x), 'lngamma': lambda x: mp.loggamma(x), 'digamma': lambda x: mp.digamma(x),
 'erf': mp.erf, 'erfc': mp.erfc, 'erfinv': mp.erfinv, 'ei': mp.ei, 'e1': mp.e1, 'si': mp.si, 'ci': mp.ci,
 'dawson': lambda x: mp.sqrt(mp.pi)/2*mp.exp(-mp.mpf(x)**2)*mp.erfi(x),
 'airyai': mp.airyai, 'airybi': mp.airybi,
 'fresnels': mp.fresnels, 'fresnelc': mp.fresnelc, 'zeta': mp.zeta,
 'lambertw0': lambda y: mp.re(mp.lambertw(y, 0)), 'lambertwm1': lambda y: mp.re(mp.lambertw(y, -1)),
 'beta': mp.beta, 'lnbeta': lambda a, b: mp.log(mp.beta(a, b)),
 'gammainc': lambda a, x: mp.gammainc(a, 0, x, regularized=True), 'gammaincc': lambda a, x: mp.gammainc(a, x, mp.inf, regularized=True),
 'betainc': lambda a, b, x: mp.betainc(a, b, 0, x, regularized=True),
 'en': lambda n, x: mp.expint(I(n), x),
 'besselj': mp.besselj, 'bessely': mp.bessely, 'besseli': mp.besseli, 'besselk': mp.besselk,
 'besseljn': lambda n, x: mp.besselj(I(n), x), 'besselyn': lambda n, x: mp.bessely(I(n), x),
 'besselin': lambda n, x: mp.besseli(I(n), x), 'besselkn': lambda n, x: mp.besselk(I(n), x),
 'sphj': lambda n, x: mp.sqrt(mp.pi/(2*mp.mpf(x)))*mp.besselj(I(n)+mp.mpf(1)/2, x),
 'sphy': lambda n, x: mp.sqrt(mp.pi/(2*mp.mpf(x)))*mp.bessely(I(n)+mp.mpf(1)/2, x),
 'fermidirac': fd,
 'ellipk': lambda k: mp.ellipk(mp.mpf(k)**2), 'ellipe': lambda k: mp.ellipe(mp.mpf(k)**2),
 'ellippi': lambda n, k: mp.ellippi(n, mp.mpf(k)**2), 'ellipf': lambda ph, k: mp.ellipf(ph, mp.mpf(k)**2),
 'ellipeinc': lambda ph, k: mp.ellipe(ph, mp.mpf(k)**2), 'ellippiinc': lambda ph, n, k: mp.ellippi(n, ph, mp.mpf(k)**2),
 'sn': lambda u, m: mp.ellipfun('sn', u, m=m), 'cn': lambda u, m: mp.ellipfun('cn', u, m=m), 'dn': lambda u, m: mp.ellipfun('dn', u, m=m),
 'hyp2f1': mp.hyp2f1, 'hyp1f1': mp.hyp1f1, 'hyp0f1': mp.hyp0f1,
 'polylog': lambda s, z: mp.re(mp.polylog(s, z)) if s == int(s) else mp.re(mp.polylog(s, z)),
 'hurwitz': mp.zeta,
 'legendre': lambda n, x: mp.legendre(I(n), x), 'chebyt': lambda n, x: mp.chebyt(I(n), x), 'chebyu': lambda n, x: mp.chebyu(I(n), x),
 'hermite': lambda n, x: mp.hermite(I(n), x), 'hermitee': lambda n, x: 2**(-mp.mpf(I(n))/2)*mp.hermite(I(n), mp.mpf(x)/mp.sqrt(2)),
 'laguerre': lambda n, a, x: mp.laguerre(I(n), a, x), 'jacobi': lambda n, a, b, x: mp.jacobi(I(n), a, b, x),
 'legenp': lambda l, m, x: mp.legenp(I(l), I(m), x),
 'ylmre': lambda l, m, t, p: mp.re(mp.spherharm(I(l), I(m), t, p)), 'ylmim': lambda l, m, t, p: mp.im(mp.spherharm(I(l), I(m), t, p)),
 'ylmreal': lambda l, m, t, p: realylm(I(l), I(m), t, p),
 'normcdf': mp.ncdf, 'tcdf': tcdf,
 'fcdf': lambda x, a, b: mp.betainc(mp.mpf(a)/2, mp.mpf(b)/2, 0, a*x/(a*x+b), regularized=True),
 'chi2sf': lambda x, k: mp.gammainc(mp.mpf(k)/2, mp.mpf(x)/2, mp.inf, regularized=True),
 'poispmf': lambda k, l: mp.exp(I(k)*mp.log(l) - l - mp.loggamma(I(k)+1)) if l > 0 else 0,
 'poiscdf': lambda k, l: mp.gammainc(I(k)+1, l, mp.inf, regularized=True),
 'binpmf': lambda k, n, q: mp.binomial(I(n), I(k))*mp.mpf(q)**I(k)*(1-mp.mpf(q))**(I(n)-I(k)),
 'bincdf': lambda k, n, q: mp.betainc(I(n)-I(k), I(k)+1, 0, 1-mp.mpf(q), regularized=True) if I(k) < I(n) else 1,
 'lwre': lambda k, a, b: mp.re(mp.lambertw(mp.mpc(a, b), I(k))), 'lwim': lambda k, a, b: mp.im(mp.lambertw(mp.mpc(a, b), I(k))),
 'ixlogx': lambda y: mp.exp(mp.re(mp.lambertw(y, 0))), 'ixlogxlo': lambda y: mp.exp(mp.re(mp.lambertw(y, -1))),
 'factf': lambda n: mp.factorial(I(n)), 'lnfact': lambda n: mp.loggamma(I(n)+1), 'lnbinom': lambda n, k: mp.log(mp.binomial(I(n), I(k))),
}

def quantile(cdf, q, lo, hi):
    """Bisection at 50 digits for the x with cdf(x) = q, cdf increasing on [lo, hi]."""
    q, lo, hi = mp.mpf(q), mp.mpf(lo), mp.mpf(hi)
    for _ in range(400):
        mid = (lo + hi) / 2
        if cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

def hermite_he(n, x):
    """Probabilists' Hermite polynomial by its recurrence (exact zeros stay exact)."""
    x = mp.mpf(x)
    a, b = mp.mpf(1), x
    if n == 0:
        return a
    for k in range(1, n):
        a, b = b, x * b - k * a
    return b

R['hermitee'] = lambda n, x: hermite_he(I(n), x)
R['normq'] = lambda q: mp.sqrt(2)*mp.erfinv(2*mp.mpf(q)-1)
R['chi2q'] = lambda q, k: quantile(lambda x: mp.gammainc(mp.mpf(k)/2, 0, x/2, regularized=True), q, 0, 2000)
R['tq'] = lambda q, k: quantile(lambda t: tcdf(t, k), q, -1e12, 1e12)

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'special_functions_reference.csv')
lines = open(path).read().splitlines()
out = [lines[0]]
for line in lines[1:]:
    name, args, _ = line.split(',')
    a = [float(v) for v in args.split()]
    if name == 'polylog' and a[1] == 1.0 and a[0] <= 1.0:
        continue  # the series diverges; mpmath returns the analytic continuation instead
    if name == 'polylog' and a[1] < -1.0 and a[0] <= 0.0 and a[0] != int(a[0]):
        continue  # documented as unsupported (the library returns NaN)
    if name == 'tq' and a[0] == 0.5:
        continue  # exactly zero
    if name == 'cn' and abs(float(R['cn'](*a))) < 1e-12:
        continue  # at a zero of cn, relative error is meaningless
    try:
        r = complex(R[name](*a))
    except Exception:
        continue
    if name not in ('lwre', 'lwim') and abs(r.imag) > 1e-30*max(1.0, abs(r.real)):
        continue
    r = r.real
    if not math.isfinite(r) or abs(r) < 1e-280:
        continue
    out.append(f"{name},{args},{r!r}")
open(path, 'w').write("\n".join(out) + "\n")
print(len(out) - 1, "reference values")
