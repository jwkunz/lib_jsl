"""SciPy references for release 2.14.0: multivariate normal and t distribution functions (Genz
QMC; the bivariate normal exactly), singular covariances, Dunnett's test, MultivariateNormalQMC
and Poisson-disk sampling. QMC results are random: each case also records a high-accuracy value
(many more points, or the exact bivariate formula) as the reference. Writes mv214_reference.txt."""
import numpy as np
from scipy import stats
from scipy.stats import qmc

out = []
def put(key, *lists):
    out.append(key + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))

rng = np.random.default_rng(214)
def corr(d):
    a = rng.normal(size=(d, d + 2))
    c = a @ a.T
    s = np.sqrt(np.diag(c))
    return c / s[:, None] / s[None, :]

# Bivariate: exact.
bv = []
for h, k, r in [(0.3, -0.5, 0.2), (-1.0, 2.0, -0.6), (1.5, 1.5, 0.95), (-2.0, -3.0, -0.97), (0.0, 0.7, 0.999), (2.5, -0.4, 0.5)]:
    p = stats.multivariate_normal([0, 0], [[1, r], [r, 1]]).cdf([h, k])
    bv += [h, k, r, p]
put("bvn", bv)

for d, name in [(3, "mvn3"), (5, "mvn5"), (8, "mvn8")]:
    c = corr(d)
    x = rng.uniform(-0.5, 1.5, d)
    lo = x - rng.uniform(1.0, 3.0, d)
    m = stats.multivariate_normal(np.zeros(d), c)
    scipy_val = m.cdf(x, lower_limit=lo, rng=1)
    truth = stats.multivariate_normal.cdf(x, np.zeros(d), c, lower_limit=lo, abseps=1e-9, maxpts=40_000_000 * d, rng=2)
    put(name, c, x, lo, [scipy_val, truth])

for d, df, name in [(2, 4.0, "mvt2"), (3, 3.0, "mvt3"), (5, 10.0, "mvt5")]:
    c = corr(d)
    x = rng.uniform(0.0, 2.0, d)
    t = stats.multivariate_t(np.zeros(d), c, df=df)
    scipy_val = t.cdf(x, random_state=1)
    truth = np.mean([t.cdf(x, maxpts=2_000_000 * d, random_state=s) for s in range(4)])
    put(name, c, x, [df, scipy_val, truth])

# Singular covariance.
v = rng.normal(size=(4, 2))
cs = v @ v.T
ms = stats.multivariate_normal(np.array([1.0, 0.0, -1.0, 0.5]), cs, allow_singular=True)
xs = np.array([1.0, 0.0, -1.0, 0.5]) + v @ np.array([0.3, -0.7])
put("singular", cs, xs, [ms.logpdf(xs), ms.entropy(), ms.logpdf(xs + np.array([0.0, 0.1, 0.0, 0.0]))])

# Dunnett: SciPy's documentation example.
control = np.array([7.40, 8.50, 7.20, 8.24, 9.84, 8.32])
drug_a = np.array([9.76, 8.80, 7.68, 9.36])
drug_b = np.array([12.80, 9.68, 12.16, 9.20, 10.55])
for alt in ["two-sided", "greater", "less"]:
    r = stats.dunnett(drug_a, drug_b, control=control, alternative=alt, rng=1)
    put(f"dunnett {alt}", r.statistic, r.pvalue)

# MultivariateNormalQMC with an unscrambled Sobol engine.
cq = np.array([[2.0, 0.6, 0.1], [0.6, 1.0, -0.3], [0.1, -0.3, 0.5]])
g = qmc.MultivariateNormalQMC([1.0, -1.0, 0.0], cq, engine=qmc.Sobol(3, scramble=False))
put("mvnqmc", g.random(16))
g = qmc.MultivariateNormalQMC([1.0, -1.0, 0.0], cq, inv_transform=False, engine=qmc.Sobol(4, scramble=False))
put("mvnqmc bm", g.random(16)[1:])

# Poisson disk: the number of points fill_space places, over seeds.
counts = [len(qmc.PoissonDisk(2, radius=0.05, rng=s).fill_space()) for s in range(5)]
counts_s = [len(qmc.PoissonDisk(2, radius=0.05, hypersphere="surface", rng=s).fill_space()) for s in range(5)]
put("poisson counts", counts, counts_s)
open("mv214_reference.txt", "w").write("\n".join(out) + "\n")
