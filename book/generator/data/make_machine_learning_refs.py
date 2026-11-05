"""Reference fits for the machine learning chapters, computed with SciPy's linear algebra,
optimisers and distributions (the book used SciPy 1.17.1). Each library objective is reproduced
exactly, so the references are the same mathematical solutions computed independently.

    python book/generator/data/make_machine_learning_refs.py > book/generator/data/machine_learning_reference.txt

Data sets are written as `data <name>|rows;cols;values` and results as `case|values`.
"""

import numpy as np
from scipy import cluster, linalg, optimize, special, stats

rng = np.random.default_rng(2026)


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


def data(name, a):
    a = np.round(np.atleast_2d(a), 6)
    print(f"data {name}|{a.shape[0]};{a.shape[1]};" + ",".join(repr(float(v)) for v in a.ravel()))
    return a


# Regression: 40 samples, 6 features (one nearly a copy of another).
X = rng.normal(size=(40, 6))
X[:, 3] = X[:, 0] + 0.1 * X[:, 3]
X = data("X", X)
y = data("y", 3 * X[:, 0] - 2 * X[:, 1] + 1.5 * X[:, 2] + 0.5 + rng.normal(size=40))[0]
A = np.column_stack([X, np.ones(40)])
out("ols", linalg.lstsq(A, y)[0])
xm, ym = X.mean(axis=0), y.mean()
Xc, yc = X - xm, y - ym
w = linalg.solve(Xc.T @ Xc + 10.0 * np.eye(6), Xc.T @ yc)
out("ridge 10", w, [ym - xm @ w])


def elastic_net(alpha, rho):
    # Minimise 1/(2n)|y - Xw - b|^2 + alpha (rho |w|_1 + (1 - rho)/2 |w|^2) with w = p - q, p, q >= 0.
    n = len(y)

    def f(v):
        p, q, b = v[:6], v[6:12], v[12]
        r = y - X @ (p - q) - b
        g_w = -X.T @ r / n + alpha * (1 - rho) * (p - q)
        value = r @ r / (2 * n) + alpha * (rho * (p + q).sum() + (1 - rho) / 2 * ((p - q) ** 2).sum())
        return value, np.concatenate([g_w + alpha * rho, -g_w + alpha * rho, [-r.sum() / n]])

    res = optimize.minimize(f, np.zeros(13), jac=True, method="L-BFGS-B", bounds=[(0, None)] * 12 + [(None, None)], options=dict(ftol=1e-16, gtol=1e-12, maxiter=100000))
    return res.x[:6] - res.x[6:12], res.x[12]


w, b = elastic_net(0.2, 1.0)
out("lasso 0.2", w, [b])
w, b = elastic_net(0.2, 0.5)
out("elastic net 0.2 0.5", w, [b])

# Logistic regression: two features, two classes, objective of the library (multinomial softmax with
# a full weight matrix, mean negative log-likelihood + lambda/2 |W|^2, intercepts unpenalised).
Z = data("Z", np.column_stack([rng.uniform(0, 10, 200), rng.uniform(4, 9, 200)]))
p = 1 / (1 + np.exp(-(-6 + 0.8 * Z[:, 0] + 0.5 * Z[:, 1])))
c = data("c", (rng.uniform(size=200) < p).astype(float))[0].astype(int)
lam = 1e-2
Zb = np.column_stack([Z, np.ones(200)])
Y = np.eye(2)[c]


def logistic(v):
    W = v.reshape(2, 3)
    s = Zb @ W.T
    log_p = s - special.logsumexp(s, axis=1, keepdims=True)
    pen = W[:, :2]
    value = -(Y * log_p).sum() / 200 + lam / 2 * (pen**2).sum()
    grad = (np.exp(log_p) - Y).T @ Zb / 200
    grad[:, :2] += lam * pen
    return value, grad.ravel()


res = optimize.minimize(logistic, np.zeros(6), jac=True, method="BFGS", options=dict(gtol=1e-12))
W = res.x.reshape(2, 3)
grid = np.array([[2.0, 8.0], [5.0, 7.0], [8.0, 5.0]])
s = np.column_stack([grid, np.ones(3)]) @ W.T
out("logistic", W[1] - W[0], np.exp(s - special.logsumexp(s, axis=1, keepdims=True))[:, 1])

# Gaussian naive Bayes on three classes in three dimensions, as in scikit-learn: variances get
# var_smoothing times the largest feature variance added.
G = data("G", np.vstack([rng.normal([0, 0, 0], [1, 2, 0.5], (30, 3)), rng.normal([2, 1, 0], [1, 1, 1], (30, 3)), rng.normal([0, 3, 2], [0.5, 1, 1], (30, 3))]))
g = np.repeat([0, 1, 2], 30)
eps = 1e-9 * G.var(axis=0).max()
Q = data("Q", rng.normal([1, 1, 1], 1.5, (10, 3)))
log_joint = np.column_stack([np.log(1 / 3) + stats.norm.logpdf(Q, G[g == k].mean(0), np.sqrt(G[g == k].var(0) + eps)).sum(1) for k in range(3)])
out("gaussian nb", np.exp(log_joint - special.logsumexp(log_joint, axis=1, keepdims=True)))

# Fisher discriminant: the generalised eigenproblem S_b w = lambda S_w w on the same three classes.
mean = G.mean(0)
Sw = sum((G[g == k] - G[g == k].mean(0)).T @ (G[g == k] - G[g == k].mean(0)) for k in range(3))
Sb = sum(30 * np.outer(G[g == k].mean(0) - mean, G[g == k].mean(0) - mean) for k in range(3))
lam_, V = linalg.eigh(Sb, Sw)
order = np.argsort(lam_)[::-1][:2]
V = V[:, order] / np.linalg.norm(V[:, order], axis=0)
V *= np.sign(V[0])
out("lda", lam_[order], V)

# k-means on four separated blobs: the best of 20 k-means++ starts of SciPy's kmeans2.
K = data("K", np.vstack([rng.normal(m, 0.6, (25, 2)) for m in [[0, 0], [6, 0.5], [1, 6], [7, 6]]]))
best = None
for seed in range(20):
    centroids, labels = cluster.vq.kmeans2(K, 4, minit="++", seed=seed, iter=100)
    inertia = ((K - centroids[labels]) ** 2).sum()
    if best is None or inertia < best[0]:
        best = (inertia, centroids)
centroids = best[1][np.lexsort((best[1][:, 1], best[1][:, 0]))]
out("kmeans", centroids, [best[0]])
