"""Reference values for the clustering and dimensionality reduction chapter, from SciPy and NumPy.

Run with a Python that has SciPy (the book used SciPy 1.17.1):

    python book/generator/data/make_cluster_refs.py > book/generator/data/cluster_reference.txt

Each output line is `case|method|data|values`. `data` holds the rows of a point set, separated by
`;`, with comma-separated coordinates; `values` is a comma-separated list. For `linkage` cases the
values are the merge heights from `scipy.cluster.hierarchy.linkage`; for `pca` cases the
explained variances followed by the absolute values of the principal axes' entries (row by row),
from NumPy's SVD of the centred data. The generator recomputes every case with the library and
tabulates the differences.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage

rng = np.random.default_rng(20251103)


def rows(x):
    return ";".join(",".join(repr(float(v)) for v in r) for r in x)


def vals(v):
    return ",".join(repr(float(t)) for t in v)


blobs = np.round(np.vstack([rng.normal(c, 0.8, (10, 2)) for c in ([0, 0], [5, 1], [2, 5])]), 3)
line = np.round(rng.uniform(0, 10, (25, 3)), 3)
for name, x in [("three blobs, 30 points", blobs), ("uniform cube, 25 points", line)]:
    for method in ["single", "complete", "average", "ward"]:
        z = linkage(x, method=method)
        print(f"{name}|{method}|{rows(x)}|{vals(z[:, 2])}")

latent = rng.normal(0, 1, (200, 2))
mix = np.array([[3, 1, 0, 2, -1], [0, 2, 1, -1, 1]])
x = np.round(latent @ mix + rng.normal(0, 0.3, (200, 5)), 4)
c = x - x.mean(axis=0)
_, s, vt = np.linalg.svd(c, full_matrices=False)
print(f"two factors in five dimensions|pca|{rows(x)}|{vals(np.concatenate([s**2 / (len(x) - 1), np.abs(vt).ravel()]))}")
