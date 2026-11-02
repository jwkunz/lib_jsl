"""Reference singular values for the SVD accuracy figure of the lib_jsl book.

The matrix is A = B D with B = lib_jsl_linalg_reference::random_matrix(40, 20, 61) and
D = diag(2^-k_p(j)), k_i = round(50 i / 19) and p(j) = 7 j mod 20 (the scales in a scrambled
order): every entry of A is a double exactly (a scaling by a
power of two), so mpmath at 50 digits gives its exact singular values to double precision.

Run with:  python3 -I book/generator/data/make_svd_ref.py   (needs mpmath)
"""
import os
import mpmath as mp

mp.mp.dps = 50
M64 = (1 << 64) - 1


def random_matrix(rows, cols, seed):
    """The deterministic generator of lib_jsl_linalg_reference::random_matrix (row-major)."""
    s = seed
    out = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            s = (s * 6364136223846793005 + 1442695040888963407) & M64
            row.append((s >> 11) / float(1 << 53) * 2.0 - 1.0)
        out.append(row)
    return out


m, n = 40, 20
b = random_matrix(m, n, 61)
k = [round(50 * j / 19) for j in range(n)]
a = mp.matrix([[mp.mpf(b[i][j]) * mp.mpf(2) ** (-k[(7 * j) % n]) for j in range(n)] for i in range(m)])
sigma = sorted((mp.mpf(s) for s in mp.svd_r(a, compute_uv=False)), reverse=True)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'svd_reference.csv')
with open(path, 'w') as f:
    f.write("# singular values of B D (see make_svd_ref.py), descending\n")
    for s in sigma:
        f.write(repr(float(s)) + "\n")
print(len(sigma), "singular values from", float(sigma[0]), "to", float(sigma[-1]))
