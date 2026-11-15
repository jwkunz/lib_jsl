"""Writes crates/random/data/sobol_directions.bin: the primitive polynomials and initial direction
numbers of Joe and Kuo's `new-joe-kuo-6.21201` table for Sobol dimensions 2 to 21201, read from the
copy shipped with SciPy (`scipy/stats/_sobol_direction_numbers.npz`).

    python crates/random/data/make_sobol_directions.py

Format: for each dimension, unsigned LEB128 varints `s`, `a`, `m_1`, ..., `m_s` (the degree, the
interior coefficients of the polynomial as bits, and the initial direction numbers).
"""

import glob
import os

import numpy as np
import scipy.stats.qmc as qmc

_z = np.load(glob.glob(os.path.join(os.path.dirname(qmc.__file__), "_sobol_direction_numbers.npz"))[0])
z = {"poly": _z["poly"], "vinit": _z["vinit"]}


def varint(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return out


data = bytearray()
for k in range(1, len(z["poly"])):
    poly = int(z["poly"][k])
    s = poly.bit_length() - 1
    a = (poly >> 1) & ((1 << (s - 1)) - 1)
    data += varint(s) + varint(a)
    for v in z["vinit"][k][:s]:
        data += varint(int(v))
path = os.path.join(os.path.dirname(__file__), "sobol_directions.bin")
with open(path, "wb") as f:
    f.write(data)
print(f"wrote {path}: {len(z['poly']) - 1} dimensions, {len(data)} bytes")
