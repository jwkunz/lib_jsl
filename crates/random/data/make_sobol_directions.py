"""Writes crates/random/src/sobol_directions.rs: the primitive polynomials and initial direction
numbers of Joe and Kuo's `new-joe-kuo-6.21201` table for Sobol dimensions 2 to 1111, read from the
copy shipped with SciPy (`scipy/stats/_sobol_direction_numbers.npz`).

    python crates/random/data/make_sobol_directions.py > crates/random/src/sobol_directions.rs
"""

import glob
import os

import numpy as np
import scipy.stats.qmc as qmc

DIMS = 1111
z = np.load(glob.glob(os.path.join(os.path.dirname(qmc.__file__), "_sobol_direction_numbers.npz"))[0])
print("//! Joe-Kuo direction numbers for Sobol dimensions 2 to 1111 (`new-joe-kuo-6.21201`), written by")
print("//! `crates/random/data/make_sobol_directions.py`: `(degree s, coefficients a, m_1..m_s)`.")
print()
print("#[rustfmt::skip]")
print(f"pub(crate) static JOE_KUO: [(u32, u32, &[u64]); {DIMS - 1}] = [")
for k in range(1, DIMS):
    poly = int(z["poly"][k])
    s = poly.bit_length() - 1
    a = (poly >> 1) & ((1 << (s - 1)) - 1)
    m = ", ".join(str(int(v)) for v in z["vinit"][k][:s])
    print(f"    ({s}, {a}, &[{m}]),")
print("];")
