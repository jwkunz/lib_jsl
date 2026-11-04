"""Reference values for the chapter on information theory, from Python's standard library and SciPy.

Run with a Python that has SciPy (the book used Python 3.13 and SciPy 1.17.1):

    python book/generator/data/make_information_refs.py > book/generator/data/information_reference.txt

Each output line is `case|value` (or comma-separated values). The text is the King James Bible
test asset of `lib_jsl_information`.
"""

import bz2
import hashlib
import lzma
import pathlib
import zlib

import numpy as np
from scipy import stats

root = pathlib.Path(__file__).resolve().parents[3]
kjv = (root / "crates/information/test_assets/kjv.txt").read_bytes()


def out(name, *values):
    print(name + "|" + ",".join(str(v) for v in values))


for label, data in [("empty", b""), ("abc", b"abc"), ("check", b"123456789"), ("fox", b"The quick brown fox jumps over the lazy dog"), ("kjv", kjv)]:
    out(f"crc32 {label}", zlib.crc32(data))
    out(f"sha256 {label}", hashlib.sha256(data).hexdigest())
# A million repetitions of "a", the FIPS 180 long test vector.
out("sha256 million", hashlib.sha256(b"a" * 1_000_000).hexdigest())

counts = np.bincount(np.frombuffer(kjv, dtype=np.uint8), minlength=256)
out("entropy kjv", repr(float(stats.entropy(counts, base=2))))
p = np.array([0.1, 0.2, 0.3, 0.4])
q = np.array([0.25, 0.25, 0.25, 0.25])
out("kl p q", repr(float(stats.entropy(p, q, base=2))))
out("kl q p", repr(float(stats.entropy(q, p, base=2))))

out("length kjv", len(kjv))
out("zlib kjv", len(zlib.compress(kjv, 9)))
out("bz2 kjv", len(bz2.compress(kjv, 9)))
out("lzma kjv", len(lzma.compress(kjv, preset=9)))
