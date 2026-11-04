"""Reference Reed-Solomon parity bytes for the chapter on error-correcting codes, from the
`reedsolo` package (the book used reedsolo 1.7), configured like the library: field polynomial
0x11D, primitive element 2, generator roots α^0 … α^(n-k-1).

    python book/generator/data/make_coding_refs.py > book/generator/data/coding_reference.txt

Each output line is `n k|parity bytes` for the message bytes 1, 2, …, k (mod 256).
"""

import reedsolo

for n, k in [(255, 223), (255, 239), (26, 16), (15, 9), (7, 3)]:
    rs = reedsolo.RSCodec(n - k, nsize=n, fcr=0, prim=0x11D, generator=2)
    message = bytes((i + 1) % 256 for i in range(k))
    print(f"{n} {k}|" + ",".join(str(b) for b in rs.encode(message)[k:]))
