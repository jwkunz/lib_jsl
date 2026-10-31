#!/usr/bin/env python3
"""Extract the MPEG-1 Layer III synthesis window and Huffman tables into layer3_tables.bin.

Source data: minimp3 (https://github.com/lieff/minimp3), `minimp3.h`, which its author has
dedicated to the public domain under CC0 1.0. The script only *parses text* from that header; it
never compiles or runs it. minimp3 stores the tables in a packed form tuned for its decoder
(multi-level lookup tables), so this script expands them into plain (code, length) listings by
walking them exactly as the minimp3 decoder does.

Usage:  extract_layer3_tables.py <minimp3.h> <output.bin>

Binary layout (all integers little-endian):
    magic    b"L3TB"
    version  u8 (= 1)
    window   512 x i32          ISO 11172-3 synthesis window D[i] scaled by 65536
    huffman  u8 count (= 32), then per table 0..31:
                 linbits u8, xlen u8, ylen u8, n u16,
                 n x { x u8, y u8, length u8, code u32 }   (code right-aligned in `length` bits)
    count1   u8 count (= 2: table A, table B), then per table:
                 n u8 (= 16), n x { values u8, length u8, code u8 }
             `values` has bit 3 = v, bit 2 = w, bit 1 = x, bit 0 = y (1 = nonzero, magnitude 1).
"""
import hashlib
import re
import struct
import sys


def int_list(text):
    return [int(v) for v in re.findall(r"-?\d+", text)]


def grab(src, name):
    m = re.search(r"static const \w+ " + re.escape(name) + r"\[[^\]]*\]\s*=\s*\{(.*?)\};", src, re.S)
    if not m:
        sys.exit(f"table {name} not found")
    return int_list(m.group(1))


def build_window(src):
    vals = grab(src, "g_win")
    assert len(vals) == 240
    d = [None] * 512
    # Each row holds D[j + 32m] for j = 1..15 as pairs (m, 15-m), m = 0..7 (ordered by distance
    # from the window centre).
    for j in range(1, 16):
        row = vals[(j - 1) * 16:j * 16]
        for m in range(16):
            d[j + 32 * m] = row[2 * m] if m < 8 else row[2 * (15 - m) + 1]
    sgn = lambda i: -1 if (i // 64) % 2 else 1          # D[i] = (-1)^floor(i/64) * h[i]
    # j = 17..31 follow from D[512 - i] = -D[i] for i not a multiple of 64.
    for j in range(17, 32):
        for m in range(16):
            d[j + 32 * m] = -d[512 - (j + 32 * m)]
    h = [None if v is None else sgn(i) * v for i, v in enumerate(d)]     # symmetric prototype
    for i in range(1, 512):
        if h[i] is not None and h[512 - i] is not None:
            assert h[i] == h[512 - i], "prototype is not symmetric"
    # Column j = 0: magnitudes by distance from the centre (from mp3d_synth_pair); the sign is
    # taken from the (smooth) neighbours.
    for q, mag in enumerate([75038, 37489, 6574, 5153, 2037, 459, 213, 29]):
        for i in {256 - 32 * q, 256 + 32 * q}:
            nb = h[i - 1] + h[i + 1]
            h[i] = mag if q == 0 or nb > 0 else -mag
    # Column j = 16: the eight magnitudes {64019, 9975, 9727, 45, 1567, 146, 104, 5} at indices
    # 240, 208, ..., 16; assignment and signs resolved from the neighbouring taps.
    for i, v in {240: 64019, 208: 9975, 176: -9727, 144: -45, 112: 1567, 80: -146, 48: -104, 16: -5}.items():
        nb = h[i - 1] + h[i + 1]
        assert (nb > 0) == (v > 0) and abs(abs(v) - abs(nb) / 2) < 0.15 * abs(v) + 8, (i, v, nb)
        h[i] = h[512 - i] = v
    h[0] = 0
    assert all(v is not None for v in h)
    return [sgn(i) * h[i] for i in range(512)]


def expand_huffman(tabs, base):
    """All (bitstring, x, y) leaves of the lookup table starting at tabs[base]."""
    found = {}

    def rec(prefix, w, offset):
        for v in range(1 << w):
            leaf = tabs[base + v - offset]
            bits = format(v, "0%db" % w)
            if leaf < 0:
                rec(prefix + bits, leaf & 7, leaf >> 3)
            else:
                length = leaf >> 8
                if length == 0 and leaf == 0 and prefix == "" and base != 0:
                    continue
                code = prefix + bits[:length]
                found[code] = (leaf & 15, (leaf >> 4) & 15)

    rec("", 5, 0)
    return found


def expand_count1(tab):
    out = {}
    for first in range(16):
        leaf = tab[first]
        if leaf & 8:
            code = format(first, "04b")[:leaf & 7]
            out[code] = leaf >> 4
        else:
            nbits = leaf & 3
            for extra in range(1 << nbits):
                leaf2 = tab[(leaf >> 3) + extra]
                length = leaf2 & 7
                code = (format(first, "04b") + format(extra, "0%db" % nbits))[:length]
                out[code] = leaf2 >> 4
    return out


def check_prefix_free(codes, what):
    items = sorted(codes)
    for a, b in zip(items, items[1:]):
        assert not b.startswith(a), f"{what}: code {a} is a prefix of {b}"
    assert sum(2.0 ** -len(c) for c in codes if c) <= 1.0 + 1e-12, f"{what}: Kraft sum above 1"


def main():
    src = open(sys.argv[1]).read()
    window = build_window(src)

    tabs = grab(src, "tabs")
    tabindex = grab(src, "tabindex")
    linbits = grab(src, "g_linbits")
    assert len(tabindex) == 32 and len(linbits) == 32

    blob = bytearray(b"L3TB" + bytes([1]))
    blob += struct.pack("<512i", *window)
    blob += bytes([32])
    expected_dims = {1: 2, 2: 3, 3: 3, 5: 4, 6: 4, 7: 6, 8: 6, 9: 6, 10: 8, 11: 8, 12: 8, 13: 16, 15: 16}
    for t in range(32):
        if t in (4, 14):                       # unused table numbers
            entries = {}
        elif t == 0:
            entries = {"": (0, 0)}
        else:
            entries = expand_huffman(tabs, tabindex[t])
        check_prefix_free(entries, f"table {t}")
        if entries and t != 0:
            xs = [v[0] for v in entries.values()]
            ys = [v[1] for v in entries.values()]
            xlen, ylen = max(xs) + 1, max(ys) + 1
            assert len(entries) == xlen * ylen, f"table {t}: {len(entries)} leaves for {xlen}x{ylen}"
            assert len(set(entries.values())) == len(entries), f"table {t}: duplicate (x, y)"
            if t in expected_dims:
                assert xlen == ylen == expected_dims[t], f"table {t}: unexpected size {xlen}x{ylen}"
            elif t >= 16:
                assert xlen == ylen == 16
        else:
            xlen = ylen = 1 if t == 0 else 0
        rows = sorted(((v[0], v[1], len(c), int(c, 2) if c else 0) for c, v in entries.items()))
        blob += struct.pack("<BBBH", linbits[t], xlen, ylen, len(rows))
        for x, y, ln, code in rows:
            blob += struct.pack("<BBBI", x, y, ln, code)
    # Tables 16..23 (and 24..31) must share one code set; they differ only in linbits.
    for group in (range(16, 24), range(24, 32)):
        sets = {frozenset(expand_huffman(tabs, tabindex[t]).items()) for t in group}
        assert len(sets) == 1

    blob += bytes([2])
    for name in ("tab32", "tab33"):
        codes = expand_count1(grab(src, name))
        check_prefix_free(codes, name)
        assert len(codes) == 16 and len(set(codes.values())) == 16, f"{name}: expected 16 distinct quadruples"
        blob += bytes([len(codes)])
        for code, vals in sorted(codes.items(), key=lambda kv: (len(kv[0]), kv[0])):
            blob += struct.pack("<BBB", vals, len(code), int(code, 2))
    open(sys.argv[2], "wb").write(bytes(blob))
    print(f"wrote {len(blob)} bytes, sha256 {hashlib.sha256(blob).hexdigest()}")
    print(f"source minimp3.h sha256 {hashlib.sha256(src.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
