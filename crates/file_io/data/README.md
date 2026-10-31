# Layer III tables

`layer3_tables.bin` holds the standard MPEG-1 Layer III constants that cannot be derived from a
formula:

- the 512-tap polyphase synthesis window `D[i]` (the analysis window is `C[i] = D[i] / 32`), as
  integers scaled by 65536;
- the 32 Huffman code tables used for the "big values" region (tables 4 and 14 are unused and
  empty), with their `linbits`;
- the two Huffman code tables for the "count1" region.

The byte layout is documented at the top of `extract_layer3_tables.py`; `src/mp3_tables.rs` reads it.

## Provenance and license

The data was extracted from `minimp3.h` in [lieff/minimp3](https://github.com/lieff/minimp3),
which its author dedicated to the public domain under
[CC0 1.0](http://creativecommons.org/publicdomain/zero/1.0/). minimp3 stores these tables in a
packed form for its decoder; `extract_layer3_tables.py` expands them (parsing the text only, never
compiling or running the header). Regenerate with:

    python3 extract_layer3_tables.py minimp3.h layer3_tables.bin

| item | sha256 |
|------|--------|
| `minimp3.h` used | `57e437c5c1f0e8b243885d3929c8973b5e6c778451e0100ab4251d19915cb3ad` |
| `layer3_tables.bin` | `7b54755ce4bfb1e6eedbc6e50f8910edc1562966669a5dc958e30ff9eebc85a2` |

## How it was checked

- The extractor asserts that every Huffman table is prefix-free, covers its full `xlen x ylen`
  grid with no duplicate pairs, has the expected dimensions, and that tables 16-23 (and 24-31)
  share one code set.
- The window is exactly symmetric, and a 32-band analysis/synthesis round trip reconstructs a
  signal at the ISO delay of 481 samples with about -85 dB error. That test is insensitive to the
  few smallest taps (|D| below roughly 150/65536), which were fixed by smoothness of the prototype
  filter rather than by an exact comparison.
- The Huffman tables were used to decode the Huffman data of 8,120 granules produced by an
  independent encoder (LAME, mono, 64-320 kbps, several signal types). Every granule's bit
  position landed exactly on its declared length, exercising 26 of the 29 distinct tables and
  both count1 tables. Tables 19-21 did not occur in that data.
