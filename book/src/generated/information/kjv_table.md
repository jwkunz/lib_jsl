| coder | compressed bytes | bits per byte | compress (MB/s) | decompress (MB/s) |
|---|---:|---:|---:|---:|
| Huffman | 2682563 | 4.658 | 47 | 31 |
| Shannon-Fano | 2687305 | 4.667 | 38 | 27 |
| arithmetic | 2661602 | 4.622 | 8 | 6 |
| LZ77 | 1673615 | 2.906 | 1 | 173 |
| LZ78 | 2957251 | 5.135 | 3 | 26 |
| LZW | 1811014 | 3.145 | 3 | 21 |
| zlib, level 9 (Python) | 1350639 | 2.345 |  |  |
| bzip2, level 9 (Python) | 961651 | 1.670 |  |  |
| xz/LZMA, preset 9 (Python) | 1011092 | 1.756 |  |  |
