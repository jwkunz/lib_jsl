| coder | compressed bytes | bits per byte | compress (MB/s) | decompress (MB/s) |
|---|---:|---:|---:|---:|
| Huffman | 2682563 | 4.658 | 106 | 74 |
| Shannon-Fano | 2687305 | 4.667 | 98 | 61 |
| arithmetic | 2661602 | 4.622 | 22 | 17 |
| LZ77 | 1673615 | 2.906 | 4 | 538 |
| LZ78 | 2957251 | 5.135 | 12 | 99 |
| LZW | 1811014 | 3.145 | 12 | 73 |
| zlib, level 9 (Python) | 1350639 | 2.345 |  |  |
| bzip2, level 9 (Python) | 961651 | 1.670 |  |  |
| xz/LZMA, preset 9 (Python) | 1011092 | 1.756 |  |  |
