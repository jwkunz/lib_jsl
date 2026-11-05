| file written by Python | bytes | read by the library |
|---|---:|---:|
| png grey | 635 | identical |
| png RGB | 1866 | identical |
| png grey + alpha | 1237 | identical |
| png RGBA | 2517 | identical |
| png palette | 360 | identical |
| png palette + tRNS | 376 | identical |
| png 1-bit | 136 | identical |
| png 16-bit | 1540 | identical |
| ppm P6 | 2317 | identical |
| pgm P5 | 781 | identical |
| pbm P4 | 105 | identical |
| ppm P3 (text) | 8223 | identical |
| pgm 16-bit | 1551 | identical |
| jpeg 4:4:4 | 1318 | PSNR 30.3 dB against the original (Pillow: 30.3 dB) |
| jpeg 4:2:0 | 1014 | PSNR 25.1 dB against the original (Pillow: 25.1 dB) |
| jpeg 4:2:0 with restarts | 933 | PSNR 24.7 dB against the original (Pillow: 24.7 dB) |
| jpeg greyscale | 613 | PSNR 36.5 dB against the original (Pillow: 36.5 dB) |
| jpeg progressive | 917 | refused: Only baseline sequential Huffman JPEG is supported. |
| zlib level 0 | 6011 | identical |
| zlib level 1 | 1670 | identical |
| zlib level 6 | 1645 | identical |
| zlib level 9 | 1645 | identical |
