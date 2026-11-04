| file written by the library (read by) | result |
|---|---:|
| PNG, L, 8-bit (Pillow) | read as L, identical: True |
| PNG, LA, 8-bit (Pillow) | read as LA, identical: True |
| PNG, RGB, 8-bit (Pillow) | read as RGB, identical: True |
| PNG, RGBA, 8-bit (Pillow) | read as RGBA, identical: True |
| PNG, greyscale, 16-bit (Pillow) | read as I;16, identical: True |
| PGM (Pillow) | read as L, identical: True |
| PPM (Pillow) | read as RGB, identical: True |
| JPEG, greyscale, quality 50 (Pillow) | decodes; PSNR 28.0 dB |
| JPEG, greyscale, quality 90 (Pillow) | decodes; PSNR 38.4 dB |
| JPEG, RGB, quality 50 (Pillow) | decodes; PSNR 21.1 dB |
| JPEG, RGB, quality 90 (Pillow) | decodes; PSNR 30.9 dB |
| zlib stream, level 0 (zlib) | decompresses: True; 3181 bytes (zlib's own: 3181) |
| zlib stream, level 6 (zlib) | decompresses: True; 278 bytes (zlib's own: 212) |
| zlib stream, level 9 (zlib) | decompresses: True; 278 bytes (zlib's own: 212) |
