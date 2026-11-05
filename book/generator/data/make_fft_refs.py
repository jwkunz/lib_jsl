"""Reference transforms for the FFT chapter, from `scipy.fft` (the book used SciPy 1.17.1).

    python book/generator/data/make_fft_refs.py > book/generator/data/fft_reference.txt

The inputs are formulas the generator repeats. Each output line is `case|values`, complex results
as `re;im`.
"""

import numpy as np
from scipy import fft


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


def signal(n):
    k = np.arange(n)
    return np.sin(0.37 * k) + 0.5 * np.cos(1.3 * k * k / n) + 1j * 0.2 * np.sin(0.11 * k)


def real_signal(n):
    k = np.arange(n)
    return np.sin(0.37 * k) + 0.5 * np.cos(1.3 * k * k / n)


for n in [8, 64, 512, 1000, 997]:
    y = fft.fft(signal(n))
    out(f"fft {n}", y.real, y.imag)
for n in [64, 997]:
    y = fft.ifft(signal(n))
    out(f"ifft {n}", y.real, y.imag)
for n in [16, 17]:
    y = fft.rfft(real_signal(n))
    out(f"rfft {n}", y.real, y.imag)
grid = signal(8 * 12).reshape(8, 12)
y = fft.fft2(grid)
out("fft2 8x12", y.real, y.imag)
cube = signal(4 * 6 * 5).reshape(4, 6, 5)
y = fft.fftn(cube)
out("fftn 4x6x5", y.real, y.imag)
y = fft.fftshift(signal(9))
out("fftshift 9", y.real, y.imag)
x = real_signal(32)
for kind in [1, 2, 3, 4]:
    for norm in ["backward", "ortho"]:
        out(f"dct {kind} {norm}", fft.dct(x, type=kind, norm=norm))
        out(f"dst {kind} {norm}", fft.dst(x, type=kind, norm=norm))
        out(f"idct {kind} {norm}", fft.idct(x, type=kind, norm=norm))
