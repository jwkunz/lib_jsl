"""Reference values for the DSP filter chapter, from SciPy.

Run with a Python that has SciPy (the book used SciPy 1.17.1):

    python book/generator/data/make_filter_refs.py > book/generator/data/filter_reference.txt

Each output line is `case|values`, where `values` is one or more comma-separated lists separated
by `;` (for example a numerator and a denominator). The generator recomputes each named case with
the library and tabulates the differences.
"""

import numpy as np
from scipy import signal


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.atleast_1d(l)) for l in lists))


def ba(name, b, a):
    out(name, b, a)


ba("butter 4 lowpass 0.2", *signal.butter(4, 0.2))
ba("butter 3 highpass 0.3", *signal.butter(3, 0.3, "highpass"))
ba("butter 4 bandpass 0.2-0.5", *signal.butter(4, [0.2, 0.5], "bandpass"))
ba("butter 2 bandstop 0.2-0.4", *signal.butter(2, [0.2, 0.4], "bandstop"))
ba("cheby1 5 1 dB lowpass 0.3", *signal.cheby1(5, 1, 0.3))
ba("cheby1 4 0.5 dB bandpass 0.2-0.5", *signal.cheby1(4, 0.5, [0.2, 0.5], "bandpass"))
ba("cheby2 5 40 dB lowpass 0.3", *signal.cheby2(5, 40, 0.3))
ba("cheby2 4 30 dB highpass 0.4", *signal.cheby2(4, 30, 0.4, "highpass"))
ba("ellip 5 1/50 dB lowpass 0.3", *signal.ellip(5, 1, 50, 0.3))
ba("ellip 4 0.5/40 dB bandpass 0.2-0.5", *signal.ellip(4, 0.5, 40, [0.2, 0.5], "bandpass"))
ba("ellip 3 1/60 dB bandstop 0.2-0.6", *signal.ellip(3, 1, 60, [0.2, 0.6], "bandstop"))
for norm in ["phase", "delay", "mag"]:
    ba(f"bessel 4 lowpass 0.2 ({norm})", *signal.bessel(4, 0.2, norm=norm))
for fn in ["buttord", "cheb1ord", "cheb2ord", "ellipord"]:
    f = getattr(signal, fn)
    n, wn = f(0.2, 0.3, 1, 40)
    out(f"{fn} lowpass", [n], wn)
    n, wn = f([0.2, 0.5], [0.1, 0.6], 1, 40)
    out(f"{fn} bandpass", [n], wn)
    n, wn = f([0.1, 0.6], [0.2, 0.5], 1, 40)
    out(f"{fn} bandstop", [n], wn)
out("firwin 51 lowpass 0.3 hamming", signal.firwin(51, 0.3))
out("firwin 51 bandpass 0.2-0.5", signal.firwin(51, [0.2, 0.5], pass_zero=False))
out("firwin 61 highpass 0.4 kaiser 6", signal.firwin(61, 0.4, window=("kaiser", 6.0), pass_zero=False))
out("firwin2 51", signal.firwin2(51, [0, 0.3, 0.5, 1], [1, 1, 0, 0]))
out("remez 51 lowpass", signal.remez(51, [0, 0.2, 0.3, 0.5], [1, 0], fs=1))
out("remez 41 bandpass weighted", signal.remez(41, [0, 0.1, 0.15, 0.3, 0.35, 0.5], [0, 1, 0], weight=[10, 1, 10], fs=1))
out("remez 31 hilbert", signal.remez(31, [0.05, 0.45], [1], type="hilbert", fs=1))
out("remez 30 differentiator", signal.remez(30, [0.0, 0.4], [1], type="differentiator", fs=1))
ba("iirnotch 0.25 Q 30", *signal.iirnotch(0.25, 30))
ba("iirpeak 0.25 Q 30", *signal.iirpeak(0.25, 30))
ba("iircomb 20 Hz Q 30 fs 200 notch", *signal.iircomb(20, 30, ftype="notch", fs=200))
ba("bilinear 1/(s+1) fs 2", *signal.bilinear([1], [1, 1], fs=2))
b, a = signal.butter(4, 0.2)
w, h = signal.freqz(b, a, worN=8)
out("freqz butter 4, 8 points", h.real, h.imag)
w, h = signal.freqs([1], [1, 2.6131259, 3.4142136, 2.6131259, 1], worN=[0.1, 0.5, 1, 2, 5])
out("freqs Butterworth prototype", h.real, h.imag)
x = np.cos(0.37 * np.arange(64)) + 0.5 * np.sin(1.3 * np.arange(64))
b, a = signal.ellip(4, 0.5, 40, 0.3)
out("lfilter ellip 4 on 64 samples", signal.lfilter(b, a, x))
y = np.arange(1, 9) ** 0.5
out("convolve full", np.convolve(x[:20], y, "full"))
out("convolve same", np.convolve(x[:20], y, "same"))
out("convolve valid", np.convolve(x[:20], y, "valid"))
out("correlate full", signal.correlate(x[:20], y, "full"))
