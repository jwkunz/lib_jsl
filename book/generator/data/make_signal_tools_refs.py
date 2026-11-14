"""Reference results for the entry on signal-processing utilities, from `scipy.signal` (the book
used SciPy 1.17.1).

    python book/generator/data/make_signal_tools_refs.py > book/generator/data/signal_tools_reference.txt
"""

import numpy as np
from scipy import signal

rng = np.random.default_rng(12)


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


b, a = signal.cheby1(6, 1.0, 0.3)
out("data b", b)
out("data a", a)
w = np.linspace(0.01, 3.1, 60)
out("data w", w)
out("group_delay iir", signal.group_delay((b, a), w)[1])
bf = signal.firwin(41, 0.3)
out("data fir", bf)
out("group_delay fir", signal.group_delay((bf, [1.0]), w)[1])
h = signal.remez(31, [0, 0.2, 0.3, 0.5], [1, 0])
out("data remez", h)
out("minimum_phase half", signal.minimum_phase(h))
out("minimum_phase full", signal.minimum_phase(h, half=False))
out("firls lowpass", signal.firls(41, [0, 0.3, 0.4, 1], [1, 1, 0, 0]))
out("firls bandpass", signal.firls(63, [0, 0.1, 0.2, 0.4, 0.5, 1], [0, 0, 1, 1, 0, 0], weight=[10, 1, 10]))
out("firls differentiator", signal.firls(31, [0, 0.9], [0, 0.9 * np.pi], fs=2))
t = np.arange(400)
x = np.round(np.sin(0.05 * t) + 0.5 * np.sin(0.7 * t) + 0.2 * rng.normal(size=400), 6)
out("data x", x)
for q in (2, 5):
    out(f"decimate {q}", signal.decimate(x, q))
out("decimate 3 causal", signal.decimate(x, 3, zero_phase=False))
for num in (150, 401, 1000):
    out(f"resample {num}", signal.resample(x, num))
xs = np.round(x + np.where(rng.random(400) < 0.05, 4.0, 0.0), 6)
out("data spiky", xs)
for k in (3, 7):
    out(f"medfilt {k}", signal.medfilt(xs, k))
out("envelope", np.abs(signal.hilbert(x)))
y = np.round(signal.lfilter([0.3, 0.4, 0.3], [1, -0.5], x) + 0.3 * rng.normal(size=400), 6)
out("data y", y)
win = signal.get_window("hann", 64)
out("data window", win)
f, p = signal.csd(x, y, fs=10.0, window=win, noverlap=32)
out("csd", p.real, p.imag)
f, c = signal.coherence(x, y, fs=10.0, window=win, noverlap=32)
out("coherence", c)
