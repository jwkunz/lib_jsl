"""Writes the remez_*.csv reference taps from SciPy (1.17.1 was used) for the remez tests.

Run from crates/dsp with a Python that has SciPy:  python src/discrete/test_data/make_remez_goldens.py
"""

from scipy import signal

cases = {
    "remez_lowpass_numtaps17.csv": signal.remez(17, [0, 0.2, 0.3, 0.5], [1, 0], fs=1),
    "remez_highpass_numtaps21_weighted.csv": signal.remez(21, [0, 0.14, 0.2, 0.5], [0, 1], weight=[2, 1], fs=1),
    "remez_bandpass_numtaps20_even.csv": signal.remez(20, [0, 0.1, 0.2, 0.3, 0.4, 0.5], [0, 1, 0], fs=1),
    "remez_hilbert_numtaps31.csv": signal.remez(31, [0.05, 0.45], [1], type="hilbert", fs=1),
    "remez_hilbert_numtaps30.csv": signal.remez(30, [0.05, 0.5], [1], type="hilbert", fs=1),
    "remez_differentiator_numtaps30.csv": signal.remez(30, [0.0, 0.4], [1], type="differentiator", fs=1),
}
for name, taps in cases.items():
    with open("src/discrete/test_data/" + name, "w") as f:
        f.write("".join(f"{float(v)!r}\n" for v in taps))
