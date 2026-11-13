| SciPy function and case | largest difference ÷ largest coefficient |
|---|---:|
| butter 4 lowpass 0.2 | 3.7e-16 |
| butter 3 highpass 0.3 | 9.6e-17 |
| butter 4 bandpass 0.2-0.5 | 8.5e-16 |
| butter 2 bandstop 0.2-0.4 | 2.0e-16 |
| cheby1 5 1 dB lowpass 0.3 | 1.9e-16 |
| cheby1 4 0.5 dB bandpass 0.2-0.5 | 5.5e-16 |
| cheby2 5 40 dB lowpass 0.3 | 2.4e-16 |
| cheby2 4 30 dB highpass 0.4 | 1.5e-16 |
| ellip 5 1/50 dB lowpass 0.3 | 1.6e-14 |
| ellip 4 0.5/40 dB bandpass 0.2-0.5 | 6.1e-15 |
| ellip 3 1/60 dB bandstop 0.2-0.6 | 5.7e-13 |
| bessel 4 lowpass 0.2 (phase) | 3.0e-16 |
| bessel 4 lowpass 0.2 (delay) | 6.1e-16 |
| bessel 4 lowpass 0.2 (mag) | 3.1e-15 |
| buttord lowpass | order 12 (SciPy 12), 0.0e0 |
| buttord bandpass | order 11 (SciPy 11), 2.5e-18 |
| buttord bandstop | order 11 (SciPy 11), 1.2e-7 |
| cheb1ord lowpass | order 6 (SciPy 6), 0.0e0 |
| cheb1ord bandpass | order 6 (SciPy 6), 0.0e0 |
| cheb1ord bandstop | order 6 (SciPy 6), 2.2e-7 |
| cheb2ord lowpass | order 6 (SciPy 6), 0.0e0 |
| cheb2ord bandpass | order 6 (SciPy 6), 0.0e0 |
| cheb2ord bandstop | order 6 (SciPy 6), 2.6e-7 |
| ellipord lowpass | order 4 (SciPy 4), 0.0e0 |
| ellipord bandpass | order 4 (SciPy 4), 0.0e0 |
| ellipord bandstop | order 4 (SciPy 4), 3.2e-7 |
| firwin 51 lowpass 0.3 hamming | 0.0e0 |
| firwin 51 bandpass 0.2-0.5 | 1.8e-16 |
| firwin 61 highpass 0.4 kaiser 6 | 5.6e-16 |
| firwin2 51 | 1.4e-16 |
| remez 51 lowpass | 7.1e-13 |
| remez 41 bandpass weighted | 6.1e-15 |
| remez 31 hilbert | 1.9e-13 |
| remez 30 differentiator | 7.0e-13 |
| iirnotch 0.25 Q 30 | 0.0e0 |
| iirpeak 0.25 Q 30 | 0.0e0 |
| iircomb 20 Hz Q 30 fs 200 notch | 0.0e0 |
| bilinear 1/(s+1) fs 2 | 0.0e0 |
| freqz butter 4, 8 points | 2.0e-15 |
| freqs Butterworth prototype | 1.1e-16 |
| lfilter ellip 4 on 64 samples | 9.0e-14 |
| convolve full | 1.6e-16 |
| convolve same | 1.6e-16 |
| convolve valid | 0.0e0 |
| correlate full | 1.6e-16 |
| zpk2sos butter 8 lowpass 0.15 | 1.1e-16 |
| zpk2sos ellip 5 1/50 dB lowpass 0.3 | 3.1e-13 |
| zpk2sos cheby1 4 0.5 dB bandpass 0.2-0.5 | 2.8e-16 |
| sosfilt_zi butter 8 lowpass 0.15 | 8.2e-16 |
| sosfilt butter 8 lowpass 0.15 | 3.6e-15 |
| sosfiltfilt butter 8 lowpass 0.15 | 5.5e-15 |
| lfilter_zi butter 4 lowpass 0.2 | 3.8e-15 |
| filtfilt butter 4 lowpass 0.2 (odd) | 6.3e-15 |
| filtfilt butter 4 lowpass 0.2 (even) | 5.2e-15 |
| filtfilt butter 4 lowpass 0.2 (constant, 20) | 5.6e-15 |
