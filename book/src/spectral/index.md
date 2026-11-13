# DSP: spectral estimation, windows, resampling and multirate processing

The [previous chapter](../filters/index.md) shaped signals with filters. This one measures them and changes their sampling rate. **Spectral estimation** asks
how a signal's power is distributed over frequency: the periodogram and Welch's average, the spectrogram for signals that change, the Lomb-Scargle periodogram
for irregular samples, models that resolve finer than the DFT allows, and wavelets that analyse many scales at once. **Windows** control
the leakage that every finite record suffers. **Multirate processing** changes the sampling rate efficiently and splits signals into channels.

## The central questions

- **Resolution or dynamic range?** A window's main lobe sets how close two components can be; its side lobes set how weak one can be next to a
  strong one. Parametric methods buy resolution with assumptions.
- **Bias or variance?** Longer segments give finer frequency resolution; more segments give steadier estimates. The record length fixes the
  product.
- **Time or frequency?** A spectrogram trades one for the other uniformly; wavelets trade them differently at each scale; the analytic signal
  gives an instantaneous frequency for narrow-band signals.
- **How to change the rate cheaply?** Compute only the samples that are kept, from only the samples that are not zero: polyphase filters,
  half-band stages and multiplier-free CIC filters.

## Testing against SciPy

Every window function, in symmetric and periodic forms, is compared with `scipy.signal.windows` in the [window entry](windows.md#testing-against-scipy). The other computations:

<p class="caption" id="tab-15-1"><strong>Table 15.1.</strong> Columns: SciPy function and case, largest difference ÷ largest value.</p>

{{#include ../generated/spectral/reference.md}}

## In the library

All of these are in `lib_jsl_dsp` ([API reference](../../api/lib_jsl_dsp/index.html)): windows in `discrete::windows`, spectral methods in `discrete::spectral`,
resamplers in `discrete::resampling`, and Fourier integrals in `continuous::spectral`.

| You need | Use | Entry |
|---|---|---|
| A taper for spectral analysis or filter design | `generate_window`, `WindowType`, `hann`, `kaiser`, `chebwin`, `dpss`, … | [Window functions](windows.md) |
| A power spectrum, a spectrogram, or a spectrum of irregular samples | `periodogram`, `welch`, `stft`, `istft`, `spectrogram`, `lomb_scargle` | [Power spectra](power-spectra.md) |
| A few DFT bins, a zoomed band, or a Fourier integral | `goertzel`, `chirpz`, `fourier_integral` | [Spectra at chosen frequencies](chosen-frequencies.md) |
| An envelope or instantaneous frequency | `analytic_signal`, `hilbert_transform`, `instantaneous_attributes` | [Hilbert transform](hilbert.md) |
| Resolution beyond the DFT; a linear-prediction model | `levinson_durbin`, `linear_predictive_coding`, `burg`, `music` | [Parametric spectra](parametric.md) |
| Multiscale analysis, compression or denoising | `dwt`, `wavedec`, `waverec`, `continuous_wavelet_transform` | [Wavelet transforms](wavelets.md) |
| A different sampling rate | `PolyphaseIntegerResampler`, `PolyphaseArbitraryResampler`, `HalfBandDecimator`, `CicDecimator` | [Resampling](resampling.md) |
| Many frequency channels, and recombining them | `*PolyphaseAnalysisFilterBank`, `*PolyphaseSynthesisFilterBank` | [Polyphase filter banks](filter-banks.md) |
