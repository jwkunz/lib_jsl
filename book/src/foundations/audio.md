# Audio files: WAV

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Store sampled sound losslessly as PCM or floating-point samples in RIFF/WAVE files, the common exchange format for audio |
| **Devised by** | Electronic Arts (1985), the IFF chunk format; IBM and Microsoft (1991), RIFF/WAVE |
| **Accuracy** | WAV files from Python's `wave` module and hand-built headers read exactly; SciPy reads the library's WAV files to within half a quantisation step (tables below) |
| **Cost** | $O(n)$ |
| **API** | [`parse_wav`](../../api/lib_jsl_file_io/wav/fn.parse_wav.html), [`encode_wav`](../../api/lib_jsl_file_io/wav/fn.encode_wav.html), [`read_wav_file`](../../api/lib_jsl_file_io/wav/fn.read_wav_file.html), [`write_wav_file`](../../api/lib_jsl_file_io/wav/fn.write_wav_file.html), [`WavAudio`](../../api/lib_jsl_file_io/wav/struct.WavAudio.html), [`WavSampleFormat`](../../api/lib_jsl_file_io/wav/enum.WavSampleFormat.html) |

</div>

## Overview

A **WAV** file is the plainest way to keep sound: a short header saying how many channels, how many samples per second and how many bits per sample, then the samples themselves, interleaved frame by frame. Nothing is lost, and nothing is saved: CD-quality stereo takes 1411 kbit/s.

Uncompressed PCM is what audio software exchanges and what signal-processing code works on, so WAV is the format the library reads and writes. Compressed formats such as MP3, AAC and FLAC are outside its scope.

## History

The RIFF container and its WAVE form were defined by IBM and Microsoft in 1991 for Windows multimedia. Later extensions added a "WAVE_FORMAT_EXTENSIBLE" header for more channels and sample formats. RIFF itself followed Electronic Arts' Interchange File Format (1985), which introduced the idea of a file as a sequence of tagged, length-prefixed chunks that a reader can skip if it does not understand them. That design is why WAV files survive the metadata chunks that recorders and editors add.

## The idea

### WAV files

A WAV file is a list of **chunks**, each with a four-character identifier and a length. The `fmt ` chunk gives the sample format (integer PCM or IEEE float), the channel count, the sample rate and the bits per sample. The `data` chunk holds the samples, little-endian. Other chunks, such as metadata, are skipped. The reader accepts 8-bit unsigned, 16-, 24- and 32-bit signed integer and 32- and 64-bit float samples, in plain or extensible headers. It returns them as `f32` values scaled by $2^{-(b-1)}$ for $b$-bit integers, so full scale is $[-1, 1)$. A truncated final chunk is tolerated, as many recorders leave one.

```rust,ignore
{{#include ../../../crates/file_io/src/wav.rs:parse_wav}}
```

Writing reverses this and clips integer formats to their range. Files written by Python and read by the library:

<p class="caption" id="tab-26-3"><strong>Table 26.3.</strong> Columns: file written by Python, bytes, read by the library.</p>

{{#include ../generated/foundations/reading_wav.md}}

And files written by the library, read by SciPy's `wavfile`:

<p class="caption" id="tab-26-4"><strong>Table 26.4.</strong> Columns: file written by the library (read by), result.</p>

{{#include ../generated/foundations/interop_wav.md}}

The 32-bit and double-precision files differ by about $3 \times 10^{-8}$ because the library holds samples as `f32`, with a 24-bit significand.

## Algorithm

**Reading a WAV file.**

1. Check that the file begins with the tag `RIFF`, a length, and the tag `WAVE`.
2. Walk the **chunks** that follow. Each has a four-character tag and a length; a chunk that
   is not recognised is skipped by its length.
3. From the `fmt ` chunk, read the sample format (integer or floating point), the number of
   channels, the sampling rate and the bits per sample. If the format is the "extensible"
   one, the real format is given by a sub-format field later in the chunk.
4. From the `data` chunk, read the samples, which are little-endian and interleaved (one
   sample for each channel in turn), and convert them to `f32`: integer samples are divided
   by their full-scale value, so that the result lies in $[-1, 1)$.

**Writing a WAV file**: write the canonical 44-byte header (the `RIFF` tag and total length,
the `WAVE` tag, a `fmt ` chunk of 16 bytes and the `data` tag and length), followed by the
interleaved little-endian samples.

## Implementation

- `WavAudio` holds the sample rate, channel count and interleaved `f32` samples; `channel(i)` extracts one channel.
- Integer formats are clipped to their range on writing; floating-point formats are written as they are.

## In practice

```rust,ignore
{{#include ../../../crates/file_io/examples/file_io_audio.rs}}
```

```text
Pcm8: 44144 bytes, largest error 3.9e-3
Pcm16: 88244 bytes, largest error 1.5e-5
Pcm24: 132344 bytes, largest error 6.0e-8
Pcm32: 176444 bytes, largest error 2.3e-10
Float32: 176444 bytes, largest error 0.0e0
file: 44100 Hz, 2 channels, 22050 frames; left channel peak 0.800
```

Each extra 8 bits per sample cuts the round-trip error by $2^8$, down to the precision of the `f32` samples themselves.

**Strengths**

- WAV reading and writing for every common PCM and float layout, checked against Python's `wave` and SciPy.
- Tolerant reading: unknown chunks are skipped and a truncated final chunk is accepted.

**Weaknesses**

- No compressed formats (MP3, AAC, FLAC), and no reading of metadata chunks.
- WAV samples are held as `f32`, so 32-bit integer and 64-bit float data lose precision beyond 24 bits.

## Exercises

1. Compute the bit rate of 16-bit stereo audio at 44.1 kHz, and the size of an hour of it.
2. Explain why 8-bit WAV samples are unsigned while all wider integer samples are signed, and how the reader centres them.
3. Explain why a $b$-bit quantiser gives about $6.02 b + 1.76$ dB SNR on a full-scale sine wave.
4. Why does the reader scale $b$-bit integers by $2^{-(b-1)}$ rather than $1/(2^{b-1} - 1)$, and what happens to full-scale positive samples?
5. Write the 44-byte canonical WAV header for 2 seconds of mono 16-bit audio at 8 kHz, field by field.

<details>
<summary>Hints and answers</summary>

1. $16\cdot2\cdot44\,100 = 1\,411\,200$ bit/s, i.e. $176\,400$ bytes/s; an hour is $635\,040\,000$ bytes $\approx 635$ MB.
2. In the WAV format, 8-bit samples are unsigned (offset 128) for historical reasons; wider samples are two's complement signed. The reader subtracts 128 to centre them.
3. A full-scale sine has power $A^2/2$ with $A = 2^{b-1}$, while uniform quantisation noise has power $\Delta^2/12$ with $\Delta = 1$: SNR $= 10\log_{10}(1.5\cdot2^{2b}) = 6.02b + 1.76$ dB.
4. Dividing by $2^{b-1}$ maps $-2^{b-1}$ to exactly $-1$ and the largest positive value $2^{b-1} - 1$ to just below $+1$; dividing by $2^{b-1} - 1$ would map $-2^{b-1}$ below $-1$.
5. The 44-byte header: `RIFF`, size $36 + 32000$, `WAVE`, `fmt `, 16, format 1, 1 channel, rate 8000, byte rate 16000, block align 2, 16 bits, `data`, data size 32000.

</details>

## Further reading

- IBM and Microsoft, *Multimedia Programming Interface and Data Specifications 1.0*, 1991 (the RIFF and WAVE specification).
- Microsoft, "WAVEFORMATEXTENSIBLE structure", Windows multimedia documentation.
- K. C. Pohlmann, *Principles of Digital Audio*, 6th ed., McGraw-Hill, 2011.
