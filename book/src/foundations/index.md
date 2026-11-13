# Foundations: shared types, utilities and file formats

Underneath the numerical chapters are the pieces every program needs: one error type, numeric traits and constants that all crates share, small utilities, and ways to get data in and out. This chapter covers:
- `lib_jsl_core`: the error type, numeric traits, array aliases and named constants;
- `lib_jsl_utils`: floating-point spacing, unit conversion and terminal plots;
- `lib_jsl_file_io`: readers and writers for WAV audio, PNG, JPEG and Netpbm images, CSV tables and NumPy arrays.

## The central questions

- **Will other programs read it?** A file format is only useful if other software agrees about it. Every format here is tested in both directions against the reference implementations: Pillow (libpng, libjpeg), Python's `wave`, `csv` and `zlib` modules, SciPy and NumPy.
- **Lossless or lossy?** WAV, PNG, Netpbm, CSV and `.npy` keep the data exactly; JPEG trades exactness for size, guided by a model of what the eye notices.
- **What if the file is damaged or hostile?** Decoders return errors rather than panicking, and check the sizes a header claims before allocating for them.

## Testing against Python's libraries

The references come from `book/generator/data/make_foundations_refs.py`. It writes small files with Pillow 12.2, NumPy 2.4 and the Python standard library, embeds them in the reference file for the library to read, and reads back files written by the library. Constants are checked against mpmath and SciPy's CODATA 2022 values. The results are on the [audio](audio.md), [image](images.md) and [data-file](data-files.md) pages.

## In the library

| You need | Use | Entry |
|---|---|---|
| The workspace's error type, numeric traits and constants; ULPs, unit conversion, terminal plots | `ErrorsJSL`, `constants`, `ulp`, `unit_conversion`, `render_terminal_plot` | [Shared types, constants and utilities](core-utils.md) |
| WAV files | `read_wav_file`, `write_wav_file`, `parse_wav`, `encode_wav` | [Audio files](audio.md) |
| PNG, JPEG or Netpbm images; zlib streams | `read_png_file`, `jpeg_encode`, `ppm_decode`, `zlib_compress` | [Image files](images.md) |
| Tables and arrays to share with spreadsheets and Python | `read_csv_file`, `read_npy_file`, `write_npy_file` | [Data files](data-files.md) |
