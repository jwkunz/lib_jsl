"""Reference values for the chapter on shared types, utilities and file formats (the book used
mpmath 1.3, SciPy 1.17.1 with CODATA 2022, Pillow 12.2, NumPy 2.4 and Python's wave, csv and
zlib modules).

    python book/generator/data/make_foundations_refs.py > book/generator/data/foundations_reference.txt

Lines are `<kind> <name>|<payload>`:
- `const <module::NAME>|value`: reference values of library constants (mpmath or CODATA 2022);
- `file <name>|base64`: small files written by the Python libraries, for the library to read;
- `expect <name>|v,v,...`: the samples those files hold, as the Python libraries decode them;
- `pillowpsnr <name>|psnr`: Pillow's own decoding of the JPEG test files, against the original;
- `jpeg <quality>|bytes,psnr`: Pillow's JPEG size and PSNR on the test photograph;
- `zlib <name>|level-6 size,level-9 size`;
- `interop <description>|result`: how the Python libraries read files the library wrote. These
  are made by running the generator with `INTEROP_OUT` set (it needs `cargo` on the path).
"""

import base64, csv, io, os, struct, subprocess, sys, tempfile, wave, zlib
import mpmath as mp
import numpy as np
import scipy.constants as sc
from PIL import Image

mp.mp.dps = 40
rng = np.random.default_rng(2026)


def out(kind, name, payload):
    print(f"{kind} {name}|{payload}")


def floats(values):
    return ",".join(f"{float(v):.17g}" for v in np.ravel(values))


# ---------------------------------------------------------------- constants
pi = mp.pi
math_refs = {
    "TWO_PI_OVER_3": 2 * pi / 3, "THREE_PI_OVER_2": 3 * pi / 2, "PI_SQUARED": pi**2, "SQRT_PI": mp.sqrt(pi), "SQRT_TAU": mp.sqrt(2 * pi),
    "FRAC_1_SQRT_TAU": 1 / mp.sqrt(2 * pi), "FRAC_1_SQRT_PI": 1 / mp.sqrt(pi), "LN_PI": mp.log(pi), "LN_SQRT_TAU": mp.log(2 * pi) / 2,
    "LN_TAU": mp.log(2 * pi), "FOUR_PI": 4 * pi, "SQRT_3": mp.sqrt(3), "SQRT_5": mp.sqrt(5), "SQRT_6": mp.sqrt(6), "CBRT_2": mp.cbrt(2),
    "CBRT_3": mp.cbrt(3), "PHI": mp.phi, "FRAC_1_PHI": 1 / mp.phi, "LN_PHI": mp.log(mp.phi), "SILVER_RATIO": 1 + mp.sqrt(2),
    "PLASTIC_NUMBER": mp.findroot(lambda x: x**3 - x - 1, 1.3), "EULER_GAMMA": mp.euler, "APERY": mp.zeta(3), "ZETA_2": mp.zeta(2),
    "ZETA_4": mp.zeta(4), "CATALAN": mp.catalan, "GLAISHER": mp.glaisher, "KHINCHIN": mp.khinchin, "OMEGA": mp.lambertw(1).real,
    "TWIN_PRIME": mp.twinprime, "GIBBS": 2 / pi * mp.si(pi), "FEIGENBAUM_DELTA": mp.mpf("4.669201609102990671853203820466"),
    "FEIGENBAUM_ALPHA": mp.mpf("2.502907875095892822283902873218"), "SIGMA_TO_FWHM": 2 * mp.sqrt(2 * mp.log(2)),
    "FWHM_TO_SIGMA": 1 / (2 * mp.sqrt(2 * mp.log(2))), "Z_90": mp.sqrt(2) * mp.erfinv(mp.mpf("0.9")), "Z_95": mp.sqrt(2) * mp.erfinv(mp.mpf("0.95")),
    "Z_99": mp.sqrt(2) * mp.erfinv(mp.mpf("0.99")), "Z_999": mp.sqrt(2) * mp.erfinv(mp.mpf("0.999")), "P_1_SIGMA": mp.erf(1 / mp.sqrt(2)),
    "P_2_SIGMA": mp.erf(2 / mp.sqrt(2)), "P_3_SIGMA": mp.erf(3 / mp.sqrt(2)),
}
for k, v in math_refs.items():
    out("const", "math::" + k, f"{float(v):.17g}")
cod = lambda n: sc.physical_constants[n][0]
codata = {
    "physics::HBAR": sc.hbar, "physics::R_GAS": sc.R, "physics::FARADAY": cod("Faraday constant"), "physics::SIGMA_SB": sc.Stefan_Boltzmann,
    "physics::C1_RADIATION": cod("first radiation constant"), "physics::C2_RADIATION": cod("second radiation constant"), "physics::G": sc.G,
    "physics::ALPHA": sc.alpha, "physics::INV_ALPHA": cod("inverse fine-structure constant"), "physics::M_E": sc.m_e, "physics::M_P": sc.m_p,
    "physics::M_N": sc.m_n, "physics::AMU": cod("atomic mass constant"), "physics::MP_OVER_ME": cod("proton-electron mass ratio"),
    "physics::WIEN_B": cod("Wien wavelength displacement law constant"), "physics::WIEN_B_FREQ": cod("Wien frequency displacement law constant"),
    "physics::BOHR_RADIUS": cod("Bohr radius"), "physics::COMPTON_WAVELENGTH": cod("Compton wavelength"),
    "physics::CLASSICAL_ELECTRON_RADIUS": cod("classical electron radius"), "physics::THOMSON_CROSS_SECTION": cod("Thomson cross section"),
    "physics::RYDBERG": sc.Rydberg, "physics::HARTREE": cod("Hartree energy"), "physics::PLANCK_LENGTH": cod("Planck length"),
    "physics::PLANCK_MASS": cod("Planck mass"), "physics::PLANCK_TIME": cod("Planck time"), "physics::PLANCK_TEMPERATURE": cod("Planck temperature"),
    "physics::MOLAR_VOLUME_STP": cod("molar volume of ideal gas (273.15 K, 101.325 kPa)"), "physics::PARSEC": sc.parsec,
    "em::MU_0": sc.mu_0, "em::EPS_0": sc.epsilon_0, "em::Z_0": cod("characteristic impedance of vacuum"), "em::FLUX_QUANTUM": cod("mag. flux quantum"),
    "em::JOSEPHSON": cod("Josephson constant"), "em::VON_KLITZING": cod("von Klitzing constant"), "em::CONDUCTANCE_QUANTUM": cod("conductance quantum"),
    "em::BOHR_MAGNETON": cod("Bohr magneton"), "em::NUCLEAR_MAGNETON": cod("nuclear magneton"), "em::G_ELECTRON": -cod("electron g factor"),
    "units::BTU": sc.Btu_IT, "units::CALORIE": sc.calorie_th, "units::NAUTICAL_MILE": sc.nautical_mile, "units::PSI": sc.psi,
    "units::HORSEPOWER": sc.hp, "units::LBF": sc.pound_force, "units::OUNCE": sc.ounce, "units::GALLON_US": sc.gallon,
}
for k, v in codata.items():
    out("const", k, f"{float(v):.17g}")

# ---------------------------------------------------------------- files written by Python
files, expects = {}, {}
h, w = 24, 32
y, x = np.mgrid[0:h, 0:w]
rgb = np.stack([(x * 8) % 256, (y * 10) % 256, ((x + y) * 6) % 256], -1)
rgb = (rgb + rng.integers(-20, 20, rgb.shape)).clip(0, 255).astype(np.uint8)
gray = rgb.mean(-1).astype(np.uint8)
alpha = ((x * y * 3) % 256).astype(np.uint8)


def save_image(name, image, fmt, **kw):
    b = io.BytesIO()
    image.save(b, fmt, **kw)
    files[name] = b.getvalue()
    expects[name] = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert({"P": "RGB", "1": "L"}.get(image.mode, image.mode)) if image.mode in "P1" else Image.open(io.BytesIO(b.getvalue())))


save_image("png grey", Image.fromarray(gray, "L"), "PNG")
save_image("png RGB", Image.fromarray(rgb, "RGB"), "PNG")
save_image("png grey + alpha", Image.fromarray(np.stack([gray, alpha], -1), "LA"), "PNG")
save_image("png RGBA", Image.fromarray(np.dstack([rgb, alpha]), "RGBA"), "PNG")
save_image("png palette", Image.fromarray(rgb, "RGB").quantize(16), "PNG")
b = io.BytesIO()
Image.fromarray(rgb, "RGB").quantize(16).save(b, "PNG", transparency=3)
files["png palette + tRNS"] = b.getvalue()
expects["png palette + tRNS"] = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGBA"))
bits = Image.fromarray(gray > 128).convert("1")
b = io.BytesIO(); bits.save(b, "PNG"); files["png 1-bit"] = b.getvalue(); expects["png 1-bit"] = np.asarray(bits).astype(np.uint8) * 255
g16 = (gray.astype(np.uint16) * 257 + rng.integers(0, 200, gray.shape)).astype(np.uint16)
b = io.BytesIO(); Image.fromarray(g16).save(b, "PNG"); files["png 16-bit"] = b.getvalue(); expects["png 16-bit"] = g16
save_image("ppm P6", Image.fromarray(rgb, "RGB"), "PPM")
save_image("pgm P5", Image.fromarray(gray, "L"), "PPM")
b = io.BytesIO(); bits.save(b, "PPM"); files["pbm P4"] = b.getvalue(); expects["pbm P4"] = np.asarray(bits).astype(np.uint8) * 255
files["ppm P3 (text)"] = ("P3\n# made by hand\n%d %d\n255\n" % (w, h) + "\n".join(" ".join(str(v) for v in row) for row in rgb.reshape(h, -1))).encode()
expects["ppm P3 (text)"] = rgb
files["pgm 16-bit"] = b"P5\n%d %d\n65535\n" % (w, h) + g16.astype(">u2").tobytes()
expects["pgm 16-bit"] = g16
for name, kw in [("jpeg 4:4:4", dict(quality=90, subsampling=0)), ("jpeg 4:2:0", dict(quality=90, subsampling=2)), ("jpeg 4:2:0 with restarts", dict(quality=85, subsampling=2, restart_marker_blocks=3))]:
    b = io.BytesIO(); Image.fromarray(rgb, "RGB").save(b, "JPEG", **kw); files[name] = b.getvalue(); expects[name] = rgb
b = io.BytesIO(); Image.fromarray(gray, "L").save(b, "JPEG", quality=90); files["jpeg greyscale"] = b.getvalue(); expects["jpeg greyscale"] = gray
b = io.BytesIO(); Image.fromarray(rgb, "RGB").save(b, "JPEG", quality=90, progressive=True); files["jpeg progressive"] = b.getvalue(); expects["jpeg progressive"] = np.zeros(0)

t = np.arange(400) / 8000
audio = np.stack([0.8 * np.sin(2 * np.pi * 440 * t), -0.4 * np.sin(2 * np.pi * 660 * t)], 1)
for name, (width, fmt) in {"wav 8-bit": (1, 1), "wav 16-bit": (2, 1), "wav 24-bit": (3, 1), "wav 32-bit": (4, 1), "wav float": (4, 3), "wav double": (8, 3)}.items():
    if fmt == 3:
        raw = audio.astype("<f4" if width == 4 else "<f8").tobytes(); expect = audio.astype("<f4" if width == 4 else "<f8").astype(float)
    elif width == 1:
        q = np.round(audio * 128 + 128).clip(0, 255).astype(np.uint8); raw = q.tobytes(); expect = (q.astype(float) - 128) / 128
    elif width == 2:
        q = np.round(audio * 32768).clip(-32768, 32767).astype("<i2"); raw = q.tobytes(); expect = q / 32768
    elif width == 3:
        q = np.round(audio * 8388608).astype("<i4"); raw = b"".join(int(v).to_bytes(3, "little", signed=True) for v in q.ravel()); expect = q / 8388608
    else:
        q = np.round(audio * 2147483648).clip(-2**31, 2**31 - 1).astype("<i4"); raw = q.tobytes(); expect = q / 2**31
    if fmt == 1 and width in (1, 2):
        b = io.BytesIO()
        with wave.open(b, "wb") as wf:
            wf.setnchannels(2); wf.setsampwidth(width); wf.setframerate(8000); wf.writeframes(raw)
        files[name] = b.getvalue()
    else:
        header = struct.pack("<4sI4s4sIHHIIHH", b"RIFF", 36 + len(raw), b"WAVE", b"fmt ", 16, fmt, 2, 8000, 8000 * 2 * width, 2 * width, 8 * width)
        files[name] = header + b"data" + struct.pack("<I", len(raw)) + raw
    expects[name] = expect

table = [["name", "value", "note"], ["a,b", "1.5", 'say "hi"'], ["two\nlines", "-2e3", ""], ["plain", "3", "x"]]
b = io.StringIO(); csv.writer(b).writerows(table); files["csv (Excel dialect)"] = b.getvalue().encode()
b = io.StringIO(); csv.writer(b, delimiter=";", lineterminator="\n").writerows(table); files["csv (semicolons)"] = b.getvalue().encode()

arrays = {"npy float64": rng.normal(size=(3, 4)), "npy float32": rng.normal(size=5).astype(np.float32), "npy int16": np.arange(-6, 6, dtype=np.int16).reshape(3, 4),
          "npy uint8 3-D": np.arange(24, dtype=np.uint8).reshape(2, 3, 4), "npy bool": np.array([True, False, True]), "npy big-endian": rng.normal(size=(2, 3)).astype(">f8"),
          "npy Fortran order": np.asfortranarray(np.arange(12.0).reshape(3, 4)), "npy int64": np.array([-(2**62), 2**62]), "npy scalar": np.array(3.5)}
for name, a in arrays.items():
    b = io.BytesIO(); np.save(b, a); files[name] = b.getvalue(); expects[name] = a.astype(float)
for version in [(2, 0), (3, 0)]:
    b = io.BytesIO(); np.lib.format.write_array(b, np.arange(5.0), version=version); files[f"npy version {version[0]}.0"] = b.getvalue(); expects[f"npy version {version[0]}.0"] = np.arange(5.0)

text = (b"the quick brown fox jumps over the lazy dog. " * 100) + bytes(rng.integers(0, 256, 1500, dtype=np.uint8))
for level in [0, 1, 6, 9]:
    files[f"zlib level {level}"] = zlib.compress(text, level)

out("text", "zlib", base64.b64encode(text).decode())
for name, data in files.items():
    out("file", name, base64.b64encode(data).decode())
    if name.startswith("jpeg") and "progressive" not in name:
        decoded = np.asarray(Image.open(io.BytesIO(data))).astype(float)
        out("pillowpsnr", name, f"{10 * np.log10(255**2 / ((decoded - expects[name]) ** 2).mean()):.4f}")
    e = expects.get(name)
    if e is not None and np.asarray(e).size:
        out("expect", name, floats(e))

# ---------------------------------------------------------------- JPEG and zlib on a natural image
from skimage import data as skdata
photo = np.asarray(Image.fromarray(skdata.astronaut()).resize((128, 128), Image.LANCZOS))
out("image", "astronaut", base64.b64encode(photo.tobytes()).decode())
for q in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
    for sub, label in [(0, "4:4:4"), (2, "4:2:0")]:
        b = io.BytesIO(); Image.fromarray(photo).save(b, "JPEG", quality=q, subsampling=sub)
        dec = np.asarray(Image.open(io.BytesIO(b.getvalue()))).astype(float)
        psnr = 10 * np.log10(255**2 / ((dec - photo) ** 2).mean())
        out("jpeg", f"{label} {q}", f"{len(b.getvalue())},{psnr:.4f}")
# Compression sizes: the photograph's pixels and the test text are already above; add random bytes.
randoms = bytes(rng.integers(0, 256, 5000, dtype=np.uint8))
out("zlibdata", "random bytes", base64.b64encode(randoms).decode())
for name, data in [("photograph pixels", photo.tobytes()), ("English text", text[:4500]), ("random bytes", randoms)]:
    out("zlib", name, f"{len(zlib.compress(data, 6))},{len(zlib.compress(data, 9))}")

# ---------------------------------------------------------------- files written by the library
with tempfile.TemporaryDirectory() as d:
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    subprocess.run(["cargo", "run", "-q", "--release", "-p", "lib_jsl_book_generator"], cwd=root, env={**os.environ, "INTEROP_OUT": d, "GEN_ONLY": "none"}, check=True, stdout=subprocess.DEVNULL)
    src = lambda name: np.load(os.path.join(d, name + ".npy"))
    for ch, mode in [(1, "L"), (2, "LA"), (3, "RGB"), (4, "RGBA")]:
        im = Image.open(os.path.join(d, f"image_{ch}.png"))
        same = np.array_equal(np.asarray(im).reshape(src(f"image_{ch}").shape), src(f"image_{ch}"))
        out("interop", f"PNG, {mode}, 8-bit (Pillow)", f"read as {im.mode}, identical: {same}")
    im = Image.open(os.path.join(d, "image_16.png"))
    out("interop", "PNG, greyscale, 16-bit (Pillow)", f"read as {im.mode}, identical: {np.array_equal(np.asarray(im).reshape(src('image_16').shape), src('image_16'))}")
    for ch, name in [(1, "PGM"), (3, "PPM")]:
        im = Image.open(os.path.join(d, f"image_{ch}.ppm"))
        out("interop", f"{name} (Pillow)", f"read as {im.mode}, identical: {np.array_equal(np.asarray(im).reshape(src(f'image_{ch}').shape), src(f'image_{ch}'))}")
    for ch, name in [(1, "greyscale"), (3, "RGB")]:
        for q in [50, 90]:
            path = os.path.join(d, f"image_{ch}_q{q}.jpg")
            dec = np.asarray(Image.open(path)).astype(float).reshape(src(f"image_{ch}").shape)
            psnr = 10 * np.log10(255**2 / ((dec - src(f"image_{ch}")) ** 2).mean())
            out("interop", f"JPEG, {name}, quality {q} (Pillow)", f"decodes; PSNR {psnr:.1f} dB")
    import scipy.io.wavfile as wavfile
    reference = src("audio")
    for name, scale in [("8-bit", 128), ("16-bit", 32768), ("24-bit", 2**31), ("32-bit", 2**31), ("float", 1), ("double", 1)]:
        rate, data = wavfile.read(os.path.join(d, f"audio_{name}.wav"))
        values = (data.astype(float) - 128) / 128 if data.dtype == np.uint8 else data.astype(float) / scale
        out("interop", f"WAV, {name} (SciPy)", f"{data.dtype} at {rate} Hz; largest difference {np.abs(values - reference).max():.2g}")
    rows = list(csv.reader(open(os.path.join(d, "table.csv"), newline="")))
    out("interop", "CSV with quotes, delimiters and line breaks in fields (csv module)", f"identical: {rows == [['a', 'b,c'], ['x \"q\"', '1' + chr(10) + '2'], ['', 'plain']]}")
    for name, expect in [("float64", src("npy_float64")), ("int32", np.array([[-1, 2, 3], [4, 5, 6]], dtype=np.int32)), ("complex128", np.array([1 - 2j, 3.5 + 0j])), ("bool", np.array([True, False]))]:
        a = np.load(os.path.join(d, f"array_{name}.npy"))
        out("interop", f"NPY, {name} (NumPy)", f"{a.dtype} {a.shape}, identical: {np.array_equal(a, expect)}")
    raw = open(os.path.join(d, "text.bin"), "rb").read()
    for level in [0, 6, 9]:
        z = open(os.path.join(d, f"zlib_{level}.zlib"), "rb").read()
        out("interop", f"zlib stream, level {level} (zlib)", f"decompresses: {zlib.decompress(z) == raw}; {len(z)} bytes (zlib's own: {len(zlib.compress(raw, level))})")
