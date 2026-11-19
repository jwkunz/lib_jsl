"""SciPy references for release 2.12.0: files written by scipy.io (Matrix Market, MATLAB,
Harwell-Boeing, NetCDF-3, Fortran unformatted, and an ARFF file) for the library to read, and
the reverse check: SciPy reading the files the library wrote (data/io/library_*, produced by
running the generator first; rerun the generator afterwards to update the table).
Writes io_reference.txt and io_scipy_reads.txt."""
import os
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.io import arff, FortranFile, netcdf_file

D = "io"
out = []
def put(key, *lists):
    out.append(key + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))

rng = np.random.default_rng(212)
dense = rng.normal(size=(4, 5))
sparse = sp.random(30, 20, density=0.1, random_state=3, format="coo")
sym = sp.random(12, 12, density=0.2, random_state=4); sym = (sym + sym.T).tocoo()
cplx = (sp.random(6, 6, density=0.3, random_state=5) + 1j * sp.random(6, 6, density=0.3, random_state=6)).tocoo()
put("dense", dense)
put("sparse", sparse.toarray())
put("sym", sym.toarray())
put("cplx re", cplx.toarray().real); put("cplx im", cplx.toarray().imag)

sio.mmwrite(f"{D}/scipy_general.mtx", sparse)
sio.mmwrite(f"{D}/scipy_symmetric.mtx", sym, symmetry="symmetric")
sio.mmwrite(f"{D}/scipy_dense.mtx", dense)
sio.mmwrite(f"{D}/scipy_complex.mtx", cplx)
sio.hb_write(f"{D}/scipy.hb", sparse.tocsc())

matvars = dict(a=dense, z=np.array([1 - 2j, 3 + 0.5j]), i16=np.array([[-7], [300]], dtype=np.int16),
               flag=np.array([True, False, True]), s="hello world", sp=sp.csc_matrix(sym),
               c=np.array([np.array([1.0]), "x"], dtype=object), st={"alpha": np.array([2.5, 3.5]), "beta": "b"},
               nd=np.arange(24.0).reshape(2, 3, 4))
sio.savemat(f"{D}/scipy_plain.mat", matvars)
sio.savemat(f"{D}/scipy_compressed.mat", matvars, do_compression=True)

t = np.arange(5.0)
temp = rng.normal(15, 3, (5, 4)).astype(np.float32)
with netcdf_file(f"{D}/scipy.nc", "w") as f:
    f.history = "written by scipy.io.netcdf_file"
    f.createDimension("time", None)
    f.createDimension("station", 4)
    v = f.createVariable("time", "f8", ("time",)); v[:] = t; v.units = "days"
    v = f.createVariable("temp", "f4", ("time", "station")); v[:] = temp
    v = f.createVariable("station_id", "i4", ("station",)); v[:] = [11, 12, 13, 14]
    v = f.createVariable("flag", "i2", ("time",)); v[:] = [1, 0, 1, 1, 0]
put("nc temp", temp); put("nc time", t)

with FortranFile(f"{D}/scipy_fortran.dat", "w") as f:
    f.write_record(np.array([3, -4, 5], dtype=np.int32))
    f.write_record(dense)
    f.write_record(np.array([0.5, 1.5], dtype=np.float32))

arff_text = """% A small ARFF file
@relation weather
@attribute outlook {sunny, overcast, rainy}
@attribute temperature numeric
@attribute humidity real
@attribute windy {TRUE, FALSE}
@attribute play {yes, no}
@data
sunny,85,85,FALSE,no
sunny,80,90,TRUE,no
overcast,83,?,FALSE,yes
rainy,70,96,FALSE,yes
rainy,?,80,TRUE,?
"""
open(f"{D}/weather.arff", "w").write(arff_text)
data, meta = arff.loadarff(f"{D}/weather.arff")
put("arff temperature", data["temperature"]); put("arff humidity", data["humidity"])
open("io_reference.txt", "w").write("\n".join(out) + "\n")

# The reverse direction: SciPy reading the library's files.
res = []
def check(name, fn):
    path = f"{D}/{name}"
    if not os.path.exists(path):
        return
    try:
        ok = bool(fn(path))
    except Exception as e:
        print(name, "failed:", e)
        ok = False
    res.append(f"{name}|{int(ok)}")

check("library_general.mtx", lambda p: np.array_equal(sio.mmread(p).toarray(), sparse.toarray()))
check("library_symmetric.mtx", lambda p: np.array_equal(sio.mmread(p).toarray(), sym.toarray()) and sio.mminfo(p)[5] == "symmetric")
check("library_dense.mtx", lambda p: np.array_equal(sio.mmread(p), dense))
check("library_complex.mtx", lambda p: np.array_equal(sio.mmread(p).toarray(), cplx.toarray()))
check("library.hb", lambda p: np.array_equal(sio.hb_read(p).toarray(), sparse.toarray()))
def mat_ok(p):
    a = sio.loadmat(p)
    return (np.array_equal(a["a"], dense) and np.array_equal(a["z"], matvars["z"].reshape(1, -1)) and a["i16"].dtype == np.int16
            and a["s"][0] == "hello world" and np.array_equal(a["sp"].toarray(), sym.toarray()) and np.array_equal(a["flag"], [[1, 0, 1]])
            and a["st"]["beta"][0, 0][0] == "b" and np.array_equal(a["nd"], matvars["nd"]))
check("library_plain.mat", mat_ok)
check("library_compressed.mat", mat_ok)
def nc_ok(p):
    with netcdf_file(p, "r", mmap=False) as f:
        return (np.array_equal(f.variables["temp"][:], temp) and np.array_equal(f.variables["time"][:], t)
                and f.variables["time"].units == b"days" and list(f.variables["station_id"][:]) == [11, 12, 13, 14])
check("library.nc", nc_ok)
def fortran_ok(p):
    with FortranFile(p, "r") as f:
        return (list(f.read_ints(np.int32)) == [3, -4, 5] and np.array_equal(f.read_reals(float).reshape(4, 5), dense)
                and list(f.read_reals(np.float32)) == [0.5, 1.5])
check("library_fortran.dat", fortran_ok)
open("io_scipy_reads.txt", "w").write("\n".join(res) + "\n")
print(res)
