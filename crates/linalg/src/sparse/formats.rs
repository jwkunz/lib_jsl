//! Sparse matrix storage: coordinate (COO) for assembly, compressed sparse rows (CSR) and
//! columns (CSC) for computation, as in `scipy.sparse`.

use crate::error::{LinalgError, Result};
use ndarray::Array2;

/// A matrix being assembled from `(row, column, value)` triplets; duplicates are summed when
/// converting (SciPy's `coo_array`).
#[derive(Clone, Debug, Default)]
pub struct CooMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub values: Vec<f64>,
}

impl CooMatrix {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        CooMatrix { nrows, ncols, ..Default::default() }
    }

    /// Adds `value` at `(row, col)` (accumulating with any earlier entry there).
    pub fn push(&mut self, row: usize, col: usize, value: f64) {
        self.rows.push(row);
        self.cols.push(col);
        self.values.push(value);
    }

    pub fn to_csr(&self) -> Result<CsrMatrix> {
        CsrMatrix::from_triplets(self.nrows, self.ncols, &self.rows, &self.cols, &self.values)
    }
}

/// Compressed sparse rows: the entries of row `i` are `indices[indptr[i]..indptr[i+1]]` (columns,
/// increasing) and the matching `data` (SciPy's `csr_array`).
#[derive(Clone, Debug, PartialEq)]
pub struct CsrMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
}

/// Compressed sparse columns: the transpose layout of [`CsrMatrix`] (SciPy's `csc_array`).
#[derive(Clone, Debug, PartialEq)]
pub struct CscMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
}

// ANCHOR: from_triplets
impl CsrMatrix {
    /// Builds a CSR matrix from triplets, summing duplicates and sorting the column indices of
    /// each row: a counting sort by row, then by column within each row.
    pub fn from_triplets(nrows: usize, ncols: usize, rows: &[usize], cols: &[usize], values: &[f64]) -> Result<Self> {
        if rows.len() != cols.len() || rows.len() != values.len() {
            return Err(LinalgError::Shape("triplet arrays of different lengths".into()));
        }
        if rows.iter().any(|&r| r >= nrows) || cols.iter().any(|&c| c >= ncols) {
            return Err(LinalgError::Shape("an index lies outside the matrix".into()));
        }
        let mut count = vec![0usize; nrows + 1];
        for &r in rows {
            count[r + 1] += 1;
        }
        for i in 0..nrows {
            count[i + 1] += count[i];
        }
        let mut next = count.clone();
        let mut cols_sorted = vec![0usize; rows.len()];
        let mut vals_sorted = vec![0.0; rows.len()];
        for k in 0..rows.len() {
            let p = next[rows[k]];
            cols_sorted[p] = cols[k];
            vals_sorted[p] = values[k];
            next[rows[k]] += 1;
        }
        let mut indptr = vec![0usize; nrows + 1];
        let mut indices = Vec::with_capacity(rows.len());
        let mut data = Vec::with_capacity(rows.len());
        for i in 0..nrows {
            let mut entries: Vec<(usize, f64)> = (count[i]..count[i + 1]).map(|p| (cols_sorted[p], vals_sorted[p])).collect();
            entries.sort_by_key(|e| e.0);
            for (c, v) in entries {
                if indices.len() > indptr[i] && *indices.last().expect("non-empty") == c {
                    *data.last_mut().expect("non-empty") += v;
                } else {
                    indices.push(c);
                    data.push(v);
                }
            }
            indptr[i + 1] = indices.len();
        }
        Ok(CsrMatrix { nrows, ncols, indptr, indices, data })
    }
    // ANCHOR_END: from_triplets

    /// The entries of `a` larger than `tol` in magnitude.
    pub fn from_dense(a: &Array2<f64>, tol: f64) -> Self {
        let (m, n) = a.dim();
        let mut indptr = vec![0];
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        for i in 0..m {
            for j in 0..n {
                if a[(i, j)].abs() > tol {
                    indices.push(j);
                    data.push(a[(i, j)]);
                }
            }
            indptr.push(indices.len());
        }
        CsrMatrix { nrows: m, ncols: n, indptr, indices, data }
    }

    /// The `n × n` identity.
    pub fn identity(n: usize) -> Self {
        CsrMatrix { nrows: n, ncols: n, indptr: (0..=n).collect(), indices: (0..n).collect(), data: vec![1.0; n] }
    }

    /// A matrix with the given diagonals (SciPy's `diags`): `diagonals[k]` holds the values of the
    /// diagonal at `offsets[k]` (0 main, positive above), starting from its first element.
    pub fn diags(diagonals: &[Vec<f64>], offsets: &[isize], nrows: usize, ncols: usize) -> Result<Self> {
        let mut coo = CooMatrix::new(nrows, ncols);
        for (d, &off) in diagonals.iter().zip(offsets) {
            let (r0, c0) = if off >= 0 { (0usize, off as usize) } else { ((-off) as usize, 0usize) };
            for (t, &v) in d.iter().enumerate() {
                let (r, c) = (r0 + t, c0 + t);
                if r < nrows && c < ncols {
                    coo.push(r, c, v);
                }
            }
        }
        coo.to_csr()
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn to_dense(&self) -> Array2<f64> {
        let mut a = Array2::zeros((self.nrows, self.ncols));
        for i in 0..self.nrows {
            for p in self.indptr[i]..self.indptr[i + 1] {
                a[(i, self.indices[p])] += self.data[p];
            }
        }
        a
    }

    /// The entry at `(i, j)` (zero if not stored).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let row = &self.indices[self.indptr[i]..self.indptr[i + 1]];
        row.binary_search(&j).map_or(0.0, |p| self.data[self.indptr[i] + p])
    }

    /// `A x`.
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        (0..self.nrows).map(|i| (self.indptr[i]..self.indptr[i + 1]).map(|p| self.data[p] * x[self.indices[p]]).sum()).collect()
    }

    /// `Aᵀ x`.
    pub fn rmatvec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.ncols];
        for i in 0..self.nrows {
            for p in self.indptr[i]..self.indptr[i + 1] {
                y[self.indices[p]] += self.data[p] * x[i];
            }
        }
        y
    }

    /// The transpose, again in CSR form.
    pub fn transpose(&self) -> CsrMatrix {
        let c = self.to_csc();
        CsrMatrix { nrows: self.ncols, ncols: self.nrows, indptr: c.indptr, indices: c.indices, data: c.data }
    }

    /// The same matrix in CSC form.
    pub fn to_csc(&self) -> CscMatrix {
        let mut count = vec![0usize; self.ncols + 1];
        for &j in &self.indices {
            count[j + 1] += 1;
        }
        for j in 0..self.ncols {
            count[j + 1] += count[j];
        }
        let mut next = count.clone();
        let mut indices = vec![0usize; self.nnz()];
        let mut data = vec![0.0; self.nnz()];
        for i in 0..self.nrows {
            for p in self.indptr[i]..self.indptr[i + 1] {
                let j = self.indices[p];
                indices[next[j]] = i;
                data[next[j]] = self.data[p];
                next[j] += 1;
            }
        }
        CscMatrix { nrows: self.nrows, ncols: self.ncols, indptr: count, indices, data }
    }

    /// `A B` by Gustavson's row-by-row algorithm.
    pub fn matmul(&self, b: &CsrMatrix) -> Result<CsrMatrix> {
        if self.ncols != b.nrows {
            return Err(LinalgError::Shape(format!("{} × {} times {} × {}", self.nrows, self.ncols, b.nrows, b.ncols)));
        }
        let mut acc = vec![0.0; b.ncols];
        let mut marker = vec![usize::MAX; b.ncols];
        let mut indptr = vec![0];
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        for i in 0..self.nrows {
            let mut cols = Vec::new();
            for p in self.indptr[i]..self.indptr[i + 1] {
                let (k, a) = (self.indices[p], self.data[p]);
                for q in b.indptr[k]..b.indptr[k + 1] {
                    let j = b.indices[q];
                    if marker[j] != i {
                        marker[j] = i;
                        acc[j] = 0.0;
                        cols.push(j);
                    }
                    acc[j] += a * b.data[q];
                }
            }
            cols.sort_unstable();
            for j in cols {
                indices.push(j);
                data.push(acc[j]);
            }
            indptr.push(indices.len());
        }
        Ok(CsrMatrix { nrows: self.nrows, ncols: b.ncols, indptr, indices, data })
    }

    /// `α A + β B`.
    pub fn add(&self, alpha: f64, b: &CsrMatrix, beta: f64) -> Result<CsrMatrix> {
        if (self.nrows, self.ncols) != (b.nrows, b.ncols) {
            return Err(LinalgError::Shape("matrices of different shapes".into()));
        }
        let mut coo = CooMatrix::new(self.nrows, self.ncols);
        for (m, s) in [(self, alpha), (b, beta)] {
            for i in 0..m.nrows {
                for p in m.indptr[i]..m.indptr[i + 1] {
                    coo.push(i, m.indices[p], s * m.data[p]);
                }
            }
        }
        coo.to_csr()
    }

    /// The main diagonal.
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.nrows.min(self.ncols)).map(|i| self.get(i, i)).collect()
    }
}

impl CscMatrix {
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// `A x`.
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.nrows];
        for j in 0..self.ncols {
            for p in self.indptr[j]..self.indptr[j + 1] {
                y[self.indices[p]] += self.data[p] * x[j];
            }
        }
        y
    }

    pub fn to_csr(&self) -> CsrMatrix {
        let t = CsrMatrix { nrows: self.ncols, ncols: self.nrows, indptr: self.indptr.clone(), indices: self.indices.clone(), data: self.data.clone() };
        t.transpose()
    }
}
