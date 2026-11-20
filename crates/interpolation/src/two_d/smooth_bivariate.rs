//! Smoothing spline surfaces for scattered data (SciPy's `bisplrep` / `bisplev` and
//! `SmoothBivariateSpline`, `LSQBivariateSpline`).
//!
//! See the book entry [FITPACK smoothing splines](https://jwkunz.github.io/lib_jsl/book/interpolation/fitpack.html#periodic-splines-and-smoothing-surfaces).

use crate::one_d::b_spline::nonzero_basis;
use crate::one_d::fitpack::{discontinuity_jumps, fprati};
use lib_jsl_core::ErrorsJSL;

/// A tensor-product spline surface `s(x, y) = Σᵢⱼ c[i][j] Bᵢ(x) Bⱼ(y)` (SciPy's `tck` from
/// `bisplrep`), with the weighted sum of squared residuals of its fit.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SmoothBivariateSpline {
    pub tx: Vec<f64>,
    pub ty: Vec<f64>,
    pub kx: usize,
    pub ky: usize,
    /// `c[i][j]` for `i < nx − kx − 1`, `j < ny − ky − 1`.
    pub c: Vec<Vec<f64>>,
    pub fp: f64,
}

fn interval(t: &[f64], k: usize, v: f64) -> usize {
    let n = t.len();
    let mut l = k;
    while l < n - k - 2 && v >= t[l + 1] {
        l += 1;
    }
    l
}

impl SmoothBivariateSpline {
    /// The surface's value (SciPy's `bisplev` at one point). Outside the knot span the
    /// arguments are clamped to it, as FITPACK's `fpbisp` does.
    pub fn evaluate(&self, x: f64, y: f64) -> f64 {
        let (nx, ny) = (self.tx.len(), self.ty.len());
        let x = x.clamp(self.tx[self.kx], self.tx[nx - self.kx - 1]);
        let y = y.clamp(self.ty[self.ky], self.ty[ny - self.ky - 1]);
        let (lx, ly) = (interval(&self.tx, self.kx, x), interval(&self.ty, self.ky, y));
        let bx = nonzero_basis(&self.tx, self.kx, lx, x);
        let by = nonzero_basis(&self.ty, self.ky, ly, y);
        let mut s = 0.0;
        for (a, u) in bx.iter().enumerate() {
            for (b, v) in by.iter().enumerate() {
                s += u * v * self.c[lx - self.kx + a][ly - self.ky + b];
            }
        }
        s
    }

    /// Values on the grid `x × y` (SciPy's `bisplev` with arrays): `out[i][j] = s(xᵢ, yⱼ)`.
    pub fn evaluate_grid(&self, x: &[f64], y: &[f64]) -> Vec<Vec<f64>> {
        x.iter().map(|&a| y.iter().map(|&b| self.evaluate(a, b)).collect()).collect()
    }
}

/// Dense Householder least squares (the problem sizes are a few hundred unknowns).
#[allow(clippy::needless_range_loop)]
fn dense_lsq(mut a: Vec<Vec<f64>>, mut b: Vec<f64>, nc: usize) -> Vec<f64> {
    let m = a.len();
    for j in 0..nc.min(m) {
        let norm: f64 = (j..m).map(|i| a[i][j] * a[i][j]).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        let alpha = if a[j][j] > 0.0 { -norm } else { norm };
        let mut v: Vec<f64> = (j..m).map(|i| a[i][j]).collect();
        v[0] -= alpha;
        let vn: f64 = v.iter().map(|x| x * x).sum();
        if vn == 0.0 {
            continue;
        }
        for col in j..nc {
            let s: f64 = (j..m).map(|i| v[i - j] * a[i][col]).sum::<f64>() * 2.0 / vn;
            if s != 0.0 {
                for i in j..m {
                    a[i][col] -= s * v[i - j];
                }
            }
        }
        let s: f64 = (j..m).map(|i| v[i - j] * b[i]).sum::<f64>() * 2.0 / vn;
        for i in j..m {
            b[i] -= s * v[i - j];
        }
    }
    let scale = (0..nc.min(m)).map(|i| a[i][i].abs()).fold(0.0f64, f64::max);
    let mut c = vec![0.0; nc];
    for i in (0..nc.min(m)).rev() {
        // Rank deficiency (a panel without data): leave that coefficient at zero, as FITPACK's
        // minimum-norm treatment does for negligible pivots.
        if a[i][i].abs() <= 1e-12 * scale {
            continue;
        }
        let s: f64 = (i + 1..nc).map(|j| a[i][j] * c[j]).sum();
        c[i] = (b[i] - s) / a[i][i];
    }
    c
}

struct Scattered<'a> {
    x: &'a [f64],
    y: &'a [f64],
    z: &'a [f64],
    w: Vec<f64>,
    kx: usize,
    ky: usize,
}

/// The fit on knots `tx`, `ty` with the jump penalty weighted by `1/p` (`None`: least squares).
fn fit(d: &Scattered, tx: &[f64], ty: &[f64], p: Option<f64>) -> (SmoothBivariateSpline, Vec<f64>) {
    let (kx, ky) = (d.kx, d.ky);
    let (ncx, ncy) = (tx.len() - kx - 1, ty.len() - ky - 1);
    let nc = ncx * ncy;
    let mut rows = Vec::with_capacity(d.x.len());
    let mut rhs = Vec::new();
    for i in 0..d.x.len() {
        let (lx, ly) = (interval(tx, kx, d.x[i]), interval(ty, ky, d.y[i]));
        let bx = nonzero_basis(tx, kx, lx, d.x[i]);
        let by = nonzero_basis(ty, ky, ly, d.y[i]);
        let mut row = vec![0.0; nc];
        for (a, u) in bx.iter().enumerate() {
            for (b, v) in by.iter().enumerate() {
                row[(lx - kx + a) * ncy + ly - ky + b] = u * v * d.w[i];
            }
        }
        rows.push(row);
        rhs.push(d.z[i] * d.w[i]);
    }
    if let Some(p) = p {
        // Jumps of the kx-th x-derivative across interior x knots, for every y coefficient, and
        // likewise in y (FITPACK's surfit penalty).
        if tx.len() > 2 * kx + 2 {
            for jump in discontinuity_jumps(tx, kx).iter().enumerate() {
                for j in 0..ncy {
                    let mut row = vec![0.0; nc];
                    for (r, v) in jump.1.iter().enumerate() {
                        row[(jump.0 + r) * ncy + j] = v / p;
                    }
                    rows.push(row);
                    rhs.push(0.0);
                }
            }
        }
        if ty.len() > 2 * ky + 2 {
            for jump in discontinuity_jumps(ty, ky).iter().enumerate() {
                for i in 0..ncx {
                    let mut row = vec![0.0; nc];
                    for (r, v) in jump.1.iter().enumerate() {
                        row[i * ncy + jump.0 + r] = v / p;
                    }
                    rows.push(row);
                    rhs.push(0.0);
                }
            }
        }
    }
    let flat = dense_lsq(rows, rhs, nc);
    let c: Vec<Vec<f64>> = flat.chunks(ncy).map(|r| r.to_vec()).collect();
    let mut spl = SmoothBivariateSpline { tx: tx.to_vec(), ty: ty.to_vec(), kx, ky, c, fp: 0.0 };
    let res: Vec<f64> = (0..d.x.len()).map(|i| (d.w[i] * (d.z[i] - spl.evaluate(d.x[i], d.y[i]))).powi(2)).collect();
    spl.fp = res.iter().sum();
    (spl, res)
}

fn validate<'a>(x: &'a [f64], y: &'a [f64], z: &'a [f64], w: Option<&[f64]>, kx: usize, ky: usize) -> Result<Scattered<'a>, ErrorsJSL> {
    let m = x.len();
    if y.len() != m || z.len() != m {
        return Err(ErrorsJSL::IncompatibleArraySizes((y.len().min(z.len()), m)));
    }
    if !(1..=5).contains(&kx) || !(1..=5).contains(&ky) || m < (kx + 1) * (ky + 1) {
        return Err(ErrorsJSL::InvalidInputRange("bisplrep: need 1 ≤ kx, ky ≤ 5 and at least (kx+1)(ky+1) points"));
    }
    let w = match w {
        Some(w) if w.len() != m => return Err(ErrorsJSL::IncompatibleArraySizes((w.len(), m))),
        Some(w) => w.to_vec(),
        None => vec![1.0; m],
    };
    Ok(Scattered { x, y, z, w, kx, ky })
}

// ANCHOR: bisplrep_lsq
/// The weighted least-squares spline surface on given knots (SciPy's `bisplrep(task=-1)` and
/// `LSQBivariateSpline`). As in FITPACK, only the interior knots of `tx` and `ty` are used: the
/// `k + 1` boundary knots at each end are placed at the data's range (SciPy's default `xb`, `xe`,
/// `yb`, `ye`). Coefficients that no data point influences are set to zero.
#[allow(clippy::too_many_arguments)]
pub fn bisplrep_lsq(x: &[f64], y: &[f64], z: &[f64], w: Option<&[f64]>, tx: &[f64], ty: &[f64], kx: usize, ky: usize) -> Result<SmoothBivariateSpline, ErrorsJSL> {
    let d = validate(x, y, z, w, kx, ky)?;
    if tx.len() < 2 * kx + 2 || ty.len() < 2 * ky + 2 {
        return Err(ErrorsJSL::InvalidInputRange("bisplrep_lsq: need at least 2k + 2 knots in each direction"));
    }
    let ends = |t: &[f64], k: usize, v: &[f64]| -> Vec<f64> {
        let (b, e) = (v.iter().copied().fold(f64::INFINITY, f64::min), v.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        std::iter::repeat_n(b, k + 1).chain(t[k + 1..t.len() - k - 1].iter().copied()).chain(std::iter::repeat_n(e, k + 1)).collect()
    };
    Ok(fit(&d, &ends(tx, kx, x), &ends(ty, ky, y), None).0)
}
// ANCHOR_END: bisplrep_lsq

// ANCHOR: bisplrep
/// A smoothing spline surface through scattered data with automatically placed knots (SciPy's
/// `bisplrep` / `SmoothBivariateSpline`, after FITPACK's `surfit`): of the splines with
/// `Σ (wᵢ (zᵢ − s(xᵢ, yᵢ)))² ≤ smoothing`, the one whose `kx`-th x- and `ky`-th y-derivatives
/// jump least across the knots. `smoothing = None` gives SciPy's default `m − √(2m)`.
///
/// Starting from a single polynomial patch, knots are added at the middle of the panel with the
/// largest residual sum, in the direction with fewer interior knots so far, until the
/// least-squares surface reaches the smoothing target or `nxest`, `nyest` knots (SciPy's
/// defaults); then the penalty weight `p` is found by FITPACK's rational iteration so that the
/// residual sum equals the target.
#[allow(clippy::too_many_arguments)]
pub fn bisplrep(x: &[f64], y: &[f64], z: &[f64], w: Option<&[f64]>, kx: usize, ky: usize, smoothing: Option<f64>) -> Result<SmoothBivariateSpline, ErrorsJSL> {
    let d = validate(x, y, z, w, kx, ky)?;
    let m = x.len() as f64;
    let s = smoothing.unwrap_or(m - (2.0 * m).sqrt());
    let acc = 0.001 * s;
    let (xb, xe) = (x.iter().copied().fold(f64::INFINITY, f64::min), x.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let (yb, ye) = (y.iter().copied().fold(f64::INFINITY, f64::min), y.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let nxest = ((kx as f64 + 1.0 + (m / 2.0).sqrt()) as usize).max(2 * kx + 3);
    let nyest = ((ky as f64 + 1.0 + (m / 2.0).sqrt()) as usize).max(2 * ky + 3);
    let ends = |b: f64, e: f64, k: usize| -> Vec<f64> { std::iter::repeat_n(b, k + 1).chain(std::iter::repeat_n(e, k + 1)).collect() };
    let (mut tx, mut ty) = (ends(xb, xe, kx), ends(yb, ye, ky));
    let (mut spl, mut res) = fit(&d, &tx, &ty, None);
    let poly_fp = spl.fp;
    while spl.fp > s + acc && (tx.len() < nxest || ty.len() < nyest) {
        // The panel with the largest residual sum.
        let (px, py) = (tx.len() - 2 * kx - 1, ty.len() - 2 * ky - 1);
        let mut sums = vec![vec![0.0; py]; px];
        let mut counts = vec![vec![Vec::new(); py]; px];
        for i in 0..x.len() {
            let (lx, ly) = (interval(&tx, kx, x[i]) - kx, interval(&ty, ky, y[i]) - ky);
            sums[lx][ly] += res[i];
            counts[lx][ly].push(i);
        }
        let mut order: Vec<(usize, usize)> = (0..px).flat_map(|a| (0..py).map(move |b| (a, b))).collect();
        order.sort_by(|a, b| sums[b.0][b.1].total_cmp(&sums[a.0][a.1]));
        let mut added = false;
        for (a, b) in order {
            let pts = &counts[a][b];
            let distinct = |v: &dyn Fn(usize) -> f64| {
                let mut c: Vec<f64> = pts.iter().map(|&i| v(i)).collect();
                c.sort_by(f64::total_cmp);
                c.dedup();
                c.len()
            };
            let (nxd, nyd) = (distinct(&|i| x[i]), distinct(&|i| y[i]));
            let can_x = tx.len() < nxest && nxd > 1;
            let can_y = ty.len() < nyest && nyd > 1;
            // Keep the two directions balanced, as FITPACK does: split in the direction with
            // fewer interior knots (x on a tie).
            let in_x = match (can_x, can_y) {
                (true, true) => tx.len() - 2 * kx <= ty.len() - 2 * ky,
                (true, false) => true,
                (false, true) => false,
                _ => continue,
            };
            if in_x {
                let knot = 0.5 * (tx[a + kx] + tx[a + kx + 1]);
                tx.insert(a + kx + 1, knot);
            } else {
                let knot = 0.5 * (ty[b + ky] + ty[b + ky + 1]);
                ty.insert(b + ky + 1, knot);
            }
            added = true;
            break;
        }
        if !added {
            break;
        }
        (spl, res) = fit(&d, &tx, &ty, None);
    }
    // A single polynomial patch, a fit already at the target, or a least-squares surface that
    // cannot reach it: nothing to smooth.
    if (spl.fp - s).abs() < acc || (tx.len() == 2 * kx + 2 && ty.len() == 2 * ky + 2) || spl.fp > s {
        return Ok(spl);
    }
    // Rational iteration on p for fp(p) = s, between the polynomial (p = 0) and the
    // least-squares surface (p = ∞).
    let (mut p1, mut f1) = (0.0, poly_fp - s);
    let (mut p3, mut f3) = (f64::INFINITY, spl.fp - s);
    let mut p = 1.0;
    let (con1, con4, con9) = (0.1, 0.04, 0.9);
    let (mut ich1, mut ich3) = (false, false);
    let mut best = spl;
    for _ in 0..40 {
        let (cand, _) = fit(&d, &tx, &ty, Some(p));
        let (p2, f2) = (p, cand.fp - s);
        best = cand;
        if f2.abs() < acc {
            break;
        }
        if !ich3 {
            if f2 - f3 <= acc {
                (p3, f3) = (p2, f2);
                p *= con4;
                if p <= p1 {
                    p = p1 * con9 + p2 * con1;
                }
                continue;
            }
            if f2 < 0.0 {
                ich3 = true;
            }
        }
        if !ich1 {
            if f1 - f2 <= acc {
                (p1, f1) = (p2, f2);
                p /= con4;
                if p3.is_finite() && p <= p3 {
                    p = p2 * con1 + p3 * con9;
                }
                continue;
            }
            if f2 > 0.0 {
                ich1 = true;
            }
        }
        if f1 <= f2 || f2 <= f3 {
            break;
        }
        p = fprati(p1, f1, p2, f2, p3, f3);
        if f2 < 0.0 {
            (p3, f3) = (p2, f2);
        } else {
            (p1, f1) = (p2, f2);
        }
    }
    Ok(best)
}
// ANCHOR_END: bisplrep

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_a_bicubic_and_smooths_noise() {
        let mut s = 3u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let n = 300;
        let x: Vec<f64> = (0..n).map(|_| next()).collect();
        let y: Vec<f64> = (0..n).map(|_| next()).collect();
        let cubic = |a: f64, b: f64| 1.0 + a - 2.0 * b + a * a * b - 0.5 * b.powi(3) + a.powi(3) * b * b;
        let z: Vec<f64> = x.iter().zip(&y).map(|(&a, &b)| cubic(a, b)).collect();
        let t = vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let spl = bisplrep_lsq(&x, &y, &z, None, &t, &t, 3, 3).unwrap();
        assert!(spl.fp < 1e-20 && (spl.evaluate(0.3, 0.7) - cubic(0.3, 0.7)).abs() < 1e-12, "{}", spl.fp);
        let zn: Vec<f64> = x.iter().zip(&y).map(|(&a, &b)| (6.0 * a).sin() * (4.0 * b).cos() + 0.05 * (next() - 0.5)).collect();
        let target = n as f64 * 0.05f64.powi(2) / 12.0;
        let spl = bisplrep(&x, &y, &zn, None, 3, 3, Some(target)).unwrap();
        assert!((spl.fp - target).abs() <= 0.002 * target + 1e-12, "{} {}", spl.fp, target);
        assert!(((spl.evaluate(0.5, 0.5) - (3.0f64).sin() * (2.0f64).cos()).abs()) < 0.05);
    }
}
