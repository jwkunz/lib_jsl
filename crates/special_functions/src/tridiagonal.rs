//! Eigenvalues and eigenvectors of real tridiagonal matrices whose off-diagonal products are
//! positive (so that they are similar to symmetric matrices), as in the three-term recurrences
//! of Mathieu and spheroidal functions.

/// The eigenvalues, ascending, of the tridiagonal matrix with diagonal `d`, superdiagonal
/// `sup` and subdiagonal `sub` (`sup[i] · sub[i] > 0`), by the implicit QL method on its
/// symmetric form.
pub(crate) fn eigenvalues(d: &[f64], sup: &[f64], sub: &[f64]) -> Vec<f64> {
    let n = d.len();
    let mut dd = d.to_vec();
    let mut e: Vec<f64> = (0..n).map(|i| if i + 1 < n { (sup[i] * sub[i]).abs().sqrt() } else { 0.0 }).collect();
    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                let s = dd[m].abs() + dd[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * s {
                    break;
                }
                m += 1;
            }
            if m == l || iter > 60 {
                break;
            }
            iter += 1;
            let mut g = (dd[l + 1] - dd[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = dd[m] - dd[l] + e[l] / (g + r.copysign(g));
            let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut i = m;
            let mut early = false;
            while i > l {
                i -= 1;
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    dd[i + 1] -= p;
                    e[m] = 0.0;
                    early = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = dd[i + 1] - p;
                r = (dd[i] - g) * s + 2.0 * c * b;
                p = s * r;
                dd[i + 1] = g + p;
                g = c * r - b;
            }
            if early {
                continue;
            }
            dd[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    dd.sort_by(f64::total_cmp);
    dd
}

/// An eigenvector of the (non-symmetric) tridiagonal matrix for the eigenvalue `lambda`, by
/// inverse iteration with a slightly perturbed shift; normalised to unit Euclidean norm.
pub(crate) fn eigenvector(d: &[f64], sup: &[f64], sub: &[f64], lambda: f64) -> Vec<f64> {
    let n = d.len();
    let shift = lambda + 1e-12 * lambda.abs().max(1.0);
    let mut v = vec![1.0; n];
    for _ in 0..3 {
        // Solve (T − shift I) x = v by Gaussian elimination with partial pivoting on the band.
        let mut a: Vec<[f64; 4]> = (0..n).map(|i| [if i > 0 { sub[i - 1] } else { 0.0 }, d[i] - shift, if i + 1 < n { sup[i] } else { 0.0 }, 0.0]).collect();
        let mut b = v.clone();
        // a[i] = [lower, diag, upper, upper2] after pivoting.
        for i in 0..n.saturating_sub(1) {
            if a[i + 1][0].abs() > a[i][1].abs() {
                // Swap rows i and i+1 (row i+1 has entries at i, i+1, i+2).
                let ri = [a[i][1], a[i][2], a[i][3]];
                let rn = [a[i + 1][0], a[i + 1][1], a[i + 1][2]];
                a[i][1] = rn[0];
                a[i][2] = rn[1];
                a[i][3] = rn[2];
                a[i + 1][0] = ri[0];
                a[i + 1][1] = ri[1];
                a[i + 1][2] = ri[2];
                b.swap(i, i + 1);
            }
            let piv = if a[i][1] == 0.0 { 1e-300 } else { a[i][1] };
            let f = a[i + 1][0] / piv;
            a[i + 1][1] -= f * a[i][2];
            a[i + 1][2] -= f * a[i][3];
            b[i + 1] -= f * b[i];
        }
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = b[i];
            if i + 1 < n {
                s -= a[i][2] * x[i + 1];
            }
            if i + 2 < n {
                s -= a[i][3] * x[i + 2];
            }
            let piv = if a[i][1] == 0.0 { 1e-300 } else { a[i][1] };
            x[i] = s / piv;
        }
        let norm = x.iter().map(|t| t * t).sum::<f64>().sqrt();
        v = x.iter().map(|t| t / norm).collect();
    }
    v
}
