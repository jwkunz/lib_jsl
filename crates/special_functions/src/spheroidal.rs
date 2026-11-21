//! Spheroidal wave functions (SciPy's `pro_cv`, `obl_cv`, `pro_ang1`, `obl_ang1`, `pro_rad1`,
//! `obl_rad1`): solutions of `(1 − x²) y'' − 2x y' + (λ − c²x² − m²/(1 − x²)) y = 0` (prolate;
//! `c² → −c²` for oblate), in Flammer's normalisation.
//!
//! See the book entry [Mathieu, parabolic cylinder and spheroidal wave functions](https://jwkunz.github.io/lib_jsl/book/special_functions/wave-functions.html#spheroidal-wave-functions).

use crate::bessel_functions_spherical::bessel_spherical;
use crate::spherical_harmonics::associated_legendre;
use crate::tridiagonal::{eigenvalues, eigenvector};

/// `P_l^m(x)` without the Condon-Shortley phase (SciPy's and Flammer's convention).
fn legendre(l: usize, m: usize, x: f64) -> f64 {
    if l < m {
        return 0.0;
    }
    let p = associated_legendre(l as u32, m as i32, x);
    if m % 2 == 1 { -p } else { p }
}

/// `(2m + r)!/r!`.
fn ratio(m: usize, r: usize) -> f64 {
    ((r + 1)..=(r + 2 * m)).map(|k| k as f64).product()
}

// ANCHOR: flammer
/// The characteristic value `λ_mn(c)` and the expansion coefficients `d_r` (`r ≡ n − m`
/// mod 2, step 2) from Flammer's three-term recurrence
/// `α_r d_{r+2} + (β_r − λ) d_r + γ_r d_{r−2} = 0`, with `κ = c²` (prolate) or `−c²` (oblate):
/// `α_r = (2m+r+2)(2m+r+1) κ / ((2m+2r+3)(2m+2r+5))`,
/// `β_r = (m+r)(m+r+1) + (2(m+r)(m+r+1) − 2m² − 1) κ / ((2m+2r−1)(2m+2r+3))`,
/// `γ_r = r(r−1) κ / ((2m+2r−3)(2m+2r−1))`. The coefficients are normalised so that
/// `Σ (2m+r)!/r! d_r = (n+m)!/(n−m)!`: the angular function tends to `P_n^m` as `c → 0`.
fn solve(m: usize, n: usize, c: f64, oblate: bool) -> (f64, Vec<f64>, usize) {
    let kappa = if oblate { -c * c } else { c * c };
    let p = (n - m) % 2;
    let size = (n - m) / 2 + 30 + (1.5 * c.abs()) as usize;
    let r_of = |j: usize| p + 2 * j;
    let (mf, kf) = (m as f64, kappa);
    let d: Vec<f64> = (0..size)
        .map(|j| {
            let r = r_of(j) as f64;
            (mf + r) * (mf + r + 1.0) + (2.0 * (mf + r) * (mf + r + 1.0) - 2.0 * mf * mf - 1.0) * kf / ((2.0 * mf + 2.0 * r - 1.0) * (2.0 * mf + 2.0 * r + 3.0))
        })
        .collect();
    let sup: Vec<f64> = (0..size - 1)
        .map(|j| {
            let r = r_of(j) as f64;
            (2.0 * mf + r + 2.0) * (2.0 * mf + r + 1.0) * kf / ((2.0 * mf + 2.0 * r + 3.0) * (2.0 * mf + 2.0 * r + 5.0))
        })
        .collect();
    let sub: Vec<f64> = (1..size)
        .map(|j| {
            let r = r_of(j) as f64;
            r * (r - 1.0) * kf / ((2.0 * mf + 2.0 * r - 3.0) * (2.0 * mf + 2.0 * r - 1.0))
        })
        .collect();
    let lam = if c == 0.0 { (n * (n + 1)) as f64 } else { eigenvalues(&d, &sup, &sub)[(n - m) / 2] };
    let mut v = if c == 0.0 {
        let mut e = vec![0.0; size];
        e[(n - m) / 2] = 1.0;
        e
    } else {
        eigenvector(&d, &sup, &sub, lam)
    };
    let s: f64 = v.iter().enumerate().map(|(j, dj)| ratio(m, r_of(j)) * dj).sum();
    let target = ratio(m, n - m);
    v.iter_mut().for_each(|x| *x *= target / s);
    (lam, v, p)
}
// ANCHOR_END: flammer

/// The prolate characteristic value `λ_mn(c)` (SciPy's `pro_cv`), `n ≥ m`.
pub fn pro_cv(m: usize, n: usize, c: f64) -> f64 {
    if n < m { f64::NAN } else { solve(m, n, c, false).0 }
}

/// The oblate characteristic value (SciPy's `obl_cv`).
pub fn obl_cv(m: usize, n: usize, c: f64) -> f64 {
    if n < m { f64::NAN } else { solve(m, n, c, true).0 }
}

fn angular(m: usize, n: usize, c: f64, x: f64, oblate: bool) -> (f64, f64) {
    if n < m || x.abs() >= 1.0 {
        return (f64::NAN, f64::NAN);
    }
    let (_, d, p) = solve(m, n, c, oblate);
    let eval = |x: f64| -> (f64, f64) {
        let (mut s, mut ds) = (0.0, 0.0);
        for (j, dj) in d.iter().enumerate() {
            let l = m + p + 2 * j;
            s += dj * legendre(l, m, x);
            // (1 − x²) P_l^m' = (l + m) P_{l−1}^m − l x P_l^m.
            let below = if l >= 1 { legendre(l - 1, m, x) } else { 0.0 };
            ds += dj * ((l + m) as f64 * below - l as f64 * x * legendre(l, m, x)) / (1.0 - x * x);
        }
        (s, ds)
    };
    // Flammer's normalisation: S(c, 0) = P_n^m(0) for even n − m, S'(c, 0) = P_n^m'(0) for odd.
    let (s0, ds0) = eval(0.0);
    let scale = if p == 0 {
        legendre(n, m, 0.0) / s0
    } else {
        let below = if n >= 1 { legendre(n - 1, m, 0.0) } else { 0.0 };
        (n + m) as f64 * below / ds0
    };
    let (s, ds) = eval(x);
    (scale * s, scale * ds)
}

/// The prolate angular function of the first kind `S_mn^{(1)}(c, x)`, `|x| < 1`, and its
/// derivative (SciPy's `pro_ang1`): `Σ' d_r P_{m+r}^m(x)`, scaled as Flammer's so that
/// `S(c, 0) = P_n^m(0)` (`n − m` even) or `S'(c, 0) = P_n^m'(0)` (odd).
pub fn pro_ang1(m: usize, n: usize, c: f64, x: f64) -> (f64, f64) {
    angular(m, n, c, x, false)
}

/// The oblate angular function of the first kind and its derivative (SciPy's `obl_ang1`).
pub fn obl_ang1(m: usize, n: usize, c: f64, x: f64) -> (f64, f64) {
    angular(m, n, c, x, true)
}

// ANCHOR: radial
/// The radial function of the first kind (Flammer 4.1.17, 5.2.?): with `N = Σ' (2m+r)!/r! d_r`,
/// `R_mn^{(1)}(c, ξ) = (1/N) (1 ∓ 1/ξ²)^{m/2} Σ' i^{r+m−n} (2m+r)!/r! d_r j_{m+r}(cξ)`, the upper
/// sign for prolate (`ξ > 1`), the lower for oblate (`ξ ≥ 0`).
fn radial(m: usize, n: usize, c: f64, xi: f64, oblate: bool) -> (f64, f64) {
    if n < m || (!oblate && xi <= 1.0) || (oblate && xi < 0.0) || c <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let (_, d, p) = solve(m, n, c, oblate);
    let norm: f64 = d.iter().enumerate().map(|(j, dj)| ratio(m, p + 2 * j) * dj).sum();
    let (mut s, mut ds) = (0.0, 0.0);
    for (j, dj) in d.iter().enumerate() {
        let r = p + 2 * j;
        let k = m + r;
        let half = ((r + m) as i64 - n as i64) / 2;
        let sign = if half.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
        let w = sign * ratio(m, r) * dj;
        let z = c * xi;
        let jk = bessel_spherical(k as i32, z);
        // j_k'(z) = j_{k−1}(z) − (k+1)/z j_k(z) (and j_0' = −j_1).
        let djk = if k == 0 { -bessel_spherical(1, z) } else { bessel_spherical(k as i32 - 1, z) - (k + 1) as f64 / z * jk };
        s += w * jk;
        ds += w * c * djk;
    }
    let sgn = if oblate { 1.0 } else { -1.0 };
    let base = 1.0 + sgn / (xi * xi);
    let f = base.powf(m as f64 / 2.0);
    let df = if m == 0 { 0.0 } else { (m as f64 / 2.0) * base.powf(m as f64 / 2.0 - 1.0) * (-2.0 * sgn / xi.powi(3)) };
    ((f * s) / norm, (df * s + f * ds) / norm)
}
// ANCHOR_END: radial

/// The prolate radial function of the first kind `R_mn^{(1)}(c, ξ)`, `ξ > 1`, and its
/// derivative (SciPy's `pro_rad1`).
pub fn pro_rad1(m: usize, n: usize, c: f64, xi: f64) -> (f64, f64) {
    radial(m, n, c, xi, false)
}

/// The oblate radial function of the first kind, `ξ ≥ 0`, and its derivative (SciPy's
/// `obl_rad1`).
pub fn obl_rad1(m: usize, n: usize, c: f64, xi: f64) -> (f64, f64) {
    radial(m, n, c, xi, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_and_ordering() {
        // c → 0: λ = n(n+1) and S → P_n^m.
        assert!((pro_cv(1, 3, 1e-8) - 12.0).abs() < 1e-10);
        let (s, _) = pro_ang1(2, 3, 1e-6, 0.3);
        assert!((s - 15.0 * 0.3 * (1.0 - 0.09)).abs() < 1e-8, "{s}");
        // Prolate values rise and oblate values fall with c; both keep the order in n.
        assert!(pro_cv(0, 2, 3.0) > 6.0 && obl_cv(0, 2, 3.0) < 6.0);
        assert!(pro_cv(1, 2, 5.0) < pro_cv(1, 3, 5.0) && obl_cv(1, 2, 5.0) < obl_cv(1, 3, 5.0));
        // R^(1) behaves like j_n(cξ) for large ξ.
        let (r, _) = pro_rad1(0, 2, 1.0, 60.0);
        let j = crate::bessel_functions_spherical::bessel_spherical(2, 60.0);
        assert!((r - j).abs() < 2e-3, "{r} {j}");
    }
}
