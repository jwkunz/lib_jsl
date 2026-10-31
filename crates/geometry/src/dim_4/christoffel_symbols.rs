//! Christoffel symbols of the Levi-Civita connection in four dimensions.
//!
//! For a metric `g_{μν}(x)` the symbols are
//! `Γ^λ_{μν} = ½ g^{λσ} (∂_μ g_{σν} + ∂_ν g_{σμ} - ∂_σ g_{μν})`.
//! They are stored as `gamma[λ][μ][ν]` (symmetric in the lower pair) and give the geodesic
//! equation `d²x^λ/dτ² = -Γ^λ_{μν} dx^μ/dτ dx^ν/dτ`.

/// A 4×4 metric tensor `g[μ][ν]`.
pub type Metric4 = [[f64; 4]; 4];

/// The Christoffel symbols `Γ^λ_{μν}` at one point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChristoffelSymbols4D {
    pub gamma: [[[f64; 4]; 4]; 4],
}

/// Inverse of a 4×4 matrix by Gauss-Jordan elimination with partial pivoting; `None` if singular.
pub fn invert_4x4(m: &Metric4) -> Option<Metric4> {
    let mut a = *m;
    let mut inv = [[0.0; 4]; 4];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for col in 0..4 {
        let pivot = (col..4).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let d = a[col][col];
        for j in 0..4 {
            a[col][j] /= d;
            inv[col][j] /= d;
        }
        for i in 0..4 {
            if i != col {
                let f = a[i][col];
                for j in 0..4 {
                    a[i][j] -= f * a[col][j];
                    inv[i][j] -= f * inv[col][j];
                }
            }
        }
    }
    Some(inv)
}

impl ChristoffelSymbols4D {
    /// The symbols from the metric `g` and its partial derivatives `dg[σ][μ][ν] = ∂_σ g_{μν}`
    /// at a point. `None` if the metric is singular there.
    pub fn from_metric_derivatives(g: &Metric4, dg: &[Metric4; 4]) -> Option<Self> {
        let inv = invert_4x4(g)?;
        let mut gamma = [[[0.0; 4]; 4]; 4];
        for (l, plane) in gamma.iter_mut().enumerate() {
            for (m, row) in plane.iter_mut().enumerate() {
                for (n, entry) in row.iter_mut().enumerate() {
                    *entry = 0.5 * (0..4).map(|s| inv[l][s] * (dg[m][s][n] + dg[n][s][m] - dg[s][m][n])).sum::<f64>();
                }
            }
        }
        Some(ChristoffelSymbols4D { gamma })
    }

    /// The symbols of the metric field `metric(x)` at `point`, with the derivatives taken by
    /// fourth-order central differences of step `h` (a relative step near `1e-3` works well for
    /// smooth metrics). `None` if the metric is singular at `point`.
    pub fn from_metric_function(metric: impl Fn(&[f64; 4]) -> Metric4, point: [f64; 4], h: f64) -> Option<Self> {
        let g = metric(&point);
        let mut dg = [[[0.0; 4]; 4]; 4];
        for (s, d) in dg.iter_mut().enumerate() {
            let shifted = |k: f64| {
                let mut p = point;
                p[s] += k * h;
                metric(&p)
            };
            let (m2, m1, p1, p2) = (shifted(-2.0), shifted(-1.0), shifted(1.0), shifted(2.0));
            for i in 0..4 {
                for j in 0..4 {
                    d[i][j] = (m2[i][j] - 8.0 * m1[i][j] + 8.0 * p1[i][j] - p2[i][j]) / (12.0 * h);
                }
            }
        }
        Self::from_metric_derivatives(&g, &dg)
    }

    /// Flat Minkowski space in Cartesian coordinates: all symbols vanish.
    pub fn minkowski() -> Self {
        ChristoffelSymbols4D { gamma: [[[0.0; 4]; 4]; 4] }
    }

    /// Exact symbols of the Schwarzschild metric of mass `mass` (units `G = c = 1`) in coordinates
    /// `(t, r, θ, φ)` at radius `r > 2·mass` and polar angle `theta` (not a pole). `None` outside
    /// that domain.
    pub fn schwarzschild(mass: f64, r: f64, theta: f64) -> Option<Self> {
        let s = theta.sin();
        if r <= 2.0 * mass || s.abs() < 1e-12 {
            return None;
        }
        let f = 1.0 - 2.0 * mass / r;
        let mut g = [[[0.0; 4]; 4]; 4];
        let set = |g: &mut [[[f64; 4]; 4]; 4], l: usize, a: usize, b: usize, v: f64| {
            g[l][a][b] = v;
            g[l][b][a] = v;
        };
        set(&mut g, 0, 0, 1, mass / (r * r * f));
        set(&mut g, 1, 0, 0, mass * f / (r * r));
        set(&mut g, 1, 1, 1, -mass / (r * r * f));
        set(&mut g, 1, 2, 2, -r * f);
        set(&mut g, 1, 3, 3, -r * f * s * s);
        set(&mut g, 2, 1, 2, 1.0 / r);
        set(&mut g, 2, 3, 3, -s * theta.cos());
        set(&mut g, 3, 1, 3, 1.0 / r);
        set(&mut g, 3, 2, 3, theta.cos() / s);
        Some(ChristoffelSymbols4D { gamma: g })
    }

    /// The coordinate acceleration `d²x/dτ² = -Γ^λ_{μν} u^μ u^ν` of a free particle with
    /// coordinate velocity `u = dx/dτ`.
    pub fn geodesic_acceleration(&self, u: [f64; 4]) -> [f64; 4] {
        std::array::from_fn(|l| -(0..4).flat_map(|m| (0..4).map(move |n| (m, n))).map(|(m, n)| self.gamma[l][m][n] * u[m] * u[n]).sum::<f64>())
    }

    /// Whether the symbols are symmetric in their lower indices (always true for Levi-Civita).
    pub fn is_torsion_free(&self, tol: f64) -> bool {
        (0..4).all(|l| (0..4).all(|m| (0..4).all(|n| (self.gamma[l][m][n] - self.gamma[l][n][m]).abs() <= tol)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_spherical(x: &[f64; 4]) -> Metric4 {
        // ds² = -dt² + dr² + r² dθ² + r² sin²θ dφ²
        let (r, th) = (x[1], x[2]);
        let mut g = [[0.0; 4]; 4];
        g[0][0] = -1.0;
        g[1][1] = 1.0;
        g[2][2] = r * r;
        g[3][3] = r * r * th.sin().powi(2);
        g
    }

    fn schwarzschild_metric(mass: f64) -> impl Fn(&[f64; 4]) -> Metric4 {
        move |x| {
            let (r, th) = (x[1], x[2]);
            let f = 1.0 - 2.0 * mass / r;
            let mut g = [[0.0; 4]; 4];
            g[0][0] = -f;
            g[1][1] = 1.0 / f;
            g[2][2] = r * r;
            g[3][3] = r * r * th.sin().powi(2);
            g
        }
    }

    #[test]
    fn flat_space_in_spherical_coordinates() {
        let c = ChristoffelSymbols4D::from_metric_function(flat_spherical, [0.0, 2.0, 1.0, 0.3], 1e-3).unwrap();
        let th: f64 = 1.0;
        assert!((c.gamma[1][2][2] + 2.0).abs() < 1e-8); // Γ^r_θθ = -r
        assert!((c.gamma[1][3][3] + 2.0 * th.sin().powi(2)).abs() < 1e-8); // Γ^r_φφ = -r sin²θ
        assert!((c.gamma[2][1][2] - 0.5).abs() < 1e-8); // Γ^θ_rθ = 1/r
        assert!((c.gamma[3][2][3] - th.cos() / th.sin()).abs() < 1e-8); // Γ^φ_θφ = cot θ
        assert!(c.gamma[0].iter().flatten().all(|v| v.abs() < 1e-9));
        assert!(c.is_torsion_free(1e-12));
        assert_eq!(ChristoffelSymbols4D::minkowski().geodesic_acceleration([1.0, 0.3, 0.2, 0.1]), [0.0; 4]);
    }

    #[test]
    fn numerical_symbols_match_the_exact_schwarzschild_ones() {
        let (m, r, th) = (1.0, 6.0, 1.2);
        let exact = ChristoffelSymbols4D::schwarzschild(m, r, th).unwrap();
        let num = ChristoffelSymbols4D::from_metric_function(schwarzschild_metric(m), [0.0, r, th, 0.0], 1e-3).unwrap();
        for l in 0..4 {
            for a in 0..4 {
                for b in 0..4 {
                    assert!((exact.gamma[l][a][b] - num.gamma[l][a][b]).abs() < 1e-8, "Γ^{l}_{a}{b}: {} vs {}", exact.gamma[l][a][b], num.gamma[l][a][b]);
                }
            }
        }
        assert!(exact.is_torsion_free(0.0));
        // Γ^t_tr = M / (r (r - 2M)).
        assert!((exact.gamma[0][0][1] - m / (r * (r - 2.0 * m))).abs() < 1e-15);
    }

    #[test]
    fn geodesic_motion_in_schwarzschild() {
        let (m, r) = (1.0, 10.0);
        let c = ChristoffelSymbols4D::schwarzschild(m, r, std::f64::consts::FRAC_PI_2).unwrap();
        // A particle momentarily at rest (u = (u^t, 0, 0, 0)) falls inward at M (r - 2M) / r³ (u^t)².
        let f = 1.0 - 2.0 * m / r;
        let ut = 1.0 / f.sqrt();
        let a = c.geodesic_acceleration([ut, 0.0, 0.0, 0.0]);
        assert!(a[1] < 0.0);
        assert!((a[1] + m * f / (r * r) * ut * ut).abs() < 1e-15);
        // Newtonian limit: acceleration ≈ -M/r² at large r.
        let far = ChristoffelSymbols4D::schwarzschild(m, 1.0e4, std::f64::consts::FRAC_PI_2).unwrap();
        let af = far.geodesic_acceleration([1.0, 0.0, 0.0, 0.0]);
        assert!((af[1] * 1.0e8 + 1.0).abs() < 1e-3);
        // Domain checks.
        assert!(ChristoffelSymbols4D::schwarzschild(1.0, 1.5, 1.0).is_none() && ChristoffelSymbols4D::schwarzschild(1.0, 5.0, 0.0).is_none());
        let singular = |_: &[f64; 4]| [[0.0; 4]; 4];
        assert!(ChristoffelSymbols4D::from_metric_function(singular, [0.0; 4], 1e-3).is_none());
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn matrix_inverse() {
        let g = schwarzschild_metric(1.0)(&[0.0, 5.0, 1.0, 0.0]);
        let inv = invert_4x4(&g).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let p: f64 = (0..4).map(|k| g[i][k] * inv[k][j]).sum();
                assert!((p - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }
}
