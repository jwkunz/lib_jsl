//! Fermi-Dirac integrals.

use std::sync::OnceLock;

use crate::gamma::gamma;

const GL_ORDER: usize = 20;

/// Gauss-Legendre nodes and weights on [-1, 1], computed by Newton iteration.
fn gauss_legendre() -> &'static [(f64, f64); GL_ORDER] {
    static NODES: OnceLock<[(f64, f64); GL_ORDER]> = OnceLock::new();
    NODES.get_or_init(|| {
        let n = GL_ORDER;
        let mut out = [(0.0, 0.0); GL_ORDER];
        for i in 0..n {
            let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
            let mut dp = 1.0;
            for _ in 0..100 {
                let (mut p0, mut p1) = (1.0, x);
                for k in 2..=n {
                    let kf = k as f64;
                    let p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
                    p0 = p1;
                    p1 = p2;
                }
                dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
                let dx = p1 / dp;
                x -= dx;
                if dx.abs() < 1e-16 {
                    break;
                }
            }
            out[i] = (x, 2.0 / ((1.0 - x * x) * dp * dp));
        }
        out
    })
}

/// Chooses the exponent `k` of the substitution `t = u^k` that removes the endpoint singularity
/// of `t^j` (smooth integrand when `k (j + 1)` is an integer >= 1).
fn substitution_power(j: f64) -> u32 {
    for k in 1..=12u32 {
        let e = k as f64 * (j + 1.0);
        if (e - e.round()).abs() < 1e-9 && e.round() >= 1.0 {
            return k;
        }
    }
    ((3.0 / (j + 1.0)).ceil() as u32).clamp(1, 400)
}

/// The complete Fermi-Dirac integral of real order `j > -1`,
/// `F_j(x) = 1/Γ(j+1) ∫₀^∞ t^j / (1 + e^(t-x)) dt`.
///
/// Evaluated by composite 20-point Gauss-Legendre quadrature after a substitution that removes the
/// endpoint singularity. Accuracy is about 1e-13 for orders where `k(j+1)` is an integer for a small
/// `k` (all integer and half-integer orders) and degrades gradually otherwise (and as `j → -1`).
/// Returns `NaN` for `j <= -1`.
pub fn fermi_dirac_integral(j: f64, x: f64) -> f64 {
    if j.is_nan() || x.is_nan() || j <= -1.0 {
        return f64::NAN;
    }
    if x == f64::NEG_INFINITY {
        return 0.0;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if j == 0.0 {
        // F_0(x) = ln(1 + e^x), exact.
        return if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() };
    }
    let k = substitution_power(j);
    let kf = k as f64;
    let top = x.max(0.0) + 45.0;
    let umax = top.powf(1.0 / kf);
    let panels = ((top / 1.5).ceil() as usize * k.min(20) as usize).clamp(32, 20_000);
    let h = umax / panels as f64;
    let nodes = gauss_legendre();
    let expo = kf * (j + 1.0) - 1.0;
    let mut sum = 0.0;
    for p in 0..panels {
        let a = p as f64 * h;
        let mid = a + 0.5 * h;
        for &(xn, w) in nodes.iter() {
            let u = mid + 0.5 * h * xn;
            let t = u.powf(kf);
            let z = t - x;
            let f = if z > 700.0 { 0.0 } else { 1.0 / (1.0 + z.exp()) };
            sum += w * kf * u.powf(expo) * f;
        }
    }
    sum * 0.5 * h / gamma(j + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs()
    }

    #[test]
    fn order_zero_exact() {
        assert!(close(fermi_dirac_integral(0.0, 3.0), 3.048587351573742, 1e-15));
        assert!(close(fermi_dirac_integral(0.0, -40.0), (-40.0f64).exp(), 1e-15));
    }

    #[test]
    fn integer_orders() {
        assert!(close(fermi_dirac_integral(1.0, 0.0), std::f64::consts::PI.powi(2) / 12.0, 1e-13));
        assert!(close(fermi_dirac_integral(2.0, 0.0), 0.9015426773696957, 1e-13));
        assert!(close(fermi_dirac_integral(1.0, 30.0), 451.64493406684813, 1e-13));
        assert!(close(fermi_dirac_integral(3.0, -0.5), 0.585866475325089, 1e-13));
    }

    #[test]
    fn half_integer_orders() {
        assert!(close(fermi_dirac_integral(0.5, 1.0), 1.5756407761513002, 1e-12));
        assert!(close(fermi_dirac_integral(0.5, -2.0), 0.12929851332007559, 1e-12));
        assert!(close(fermi_dirac_integral(-0.5, 0.3), 0.7238460148294637, 1e-12));
        assert!(close(fermi_dirac_integral(1.5, 5.0), 20.914467402762629, 1e-12));
        assert!(close(fermi_dirac_integral(-0.5, -1.0), 0.29402761761145122, 1e-12));
        assert!(close(fermi_dirac_integral(0.5, 100.0), 752.3455915521961, 1e-12));
    }

    #[test]
    fn limits_and_invalid() {
        assert_eq!(fermi_dirac_integral(1.0, f64::NEG_INFINITY), 0.0);
        assert!(fermi_dirac_integral(-1.0, 0.0).is_nan());
        // Non-degenerate limit F_j(x) ~ e^x for x -> -inf
        assert!(close(fermi_dirac_integral(1.3, -20.0), (-20.0f64).exp(), 1e-6));
    }
}
