//! Classical orthogonal polynomials, evaluated by their stable three-term recurrences.
//!
//! | Family | Weight | Interval | Normalisation |
//! |--------|--------|----------|---------------|
//! | Legendre `P_n` | 1 | [−1, 1] | `P_n(1) = 1` |
//! | Chebyshev `T_n`, `U_n` | `(1−x²)^{∓½}` | [−1, 1] | `T_n(cos t) = cos nt`, `U_n(cos t) = sin((n+1)t)/sin t` |
//! | Jacobi `P_n^{(α,β)}` | `(1−x)^α (1+x)^β` | [−1, 1] | `P_n^{(α,β)}(1) = C(n+α, n)` |
//! | Hermite `H_n` (physicists') | `e^{−x²}` | ℝ | leading coefficient `2ⁿ` |
//! | Hermite `He_n` (probabilists') | `e^{−x²/2}` | ℝ | monic |
//! | Laguerre `L_n^{(α)}` | `x^α e^{−x}` | [0, ∞) | `L_n^{(α)}(0) = C(n+α, n)` |

/// Runs `p_{k+1} = a_k(x) p_k − b_k p_{k−1}` from `p_0 = 1`, `p_1` and returns `(p_n, p_{n−1})`
/// (`p_{−1} = 0`).
fn recurrence(n: u32, p1: f64, step: impl Fn(u32, f64, f64) -> f64) -> (f64, f64) {
    match n {
        0 => (1.0, 0.0),
        _ => {
            let (mut prev, mut cur) = (1.0, p1);
            for k in 1..n {
                let next = step(k, cur, prev);
                prev = cur;
                cur = next;
            }
            (cur, prev)
        }
    }
}

/// The Legendre polynomial `P_n(x)`.
pub fn legendre(n: u32, x: f64) -> f64 {
    legendre_with_derivative(n, x).0
}

/// `(P_n(x), P_n'(x))`, the derivative from `(x² − 1) P_n' = n (x P_n − P_{n−1})` (with the limit
/// `P_n'(±1) = (±1)^{n+1} n(n+1)/2` at the endpoints). Used for Gauss-Legendre nodes.
pub fn legendre_with_derivative(n: u32, x: f64) -> (f64, f64) {
    let (p, q) = recurrence(n, x, |k, cur, prev| {
        let k = k as f64;
        ((2.0 * k + 1.0) * x * cur - k * prev) / (k + 1.0)
    });
    let nf = n as f64;
    let derivative = if (x * x - 1.0).abs() < 1e-14 {
        let sign = if x > 0.0 || n % 2 == 1 { 1.0 } else { -1.0 };
        sign * nf * (nf + 1.0) / 2.0
    } else {
        nf * (x * p - q) / (x * x - 1.0)
    };
    (p, derivative)
}

/// The Chebyshev polynomial of the first kind `T_n(x)`.
pub fn chebyshev_t(n: u32, x: f64) -> f64 {
    recurrence(n, x, |_, cur, prev| 2.0 * x * cur - prev).0
}

/// The Chebyshev polynomial of the second kind `U_n(x)`.
pub fn chebyshev_u(n: u32, x: f64) -> f64 {
    recurrence(n, 2.0 * x, |_, cur, prev| 2.0 * x * cur - prev).0
}

/// The physicists' Hermite polynomial `H_n(x)`.
pub fn hermite(n: u32, x: f64) -> f64 {
    recurrence(n, 2.0 * x, |k, cur, prev| 2.0 * x * cur - 2.0 * k as f64 * prev).0
}

/// The probabilists' Hermite polynomial `He_n(x) = 2^{−n/2} H_n(x/√2)`.
pub fn hermite_he(n: u32, x: f64) -> f64 {
    recurrence(n, x, |k, cur, prev| x * cur - k as f64 * prev).0
}

/// The generalized Laguerre polynomial `L_n^{(α)}(x)` (`α > −1` for orthogonality; `α = 0` gives
/// the ordinary Laguerre polynomial).
pub fn laguerre(n: u32, alpha: f64, x: f64) -> f64 {
    recurrence(n, 1.0 + alpha - x, |k, cur, prev| {
        let k = k as f64;
        ((2.0 * k + 1.0 + alpha - x) * cur - (k + alpha) * prev) / (k + 1.0)
    })
    .0
}

/// The Jacobi polynomial `P_n^{(α,β)}(x)` (`α, β > −1` for orthogonality). Legendre is
/// `α = β = 0`, and Gegenbauer and Chebyshev polynomials are scaled special cases.
pub fn jacobi(n: u32, alpha: f64, beta: f64, x: f64) -> f64 {
    let (a, b) = (alpha, beta);
    let p1 = (a + 1.0) + (a + b + 2.0) * (x - 1.0) / 2.0;
    recurrence(n, p1, |k, cur, prev| {
        let k = k as f64;
        let s = 2.0 * k + a + b;
        let c0 = 2.0 * (k + 1.0) * (k + a + b + 1.0) * s;
        let c1 = (s + 1.0) * ((s + 2.0) * s * x + a * a - b * b);
        let c2 = 2.0 * (k + a) * (k + b) * (s + 2.0);
        (c1 * cur - c2 * prev) / c0
    })
    .0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binomial_coefficients::binomial_coefficient;
    use crate::gamma::gamma;
    use std::f64::consts::PI;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    /// ∫ f over [a, b] by the composite Simpson rule with `m` (even) panels.
    fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, m: usize) -> f64 {
        let h = (b - a) / m as f64;
        (0..=m).map(|i| f(a + i as f64 * h) * if i == 0 || i == m { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 }).sum::<f64>() * h / 3.0
    }

    #[test]
    fn low_degrees_match_explicit_formulas() {
        for &x in &[-0.9, -0.3, 0.0, 0.45, 1.0, 2.5] {
            let x2 = x * x;
            assert!(close(legendre(3, x), 0.5 * (5.0 * x2 * x - 3.0 * x), 1e-14));
            assert!(close(legendre(4, x), (35.0 * x2 * x2 - 30.0 * x2 + 3.0) / 8.0, 1e-14));
            assert!(close(chebyshev_t(4, x), 8.0 * x2 * x2 - 8.0 * x2 + 1.0, 1e-14));
            assert!(close(chebyshev_u(3, x), 8.0 * x2 * x - 4.0 * x, 1e-14));
            assert!(close(hermite(4, x), 16.0 * x2 * x2 - 48.0 * x2 + 12.0, 1e-14));
            assert!(close(hermite_he(4, x), x2 * x2 - 6.0 * x2 + 3.0, 1e-14));
            assert!(close(laguerre(3, 0.0, x), (-x2 * x + 9.0 * x2 - 18.0 * x + 6.0) / 6.0, 1e-14));
            assert!(close(laguerre(2, 1.5, x), x2 / 2.0 - 3.5 * x + 4.375, 1e-14));
            assert!(close(jacobi(1, 0.5, 2.0, x), 1.5 + 4.5 * (x - 1.0) / 2.0, 1e-14));
        }
        assert_eq!((legendre(0, 0.3), hermite(0, 7.0), laguerre(0, 2.0, 1.0)), (1.0, 1.0, 1.0));
    }

    #[test]
    fn identities_between_families() {
        for n in 0..25u32 {
            for &x in &[-0.97, -0.5, 0.1, 0.77] {
                // Jacobi(0, 0) is Legendre; Chebyshev from the trigonometric definitions.
                assert!(close(jacobi(n, 0.0, 0.0, x), legendre(n, x), 1e-12), "n={n}");
                let t = f64::acos(x);
                assert!(close(chebyshev_t(n, x), (n as f64 * t).cos(), 1e-11));
                assert!(close(chebyshev_u(n, x), ((n + 1) as f64 * t).sin() / t.sin(), 1e-10));
                // Jacobi(−½, −½) is T_n scaled by P_n^{(−½,−½)}(1) = C(n − ½, n).
                let scale = gamma(n as f64 + 0.5) / (gamma(0.5) * gamma(n as f64 + 1.0));
                assert!(close(jacobi(n, -0.5, -0.5, x), scale * chebyshev_t(n, x), 1e-10));
            }
            for &x in &[-2.0, 0.3, 1.7] {
                let he = hermite_he(n, x);
                assert!(close(hermite(n, x / 2f64.sqrt()) / 2f64.powf(n as f64 / 2.0), he, 1e-10), "n={n}");
            }
            // Endpoint values.
            assert!(close(legendre(n, 1.0), 1.0, 1e-13) && close(legendre(n, -1.0), if n % 2 == 0 { 1.0 } else { -1.0 }, 1e-13));
            assert!(close(laguerre(n, 0.0, 0.0), 1.0, 1e-13));
            assert!(close(jacobi(n, 1.0, 0.5, 1.0), binomial_coefficient(n as u64 + 1, n as u64) as f64, 1e-12));
        }
    }

    #[test]
    fn families_are_orthogonal_under_their_weights() {
        let m = 40_000;
        for (i, j) in [(2u32, 3u32), (3, 5), (4, 4), (6, 6)] {
            let leg = simpson(|x| legendre(i, x) * legendre(j, x), -1.0, 1.0, m);
            assert!(close(leg, if i == j { 2.0 / (2 * i + 1) as f64 } else { 0.0 }, 1e-9), "legendre {i} {j}: {leg}");
            let her = simpson(|x| hermite(i, x) * hermite(j, x) * (-x * x).exp(), -10.0, 10.0, m);
            let want = if i == j { PI.sqrt() * 2f64.powi(i as i32) * gamma(i as f64 + 1.0) } else { 0.0 };
            assert!(close(her, want, 1e-8), "hermite {i} {j}: {her} {want}");
            let lag = simpson(|x| laguerre(i, 2.0, x) * laguerre(j, 2.0, x) * x * x * (-x).exp(), 0.0, 80.0, m);
            let want = if i == j { gamma(i as f64 + 3.0) / gamma(i as f64 + 1.0) } else { 0.0 };
            assert!(close(lag, want, 1e-8), "laguerre {i} {j}: {lag} {want}");
            // Chebyshev T with weight (1 − x²)^{−½}: substitute x = cos t.
            let che = simpson(|t| chebyshev_t(i, t.cos()) * chebyshev_t(j, t.cos()), 0.0, PI, m);
            assert!(close(che, if i == j { PI / 2.0 } else { 0.0 }, 1e-9));
            // Jacobi (1, 2): x = cos t keeps the weight smooth.
            let jac = simpson(|t| { let x = t.cos(); jacobi(i, 1.0, 2.0, x) * jacobi(j, 1.0, 2.0, x) * (1.0 - x) * (1.0 + x).powi(2) * t.sin() }, 0.0, PI, m);
            if i != j {
                assert!(jac.abs() < 1e-9, "jacobi {i} {j}: {jac}");
            }
        }
    }

    #[test]
    fn legendre_derivative_matches_finite_differences_and_endpoints() {
        for n in 0..12u32 {
            for &x in &[-0.8, -0.2, 0.35, 0.9] {
                let h = 1e-6;
                let fd = (legendre(n, x + h) - legendre(n, x - h)) / (2.0 * h);
                assert!(close(legendre_with_derivative(n, x).1, fd, 1e-6), "n={n} x={x}");
            }
            let end = n as f64 * (n + 1) as f64 / 2.0;
            assert!(close(legendre_with_derivative(n, 1.0).1, end, 1e-12));
            assert!(close(legendre_with_derivative(n, -1.0).1, if n % 2 == 1 { end } else { -end }, 1e-12));
        }
    }
}
