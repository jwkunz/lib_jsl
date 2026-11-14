//! Krogh's interpolating polynomial through values and derivatives (SciPy's
//! `KroghInterpolator`): a node repeated `s + 1` times carries the value and the first `s`
//! derivatives there. The polynomial is kept in Newton form, whose coefficients are confluent
//! divided differences.
//!
//! See the book entry [Piecewise polynomials, Hermite data and scattered interpolation](https://jwkunz.github.io/lib_jsl/book/interpolation/piecewise-scattered.html#hermite-data-krogh-and-cubic-hermite-splines).

use lib_jsl_core::ErrorsJSL;

/// The polynomial of degree `n − 1` through `n` conditions.
#[derive(Clone, Debug, PartialEq)]
pub struct KroghInterpolator {
    xi: Vec<f64>,
    /// Newton coefficients `c₀, …, c_{n−1}`: `p(x) = c₀ + c₁(x − x₀) + c₂(x − x₀)(x − x₁) + …`.
    c: Vec<f64>,
}

fn factorial(s: usize) -> f64 {
    (1..=s).map(|i| i as f64).product()
}

impl KroghInterpolator {
    // ANCHOR: krogh
    /// Builds the interpolant. `xi` lists the nodes with each repeated node consecutive; `yi[k]`
    /// is the value at `xi[k]` if it is the first copy, else the derivative whose order is the
    /// number of earlier copies. Divided differences with repeated nodes become derivatives
    /// divided by factorials, which is how `Vk[0] = yᵢ/s!` enters.
    pub fn new(xi: &[f64], yi: &[f64]) -> Result<KroghInterpolator, ErrorsJSL> {
        let n = xi.len();
        if n == 0 || yi.len() != n {
            return Err(ErrorsJSL::InvalidInputRange("need one value per node"));
        }
        let mut c = vec![0.0; n];
        c[0] = yi[0];
        let mut vk = vec![0.0; n];
        for k in 1..n {
            let mut s = 0;
            while s <= k && xi[k - s] == xi[k] {
                s += 1;
            }
            s -= 1;
            vk[0] = yi[k] / factorial(s);
            for i in 0..k - s {
                if xi[i] == xi[k] {
                    return Err(ErrorsJSL::InvalidInputRange("copies of a node must be consecutive"));
                }
                vk[i + 1] = if s == 0 { (c[i] - vk[i]) / (xi[i] - xi[k]) } else { (vk[i + 1] - vk[i]) / (xi[i] - xi[k]) };
            }
            c[k] = vk[k - s];
        }
        Ok(KroghInterpolator { xi: xi.to_vec(), c })
    }
    // ANCHOR_END: krogh

    /// The value at `x`, by nested multiplication of the Newton form.
    pub fn evaluate(&self, x: f64) -> f64 {
        let n = self.c.len();
        let mut p = self.c[n - 1];
        for k in (0..n - 1).rev() {
            p = p * (x - self.xi[k]) + self.c[k];
        }
        p
    }

    /// The value and the first `count − 1` derivatives at `x` (SciPy's `derivatives`), by
    /// SciPy's recurrence on the products `Π (x − xᵢ)`.
    pub fn derivatives(&self, x: f64, count: usize) -> Vec<f64> {
        let n = self.c.len();
        let mut pi = vec![0.0; n];
        let mut w = vec![0.0; n];
        pi[0] = 1.0;
        let mut p = self.c[0];
        for k in 1..n {
            w[k - 1] = x - self.xi[k - 1];
            pi[k] = w[k - 1] * pi[k - 1];
            p += pi[k] * self.c[k];
        }
        let mut cn = vec![0.0; count.max(n + 1)];
        cn[..n].copy_from_slice(&self.c);
        cn[0] = p;
        for k in 1..n {
            for i in 1..n - k + 1 {
                pi[i] += w[k + i - 1] * pi[i - 1];
                cn[k] += pi[i] * cn[k + i];
            }
            cn[k] *= factorial(k);
        }
        cn[n] = 0.0;
        cn.truncate(count);
        cn
    }

    /// The `nu`-th derivative at `x`.
    pub fn derivative(&self, x: f64, nu: usize) -> f64 {
        self.derivatives(x, nu + 1)[nu]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hermite_conditions_are_met() {
        // p(0) = 1, p'(0) = 2, p(1) = 0, p'(1) = −1, p''(1) = 4: a quartic.
        let k = KroghInterpolator::new(&[0.0, 0.0, 1.0, 1.0, 1.0], &[1.0, 2.0, 0.0, -1.0, 4.0]).unwrap();
        assert!((k.evaluate(0.0) - 1.0).abs() < 1e-14);
        assert!((k.derivative(0.0, 1) - 2.0).abs() < 1e-13);
        assert!(k.evaluate(1.0).abs() < 1e-14);
        assert!((k.derivative(1.0, 1) + 1.0).abs() < 1e-13);
        assert!((k.derivative(1.0, 2) - 4.0).abs() < 1e-12);
        assert_eq!(k.derivative(0.3, 5), 0.0);
    }

    #[test]
    fn plain_nodes_reproduce_a_polynomial() {
        let f = |x: f64| 2.0 * x.powi(3) - x + 0.5;
        let xi = [-1.0, 0.2, 0.9, 2.0];
        let k = KroghInterpolator::new(&xi, &xi.map(f)).unwrap();
        let d = k.derivatives(0.7, 5);
        assert!((d[0] - f(0.7)).abs() < 1e-13);
        assert!((d[1] - (6.0 * 0.49 - 1.0)).abs() < 1e-12);
        assert!((d[2] - 6.0 * 2.0 * 0.7).abs() < 1e-12);
        assert!((d[3] - 12.0).abs() < 1e-12);
        assert_eq!(d[4], 0.0);
    }
}
