//! The matrix exponential by scaling and squaring with Padé approximants (Higham 2005).
//!
//! See the book entry [Matrix functions and matrix equations](https://jwkunz.github.io/lib_jsl/book/linear_algebra/matrix-equations.html).

use crate::dense::{Mat, lincomb, matmul, norm1, require_square};
use crate::error::Result;
use crate::lu::LuFactorization;
use crate::scalar::Scalar;
use ndarray::{Array2, ArrayBase, Data, Ix2};

/// Coefficients `b_k` of the `[m/m]` Padé approximant to `eˣ`, for m = 3, 5, 7, 9, 13.
const B3: [f64; 4] = [120.0, 60.0, 12.0, 1.0];
const B5: [f64; 6] = [30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0];
const B7: [f64; 8] = [17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0];
const B9: [f64; 10] = [17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0, 2162160.0, 110880.0, 3960.0, 90.0, 1.0];
const B13: [f64; 14] = [
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
];
/// The largest 1-norm for which each approximant is accurate to unit roundoff (Higham 2005, table 2.3).
const THETA: [(usize, f64); 4] = [(3, 1.495_585_217_958_292e-2), (5, 2.539_398_330_063_23e-1), (7, 9.504_178_996_162_932e-1), (9, 2.097_847_961_257_068)];
const THETA13: f64 = 5.371920351148152;

// ANCHOR: expm
/// The matrix exponential `eᴬ = I + A + A²/2! + …` (SciPy's `linalg.expm`).
///
/// Scaling and squaring (Higham 2005): choose the lowest Padé degree `m ∈ {3, 5, 7, 9, 13}`
/// accurate to rounding for `‖A‖₁`, or, for larger norms, scale `A` by `2⁻ˢ` until
/// `‖A‖₁ ≤ θ₁₃`; evaluate `r_m(A) = (V − U)⁻¹ (V + U)` from the even part `V` and odd part `U` of
/// the Padé numerator; and square the result `s` times.
pub fn expm<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>) -> Result<Array2<T>> {
    let a = Mat::from_array(a)?;
    require_square(a.rows, a.cols)?;
    let n = a.rows;
    let eye = Mat::<T>::identity(n);
    let norm = norm1(&a);
    let a2 = matmul(&a, &a);
    // Low degrees: U = A Σ b_{2k+1} A^{2k}, V = Σ b_{2k} A^{2k}.
    for (m, theta) in THETA {
        if norm <= theta {
            let b: &[f64] = match m {
                3 => &B3,
                5 => &B5,
                7 => &B7,
                _ => &B9,
            };
            let (mut u, mut v) = (Mat::zeros(n, n), Mat::zeros(n, n));
            let mut power = eye.clone();
            for k in 0..=m / 2 {
                u = lincomb(T::one(), &u, T::from_real(b[2 * k + 1]), &power);
                v = lincomb(T::one(), &v, T::from_real(b[2 * k]), &power);
                if k < m / 2 {
                    power = matmul(&power, &a2);
                }
            }
            return Ok(pade_quotient(&matmul(&a, &u), &v)?.into_array());
        }
    }
    // Degree 13 with scaling: A ← A / 2ˢ.
    let s = (norm / THETA13).log2().ceil().max(0.0) as i32;
    let scale = T::from_real(0.5f64.powi(s));
    let a1 = Mat { rows: n, cols: n, data: a.data.iter().map(|&v| v * scale).collect() };
    let a2 = matmul(&a1, &a1);
    let a4 = matmul(&a2, &a2);
    let a6 = matmul(&a4, &a2);
    let b = |k: usize| T::from_real(B13[k]);
    let sum3 = |c6: T, c4: T, c2: T, c0: T| {
        let t = lincomb(c6, &a6, c4, &a4);
        let t = lincomb(T::one(), &t, c2, &a2);
        lincomb(T::one(), &t, c0, &eye)
    };
    let inner_u = sum3(b(13), b(11), b(9), T::zero());
    let u = lincomb(T::one(), &matmul(&a6, &inner_u), T::one(), &sum3(b(7), b(5), b(3), b(1)));
    let u = matmul(&a1, &u);
    let inner_v = sum3(b(12), b(10), b(8), T::zero());
    let v = lincomb(T::one(), &matmul(&a6, &inner_v), T::one(), &sum3(b(6), b(4), b(2), b(0)));
    let mut r = pade_quotient(&u, &v)?;
    for _ in 0..s {
        r = matmul(&r, &r);
    }
    Ok(r.into_array())
}
// ANCHOR_END: expm

/// `(V − U)⁻¹ (V + U)`, solved column by column with one LU factorisation.
fn pade_quotient<T: Scalar>(u: &Mat<T>, v: &Mat<T>) -> Result<Mat<T>> {
    let n = u.rows;
    let lu = LuFactorization::new(lincomb(T::one(), v, -T::one(), u))?;
    let rhs = lincomb(T::one(), v, T::one(), u);
    let mut out = Mat::zeros(n, n);
    for j in 0..n {
        let col: Vec<T> = (0..n).map(|i| rhs.at(i, j)).collect();
        for (i, x) in lu.solve_vec(&col).into_iter().enumerate() {
            *out.at_mut(i, j) = x;
        }
    }
    Ok(out)
}
