//! Continuous-time linear time-invariant systems, following `scipy.signal`:
//! `x'(t) = A x(t) + B u(t)`, `y(t) = C x(t) + D u(t)`.
//!
//! - Conversions between transfer functions and state space ([`tf2ss`], [`ss2tf`]).
//! - Simulation with zero-order or linearly interpolated inputs ([`ContinuousStateSpace::lsim`]),
//!   step and impulse responses, all exact for the held input through the matrix exponential.
//! - Frequency response and Bode plots ([`ContinuousStateSpace::freqresp`],
//!   [`ContinuousStateSpace::bode`]).
//! - Discretisation by zero- and first-order hold, impulse invariance and the generalised
//!   bilinear transform (Tustin, forward and backward Euler) ([`ContinuousStateSpace::discretize`]).
//! - Pole placement by state feedback ([`place_poles`]).
//!
//! See the book entry [Continuous-time systems](https://jwkunz.github.io/lib_jsl/book/comms/continuous-time.html).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_linalg::{EigVals, Inverse, SVD, expm};
use ndarray::{Array1, Array2, s};
use num_complex::Complex;

type C = Complex<f64>;

fn linalg_error<E>(_: E) -> ErrorsJSL {
    ErrorsJSL::RuntimeError("a linear-algebra operation failed (singular or non-finite matrix)")
}

/// A continuous-time state-space model with real matrices (SciPy's `StateSpace` with `dt=None`).
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousStateSpace {
    pub a: Array2<f64>,
    pub b: Array2<f64>,
    pub c: Array2<f64>,
    pub d: Array2<f64>,
}

/// How the input is held between samples in [`ContinuousStateSpace::lsim`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputHold {
    /// Constant over each step (SciPy's `interp=False`).
    Zero,
    /// Linear between samples (SciPy's default, `interp=True`).
    Linear,
}

/// Discretisation methods for [`ContinuousStateSpace::discretize`] (SciPy's `cont2discrete`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Discretization {
    /// Zero-order hold: exact for piecewise-constant inputs.
    ZeroOrderHold,
    /// First-order hold: exact for piecewise-linear inputs.
    FirstOrderHold,
    /// Impulse invariance (strictly proper systems only).
    Impulse,
    /// The generalised bilinear transform with parameter `α ∈ [0, 1]`:
    /// `x[k+1] − x[k] = T (A ((1 − α) x[k] + α x[k+1]) + B u)`.
    Gbt(f64),
    /// Tustin's bilinear transform, `Gbt(0.5)`.
    Bilinear,
    /// Forward Euler, `Gbt(0)`.
    ForwardEuler,
    /// Backward Euler, `Gbt(1)`.
    BackwardEuler,
}

/// A discretised system `x[k+1] = A x[k] + B u[k]`, `y[k] = C x[k] + D u[k]` with sample time `dt`.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteMatrices {
    pub a: Array2<f64>,
    pub b: Array2<f64>,
    pub c: Array2<f64>,
    pub d: Array2<f64>,
    pub dt: f64,
}

/// The result of [`ContinuousStateSpace::lsim`]: outputs and states at each time, one row per time.
#[derive(Clone, Debug)]
pub struct Simulation {
    pub y: Array2<f64>,
    pub x: Array2<f64>,
}

/// The coefficients of `∏ (s − rᵢ)`, highest power first (NumPy's `poly`), with the imaginary
/// parts dropped when the roots come in conjugate pairs.
pub fn poly_from_roots(roots: &[C]) -> Vec<f64> {
    let mut p = vec![C::new(1.0, 0.0)];
    for &r in roots {
        let mut next = vec![C::new(0.0, 0.0); p.len() + 1];
        for (i, &c) in p.iter().enumerate() {
            next[i] += c;
            next[i + 1] -= c * r;
        }
        p = next;
    }
    p.iter().map(|z| z.re).collect()
}

fn normalize(num: &[f64], den: &[f64]) -> Result<(Vec<f64>, Vec<f64>), ErrorsJSL> {
    let den: Vec<f64> = den.iter().copied().skip_while(|&v| v == 0.0).collect();
    if den.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("the denominator must not be zero"));
    }
    let lead = den[0];
    let mut num: Vec<f64> = num.iter().copied().skip_while(|&v| v == 0.0).map(|v| v / lead).collect();
    if num.is_empty() {
        num.push(0.0);
    }
    Ok((num, den.iter().map(|v| v / lead).collect()))
}

// ANCHOR: tf2ss
/// The controllable canonical state-space form of the transfer function
/// `H(s) = (b₀ sᵐ + … + b_m) / (a₀ sⁿ + … + a_n)` (SciPy's `tf2ss`), for a proper `H` (`m ≤ n`):
/// `A` has `−a₁/a₀, …, −a_n/a₀` in its first row and ones below the diagonal, `B = e₁`,
/// `C = b̃₁.. − b̃₀ ã₁..` with `b̃` the numerator padded to length `n + 1`, and `D = b̃₀`.
pub fn tf2ss(num: &[f64], den: &[f64]) -> Result<ContinuousStateSpace, ErrorsJSL> {
    let (num, den) = normalize(num, den)?;
    let k = den.len();
    if num.len() > k {
        return Err(ErrorsJSL::InvalidInputRange("the transfer function must be proper (numerator degree ≤ denominator degree)"));
    }
    let mut padded = vec![0.0; k - num.len()];
    padded.extend_from_slice(&num);
    let n = k - 1;
    if n == 0 {
        return Ok(ContinuousStateSpace { a: Array2::zeros((1, 1)), b: Array2::zeros((1, 1)), c: Array2::zeros((1, 1)), d: Array2::from_elem((1, 1), padded[0]) });
    }
    let mut a = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        a[(0, j)] = -den[j + 1];
    }
    for i in 1..n {
        a[(i, i - 1)] = 1.0;
    }
    let mut b = Array2::<f64>::zeros((n, 1));
    b[(0, 0)] = 1.0;
    let c = Array2::from_shape_fn((1, n), |(_, j)| padded[j + 1] - padded[0] * den[j + 1]);
    Ok(ContinuousStateSpace { a, b, c, d: Array2::from_elem((1, 1), padded[0]) })
}
// ANCHOR_END: tf2ss

/// The transfer functions from input `input` to every output (SciPy's `ss2tf`): a common
/// denominator `det(sI − A)` and one numerator row per output, each of length `n + 1`, from
/// `det(sI − A + B Cₖ) + (Dₖ − 1) det(sI − A)`.
pub fn ss2tf(sys: &ContinuousStateSpace, input: usize) -> Result<(Array2<f64>, Vec<f64>), ErrorsJSL> {
    if input >= sys.b.ncols() {
        return Err(ErrorsJSL::InvalidInputRange("input index out of range"));
    }
    let n = sys.a.nrows();
    let den = poly_from_roots(&sys.a.eigvals().map_err(linalg_error)?.to_vec());
    let b = sys.b.column(input).to_owned();
    let p = sys.c.nrows();
    let mut num = Array2::<f64>::zeros((p, n + 1));
    for k in 0..p {
        let ck = sys.c.row(k);
        let outer = Array2::from_shape_fn((n, n), |(i, j)| b[i] * ck[j]);
        let pk = poly_from_roots(&(&sys.a - &outer).eigvals().map_err(linalg_error)?.to_vec());
        for j in 0..=n {
            num[(k, j)] = pk[j] + (sys.d[(k, input)] - 1.0) * den[j];
        }
    }
    Ok((num, den))
}

impl ContinuousStateSpace {
    /// A model from its matrices, checking that the shapes agree.
    pub fn new(a: Array2<f64>, b: Array2<f64>, c: Array2<f64>, d: Array2<f64>) -> Result<Self, ErrorsJSL> {
        let n = a.nrows();
        if a.ncols() != n || b.nrows() != n || c.ncols() != n || d.nrows() != c.nrows() || d.ncols() != b.ncols() {
            return Err(ErrorsJSL::InvalidInputRange("A (n×n), B (n×m), C (p×n) and D (p×m) must have compatible shapes"));
        }
        Ok(ContinuousStateSpace { a, b, c, d })
    }

    /// The poles, the eigenvalues of `A`.
    pub fn poles(&self) -> Result<Vec<C>, ErrorsJSL> {
        Ok(self.a.eigvals().map_err(linalg_error)?.to_vec())
    }

    // ANCHOR: lsim
    /// Simulates the response to the input samples `u` (one row per time) at equally spaced
    /// times `t`, from the initial state `x0` (zero if `None`) (SciPy's `lsim`).
    ///
    /// The input is held constant ([`InputHold::Zero`]) or interpolated linearly
    /// ([`InputHold::Linear`]) between samples, and each step is then exact: with
    /// `M = [[A T, B T, 0], [0, 0, I], [0, 0, 0]]` and `e^M`'s top blocks `Φ, Γ₁, Γ₂`,
    /// `x[i+1] = Φ x[i] + (Γ₁ − Γ₂) u[i] + Γ₂ u[i+1]`.
    pub fn lsim(&self, u: &Array2<f64>, t: &[f64], x0: Option<&[f64]>, hold: InputHold) -> Result<Simulation, ErrorsJSL> {
        let (n, m) = (self.a.nrows(), self.b.ncols());
        let steps = t.len();
        if u.nrows() != steps || u.ncols() != m {
            return Err(ErrorsJSL::IncompatibleArraySizes((u.nrows(), steps)));
        }
        if steps == 0 {
            return Err(ErrorsJSL::InvalidInputRange("need at least one time"));
        }
        let mut x = Array2::<f64>::zeros((steps, n));
        let x0 = Array1::from(x0.map_or(vec![0.0; n], |v| v.to_vec()));
        if x0.len() != n {
            return Err(ErrorsJSL::IncompatibleArraySizes((x0.len(), n)));
        }
        if t[0] < 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("the initial time must be non-negative"));
        }
        // Before t[0] the input is zero: x(t₀) = e^{A t₀} x₀.
        let start = if t[0] == 0.0 { x0 } else { expm(&(&self.a * t[0])).map_err(linalg_error)?.dot(&x0) };
        x.row_mut(0).assign(&start);
        if steps > 1 {
            let dt = t[1] - t[0];
            if !(dt > 0.0) || t.windows(2).any(|w| ((w[1] - w[0]) - dt).abs() > 1e-8 * dt.abs().max(1e-300) + 1e-12) {
                return Err(ErrorsJSL::InvalidInputRange("time steps must be positive and equally spaced"));
            }
            let size = match hold {
                InputHold::Zero => n + m,
                InputHold::Linear => n + 2 * m,
            };
            let mut big = Array2::<f64>::zeros((size, size));
            big.slice_mut(s![..n, ..n]).assign(&(&self.a * dt));
            big.slice_mut(s![..n, n..n + m]).assign(&(&self.b * dt));
            if hold == InputHold::Linear {
                big.slice_mut(s![n..n + m, n + m..]).assign(&Array2::eye(m));
            }
            let e = expm(&big).map_err(linalg_error)?;
            let phi = e.slice(s![..n, ..n]).to_owned();
            let (g0, g1) = match hold {
                InputHold::Zero => (e.slice(s![..n, n..n + m]).to_owned(), Array2::zeros((n, m))),
                InputHold::Linear => {
                    let g2 = e.slice(s![..n, n + m..]).to_owned();
                    (&e.slice(s![..n, n..n + m]) - &g2, g2)
                }
            };
            for i in 1..steps {
                let next = phi.dot(&x.row(i - 1)) + g0.dot(&u.row(i - 1)) + g1.dot(&u.row(i));
                x.row_mut(i).assign(&next);
            }
        }
        let y = x.dot(&self.c.t()) + u.dot(&self.d.t());
        Ok(Simulation { y, x })
    }
    // ANCHOR_END: lsim

    /// The step response from input `input` at times `t` (SciPy's `step`, with a zero-order hold).
    pub fn step(&self, t: &[f64], input: usize) -> Result<Array2<f64>, ErrorsJSL> {
        let m = self.b.ncols();
        if input >= m {
            return Err(ErrorsJSL::InvalidInputRange("input index out of range"));
        }
        let u = Array2::from_shape_fn((t.len(), m), |(_, j)| if j == input { 1.0 } else { 0.0 });
        Ok(self.lsim(&u, t, None, InputHold::Zero)?.y)
    }

    /// The impulse response `C e^{At} B` from input `input` at times `t` (SciPy's `impulse`,
    /// which leaves out the `D δ(t)` term).
    pub fn impulse(&self, t: &[f64], input: usize) -> Result<Array2<f64>, ErrorsJSL> {
        if input >= self.b.ncols() {
            return Err(ErrorsJSL::InvalidInputRange("input index out of range"));
        }
        let x0 = self.b.column(input).to_vec();
        let u = Array2::zeros((t.len(), self.b.ncols()));
        let sim = self.lsim(&u, t, Some(&x0), InputHold::Zero)?;
        Ok(sim.x.dot(&self.c.t()))
    }

    // ANCHOR: freqresp
    /// The frequency response `H(iω) = C (iω I − A)⁻¹ B + D` at each angular frequency, one
    /// `p × m` matrix per frequency.
    pub fn freqresp(&self, w: &[f64]) -> Result<Vec<Array2<C>>, ErrorsJSL> {
        let n = self.a.nrows();
        let a = self.a.mapv(|v| C::new(v, 0.0));
        let b = self.b.mapv(|v| C::new(v, 0.0));
        let c = self.c.mapv(|v| C::new(v, 0.0));
        let d = self.d.mapv(|v| C::new(v, 0.0));
        w.iter()
            .map(|&wk| {
                let m = Array2::from_shape_fn((n, n), |(i, j)| if i == j { C::new(0.0, wk) } else { C::new(0.0, 0.0) }) - &a;
                let lu = lib_jsl_linalg::Factorize::factorize(&m).map_err(linalg_error)?;
                let mut x = Array2::<C>::zeros(b.dim());
                for j in 0..b.ncols() {
                    x.column_mut(j).assign(&lu.solve(&b.column(j)).map_err(linalg_error)?);
                }
                Ok(c.dot(&x) + &d)
            })
            .collect()
    }
    // ANCHOR_END: freqresp

    /// The Bode diagram from input `input` to output `output` (SciPy's `bode`): the magnitude
    /// `20 log₁₀ |H(iω)|` in dB and the unwrapped phase in degrees.
    pub fn bode(&self, w: &[f64], input: usize, output: usize) -> Result<(Vec<f64>, Vec<f64>), ErrorsJSL> {
        if input >= self.b.ncols() || output >= self.c.nrows() {
            return Err(ErrorsJSL::InvalidInputRange("input or output index out of range"));
        }
        let h: Vec<C> = self.freqresp(w)?.iter().map(|m| m[(output, input)]).collect();
        let mag = h.iter().map(|z| 20.0 * z.norm().log10()).collect();
        let raw: Vec<f64> = h.iter().map(|z| z.im.atan2(z.re)).collect();
        Ok((mag, unwrap(&raw).into_iter().map(f64::to_degrees).collect()))
    }

    // ANCHOR: discretize
    /// The discrete-time system for sample time `dt` (SciPy's `cont2discrete`).
    pub fn discretize(&self, dt: f64, method: Discretization) -> Result<DiscreteMatrices, ErrorsJSL> {
        let (n, m) = (self.a.nrows(), self.b.ncols());
        let (a, b, c, d) = (&self.a, &self.b, &self.c, &self.d);
        let gbt = |alpha: f64| -> Result<DiscreteMatrices, ErrorsJSL> {
            if !(0.0..=1.0).contains(&alpha) {
                return Err(ErrorsJSL::InvalidInputRange("the bilinear parameter α must lie in [0, 1]"));
            }
            let eye = Array2::<f64>::eye(n);
            let ima = &eye - &(a * (alpha * dt));
            let inv = ima.inv().map_err(linalg_error)?;
            let ad = inv.dot(&(&eye + &(a * ((1.0 - alpha) * dt))));
            let bd = inv.dot(&(b * dt));
            let cd = c.dot(&inv);
            let dd = d + &(c.dot(&bd) * alpha);
            Ok(DiscreteMatrices { a: ad, b: bd, c: cd, d: dd, dt })
        };
        match method {
            Discretization::Gbt(alpha) => gbt(alpha),
            Discretization::Bilinear => gbt(0.5),
            Discretization::ForwardEuler => gbt(0.0),
            Discretization::BackwardEuler => gbt(1.0),
            Discretization::ZeroOrderHold => {
                let mut big = Array2::<f64>::zeros((n + m, n + m));
                big.slice_mut(s![..n, ..n]).assign(&(a * dt));
                big.slice_mut(s![..n, n..]).assign(&(b * dt));
                let e = expm(&big).map_err(linalg_error)?;
                Ok(DiscreteMatrices { a: e.slice(s![..n, ..n]).to_owned(), b: e.slice(s![..n, n..]).to_owned(), c: c.clone(), d: d.clone(), dt })
            }
            Discretization::FirstOrderHold => {
                let mut big = Array2::<f64>::zeros((n + 2 * m, n + 2 * m));
                big.slice_mut(s![..n, ..n]).assign(&(a * dt));
                big.slice_mut(s![..n, n..n + m]).assign(&(b * dt));
                big.slice_mut(s![n..n + m, n + m..]).assign(&Array2::eye(m));
                let e = expm(&big).map_err(linalg_error)?;
                let (m11, m12, m13) = (e.slice(s![..n, ..n]).to_owned(), e.slice(s![..n, n..n + m]).to_owned(), e.slice(s![..n, n + m..]).to_owned());
                let bd = &m12 - &m13 + m11.dot(&m13);
                let dd = d + &c.dot(&m13);
                Ok(DiscreteMatrices { a: m11, b: bd, c: c.clone(), d: dd, dt })
            }
            Discretization::Impulse => {
                if d.iter().any(|v| v.abs() > 1e-8) {
                    return Err(ErrorsJSL::InvalidInputRange("impulse invariance needs a strictly proper system (D = 0)"));
                }
                let ad = expm(&(a * dt)).map_err(linalg_error)?;
                let bd = ad.dot(b) * dt;
                Ok(DiscreteMatrices { a: ad, b: bd, c: c.clone(), d: c.dot(b) * dt, dt })
            }
        }
    }
    // ANCHOR_END: discretize
}

/// Removes jumps of more than π between consecutive phases by adding multiples of 2π (NumPy's
/// `unwrap`).
pub fn unwrap(phase: &[f64]) -> Vec<f64> {
    use std::f64::consts::PI;
    let mut out = phase.to_vec();
    let mut correction = 0.0;
    for i in 1..phase.len() {
        let d = phase[i] - phase[i - 1];
        let mut dmod = (d + PI).rem_euclid(2.0 * PI) - PI;
        if dmod == -PI && d > 0.0 {
            dmod = PI;
        }
        if d.abs() >= PI {
            correction += dmod - d;
        }
        out[i] = phase[i] + correction;
    }
    out
}

/// The result of [`place_poles`].
#[derive(Clone, Debug)]
pub struct PlacedPoles {
    /// The gain `K`, such that `A − B K` has the requested poles.
    pub gain: Array2<f64>,
    /// The eigenvalues of `A − B K`, as computed.
    pub computed_poles: Vec<C>,
    /// The eigenvector matrix `X` of `A − B K` (unit columns).
    pub transfer_matrix: Array2<C>,
    /// Sweeps of the orthogonalisation (0 when there is no freedom to optimise).
    pub iterations: usize,
}

/// The null space of a complex matrix (as orthonormal columns), from its SVD.
fn null_space(m: &Array2<C>, dim: usize) -> Result<Array2<C>, ErrorsJSL> {
    let n = m.ncols();
    let padded = if m.nrows() < n {
        let mut p = Array2::<C>::zeros((n, n));
        p.slice_mut(s![..m.nrows(), ..]).assign(m);
        p
    } else {
        m.clone()
    };
    let (_, _, vt) = padded.svd(false, true).map_err(linalg_error)?;
    let vt = vt.expect("requested");
    // The last `dim` rows of Vᴴ, conjugated, span the null space.
    Ok(Array2::from_shape_fn((n, dim), |(i, j)| vt[(n - dim + j, i)].conj()))
}

// ANCHOR: place_poles
/// A state-feedback gain `K` that places the eigenvalues of `A − B K` at `poles` (SciPy's
/// `place_poles`). Complex poles must come in conjugate pairs, and no pole may be repeated more
/// often than the rank of `B`.
///
/// Each closed-loop eigenvector `xⱼ` must lie in the subspace `Sⱼ` where
/// `U₁ᴴ (A − λⱼ I) x = 0`, with `B = [U₀ U₁] [Z; 0]` the QR factorisation of `B`; then
/// `K = Z⁻¹ U₀ᴴ (A X − X Λ) X⁻¹`. With one input each `Sⱼ` is a line and `K` is unique. With several,
/// the vectors are chosen to make `X` as well conditioned as possible by the method of Kautsky,
/// Nichols and Van Dooren (KNV0): each sweep replaces every `xⱼ` by the projection onto `Sⱼ` of
/// the direction orthogonal to all the other columns.
pub fn place_poles(a: &Array2<f64>, b: &Array2<f64>, poles: &[C]) -> Result<PlacedPoles, ErrorsJSL> {
    let (n, m) = (a.nrows(), b.ncols());
    if a.ncols() != n || b.nrows() != n || poles.len() != n {
        return Err(ErrorsJSL::InvalidInputRange("A must be n × n, B n × m, with n poles"));
    }
    // Complex poles in conjugate pairs; keep a partner index for each.
    let mut partner = vec![usize::MAX; n];
    for i in 0..n {
        if poles[i].im.abs() > 1e-12 * (1.0 + poles[i].norm()) && partner[i] == usize::MAX {
            let j = (0..n).find(|&j| j != i && partner[j] == usize::MAX && (poles[j] - poles[i].conj()).norm() <= 1e-10 * (1.0 + poles[i].norm()));
            let Some(j) = j else { return Err(ErrorsJSL::InvalidInputRange("complex poles must come in conjugate pairs")) };
            partner[i] = j;
            partner[j] = i;
        }
    }
    let bc = b.mapv(|v| C::new(v, 0.0));
    let ac = a.mapv(|v| C::new(v, 0.0));
    // Full QR of B through the SVD: B = U Σ Vᴴ, U₀ = U[:, ..r], U₁ = U[:, r..].
    let (u, sv, vt) = bc.svd(true, true).map_err(linalg_error)?;
    let (u, vt) = (u.expect("requested"), vt.expect("requested"));
    let rank = sv.iter().filter(|&&v| v > sv[0] * 1e-12 * n as f64).count();
    if rank == 0 {
        return Err(ErrorsJSL::InvalidInputRange("B is zero"));
    }
    for p in poles {
        if poles.iter().filter(|q| (*q - p).norm() < 1e-10 * (1.0 + p.norm())).count() > rank {
            return Err(ErrorsJSL::InvalidInputRange("a pole is repeated more often than the rank of B"));
        }
    }
    let u0 = u.slice(s![.., ..rank]).to_owned();
    let u1h = u.slice(s![.., rank..]).t().mapv(|v| v.conj());
    // Bases of the admissible eigenvector spaces.
    let eye = Array2::<C>::eye(n);
    let spaces: Vec<Array2<C>> = poles
        .iter()
        .map(|&p| if rank == n { Ok(eye.clone()) } else { null_space(&u1h.dot(&(&ac - &(&eye * p))), rank) })
        .collect::<Result<_, _>>()?;
    // Start from the first basis vector of each space.
    let mut x = Array2::<C>::zeros((n, n));
    for (j, space) in spaces.iter().enumerate() {
        x.column_mut(j).assign(&space.column(0));
    }
    let mut iterations = 0;
    if rank > 1 {
        for sweep in 0..30 {
            let mut change: f64 = 0.0;
            for j in 0..n {
                if partner[j] < j {
                    continue;
                }
                // The direction orthogonal to all other columns, projected onto Sⱼ.
                let mut others = Array2::<C>::zeros((n - 1, n));
                for (r, k) in (0..n).filter(|&k| k != j).enumerate() {
                    others.row_mut(r).assign(&x.column(k).mapv(|v| v.conj()));
                }
                let y = null_space(&others, 1)?;
                let sj = &spaces[j];
                let proj = sj.dot(&sj.t().mapv(|v| v.conj()).dot(&y.column(0)));
                let norm = proj.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
                if norm < 1e-14 {
                    continue;
                }
                let new = proj.mapv(|v| v / norm);
                // Compare up to a phase.
                let overlap = new.iter().zip(x.column(j).iter()).map(|(a, b)| a.conj() * b).sum::<C>().norm();
                change = change.max(1.0 - overlap);
                x.column_mut(j).assign(&new);
                if partner[j] != usize::MAX {
                    x.column_mut(partner[j]).assign(&new.mapv(|v| v.conj()));
                }
            }
            iterations = sweep + 1;
            if change < 1e-6 {
                break;
            }
        }
    }
    // K = Z⁻¹ U₀ᴴ (A X − X Λ) X⁻¹, with Z = Σ₀ V₀ᴴ the R factor of B in the basis U₀.
    let lambda = Array2::from_diag(&Array1::from(poles.to_vec()));
    let xinv = x.inv().map_err(|_| ErrorsJSL::RuntimeError("the eigenvectors are dependent; the poles cannot be placed"))?;
    let rhs = u0.t().mapv(|v| v.conj()).dot(&(ac.dot(&x) - x.dot(&lambda))).dot(&xinv);
    let z = Array2::from_shape_fn((rank, m), |(i, j)| vt[(i, j)] * sv[i]);
    // Minimum-norm solution of Z K = rhs (Z is rank × m with full row rank).
    let zzh = z.dot(&z.t().mapv(|v| v.conj()));
    let k = z.t().mapv(|v| v.conj()).dot(&zzh.inv().map_err(linalg_error)?.dot(&rhs));
    let gain = k.mapv(|v| v.re);
    let computed_poles = (a - &b.dot(&gain)).eigvals().map_err(linalg_error)?.to_vec();
    Ok(PlacedPoles { gain, computed_poles, transfer_matrix: x, iterations })
}
// ANCHOR_END: place_poles

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn tf2ss_and_back() {
        let sys = tf2ss(&[1.0, 3.0, 3.0], &[1.0, 2.0, 1.0]).unwrap();
        assert_eq!(sys.a, array![[-2.0, -1.0], [1.0, 0.0]]);
        assert_eq!(sys.c, array![[1.0, 2.0]]);
        assert_eq!(sys.d, array![[1.0]]);
        let (num, den) = ss2tf(&sys, 0).unwrap();
        for (got, want) in num.row(0).iter().zip([1.0, 3.0, 3.0]) {
            assert!((got - want).abs() < 1e-12);
        }
        for (got, want) in den.iter().zip([1.0, 2.0, 1.0]) {
            assert!((got - want).abs() < 1e-12);
        }
    }

    #[test]
    fn first_order_step_response_is_exact() {
        // H(s) = 1 / (s + 2): y(t) = (1 − e^{−2t}) / 2.
        let sys = tf2ss(&[1.0], &[1.0, 2.0]).unwrap();
        let t: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let y = sys.step(&t, 0).unwrap();
        for (i, &ti) in t.iter().enumerate() {
            assert!((y[(i, 0)] - (1.0 - (-2.0 * ti).exp()) / 2.0).abs() < 1e-14);
        }
        let h = sys.impulse(&t, 0).unwrap();
        assert!((h[(10, 0)] - (-2.0f64).exp()).abs() < 1e-14);
    }

    #[test]
    fn placed_poles_are_achieved() {
        let a = array![[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 11.0, 0.0]];
        let b1 = array![[0.0], [1.0], [0.0], [-1.0]];
        let b2 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [-1.0, 1.0]];
        let want = [C::new(-1.0, 1.0), C::new(-1.0, -1.0), C::new(-2.0, 0.0), C::new(-3.0, 0.0)];
        for b in [b1, b2] {
            let r = place_poles(&a, &b, &want).unwrap();
            for p in &want {
                assert!(r.computed_poles.iter().any(|q| (q - p).norm() < 1e-8), "{p} not in {:?}", r.computed_poles);
            }
        }
    }
}
