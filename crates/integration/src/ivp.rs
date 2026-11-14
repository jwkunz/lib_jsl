//! Initial value problems `y' = f(t, y)` with dense output and event location, following
//! `scipy.integrate.solve_ivp`:
//!
//! * [`IvpMethod::Radau`]: the implicit Runge-Kutta method Radau IIA of order 5 (Hairer and
//!   Wanner), L-stable, with a cubic collocation polynomial as dense output. A step-for-step port
//!   of SciPy's `Radau`.
//! * [`IvpMethod::Lsoda`]: variable-order, variable-step multistep methods in Nordsieck form that
//!   switch automatically between Adams-Moulton (orders 1 to 12, functional iteration) for
//!   non-stiff stretches and BDF (orders 1 to 5, Newton iteration) for stiff ones, after Petzold's
//!   LSODA. The method coefficients are those of Hindmarsh's ODEPACK.
//!
//! Every accepted step contributes a polynomial piece to a [`DenseOutput`], so the solution can be
//! evaluated anywhere, and [`Event`] functions `g(t, y)` are located to machine precision on it by
//! Brent's method.
//!
//! See the book entry [Radau, LSODA and event location](https://jwkunz.github.io/lib_jsl/book/integration/radau-lsoda.html).

// Guards such as `!(x > 0.0)` are deliberate: unlike `x <= 0.0` they also reject NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]
// The coefficient recurrences read most clearly with explicit indices.
#![allow(clippy::needless_range_loop)]

use lib_jsl_core::ErrorsJSL;
use lib_jsl_linalg::Factorize;
use lib_jsl_linalg::LuFactorization;
use ndarray::{Array1, Array2};
use num_complex::Complex;

type C = Complex<f64>;

/// The method of [`solve_ivp`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IvpMethod {
    /// Radau IIA, order 5 (SciPy's `Radau`).
    Radau,
    /// Adams/BDF with automatic stiffness switching (as SciPy's `LSODA`).
    Lsoda,
}

/// Tolerances and limits for [`solve_ivp`]. Each step keeps the local error of every component
/// below `atol + rtol · |yᵢ|`, measured in the root-mean-square norm as in SciPy.
#[derive(Clone, Copy, Debug)]
pub struct IvpOptions {
    pub rtol: f64,
    pub atol: f64,
    /// The first step; chosen automatically when `None`.
    pub first_step: Option<f64>,
    /// The largest step.
    pub max_step: f64,
    /// The largest number of step attempts.
    pub max_steps: usize,
}

impl Default for IvpOptions {
    /// SciPy's defaults: `rtol = 1e-3`, `atol = 1e-6`.
    fn default() -> Self {
        IvpOptions { rtol: 1e-3, atol: 1e-6, first_step: None, max_step: f64::INFINITY, max_steps: 1_000_000 }
    }
}

/// The boxed function of an [`Event`].
pub type EventFunction<'a> = Box<dyn Fn(f64, &[f64]) -> f64 + 'a>;

/// An event function `g(t, y)` whose zeros are located during the integration.
pub struct Event<'a> {
    pub function: EventFunction<'a>,
    /// `+1`: only zeros where `g` increases; `-1`: only where it decreases; `0`: both.
    pub direction: f64,
    /// Stop the integration after this many occurrences (SciPy's `terminal`); `None` never stops.
    pub terminal: Option<usize>,
}

impl<'a> Event<'a> {
    /// A non-terminal event in either direction.
    pub fn new(function: impl Fn(f64, &[f64]) -> f64 + 'a) -> Self {
        Event { function: Box::new(function), direction: 0.0, terminal: None }
    }
}

/// One polynomial piece of the dense output: `y(t) = y_ref + Σₖ cₖ sᵏ⁺¹` with
/// `s = (t − t_ref) / h`, valid on `[t_start, t_end]`.
#[derive(Clone, Debug)]
pub struct Segment {
    pub t_start: f64,
    pub t_end: f64,
    t_ref: f64,
    h: f64,
    y_ref: Vec<f64>,
    coeffs: Vec<Vec<f64>>,
}

impl Segment {
    /// The interpolant at `t` (extrapolated outside the segment).
    pub fn evaluate(&self, t: f64) -> Vec<f64> {
        let s = (t - self.t_ref) / self.h;
        let mut y = self.y_ref.clone();
        let mut p = 1.0;
        for c in &self.coeffs {
            p *= s;
            for (yi, ci) in y.iter_mut().zip(c) {
                *yi += ci * p;
            }
        }
        y
    }
}

/// The continuous solution: one polynomial piece per accepted step.
#[derive(Clone, Debug, Default)]
pub struct DenseOutput {
    pub segments: Vec<Segment>,
}

impl DenseOutput {
    /// The solution at `t`, from the piece that contains it (the first or last piece outside the
    /// integration interval).
    pub fn evaluate(&self, t: f64) -> Vec<f64> {
        let i = self.segments.partition_point(|s| s.t_end < t).min(self.segments.len().saturating_sub(1));
        self.segments[i].evaluate(t)
    }
}

/// Why the integration stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IvpStatus {
    /// The end of the interval was reached.
    Finished,
    /// A terminal event occurred.
    TerminatedByEvent,
}

/// The result of [`solve_ivp`].
#[derive(Clone, Debug)]
pub struct IvpSolution {
    /// The times of the accepted steps (and of a terminal event).
    pub t: Vec<f64>,
    pub y: Vec<Vec<f64>>,
    pub status: IvpStatus,
    /// For each event function, the times at which it occurred and the states there.
    pub t_events: Vec<Vec<f64>>,
    pub y_events: Vec<Vec<Vec<f64>>>,
    pub dense: DenseOutput,
    pub nfev: usize,
    pub njev: usize,
    pub nlu: usize,
    pub rejected_steps: usize,
    /// For LSODA, the times at which the method switched and whether it switched to BDF (`true`)
    /// or back to Adams (`false`).
    pub switches: Vec<(f64, bool)>,
}

/// The RMS norm of `v / scale` (SciPy's `norm`).
fn rms(v: &[f64], scale: &[f64]) -> f64 {
    (v.iter().zip(scale).map(|(a, s)| (a / s) * (a / s)).sum::<f64>() / v.len() as f64).sqrt()
}

/// Counts evaluations of `f` and its Jacobian.
struct Counter<'a, F, J> {
    f: &'a F,
    jac: Option<&'a J>,
    nfev: usize,
    njev: usize,
    nlu: usize,
}

impl<F, J> Counter<'_, F, J>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    fn f(&mut self, t: f64, y: &[f64]) -> Vec<f64> {
        self.nfev += 1;
        (self.f)(t, y)
    }

    /// The Jacobian, analytic or by forward differences.
    fn jacobian(&mut self, t: f64, y: &[f64], fy: &[f64]) -> Vec<Vec<f64>> {
        self.njev += 1;
        if let Some(j) = self.jac {
            return j(t, y);
        }
        let n = y.len();
        let mut jac = vec![vec![0.0; n]; n];
        let mut z = y.to_vec();
        for k in 0..n {
            let h = f64::EPSILON.sqrt() * y[k].abs().max(1e-5);
            z[k] = y[k] + h;
            let fz = self.f(t, &z);
            z[k] = y[k];
            for i in 0..n {
                jac[i][k] = (fz[i] - fy[i]) / h;
            }
        }
        jac
    }
}

/// SciPy's `select_initial_step` for a method of the given order.
#[allow(clippy::too_many_arguments)]
fn select_initial_step<F, J>(sys: &mut Counter<F, J>, t0: f64, y0: &[f64], f0: &[f64], span: f64, max_step: f64, order: f64, o: &IvpOptions) -> f64
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    let scale: Vec<f64> = y0.iter().map(|v| o.atol + v.abs() * o.rtol).collect();
    let d0 = rms(y0, &scale);
    let d1 = rms(f0, &scale);
    let h0 = if d0 < 1e-5 || d1 < 1e-5 { 1e-6 } else { 0.01 * d0 / d1 };
    let h0 = h0.min(span);
    let y1: Vec<f64> = y0.iter().zip(f0).map(|(y, f)| y + h0 * f).collect();
    let f1 = sys.f(t0 + h0, &y1);
    let diff: Vec<f64> = f1.iter().zip(f0).map(|(a, b)| a - b).collect();
    let d2 = rms(&diff, &scale) / h0;
    let h1 = if d1 <= 1e-15 && d2 <= 1e-15 { (h0 * 1e-3).max(1e-6) } else { (0.01 / d1.max(d2)).powf(1.0 / (order + 1.0)) };
    (100.0 * h0).min(h1).min(span).min(max_step)
}

/// Brent's method on `[a, b]` with SciPy's `brentq` tolerances `xtol = rtol = 4ε`.
fn brentq(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let (xtol, rtol) = (4.0 * f64::EPSILON, 4.0 * f64::EPSILON);
    let (mut xpre, mut xcur) = (a, b);
    let (mut xblk, mut fblk, mut spre, mut scur) = (0.0, 0.0, 0.0, 0.0);
    let mut fpre = f(xpre);
    let mut fcur = f(xcur);
    if fpre == 0.0 {
        return xpre;
    }
    if fcur == 0.0 {
        return xcur;
    }
    for _ in 0..100 {
        if fpre != 0.0 && fcur != 0.0 && (fpre < 0.0) != (fcur < 0.0) {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }
        let delta = (xtol + rtol * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || sbis.abs() < delta {
            return xcur;
        }
        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }
        xpre = xcur;
        fpre = fcur;
        xcur += if scur.abs() > delta { scur } else if sbis > 0.0 { delta } else { -delta };
        fcur = f(xcur);
    }
    xcur
}

/// A method that advances one accepted step at a time.
trait Stepper {
    fn t(&self) -> f64;
    fn y(&self) -> &[f64];
    /// Takes one accepted step and returns its dense-output piece.
    fn step(&mut self) -> Result<Segment, ErrorsJSL>;
    fn counts(&self) -> (usize, usize, usize, usize);
    fn switches(&self) -> Vec<(f64, bool)> {
        Vec::new()
    }
}

// ------------------------------------------------------------------------------------- Radau

const NEWTON_MAXITER: usize = 6;

/// The result of the collocation solve: converged, iterations, stage increments, rate.
type Collocation = (bool, usize, [Vec<f64>; 3], Option<f64>);

struct RadauConstants {
    c: [f64; 3],
    e: [f64; 3],
    mu_real: f64,
    mu_complex: C,
    t: [[f64; 3]; 3],
    ti: [[f64; 3]; 3],
    p: [[f64; 3]; 3],
}

// The digits are SciPy's, kept so that the two implementations take the same steps.
#[allow(clippy::excessive_precision)]
fn radau_constants() -> RadauConstants {
    let s6 = 6f64.sqrt();
    RadauConstants {
        c: [(4.0 - s6) / 10.0, (4.0 + s6) / 10.0, 1.0],
        e: [(-13.0 - 7.0 * s6) / 3.0, (-13.0 + 7.0 * s6) / 3.0, -1.0 / 3.0],
        mu_real: 3.0 + 3f64.powf(2.0 / 3.0) - 3f64.powf(1.0 / 3.0),
        mu_complex: C::new(3.0 + 0.5 * (3f64.powf(1.0 / 3.0) - 3f64.powf(2.0 / 3.0)), -0.5 * (3f64.powf(5.0 / 6.0) + 3f64.powf(7.0 / 6.0))),
        t: [
            [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
            [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
            [1.0, 1.0, 0.0],
        ],
        ti: [
            [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
            [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
            [0.50287263494578682, -2.57192694985560522, 0.59603920482822492],
        ],
        p: [
            [13.0 / 3.0 + 7.0 * s6 / 3.0, -23.0 / 3.0 - 22.0 * s6 / 3.0, 10.0 / 3.0 + 5.0 * s6],
            [13.0 / 3.0 - 7.0 * s6 / 3.0, -23.0 / 3.0 + 22.0 * s6 / 3.0, 10.0 / 3.0 - 5.0 * s6],
            [1.0 / 3.0, -8.0 / 3.0, 10.0 / 3.0],
        ],
    }
}

struct Radau<'a, F, J> {
    sys: Counter<'a, F, J>,
    k: RadauConstants,
    o: IvpOptions,
    t: f64,
    t_bound: f64,
    y: Vec<f64>,
    f: Vec<f64>,
    h_abs: f64,
    h_abs_old: Option<f64>,
    error_norm_old: Option<f64>,
    jac: Vec<Vec<f64>>,
    lu_real: Option<LuFactorization<f64>>,
    lu_complex: Option<LuFactorization<C>>,
    current_jac: bool,
    sol: Option<Segment>,
    newton_tol: f64,
    rejected: usize,
}

fn predict_factor(h_abs: f64, h_abs_old: Option<f64>, error_norm: f64, error_norm_old: Option<f64>) -> f64 {
    let multiplier = match (h_abs_old, error_norm_old) {
        (Some(ho), Some(eo)) if error_norm != 0.0 => h_abs / ho * (eo / error_norm).powf(0.25),
        _ => 1.0,
    };
    multiplier.min(1.0) * error_norm.powf(-0.25)
}

impl<'a, F, J> Radau<'a, F, J>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    fn new(mut sys: Counter<'a, F, J>, t0: f64, y0: &[f64], t_bound: f64, o: IvpOptions) -> Self {
        let f = sys.f(t0, y0);
        let h_abs = match o.first_step {
            Some(h) => h,
            None => select_initial_step(&mut sys, t0, y0, &f, t_bound - t0, o.max_step, 3.0, &o),
        };
        let jac = sys.jacobian(t0, y0, &f);
        let newton_tol = (10.0 * f64::EPSILON / o.rtol).max(0.03f64.min(o.rtol.sqrt()));
        Radau {
            sys,
            k: radau_constants(),
            o,
            t: t0,
            t_bound,
            y: y0.to_vec(),
            f,
            h_abs,
            h_abs_old: None,
            error_norm_old: None,
            jac,
            lu_real: None,
            lu_complex: None,
            current_jac: true,
            sol: None,
            newton_tol,
            rejected: 0,
        }
    }

    fn factor_real(&mut self, h: f64) -> Result<LuFactorization<f64>, ErrorsJSL> {
        let n = self.y.len();
        self.sys.nlu += 1;
        let m = Array2::from_shape_fn((n, n), |(i, j)| if i == j { self.k.mu_real / h } else { 0.0 } - self.jac[i][j]);
        m.factorize().map_err(|_| ErrorsJSL::RuntimeError("singular Newton matrix in Radau"))
    }

    fn factor_complex(&mut self, h: f64) -> Result<LuFactorization<C>, ErrorsJSL> {
        let n = self.y.len();
        self.sys.nlu += 1;
        let m = Array2::from_shape_fn((n, n), |(i, j)| if i == j { self.k.mu_complex / h } else { C::new(0.0, 0.0) } - self.jac[i][j]);
        m.factorize().map_err(|_| ErrorsJSL::RuntimeError("singular Newton matrix in Radau"))
    }

    // ANCHOR: collocation
    /// Simplified Newton iteration for the stage increments `Z` (SciPy's
    /// `solve_collocation_system`): in the eigenbasis `W = T⁻¹ Z` of the Radau matrix, the 3n × 3n
    /// system splits into one real and one complex n × n system.
    fn solve_collocation(&mut self, h: f64, z0: [Vec<f64>; 3], scale: &[f64]) -> Result<Collocation, ErrorsJSL> {
        let n = self.y.len();
        let (t, y) = (self.t, self.y.clone());
        let k = &self.k;
        let m_real = k.mu_real / h;
        let m_complex = k.mu_complex / h;
        let mut w: [Vec<f64>; 3] = std::array::from_fn(|r| (0..n).map(|i| (0..3).map(|c| k.ti[r][c] * z0[c][i]).sum()).collect());
        let mut z = z0;
        let mut dw_norm_old: Option<f64> = None;
        let mut rate: Option<f64> = None;
        let mut converged = false;
        let mut iterations = 0;
        let lu_real = self.lu_real.as_ref().expect("factorised");
        let lu_complex = self.lu_complex.as_ref().expect("factorised");
        for it in 0..NEWTON_MAXITER {
            iterations = it + 1;
            let fs: Vec<Vec<f64>> = (0..3)
                .map(|s| {
                    let ys: Vec<f64> = y.iter().zip(&z[s]).map(|(a, b)| a + b).collect();
                    self.sys.f(t + h * k.c[s], &ys)
                })
                .collect();
            if fs.iter().flatten().any(|v| !v.is_finite()) {
                break;
            }
            let f_real: Array1<f64> = (0..n).map(|i| (0..3).map(|s| fs[s][i] * k.ti[0][s]).sum::<f64>() - m_real * w[0][i]).collect();
            let f_complex: Array1<C> = (0..n)
                .map(|i| (0..3).map(|s| fs[s][i] * C::new(k.ti[1][s], k.ti[2][s])).sum::<C>() - m_complex * C::new(w[1][i], w[2][i]))
                .collect();
            let dw_real = lu_real.solve(&f_real).map_err(|_| ErrorsJSL::RuntimeError("Radau solve failed"))?;
            let dw_complex = lu_complex.solve(&f_complex).map_err(|_| ErrorsJSL::RuntimeError("Radau solve failed"))?;
            let dw: [Vec<f64>; 3] = [dw_real.to_vec(), dw_complex.iter().map(|z| z.re).collect(), dw_complex.iter().map(|z| z.im).collect()];
            let flat: Vec<f64> = dw.iter().flatten().copied().collect();
            let flat_scale: Vec<f64> = (0..3).flat_map(|_| scale.iter().copied()).collect();
            let dw_norm = rms(&flat, &flat_scale);
            if let Some(old) = dw_norm_old {
                rate = Some(dw_norm / old);
            }
            if let Some(r) = rate
                && (r >= 1.0 || r.powi((NEWTON_MAXITER - it) as i32) / (1.0 - r) * dw_norm > self.newton_tol)
            {
                break;
            }
            for r in 0..3 {
                for i in 0..n {
                    w[r][i] += dw[r][i];
                }
            }
            z = std::array::from_fn(|r| (0..n).map(|i| (0..3).map(|c| k.t[r][c] * w[c][i]).sum()).collect());
            if dw_norm == 0.0 || rate.is_some_and(|r| r / (1.0 - r) * dw_norm < self.newton_tol) {
                converged = true;
                break;
            }
            dw_norm_old = Some(dw_norm);
        }
        Ok((converged, iterations, z, rate))
    }
    // ANCHOR_END: collocation
}

impl<F, J> Stepper for Radau<'_, F, J>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    fn t(&self) -> f64 {
        self.t
    }
    fn y(&self) -> &[f64] {
        &self.y
    }
    fn counts(&self) -> (usize, usize, usize, usize) {
        (self.sys.nfev, self.sys.njev, self.sys.nlu, self.rejected)
    }

    fn step(&mut self) -> Result<Segment, ErrorsJSL> {
        let (t, y) = (self.t, self.y.clone());
        let n = y.len();
        let min_step = 10.0 * (t.next_up() - t).abs();
        let (mut h_abs, h_abs_old, error_norm_old) = if self.h_abs > self.o.max_step {
            (self.o.max_step, None, None)
        } else if self.h_abs < min_step {
            (min_step, None, None)
        } else {
            (self.h_abs, self.h_abs_old, self.error_norm_old)
        };
        let mut rejected = false;
        let (h, t_new, z, error_norm, n_iter, rate);
        loop {
            if h_abs < min_step {
                return Err(ErrorsJSL::RuntimeError("Radau: the required step size is less than the spacing between numbers"));
            }
            let mut t_try = t + h_abs;
            if t_try > self.t_bound {
                t_try = self.t_bound;
            }
            let h_try = t_try - t;
            h_abs = h_try.abs();
            let z0: [Vec<f64>; 3] = match &self.sol {
                None => std::array::from_fn(|_| vec![0.0; n]),
                Some(s) => std::array::from_fn(|i| s.evaluate(t + h_try * self.k.c[i]).iter().zip(&y).map(|(a, b)| a - b).collect()),
            };
            let scale: Vec<f64> = y.iter().map(|v| self.o.atol + v.abs() * self.o.rtol).collect();
            let mut result;
            loop {
                if self.lu_real.is_none() || self.lu_complex.is_none() {
                    self.lu_real = Some(self.factor_real(h_try)?);
                    self.lu_complex = Some(self.factor_complex(h_try)?);
                }
                result = self.solve_collocation(h_try, z0.clone(), &scale)?;
                if !result.0 {
                    if self.current_jac {
                        break;
                    }
                    self.jac = self.sys.jacobian(t, &y, &self.f.clone());
                    self.current_jac = true;
                    self.lu_real = None;
                    self.lu_complex = None;
                    continue;
                }
                break;
            }
            let (converged, it, zz, rt) = result;
            if !converged {
                h_abs *= 0.5;
                self.lu_real = None;
                self.lu_complex = None;
                continue;
            }
            let y_new: Vec<f64> = y.iter().zip(&zz[2]).map(|(a, b)| a + b).collect();
            let ze: Vec<f64> = (0..n).map(|i| (0..3).map(|s| zz[s][i] * self.k.e[s]).sum::<f64>() / h_try).collect();
            let lu = self.lu_real.as_ref().expect("factorised");
            let rhs: Array1<f64> = (0..n).map(|i| self.f[i] + ze[i]).collect();
            let mut err = lu.solve(&rhs).map_err(|_| ErrorsJSL::RuntimeError("Radau solve failed"))?.to_vec();
            let scale: Vec<f64> = y.iter().zip(&y_new).map(|(a, b)| self.o.atol + a.abs().max(b.abs()) * self.o.rtol).collect();
            let mut en = rms(&err, &scale);
            let safety = 0.9 * (2 * NEWTON_MAXITER + 1) as f64 / (2 * NEWTON_MAXITER + it) as f64;
            if rejected && en > 1.0 {
                let ye: Vec<f64> = y.iter().zip(&err).map(|(a, b)| a + b).collect();
                let fe = self.sys.f(t, &ye);
                let rhs: Array1<f64> = (0..n).map(|i| fe[i] + ze[i]).collect();
                err = self.lu_real.as_ref().expect("factorised").solve(&rhs).map_err(|_| ErrorsJSL::RuntimeError("Radau solve failed"))?.to_vec();
                en = rms(&err, &scale);
            }
            if en > 1.0 {
                let factor = predict_factor(h_abs, h_abs_old, en, error_norm_old);
                h_abs *= (safety * factor).max(0.2);
                self.lu_real = None;
                self.lu_complex = None;
                rejected = true;
                self.rejected += 1;
                continue;
            }
            h = h_try;
            t_new = t_try;
            z = zz;
            error_norm = en;
            n_iter = it;
            rate = rt;
            let recompute_jac = n_iter > 2 && rate.is_some_and(|r| r > 1e-3);
            let mut factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old);
            factor = (safety * factor).min(10.0);
            if !recompute_jac && factor < 1.2 {
                factor = 1.0;
            } else {
                self.lu_real = None;
                self.lu_complex = None;
            }
            let y_new_v = y_new;
            let f_new = self.sys.f(t_new, &y_new_v);
            if recompute_jac {
                self.jac = self.sys.jacobian(t_new, &y_new_v, &f_new);
                self.current_jac = true;
            } else {
                self.current_jac = false;
            }
            self.h_abs_old = Some(self.h_abs);
            self.error_norm_old = Some(error_norm);
            self.h_abs = h_abs * factor;
            self.t = t_new;
            self.y = y_new_v;
            self.f = f_new;
            break;
        }
        // Dense output: y(t_old + x h) = y_old + Q (x, x², x³), Q = Zᵀ P.
        let coeffs: Vec<Vec<f64>> = (0..3).map(|kk| (0..n).map(|i| (0..3).map(|s| z[s][i] * self.k.p[s][kk]).sum()).collect()).collect();
        let seg = Segment { t_start: t, t_end: t_new, t_ref: t, h, y_ref: y, coeffs };
        self.sol = Some(seg.clone());
        Ok(seg)
    }
}

// ------------------------------------------------------------------------------------- LSODA

/// The ODEPACK coefficients (`cfode`): for each order `q`, the corrector vector `l` (length
/// `q + 1`) and the test constants `(tesco₁, tesco₂, tesco₃)` for orders `q − 1`, `q`, `q + 1`.
struct Coefficients {
    el: Vec<Vec<f64>>,
    tesco: Vec<[f64; 3]>,
}

// ANCHOR: cfode
fn cfode(adams: bool) -> Coefficients {
    let max_order = if adams { 12 } else { 5 };
    let mut el = vec![Vec::new(); max_order + 1];
    let mut tesco = vec![[0.0; 3]; max_order + 2];
    if adams {
        el[1] = vec![1.0, 1.0];
        tesco[1] = [0.0, 2.0, 0.0];
        tesco[2][0] = 1.0;
        let mut pc = [0.0; 13];
        pc[0] = 1.0;
        let mut rqfac = 1.0;
        for nq in 2..=12usize {
            // p(x) = (x + 1)(x + 2)…(x + nq − 1), its integral over [−1, 0] and that of x p(x).
            let rq1fac = rqfac;
            rqfac /= nq as f64;
            let fnqm1 = (nq - 1) as f64;
            pc[nq - 1] = 0.0;
            for i in (1..nq).rev() {
                pc[i] = pc[i - 1] + fnqm1 * pc[i];
            }
            pc[0] *= fnqm1;
            let (mut pint, mut xpin, mut tsign) = (pc[0], pc[0] / 2.0, 1.0);
            for i in 1..nq {
                tsign = -tsign;
                pint += tsign * pc[i] / (i + 1) as f64;
                xpin += tsign * pc[i] / (i + 2) as f64;
            }
            let mut e = vec![0.0; nq + 1];
            e[0] = pint * rq1fac;
            e[1] = 1.0;
            for i in 1..nq {
                e[i + 1] = rq1fac * pc[i] / (i + 1) as f64;
            }
            el[nq] = e;
            let agamq = rqfac * xpin;
            let ragq = 1.0 / agamq;
            tesco[nq][1] = ragq;
            if nq < 12 {
                tesco[nq + 1][0] = ragq * rqfac / (nq + 1) as f64;
            }
            tesco[nq - 1][2] = ragq;
        }
    } else {
        let mut pc = [0.0; 7];
        pc[0] = 1.0;
        let mut rq1fac = 1.0;
        for nq in 1..=5usize {
            // p(x) = x (x + 1)…(x + nq − 1).
            let fnq = nq as f64;
            pc[nq] = 0.0;
            for i in (1..=nq).rev() {
                pc[i] = pc[i - 1] + fnq * pc[i];
            }
            pc[0] *= fnq;
            let mut e: Vec<f64> = (0..=nq).map(|i| pc[i] / pc[1]).collect();
            e[1] = 1.0;
            tesco[nq] = [rq1fac, (nq + 1) as f64 / e[0], (nq + 2) as f64 / e[0]];
            el[nq] = e;
            rq1fac /= fnq;
        }
    }
    Coefficients { el, tesco }
}
// ANCHOR_END: cfode

/// Stability limits `h ‖J‖` of the Adams methods of orders 1 to 12 (LSODA's `sm1`).
const SM1: [f64; 13] = [0.0, 0.5, 0.575, 0.55, 0.45, 0.35, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025];

struct Lsoda<'a, F, J> {
    sys: Counter<'a, F, J>,
    o: IvpOptions,
    adams: Coefficients,
    bdf: Coefficients,
    stiff: bool,
    t: f64,
    t_bound: f64,
    /// Nordsieck array: z[j] = hʲ y⁽ʲ⁾ / j!.
    z: Vec<Vec<f64>>,
    q: usize,
    h: f64,
    /// Steps until the next order/step-size decision.
    countdown: usize,
    acor_saved: Option<Vec<f64>>,
    jac: Option<Vec<Vec<f64>>>,
    jac_age: usize,
    lu: Option<(LuFactorization<f64>, f64)>,
    lipschitz: f64,
    steps: usize,
    last_switch: usize,
    switches: Vec<(f64, bool)>,
    rejected: usize,
    y_now: Vec<f64>,
}

impl<'a, F, J> Lsoda<'a, F, J>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    fn new(mut sys: Counter<'a, F, J>, t0: f64, y0: &[f64], t_bound: f64, o: IvpOptions) -> Self {
        let f0 = sys.f(t0, y0);
        let h = match o.first_step {
            Some(h) => h,
            None => select_initial_step(&mut sys, t0, y0, &f0, t_bound - t0, o.max_step, 1.0, &o),
        };
        let z = vec![y0.to_vec(), f0.iter().map(|v| v * h).collect()];
        Lsoda {
            sys,
            o,
            adams: cfode(true),
            bdf: cfode(false),
            stiff: false,
            t: t0,
            t_bound,
            z,
            q: 1,
            h,
            countdown: 2,
            acor_saved: None,
            jac: None,
            jac_age: 0,
            lu: None,
            lipschitz: 0.0,
            steps: 0,
            last_switch: 0,
            switches: Vec::new(),
            rejected: 0,
            y_now: y0.to_vec(),
        }
    }

    fn coefficients(&self) -> &Coefficients {
        if self.stiff { &self.bdf } else { &self.adams }
    }

    fn max_order(&self) -> usize {
        if self.stiff { 5 } else { 12 }
    }

    fn rescale(&mut self, r: f64) {
        let mut p = 1.0;
        for zj in self.z.iter_mut().skip(1) {
            p *= r;
            zj.iter_mut().for_each(|v| *v *= p);
        }
        self.h *= r;
    }

    /// Multiplies the Nordsieck array by the Pascal matrix: the predictor at `t + h`.
    fn predict(z: &mut [Vec<f64>]) {
        let q = z.len() - 1;
        for k in 0..q {
            for j in (k + 1..=q).rev() {
                let (lo, hi) = z.split_at_mut(j);
                for (a, b) in lo[j - 1].iter_mut().zip(&hi[0]) {
                    *a += b;
                }
            }
        }
    }

    /// The weighted norm used by ODEPACK: RMS of `v / ewt`, `ewt = rtol |y| + atol`.
    fn norm(&self, v: &[f64]) -> f64 {
        let ewt: Vec<f64> = self.z[0].iter().map(|y| self.o.rtol * y.abs() + self.o.atol).collect();
        rms(v, &ewt)
    }

    /// The weighted row norm of the Jacobian, LSODA's estimate of ‖J‖ in stiff mode.
    fn jacobian_norm(&self, jac: &[Vec<f64>]) -> f64 {
        let ewt: Vec<f64> = self.z[0].iter().map(|y| self.o.rtol * y.abs() + self.o.atol).collect();
        jac.iter().enumerate().map(|(i, row)| row.iter().zip(&ewt).map(|(a, e)| a.abs() * e).sum::<f64>() / ewt[i]).fold(0.0, f64::max)
    }
}

impl<F, J> Stepper for Lsoda<'_, F, J>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    fn t(&self) -> f64 {
        self.t
    }
    fn y(&self) -> &[f64] {
        &self.y_now
    }
    fn counts(&self) -> (usize, usize, usize, usize) {
        (self.sys.nfev, self.sys.njev, self.sys.nlu, self.rejected)
    }
    fn switches(&self) -> Vec<(f64, bool)> {
        self.switches.clone()
    }

    // ANCHOR: lsoda_step
    fn step(&mut self) -> Result<Segment, ErrorsJSL> {
        let n = self.z[0].len();
        let mut failures = 0;
        loop {
            if self.h > self.o.max_step {
                self.rescale(self.o.max_step / self.h);
            }
            if self.t + self.h > self.t_bound {
                self.rescale((self.t_bound - self.t) / self.h);
            }
            if self.h < 10.0 * (self.t.next_up() - self.t) {
                return Err(ErrorsJSL::RuntimeError("LSODA: the step size underflowed"));
            }
            let t_new = if self.t + self.h >= self.t_bound * (1.0 - 1e-15) && self.t_bound - self.t <= self.h * (1.0 + 1e-12) { self.t_bound } else { self.t + self.h };
            let q = self.q;
            let (el, tesco) = {
                let c = self.coefficients();
                (c.el[q].clone(), c.tesco[q])
            };
            let mut zp = self.z.clone();
            Self::predict(&mut zp);
            // Corrector: acor solves acor = h f(z₀ + l₀ acor) − z₁, by functional iteration
            // (Adams) or by Newton's method with P = I − h l₀ J (BDF).
            if self.stiff {
                let need_jac = self.jac.is_none() || self.jac_age >= 20;
                if need_jac {
                    let f0 = self.sys.f(t_new, &zp[0]);
                    let jac = self.sys.jacobian(t_new, &zp[0], &f0);
                    self.lipschitz = self.jacobian_norm(&jac);
                    self.jac = Some(jac);
                    self.jac_age = 0;
                    self.lu = None;
                }
                let hl0 = self.h * el[0];
                if self.lu.as_ref().is_none_or(|(_, v)| (hl0 / v - 1.0).abs() > 0.3) {
                    let jac = self.jac.as_ref().expect("evaluated");
                    let m = Array2::from_shape_fn((n, n), |(i, j)| if i == j { 1.0 } else { 0.0 } - hl0 * jac[i][j]);
                    self.sys.nlu += 1;
                    let lu = m.factorize().map_err(|_| ErrorsJSL::RuntimeError("LSODA: singular iteration matrix"))?;
                    self.lu = Some((lu, hl0));
                }
            }
            let mut acor = vec![0.0; n];
            let mut y = zp[0].clone();
            let (mut crate_, mut delp) = (0.7f64, 0.0f64);
            let conit = 0.5 / (q + 2) as f64;
            let mut converged = false;
            let (mut prev_y, mut prev_f): (Option<Vec<f64>>, Option<Vec<f64>>) = (None, None);
            let mut lipschitz_step = 0.0f64;
            for m in 0..3 {
                let fy = self.sys.f(t_new, &y);
                if fy.iter().any(|v| !v.is_finite()) {
                    break;
                }
                let del;
                if self.stiff {
                    // If the LU was formed for a different h l₀, correct the Newton step by the ratio.
                    let (lu, hl0_lu) = self.lu.as_ref().expect("factorised");
                    let r: Array1<f64> = (0..n).map(|i| self.h * fy[i] - zp[1][i] - acor[i]).collect();
                    let mut x = lu.solve(&r).map_err(|_| ErrorsJSL::RuntimeError("LSODA: solve failed"))?.to_vec();
                    let ratio = 2.0 / (1.0 + self.h * el[0] / hl0_lu);
                    if (ratio - 1.0).abs() > 1e-12 {
                        x.iter_mut().for_each(|v| *v *= ratio);
                    }
                    del = self.norm(&x);
                    for i in 0..n {
                        acor[i] += x[i];
                    }
                } else {
                    let r: Vec<f64> = (0..n).map(|i| self.h * fy[i] - zp[1][i]).collect();
                    let d: Vec<f64> = r.iter().zip(&acor).map(|(a, b)| a - b).collect();
                    del = self.norm(&d);
                    acor = r;
                    if let (Some(py), Some(pf)) = (&prev_y, &prev_f) {
                        let dy: Vec<f64> = y.iter().zip(py).map(|(a, b)| a - b).collect();
                        let df: Vec<f64> = fy.iter().zip(pf).map(|(a, b)| a - b).collect();
                        let ny = self.norm(&dy);
                        if ny > 0.0 {
                            lipschitz_step = lipschitz_step.max(self.norm(&df) / ny);
                        }
                    }
                    prev_y = Some(y.clone());
                    prev_f = Some(fy.clone());
                }
                y = (0..n).map(|i| zp[0][i] + el[0] * acor[i]).collect();
                if m > 0 {
                    crate_ = (0.2 * crate_).max(del / delp);
                }
                let dcon = del * (1.5 * crate_).min(1.0) / (tesco[1] * conit);
                if dcon <= 1.0 {
                    converged = true;
                    break;
                }
                if m >= 1 && del > 2.0 * delp {
                    break;
                }
                delp = del;
            }
            if !self.stiff && lipschitz_step > 0.0 {
                self.lipschitz = lipschitz_step;
            }
            if !converged {
                failures += 1;
                if self.stiff && self.jac_age > 0 {
                    self.jac = None;
                    continue;
                }
                self.rejected += 1;
                if failures > 10 {
                    return Err(ErrorsJSL::RuntimeError("LSODA: repeated corrector failures"));
                }
                self.rescale(0.25);
                continue;
            }
            // Local error test.
            let dsm = self.norm(&acor) / tesco[1];
            if dsm > 1.0 {
                failures += 1;
                self.rejected += 1;
                if failures >= 3 {
                    // Restart at order one from the current state.
                    let f0 = self.sys.f(self.t, &self.z[0]);
                    self.z.truncate(2);
                    self.q = 1;
                    self.h *= 0.1;
                    self.z[1] = f0.iter().map(|v| v * self.h).collect();
                    self.countdown = 2;
                    self.acor_saved = None;
                    continue;
                }
                // As ODEPACK: the larger of the steps allowed at this order and one lower; after a
                // second failure, at most a fifth of the step.
                let rhsm = 1.0 / (1.2 * dsm.powf(1.0 / (q + 1) as f64) + 1.2e-6);
                let rhdn = if q > 1 { 1.0 / (1.3 * (self.norm(&self.z[q]) / tesco[0]).powf(1.0 / q as f64) + 1.3e-6) } else { 0.0 };
                let mut rh = rhsm;
                if rhdn > rhsm {
                    rh = rhdn;
                    self.z.truncate(q);
                    self.q = q - 1;
                }
                if failures >= 2 {
                    rh = rh.min(0.2);
                }
                self.rescale(rh.clamp(0.1, 0.9));
                self.countdown = self.q + 1;
                self.acor_saved = None;
                continue;
            }
            // Accept: z ← z_pred + l acor.
            for j in 0..=q {
                for i in 0..n {
                    zp[j][i] += el[j] * acor[i];
                }
            }
            self.z = zp;
            self.t = t_new;
            self.y_now = self.z[0].clone();
            self.steps += 1;
            if self.stiff {
                self.jac_age += 1;
            }
            let seg = Segment { t_start: t_new - self.h, t_end: t_new, t_ref: t_new, h: self.h, y_ref: self.z[0].clone(), coeffs: self.z[1..].to_vec() };
            // Order and step size, every q + 1 steps (ODEPACK's `ialth`).
            self.countdown = self.countdown.saturating_sub(1);
            if self.countdown == 1 && q < self.max_order() {
                self.acor_saved = Some(acor.clone());
            }
            if self.countdown == 0 {
                let rhsm = 1.0 / (1.2 * dsm.powf(1.0 / (q + 1) as f64) + 1.2e-6);
                let rhdn = if q > 1 { 1.0 / (1.3 * (self.norm(&self.z[q]) / tesco[0]).powf(1.0 / q as f64) + 1.3e-6) } else { 0.0 };
                let rhup = match (&self.acor_saved, q < self.max_order()) {
                    (Some(prev), true) => {
                        let d: Vec<f64> = acor.iter().zip(prev).map(|(a, b)| a - b).collect();
                        1.0 / (1.4 * (self.norm(&d) / tesco[2]).powf(1.0 / (q + 2) as f64) + 1.4e-6)
                    }
                    _ => 0.0,
                };
                let (mut rh, mut new_q) = (rhsm, q);
                if rhup > rh {
                    rh = rhup;
                    new_q = q + 1;
                }
                if rhdn > rh {
                    rh = rhdn;
                    new_q = q - 1;
                }
                // Stiffness detection (Petzold): compare the step each method could take, with
                // the Adams step also limited by its stability region, h ‖J‖ ≤ sm1(q).
                if self.steps >= self.last_switch + 20 && self.lipschitz > 0.0 {
                    let cm = |c: &Coefficients, k: usize| c.tesco[k][1] * c.el[k][k];
                    if !self.stiff {
                        let rh1 = rh.min(SM1[new_q] / (self.h * self.lipschitz));
                        let q2 = q.min(5);
                        let dm2 = dsm * (cm(&self.adams, q) / cm(&self.bdf, q2));
                        let rh2 = 1.0 / (1.2 * dm2.powf(1.0 / (q2 + 1) as f64) + 1.2e-6);
                        if rh2 >= 5.0 * rh1 {
                            self.stiff = true;
                            self.switches.push((self.t, true));
                            self.last_switch = self.steps;
                            self.jac = None;
                            self.lu = None;
                            self.z.truncate(q2 + 1);
                            self.q = q2;
                            self.countdown = q2 + 1;
                            self.acor_saved = None;
                            self.rescale(rh2.min(10.0));
                            return Ok(seg);
                        }
                    } else {
                        let dm1 = dsm * (cm(&self.bdf, q) / cm(&self.adams, q));
                        let rh1 = (1.0 / (1.2 * dm1.powf(1.0 / (q + 1) as f64) + 1.2e-6)).min(SM1[q] / (self.h * self.lipschitz));
                        if rh1 >= rh {
                            self.stiff = false;
                            self.switches.push((self.t, false));
                            self.last_switch = self.steps;
                            self.lu = None;
                            self.countdown = q + 1;
                            self.acor_saved = None;
                            self.rescale(rh1.min(10.0));
                            return Ok(seg);
                        }
                    }
                }
                let rh = rh.min(10.0);
                if rh >= 1.1 {
                    if new_q > q {
                        let extra: Vec<f64> = acor.iter().map(|a| a * el[q] / (q + 1) as f64).collect();
                        self.z.push(extra);
                    } else if new_q < q {
                        self.z.truncate(new_q + 1);
                    }
                    self.q = new_q;
                    self.rescale(rh);
                } else if new_q < q && rhdn >= rhsm {
                    self.z.truncate(new_q + 1);
                    self.q = new_q;
                }
                self.countdown = self.q + 1;
                self.acor_saved = None;
            }
            return Ok(seg);
        }
    }
    // ANCHOR_END: lsoda_step
}

// ------------------------------------------------------------------------------------- driver

// ANCHOR: solve_ivp
/// Solves `y' = f(t, y)`, `y(t0) = y0` on `[t0, t1]` (`t1 > t0`) with the given method (SciPy's
/// `solve_ivp`), returning every accepted step, the dense output, and the occurrences of each
/// event. `jac`, if given, returns `∂f/∂y` as rows; otherwise forward differences are used.
///
/// After each step, every event function is evaluated at the new point; where its sign changes
/// in the requested direction, the zero is found by Brent's method on the step's interpolant.
/// A terminal event that reaches its count ends the integration there.
pub fn solve_ivp<F, J>(f: F, jac: Option<J>, t_span: (f64, f64), y0: &[f64], method: IvpMethod, options: IvpOptions, events: &[Event]) -> Result<IvpSolution, ErrorsJSL>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    let (t0, t1) = t_span;
    if y0.is_empty() || !(t1 > t0) || !t0.is_finite() || !t1.is_finite() {
        return Err(ErrorsJSL::InvalidInputRange("need a non-empty state and finite t0 < t1"));
    }
    if !(options.rtol > 0.0) || !(options.atol > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("tolerances must be positive"));
    }
    let sys = Counter { f: &f, jac: jac.as_ref(), nfev: 0, njev: 0, nlu: 0 };
    let mut stepper: Box<dyn Stepper + '_> = match method {
        IvpMethod::Radau => Box::new(Radau::new(sys, t0, y0, t1, options)),
        IvpMethod::Lsoda => Box::new(Lsoda::new(sys, t0, y0, t1, options)),
    };
    let mut ts = vec![t0];
    let mut ys = vec![y0.to_vec()];
    let mut t_events = vec![Vec::new(); events.len()];
    let mut y_events = vec![Vec::new(); events.len()];
    let mut counts = vec![0usize; events.len()];
    let mut g: Vec<f64> = events.iter().map(|e| (e.function)(t0, y0)).collect();
    let mut dense = DenseOutput::default();
    let mut status = IvpStatus::Finished;
    let mut attempts = 0;
    while stepper.t() < t1 {
        attempts += 1;
        if attempts > options.max_steps {
            return Err(ErrorsJSL::RuntimeError("solve_ivp: too many steps"));
        }
        let t_old = stepper.t();
        let seg = stepper.step()?;
        let (t_new, y_new) = (stepper.t(), stepper.y().to_vec());
        let g_new: Vec<f64> = events.iter().map(|e| (e.function)(t_new, &y_new)).collect();
        // Events active in this step, with their roots, in time order.
        let mut active: Vec<(f64, usize)> = Vec::new();
        for (i, e) in events.iter().enumerate() {
            let up = g[i] <= 0.0 && g_new[i] >= 0.0;
            let down = g[i] >= 0.0 && g_new[i] <= 0.0;
            if (up && e.direction > 0.0) || (down && e.direction < 0.0) || ((up || down) && e.direction == 0.0) {
                let root = brentq(|t| (e.function)(t, &seg.evaluate(t)), t_old, t_new);
                active.push((root, i));
            }
        }
        active.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut stop_at = None;
        for &(root, i) in &active {
            t_events[i].push(root);
            y_events[i].push(seg.evaluate(root));
            counts[i] += 1;
            if events[i].terminal.is_some_and(|k| counts[i] >= k) {
                stop_at = Some(root);
                break;
            }
        }
        g = g_new;
        dense.segments.push(seg.clone());
        if let Some(root) = stop_at {
            ts.push(root);
            ys.push(seg.evaluate(root));
            if let Some(last) = dense.segments.last_mut() {
                last.t_end = root;
            }
            status = IvpStatus::TerminatedByEvent;
            break;
        }
        ts.push(t_new);
        ys.push(y_new);
    }
    let (nfev, njev, nlu, rejected_steps) = stepper.counts();
    let switches = stepper.switches();
    Ok(IvpSolution { t: ts, y: ys, status, t_events, y_events, dense, nfev, njev, nlu, rejected_steps, switches })
}
// ANCHOR_END: solve_ivp

#[cfg(test)]
mod tests {
    use super::*;

    type Jac = fn(f64, &[f64]) -> Vec<Vec<f64>>;

    #[test]
    fn cfode_matches_the_known_coefficients() {
        let bdf = cfode(false);
        // BDF2: l = (2/3, 1, 1/3).
        assert!((bdf.el[2][0] - 2.0 / 3.0).abs() < 1e-15 && (bdf.el[2][2] - 1.0 / 3.0).abs() < 1e-15);
        let adams = cfode(true);
        // Adams-Moulton order 2 (trapezoid): l = (1/2, 1, 1/2).
        assert!((adams.el[2][0] - 0.5).abs() < 1e-15 && (adams.el[2][2] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn exponential_decay_with_both_methods() {
        for method in [IvpMethod::Radau, IvpMethod::Lsoda] {
            let o = IvpOptions { rtol: 1e-8, atol: 1e-10, ..Default::default() };
            let sol = solve_ivp(|_, y: &[f64]| vec![-y[0], -2.0 * y[1]], None::<Jac>, (0.0, 5.0), &[1.0, 1.0], method, o, &[]).unwrap();
            let y = sol.y.last().unwrap();
            assert!((y[0] - (-5f64).exp()).abs() < 1e-7, "{method:?} {y:?}");
            assert!((sol.dense.evaluate(2.3)[0] - (-2.3f64).exp()).abs() < 1e-6, "{method:?} dense");
        }
    }

    #[test]
    fn a_terminal_event_stops_at_the_ground() {
        // A ball thrown up at 10 m/s from 0 m under gravity: it lands at t = 20/9.81.
        let ground = Event { function: Box::new(|_, y: &[f64]| y[0]), direction: -1.0, terminal: Some(1) };
        for method in [IvpMethod::Radau, IvpMethod::Lsoda] {
            let sol = solve_ivp(|_, y: &[f64]| vec![y[1], -9.81], None::<Jac>, (0.0, 10.0), &[0.0, 10.0], method, IvpOptions::default(), std::slice::from_ref(&ground)).unwrap();
            assert_eq!(sol.status, IvpStatus::TerminatedByEvent);
            let t = *sol.t.last().unwrap();
            // Radau is exact for this quadratic; LSODA starts at order one, accurate to the tolerance.
            assert!((t - 20.0 / 9.81).abs() < 1e-6, "{method:?} {t}");
        }
    }

    #[test]
    fn lsoda_switches_to_bdf_on_a_stiff_problem() {
        // Robertson's chemical kinetics.
        let f = |_: f64, y: &[f64]| vec![-0.04 * y[0] + 1e4 * y[1] * y[2], 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1], 3e7 * y[1] * y[1]];
        let o = IvpOptions { rtol: 1e-6, atol: 1e-10, ..Default::default() };
        let sol = solve_ivp(f, None::<Jac>, (0.0, 1e3), &[1.0, 0.0, 0.0], IvpMethod::Lsoda, o, &[]).unwrap();
        assert!(sol.switches.iter().any(|s| s.1), "no switch to BDF");
        let y = sol.y.last().unwrap();
        assert!((y.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        assert!(sol.t.len() < 2000, "{} steps", sol.t.len());
    }
}
