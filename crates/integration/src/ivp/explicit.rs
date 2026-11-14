//! Explicit Runge-Kutta methods for [`solve_ivp_explicit`]: Dormand and Prince's RK45 (order 5
//! with an embedded order-4 error estimate and a quartic interpolant) and DOP853 (Hairer's order-8
//! method with a combined order-5/order-3 error estimate and an order-7 interpolant), ported step
//! for step from SciPy's `RK45` and `DOP853`. The coefficient tables below were generated from
//! `scipy.integrate._ivp` and are exact copies of its values.
//!
//! See the book entry [Radau, LSODA and event location](https://jwkunz.github.io/lib_jsl/book/integration/radau-lsoda.html#explicit-methods-rk45-and-dop853).

use super::{Event, IvpSolution, Segment, Stepper, drive};
use lib_jsl_core::ErrorsJSL;

#[rustfmt::skip]
mod tables {
pub(super) const RK45_A: [[f64; 5]; 6] = [[0.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0, 0.0], [0.075, 0.225, 0.0, 0.0, 0.0], [0.9777777777777777, -3.7333333333333334, 3.5555555555555554, 0.0, 0.0], [2.9525986892242035, -11.595793324188385, 9.822892851699436, -0.2908093278463649, 0.0], [2.8462752525252526, -10.757575757575758, 8.906422717743473, 0.2784090909090909, -0.2735313036020583]];
pub(super) const RK45_B: [f64; 6] = [0.09114583333333333, 0.0, 0.44923629829290207, 0.6510416666666666, -0.322376179245283, 0.13095238095238096];
pub(super) const RK45_C: [f64; 6] = [0.0, 0.2, 0.3, 0.8, 0.8888888888888888, 1.0];
pub(super) const RK45_E: [f64; 7] = [-0.0012326388888888888, 0.0, 0.0042527702905061394, -0.03697916666666667, 0.05086379716981132, -0.0419047619047619, 0.025];
pub(super) const RK45_P: [[f64; 4]; 7] = [[1.0, -2.8535800653862835, 3.0717434641059005, -1.1270175653862835], [0.0, 0.0, 0.0, 0.0], [0.0, 4.023133379230305, -6.249321565289, 2.675424484351598], [0.0, -3.7324019615885042, 10.068970589843675, -5.685526961588504], [0.0, 2.5548038301849423, -6.399112377351017, 3.5219323679207912], [0.0, -1.3744241142186024, 3.272657752246729, -1.7672812570757455], [0.0, 1.3824689317781436, -3.764937863556287, 2.382468931778144]];
pub(super) const DOP_A: [[f64; 16]; 16] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.05260015195876773, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0197250569845379, 0.0591751709536137, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.02958758547680685, 0.0, 0.08876275643042054, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2413651341592667, 0.0, -0.8845494793282861, 0.924834003261792, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.037037037037037035, 0.0, 0.0, 0.17082860872947386, 0.12546768756682242, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.037109375, 0.0, 0.0, 0.17025221101954405, 0.06021653898045596, -0.017578125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.03709200011850479, 0.0, 0.0, 0.17038392571223998, 0.10726203044637328, -0.015319437748624402, 0.008273789163814023, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6241109587160757, 0.0, 0.0, -3.3608926294469414, -0.868219346841726, 27.59209969944671, 20.154067550477894, -43.48988418106996, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.47766253643826434, 0.0, 0.0, -2.4881146199716677, -0.590290826836843, 21.230051448181193, 15.279233632882423, -33.28821096898486, -0.020331201708508627, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.9371424300859873, 0.0, 0.0, 5.186372428844064, 1.0914373489967295, -8.149787010746927, -18.52006565999696, 22.739487099350505, 2.4936055526796523, -3.0467644718982196, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.273310147516538, 0.0, 0.0, -10.53449546673725, -2.0008720582248625, -17.9589318631188, 27.94888452941996, -2.8589982771350235, -8.87285693353063, 12.360567175794303, 0.6433927460157636, 0.0, 0.0, 0.0, 0.0, 0.0], [0.054293734116568765, 0.0, 0.0, 0.0, 0.0, 4.450312892752409, 1.8915178993145003, -5.801203960010585, 0.3111643669578199, -0.1521609496625161, 0.20136540080403034, 0.04471061572777259, 0.0, 0.0, 0.0, 0.0], [0.056167502283047954, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25350021021662483, -0.2462390374708025, -0.12419142326381637, 0.15329179827876568, 0.00820105229563469, 0.007567897660545699, -0.008298, 0.0, 0.0, 0.0], [0.03183464816350214, 0.0, 0.0, 0.0, 0.0, 0.028300909672366776, 0.053541988307438566, -0.05492374857139099, 0.0, 0.0, -0.00010834732869724932, 0.0003825710908356584, -0.00034046500868740456, 0.1413124436746325, 0.0, 0.0], [-0.42889630158379194, 0.0, 0.0, 0.0, 0.0, -4.697621415361164, 7.683421196062599, 4.06898981839711, 0.3567271874552811, 0.0, 0.0, 0.0, -0.0013990241651590145, 2.9475147891527724, -9.15095847217987, 0.0]];
pub(super) const DOP_B: [f64; 12] = [0.054293734116568765, 0.0, 0.0, 0.0, 0.0, 4.450312892752409, 1.8915178993145003, -5.801203960010585, 0.3111643669578199, -0.1521609496625161, 0.20136540080403034, 0.04471061572777259];
pub(super) const DOP_C: [f64; 16] = [0.0, 0.05260015195876773, 0.0789002279381516, 0.1183503419072274, 0.2816496580927726, 0.3333333333333333, 0.25, 0.3076923076923077, 0.6512820512820513, 0.6, 0.8571428571428571, 1.0, 1.0, 0.1, 0.2, 0.7777777777777778];
pub(super) const DOP_E3: [f64; 13] = [-0.18980075407240762, 0.0, 0.0, 0.0, 0.0, 4.450312892752409, 1.8915178993145003, -5.801203960010585, -0.4226823213237919, -0.1521609496625161, 0.20136540080403034, 0.02265179219836082, 0.0];
pub(super) const DOP_E5: [f64; 13] = [0.01312004499419488, 0.0, 0.0, 0.0, 0.0, -1.2251564463762044, -0.4957589496572502, 1.6643771824549864, -0.35032884874997366, 0.3341791187130175, 0.08192320648511571, -0.022355307863886294, 0.0];
pub(super) const DOP_D: [[f64; 16]; 4] = [[-8.428938276109013, 0.0, 0.0, 0.0, 0.0, 0.5667149535193777, -3.0689499459498917, 2.38466765651207, 2.117034582445028, -0.871391583777973, 2.2404374302607883, 0.6315787787694688, -0.08899033645133331, 18.148505520854727, -9.194632392478356, -4.436036387594894], [10.427508642579134, 0.0, 0.0, 0.0, 0.0, 242.28349177525817, 165.20045171727028, -374.5467547226902, -22.113666853125306, 7.733432668472264, -30.674084731089398, -9.332130526430229, 15.697238121770845, -31.139403219565178, -9.35292435884448, 35.81684148639408], [19.985053242002433, 0.0, 0.0, 0.0, 0.0, -387.0373087493518, -189.17813819516758, 527.8081592054236, -11.57390253995963, 6.8812326946963, -1.0006050966910838, 0.7777137798053443, -2.778205752353508, -60.19669523126412, 84.32040550667716, 11.99229113618279], [-25.69393346270375, 0.0, 0.0, 0.0, 0.0, -154.18974869023643, -231.5293791760455, 357.6391179106141, 93.40532418362432, -37.45832313645163, 104.0996495089623, 29.8402934266605, -43.53345659001114, 96.32455395918828, -39.17726167561544, -149.72683625798564]];
}
use tables::*;

/// The explicit method of [`solve_ivp_explicit`]. New methods may be added in minor releases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExplicitMethod {
    /// Dormand-Prince 5(4) (SciPy's `RK45`, its default method).
    Rk45,
    /// Hairer's 8(5,3) (SciPy's `DOP853`), the method of choice at tight tolerances.
    Dop853,
}

/// Tolerances and limits for [`solve_ivp_explicit`]. Each step keeps the local error of component
/// `i` below `atolᵢ + rtol · |yᵢ|` in the RMS norm, where `atolᵢ` comes from `atol_components` if it
/// is non-empty (one entry per component) and from `atol` otherwise.
#[derive(Clone, Debug)]
pub struct ExplicitOptions {
    pub rtol: f64,
    pub atol: f64,
    /// Per-component absolute tolerances; empty means `atol` for every component.
    pub atol_components: Vec<f64>,
    /// The first step's size; chosen automatically when `None`.
    pub first_step: Option<f64>,
    /// The largest step size.
    pub max_step: f64,
    /// The largest number of accepted steps.
    pub max_steps: usize,
}

impl Default for ExplicitOptions {
    /// SciPy's defaults: `rtol = 1e-3`, `atol = 1e-6`.
    fn default() -> Self {
        ExplicitOptions { rtol: 1e-3, atol: 1e-6, atol_components: Vec::new(), first_step: None, max_step: f64::INFINITY, max_steps: 1_000_000 }
    }
}

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

struct ExplicitRk<'a, F> {
    f: &'a F,
    method: ExplicitMethod,
    t: f64,
    y: Vec<f64>,
    fy: Vec<f64>,
    t_bound: f64,
    direction: f64,
    h_abs: f64,
    rtol: f64,
    atol: Vec<f64>,
    max_step: f64,
    nfev: usize,
    rejected: usize,
}

impl<F: Fn(f64, &[f64]) -> Vec<f64>> ExplicitRk<'_, F> {
    fn eval(&mut self, t: f64, y: &[f64]) -> Vec<f64> {
        self.nfev += 1;
        (self.f)(t, y)
    }

    fn n_stages(&self) -> usize {
        match self.method {
            ExplicitMethod::Rk45 => 6,
            ExplicitMethod::Dop853 => 12,
        }
    }

    fn a(&self, s: usize, j: usize) -> f64 {
        match self.method {
            ExplicitMethod::Rk45 => RK45_A[s][j],
            ExplicitMethod::Dop853 => DOP_A[s][j],
        }
    }

    fn c(&self, s: usize) -> f64 {
        match self.method {
            ExplicitMethod::Rk45 => RK45_C[s],
            ExplicitMethod::Dop853 => DOP_C[s],
        }
    }

    fn b(&self, s: usize) -> f64 {
        match self.method {
            ExplicitMethod::Rk45 => RK45_B[s],
            ExplicitMethod::Dop853 => DOP_B[s],
        }
    }

    /// `y + h Σⱼ a[s][j] K[j]` over the first `s` stages.
    fn stage_point(&self, k: &[Vec<f64>], s: usize, y: &[f64], h: f64) -> Vec<f64> {
        let mut out = y.to_vec();
        for (j, kj) in k.iter().enumerate().take(s) {
            let a = self.a(s, j) * h;
            if a != 0.0 {
                out.iter_mut().zip(kj).for_each(|(o, v)| *o += a * v);
            }
        }
        out
    }

    // ANCHOR: rk_step
    /// One Runge-Kutta step (SciPy's `rk_step`): the stages `K₀ = f(t, y)`,
    /// `Kₛ = f(t + cₛh, y + h Σⱼ aₛⱼ Kⱼ)`, the new value `y + h Σ bₛ Kₛ`, and `K_last = f(t + h, y_new)`.
    fn rk_step(&mut self, h: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = self.n_stages();
        let mut k = vec![self.fy.clone()];
        for s in 1..n {
            let p = self.stage_point(&k, s, &self.y, h);
            let ks = self.eval(self.t + self.c(s) * h, &p);
            k.push(ks);
        }
        let mut y_new = self.y.clone();
        for (s, ks) in k.iter().enumerate() {
            let b = self.b(s) * h;
            y_new.iter_mut().zip(ks).for_each(|(o, v)| *o += b * v);
        }
        let f_new = self.eval(self.t + h, &y_new);
        k.push(f_new);
        (y_new, k)
    }
    // ANCHOR_END: rk_step

    /// The scaled error norm: RK45's `‖h Σ Eₛ Kₛ / scale‖_RMS`, or DOP853's blend of its
    /// order-5 and order-3 estimates, `|h| ‖e₅‖² / √((‖e₅‖² + 0.01 ‖e₃‖²) n)`.
    fn error_norm(&self, k: &[Vec<f64>], h: f64, scale: &[f64]) -> f64 {
        let combine = |e: &[f64]| -> Vec<f64> { (0..scale.len()).map(|i| k.iter().zip(e).map(|(ks, w)| w * ks[i]).sum::<f64>() / scale[i]).collect() };
        match self.method {
            ExplicitMethod::Rk45 => rms(&combine(&RK45_E).iter().map(|v| v * h).collect::<Vec<_>>()),
            ExplicitMethod::Dop853 => {
                let e5: f64 = combine(&DOP_E5).iter().map(|v| v * v).sum();
                let e3: f64 = combine(&DOP_E3).iter().map(|v| v * v).sum();
                if e5 == 0.0 && e3 == 0.0 {
                    return 0.0;
                }
                h.abs() * e5 / ((e5 + 0.01 * e3) * scale.len() as f64).sqrt()
            }
        }
    }

    fn error_exponent(&self) -> f64 {
        match self.method {
            ExplicitMethod::Rk45 => -1.0 / 5.0,
            ExplicitMethod::Dop853 => -1.0 / 8.0,
        }
    }

    /// The step's polynomial in `x = (t − t_old)/h`, as coefficients of `x¹ … xᵖ`.
    fn dense(&mut self, t_old: f64, y_old: &[f64], f_old_new: (&[f64], &[f64]), mut k: Vec<Vec<f64>>, h: f64) -> Vec<Vec<f64>> {
        let n = y_old.len();
        match self.method {
            ExplicitMethod::Rk45 => {
                // y = y_old + h Kᵀ P [x, x², x³, x⁴].
                (0..4).map(|p| (0..n).map(|i| h * k.iter().enumerate().map(|(s, ks)| RK45_P[s][p] * ks[i]).sum::<f64>()).collect()).collect()
            }
            ExplicitMethod::Dop853 => {
                // Three extra stages, then SciPy's nested form y_old + F₀x + F₁x(1 − x) + …
                for s in 13..16 {
                    let p = self.stage_point(&k, s, y_old, h);
                    let ks = self.eval(t_old + DOP_C[s] * h, &p);
                    k.push(ks);
                }
                let (f_old, f_new) = f_old_new;
                let dy: Vec<f64> = self.y.iter().zip(y_old).map(|(a, b)| a - b).collect();
                let mut f = vec![dy.clone(), (0..n).map(|i| h * f_old[i] - dy[i]).collect(), (0..n).map(|i| 2.0 * dy[i] - h * (f_new[i] + f_old[i])).collect()];
                for d in DOP_D.iter() {
                    f.push((0..n).map(|i| h * k.iter().enumerate().map(|(s, ks)| d[s] * ks[i]).sum::<f64>()).collect());
                }
                // Expand Σ in the nested product into powers of x: iterate F in reverse,
                // adding the row and multiplying by x (even steps) or by 1 − x (odd steps).
                let mut poly: Vec<Vec<f64>> = vec![vec![0.0; n]];
                for (step, row) in f.iter().rev().enumerate() {
                    poly[0].iter_mut().zip(row).for_each(|(a, b)| *a += b);
                    let mut next = vec![vec![0.0; n]; poly.len() + 1];
                    for (d, coef) in poly.iter().enumerate() {
                        for i in 0..n {
                            if step % 2 == 0 {
                                next[d + 1][i] += coef[i];
                            } else {
                                next[d][i] += coef[i];
                                next[d + 1][i] -= coef[i];
                            }
                        }
                    }
                    poly = next;
                }
                poly.remove(0); // the constant term is zero
                poly
            }
        }
    }
}

impl<F: Fn(f64, &[f64]) -> Vec<f64>> Stepper for ExplicitRk<'_, F> {
    fn t(&self) -> f64 {
        self.t
    }
    fn y(&self) -> &[f64] {
        &self.y
    }
    fn counts(&self) -> (usize, usize, usize, usize) {
        (self.nfev, 0, 0, self.rejected)
    }

    // ANCHOR: explicit_step
    /// SciPy's `RungeKutta._step_impl`: try `h`, accept if the error norm is below 1 and grow the
    /// step by `min(10, 0.9 err^{−1/(q+1)})` (not above 1 after a rejection), otherwise shrink it by
    /// `max(0.2, 0.9 err^{−1/(q+1)})` and retry; the last step is cut to end exactly at `t_bound`.
    fn step(&mut self) -> Result<Segment, ErrorsJSL> {
        let t = self.t;
        let min_step = 10.0 * ((t + self.direction * t.abs().max(f64::MIN_POSITIVE) * f64::EPSILON) - t).abs().max(f64::MIN_POSITIVE);
        let mut h_abs = self.h_abs.min(self.max_step).max(min_step);
        let mut rejected = false;
        loop {
            if h_abs < min_step {
                return Err(ErrorsJSL::RuntimeError("solve_ivp_explicit: the step size became too small"));
            }
            let mut t_new = t + self.direction * h_abs;
            if self.direction * (t_new - self.t_bound) > 0.0 {
                t_new = self.t_bound;
            }
            let h = t_new - t;
            h_abs = h.abs();
            let (y_new, k) = self.rk_step(h);
            let scale: Vec<f64> = (0..y_new.len()).map(|i| self.atol[i] + self.y[i].abs().max(y_new[i].abs()) * self.rtol).collect();
            let err = self.error_norm(&k, h, &scale);
            if err < 1.0 {
                let mut factor = if err == 0.0 { MAX_FACTOR } else { MAX_FACTOR.min(SAFETY * err.powf(self.error_exponent())) };
                if rejected {
                    factor = factor.min(1.0);
                }
                let y_old = std::mem::replace(&mut self.y, y_new);
                let f_new = k[k.len() - 1].clone();
                let f_old = std::mem::replace(&mut self.fy, f_new.clone());
                self.t = t_new;
                self.h_abs = h_abs * factor;
                let coeffs = self.dense(t, &y_old, (&f_old, &f_new), k, h);
                return Ok(Segment { t_start: t.min(t_new), t_end: t.max(t_new), t_ref: t, h, y_ref: y_old, coeffs });
            }
            h_abs *= MIN_FACTOR.max(SAFETY * err.powf(self.error_exponent()));
            self.rejected += 1;
            rejected = true;
        }
    }
    // ANCHOR_END: explicit_step
}

// ANCHOR: solve_ivp_explicit
/// Solves `y' = f(t, y)`, `y(t0) = y0` from `t0` to `t1` with an explicit Runge-Kutta method
/// (SciPy's `solve_ivp` with `method='RK45'` or `'DOP853'`). `t1 < t0` integrates backwards.
/// Returns the same [`IvpSolution`] as [`solve_ivp`](super::solve_ivp): every accepted step, a
/// dense output, and the occurrences of each [`Event`] located on the interpolants.
pub fn solve_ivp_explicit<F>(f: F, t_span: (f64, f64), y0: &[f64], method: ExplicitMethod, options: &ExplicitOptions, events: &[Event]) -> Result<IvpSolution, ErrorsJSL>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let (t0, t1) = t_span;
    let n = y0.len();
    if n == 0 || !t0.is_finite() || !t1.is_finite() || t0 == t1 {
        return Err(ErrorsJSL::InvalidInputRange("need a non-empty state and finite t0 ≠ t1"));
    }
    let atol = if options.atol_components.is_empty() { vec![options.atol; n] } else { options.atol_components.clone() };
    if atol.len() != n || atol.iter().any(|a| !(*a >= 0.0)) || !(options.rtol > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("need rtol > 0 and one non-negative atol per component"));
    }
    if !(options.max_step > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("max_step must be positive"));
    }
    let direction = if t1 > t0 { 1.0 } else { -1.0 };
    let mut rk = ExplicitRk { f: &f, method, t: t0, y: y0.to_vec(), fy: Vec::new(), t_bound: t1, direction, h_abs: 0.0, rtol: options.rtol, atol, max_step: options.max_step, nfev: 0, rejected: 0 };
    rk.fy = rk.eval(t0, y0);
    rk.h_abs = match options.first_step {
        Some(h) if h > 0.0 => h,
        Some(_) => return Err(ErrorsJSL::InvalidInputRange("first_step must be positive")),
        None => {
            // SciPy's select_initial_step, with the per-component tolerances.
            let order = match method {
                ExplicitMethod::Rk45 => 4.0,
                ExplicitMethod::Dop853 => 7.0,
            };
            let span = (t1 - t0).abs();
            let scale: Vec<f64> = (0..n).map(|i| rk.atol[i] + y0[i].abs() * rk.rtol).collect();
            let d0 = rms(&y0.iter().zip(&scale).map(|(a, s)| a / s).collect::<Vec<_>>());
            let d1 = rms(&rk.fy.iter().zip(&scale).map(|(a, s)| a / s).collect::<Vec<_>>());
            let h0 = if d0 < 1e-5 || d1 < 1e-5 { 1e-6 } else { 0.01 * d0 / d1 }.min(span);
            let y1: Vec<f64> = (0..n).map(|i| y0[i] + h0 * direction * rk.fy[i]).collect();
            let f1 = rk.eval(t0 + h0 * direction, &y1);
            let d2 = rms(&(0..n).map(|i| (f1[i] - rk.fy[i]) / scale[i]).collect::<Vec<_>>()) / h0;
            let h1 = if d1 <= 1e-15 && d2 <= 1e-15 { (h0 * 1e-3).max(1e-6) } else { (0.01 / d1.max(d2)).powf(1.0 / (order + 1.0)) };
            (100.0 * h0).min(h1).min(span).min(options.max_step)
        }
    };
    let (t, y, status, t_events, y_events, dense) = drive(&mut rk, t0, t1, y0, events, options.max_steps)?;
    let (nfev, _, _, rejected_steps) = rk.counts();
    Ok(IvpSolution { t, y, status, t_events, y_events, dense, nfev, njev: 0, nlu: 0, rejected_steps, switches: Vec::new() })
}
// ANCHOR_END: solve_ivp_explicit

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_methods_reach_their_order_on_the_harmonic_oscillator() {
        let f = |_: f64, y: &[f64]| vec![y[1], -y[0]];
        for method in [ExplicitMethod::Rk45, ExplicitMethod::Dop853] {
            let o = ExplicitOptions { rtol: 1e-10, atol: 1e-12, ..Default::default() };
            let sol = solve_ivp_explicit(f, (0.0, 10.0), &[1.0, 0.0], method, &o, &[]).unwrap();
            let y = sol.y.last().unwrap();
            assert!((y[0] - 10f64.cos()).abs() < 1e-8, "{method:?} {y:?}");
            // The dense output is accurate between steps too.
            let d = sol.dense.evaluate(3.3);
            assert!((d[0] - 3.3f64.cos()).abs() < 1e-8, "{method:?} dense {d:?}");
        }
        // DOP853 needs far fewer evaluations than RK45 at this tolerance.
        let o = ExplicitOptions { rtol: 1e-10, atol: 1e-12, ..Default::default() };
        let a = solve_ivp_explicit(f, (0.0, 10.0), &[1.0, 0.0], ExplicitMethod::Rk45, &o, &[]).unwrap();
        let b = solve_ivp_explicit(f, (0.0, 10.0), &[1.0, 0.0], ExplicitMethod::Dop853, &o, &[]).unwrap();
        assert!(b.nfev * 2 < a.nfev, "{} vs {}", b.nfev, a.nfev);
    }

    #[test]
    fn backward_integration_and_events() {
        // y' = y from t = 2 back to t = 0, starting at e²: y(0) = 1, crossing y = e at t = 1.
        let cross = Event { function: Box::new(|_, y: &[f64]| y[0] - std::f64::consts::E), direction: 0.0, terminal: None };
        let o = ExplicitOptions { rtol: 1e-10, atol: 1e-12, ..Default::default() };
        let sol = solve_ivp_explicit(|_, y: &[f64]| vec![y[0]], (2.0, 0.0), &[2f64.exp()], ExplicitMethod::Dop853, &o, std::slice::from_ref(&cross)).unwrap();
        assert!((sol.y.last().unwrap()[0] - 1.0).abs() < 1e-9);
        assert!((sol.t_events[0][0] - 1.0).abs() < 1e-9, "{:?}", sol.t_events);
        assert!((sol.dense.evaluate(0.5)[0] - 0.5f64.exp()).abs() < 1e-9);
        // Per-component tolerances are accepted.
        let o = ExplicitOptions { atol_components: vec![1e-9, 1e-3], ..Default::default() };
        assert!(solve_ivp_explicit(|_, y: &[f64]| vec![-y[0], -y[1]], (0.0, 1.0), &[1.0, 1.0], ExplicitMethod::Rk45, &o, &[]).is_ok());
    }
}
