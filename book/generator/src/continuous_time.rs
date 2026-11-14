//! Figures and tables for "Continuous-time systems" (the comms chapter): the SciPy comparison,
//! step and Bode responses of second-order systems, discretisation errors and pole placement for
//! an inverted pendulum.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_dsp::continuous::lti::{ContinuousStateSpace, DiscreteMatrices, Discretization, InputHold, place_poles, ss2tf, tf2ss};
use lib_jsl_linalg::{Inverse, SVD};
use ndarray::{Array2, array};
use num_complex::Complex;

const CHAPTER: &str = "comms";
type C = Complex<f64>;

pub fn generate() {
    scipy_reference();
    step_and_bode();
    discretisation_table();
    pendulum();
}

fn max_diff<'a>(a: impl IntoIterator<Item = &'a f64>, b: &[f64]) -> f64 {
    a.into_iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

fn flat(ms: &[&Array2<f64>]) -> Vec<f64> {
    ms.iter().flat_map(|m| m.iter().copied()).collect()
}

fn scipy_reference() {
    let refs = Refs::parse(include_str!("../data/lti_reference.txt"));
    let mat = |name: &str| {
        let m = refs.get(&format!("data {name}"));
        Array2::from_shape_vec((m[0][0] as usize, m[1][0] as usize), m[2].clone()).unwrap()
    };
    let all = |key: &str| refs.get(key).iter().flatten().copied().collect::<Vec<f64>>();
    let (num, den) = (refs.values("data num").to_vec(), refs.values("data den").to_vec());
    let siso = tf2ss(&num, &den).unwrap();
    let sys = ContinuousStateSpace::new(mat("A"), mat("B"), mat("C"), mat("D")).unwrap();
    let t = refs.values("data t").to_vec();
    let u = Array2::from_shape_fn((t.len(), 2), |(i, j)| if j == 0 { (1.3 * t[i]).sin() } else if t[i] < 4.0 { 1.0 } else { -0.5 });
    let x0 = [0.2, -0.1, 0.4];
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row = |what: &str, d: f64| rows.push(vec![what.to_string(), fmt_diff(d)]);
    row("`tf2ss`, third order with a second-order numerator", max_diff(&flat(&[&siso.a, &siso.b, &siso.c, &siso.d]), &all("tf2ss")));
    for i in 0..2 {
        let (n, d) = ss2tf(&sys, i).unwrap();
        let got: Vec<f64> = n.iter().copied().chain(d).collect();
        row(&format!("`ss2tf`, 3 states, input {i}"), max_diff(&got, &all(&format!("ss2tf {i}"))));
    }
    for (hold, label) in [(InputHold::Zero, "zero-order hold"), (InputHold::Linear, "linear interpolation")] {
        let sim = sys.lsim(&u, &t, Some(&x0), hold).unwrap();
        let got = flat(&[&sim.y, &sim.x]);
        row(&format!("`lsim`, 3 states, 2 inputs, 201 steps, {label}"), max_diff(&got, &all(&format!("lsim {}", (hold == InputHold::Linear) as u8))));
    }
    let t2: Vec<f64> = (0..101).map(|i| 0.5 + i as f64 * 0.1).collect();
    let u2 = u.slice(ndarray::s![..101, ..]).to_owned();
    row("`lsim` starting at t = 0.5", max_diff(sys.lsim(&u2, &t2, Some(&x0), InputHold::Linear).unwrap().y.iter(), refs.values("lsim delayed")));
    let ts: Vec<f64> = (0..401).map(|i| i as f64 * 0.05).collect();
    row("`step`", max_diff(siso.step(&ts, 0).unwrap().iter(), refs.values("step")));
    row("`impulse`", max_diff(siso.impulse(&ts, 0).unwrap().iter(), refs.values("impulse")));
    let w = refs.values("data w").to_vec();
    let (mag, phase) = siso.bode(&w, 0, 0).unwrap();
    let want = refs.get("bode");
    row("`bode`, magnitude in dB", max_diff(&mag, &want[0]));
    row("`bode`, phase in degrees", max_diff(&phase, &want[1]));
    for (method, key) in [
        (Discretization::ZeroOrderHold, "zoh"),
        (Discretization::FirstOrderHold, "foh"),
        (Discretization::Bilinear, "bilinear"),
        (Discretization::ForwardEuler, "euler"),
        (Discretization::BackwardEuler, "backward_diff"),
        (Discretization::Gbt(0.3), "gbt"),
    ] {
        let dm = sys.discretize(0.25, method).unwrap();
        row(&format!("`cont2discrete`, {key}"), max_diff(&flat(&[&dm.a, &dm.b, &dm.c, &dm.d]), &all(&format!("c2d {key}"))));
    }
    let strict = ContinuousStateSpace::new(mat("A"), mat("B"), mat("C"), Array2::zeros((2, 2))).unwrap();
    let dm = strict.discretize(0.25, Discretization::Impulse).unwrap();
    row("`cont2discrete`, impulse", max_diff(&flat(&[&dm.a, &dm.b, &dm.c, &dm.d]), &all("c2d impulse")));
    let p = refs.get("data poles");
    let poles: Vec<C> = p[0].iter().zip(&p[1]).map(|(&r, &i)| C::new(r, i)).collect();
    let r = place_poles(&mat("Ap"), &mat("Bp"), &poles).unwrap();
    row("`place_poles`, one input: the gain", max_diff(r.gain.iter(), refs.values("place single")));
    let r2 = place_poles(&mat("Ap"), &mat("Bp2"), &poles).unwrap();
    let err = poles.iter().map(|p| r2.computed_poles.iter().map(|q| (q - p).norm()).fold(f64::INFINITY, f64::min)).fold(0.0, f64::max);
    row("`place_poles`, two inputs: the achieved poles", err);
    write(CHAPTER, "lti_scipy.md", &md_table(&["computation (SciPy reference)", "largest difference"], &rows));
    let cond = |x: &Array2<C>| {
        let (_, s, _) = x.svd(false, false).unwrap();
        s[0] / s[s.len() - 1]
    };
    write(
        CHAPTER,
        "lti_place_facts.md",
        &format!(
            "With two inputs the gain is not unique. After {} sweeps the library's eigenvector matrix has condition number {:.2}, against {:.2} for SciPy's (whose default method, by Tits and Yang, maximises a different measure); the gains differ, but both place the poles to $10^{{-14}}$. With one input, the eigenvector matrix is fixed, with condition number {:.0}.",
            r2.iterations,
            cond(&r2.transfer_matrix),
            refs.values("place multi cond")[0],
            cond(&r.transfer_matrix)
        ),
    );
}

fn second_order(zeta: f64) -> ContinuousStateSpace {
    tf2ss(&[1.0], &[1.0, 2.0 * zeta, 1.0]).unwrap()
}

fn step_and_bode() {
    let t: Vec<f64> = (0..=600).map(|i| i as f64 * 0.03).collect();
    let w: Vec<f64> = (0..=200).map(|i| 10f64.powf(-1.5 + 3.0 * i as f64 / 200.0)).collect();
    let mut step = Plot::new("Step responses of 1 / (s² + 2ζs + 1)", "time t", "y(t)");
    let mut mag = Plot::new("Bode magnitude", "angular frequency ω", "|H(iω)| (dB)");
    let mut phase = Plot::new("Bode phase", "angular frequency ω", "phase (degrees)");
    mag.log_x = true;
    phase.log_x = true;
    mag.legend = Legend::BottomLeft;
    phase.legend = Legend::BottomLeft;
    for (i, zeta) in [0.1, 0.3, 0.7, 1.0].into_iter().enumerate() {
        let sys = second_order(zeta);
        let y = sys.step(&t, 0).unwrap();
        let label = format!("ζ = {zeta}");
        step.series.push(Series::new(&label, t.iter().zip(y.column(0)).map(|(&a, &b)| (a, b)).collect(), Style::Line, i));
        let (m, p) = sys.bode(&w, 0, 0).unwrap();
        mag.series.push(Series::new(&label, w.iter().zip(&m).map(|(&a, &b)| (a, b)).collect(), Style::Line, i));
        phase.series.push(Series::new(&label, w.iter().zip(&p).map(|(&a, &b)| (a, b)).collect(), Style::Line, i));
    }
    write(CHAPTER, "lti_step.svg", &step.render());
    write(CHAPTER, "lti_bode_magnitude.svg", &mag.render());
    write(CHAPTER, "lti_bode_phase.svg", &phase.render());
}

/// `H(e^{iωT}) = C (e^{iωT} I − A)⁻¹ B + D` for a single-input, single-output discrete system.
fn discrete_response(d: &DiscreteMatrices, w: f64) -> C {
    let n = d.a.nrows();
    let z = C::from_polar(1.0, w * d.dt);
    let m = Array2::from_shape_fn((n, n), |(i, j)| if i == j { z } else { C::new(0.0, 0.0) }) - d.a.mapv(|v| C::new(v, 0.0));
    let inv = m.inv().unwrap();
    d.c.mapv(|v| C::new(v, 0.0)).dot(&inv).dot(&d.b.mapv(|v| C::new(v, 0.0)))[(0, 0)] + d.d[(0, 0)]
}

fn discretisation_table() {
    let zeta = 0.3;
    let sys = second_order(zeta);
    let mut rows = Vec::new();
    for dt in [0.1, 0.5] {
        let nyquist = std::f64::consts::PI / dt;
        let error_at = |d: &DiscreteMatrices, ws: &[f64]| -> f64 {
            let hc = sys.freqresp(ws).unwrap();
            ws.iter().zip(&hc).map(|(&w, h)| (discrete_response(d, w) - h[(0, 0)]).norm()).fold(0.0, f64::max)
        };
        let low = [0.1 * nyquist];
        let band: Vec<f64> = (1..=50).map(|i| i as f64 / 50.0 * 0.5 * nyquist).collect();
        for (method, name) in [
            (Discretization::ZeroOrderHold, "zero-order hold"),
            (Discretization::FirstOrderHold, "first-order hold"),
            (Discretization::Bilinear, "bilinear (Tustin)"),
            (Discretization::ForwardEuler, "forward Euler"),
            (Discretization::BackwardEuler, "backward Euler"),
        ] {
            let d = sys.discretize(dt, method).unwrap();
            rows.push(vec![format!("{dt}"), name.to_string(), format!("{:.1e}", error_at(&d, &low)), format!("{:.1e}", error_at(&d, &band))]);
        }
    }
    write(CHAPTER, "lti_discretisation.md", &md_table(&["T", "method", "error at a tenth of the Nyquist frequency", "largest error up to half the Nyquist frequency"], &rows));
}

fn pendulum() {
    // Cart (position, velocity) with a pendulum (angle, angular velocity), linearised upright;
    // the input is the force on the cart.
    let a = array![[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -0.98, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 21.56, 0.0]];
    let b = array![[0.0], [1.0], [0.0], [-2.0]];
    let poles = [C::new(-2.0, 1.0), C::new(-2.0, -1.0), C::new(-3.0, 0.0), C::new(-4.0, 0.0)];
    let r = place_poles(&a, &b, &poles).unwrap();
    let closed = ContinuousStateSpace::new(&a - &b.dot(&r.gain), Array2::zeros((4, 1)), Array2::eye(4), Array2::zeros((4, 1))).unwrap();
    let t: Vec<f64> = (0..=400).map(|i| i as f64 * 0.01).collect();
    let sim = closed.lsim(&Array2::zeros((t.len(), 1)), &t, Some(&[0.0, 0.0, 0.2, 0.0]), InputHold::Zero).unwrap();
    let force: Vec<f64> = (0..t.len()).map(|i| -r.gain.row(0).dot(&sim.x.row(i))).collect();
    let mut plot = Plot::new("Balancing an inverted pendulum by pole placement", "time (s)", "state, force / 10");
    plot.legend = Legend::TopRight;
    plot.series.push(Series::new("pendulum angle (rad)", t.iter().zip(sim.x.column(2)).map(|(&a, &b)| (a, b)).collect(), Style::Line, 0));
    plot.series.push(Series::new("cart position (m)", t.iter().zip(sim.x.column(0)).map(|(&a, &b)| (a, b)).collect(), Style::Line, 1));
    plot.series.push(Series::new("force / 10 (N)", t.iter().zip(&force).map(|(&a, &b)| (a, b / 10.0)).collect(), Style::Dashed, 2));
    write(CHAPTER, "lti_pendulum.svg", &plot.render());
    write(CHAPTER, "lti_pendulum_gain.md", &format!("The gain is $K = ({:.2}, {:.2}, {:.2}, {:.2})$.", r.gain[(0, 0)], r.gain[(0, 1)], r.gain[(0, 2)], r.gain[(0, 3)]));
}
