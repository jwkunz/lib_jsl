//! Figures and tables for "Matrix functions and matrix equations" (the linear algebra chapter):
//! the SciPy comparison, the Taylor series against scaling and squaring, an LQR design and the
//! steady state of a Kalman filter.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_linalg::{EigVals, Inverse, expm, solve_continuous_are, solve_continuous_lyapunov, solve_discrete_are, solve_discrete_lyapunov, solve_sylvester};
use ndarray::{Array2, array, s};
use num_complex::Complex;

const CHAPTER: &str = "linear_algebra";
type C = Complex<f64>;

pub fn generate() {
    scipy_reference();
    taylor_table();
    lqr_figure();
    kalman_steady_state();
}

fn max_abs<T: Copy + Into<C>>(a: &Array2<T>) -> f64 {
    a.iter().fold(0.0, |m, &v| m.max(Into::<C>::into(v).norm()))
}

fn rel_diff<T: Copy + Into<C>>(got: &Array2<T>, want: &Array2<T>) -> f64 {
    got.iter().zip(want.iter()).fold(0.0f64, |m, (&a, &b)| m.max((Into::<C>::into(a) - Into::<C>::into(b)).norm())) / max_abs(want)
}

fn h(a: &Array2<C>) -> Array2<C> {
    a.t().mapv(|v| v.conj())
}

fn scipy_reference() {
    let refs = Refs::parse(include_str!("../data/matrix_equations_reference.txt"));
    let real = |name: &str| {
        let m = refs.get(&format!("matrix {name}"));
        Array2::from_shape_vec((m[0][0] as usize, m[1][0] as usize), m[2].clone()).unwrap()
    };
    let cplx = |name: &str| {
        let m = refs.get(&format!("matrix {name}"));
        Array2::from_shape_vec((m[0][0] as usize, m[1][0] as usize), m[2].iter().zip(&m[3]).map(|(&r, &i)| C::new(r, i)).collect()).unwrap()
    };
    let result = |key: &str, shape: (usize, usize)| Array2::from_shape_vec(shape, refs.values(key).to_vec()).unwrap();
    let cresult = |key: &str, shape: (usize, usize)| {
        let v = refs.get(key);
        Array2::from_shape_vec(shape, v[0].iter().zip(&v[1]).map(|(&r, &i)| C::new(r, i)).collect()).unwrap()
    };
    let (a, big, ac, b, sm, f, q, r) = (real("A"), real("Big"), cplx("Ac"), real("B"), real("S"), real("F"), real("Q"), real("R"));
    let (stable, contr, qh) = (real("Stable"), real("Contr"), cplx("Qh"));
    let n = a.nrows();
    let tc = |m: &Array2<f64>| m.mapv(|v| C::new(v, 0.0));
    let eye = Array2::<C>::eye(n);

    // Residuals relative to the largest entry of the solution, for the library's X and SciPy's.
    let lyap_c = |a: &Array2<C>, q: &Array2<C>, x: &Array2<C>| max_abs(&(a.dot(x) + x.dot(&h(a)) - q)) / max_abs(x);
    let lyap_d = |a: &Array2<C>, q: &Array2<C>, x: &Array2<C>| max_abs(&(a.dot(x).dot(&h(a)) - x + q)) / max_abs(x);
    let care_res = |a: &Array2<C>, b: &Array2<C>, q: &Array2<C>, r: &Array2<C>, x: &Array2<C>| {
        let g = b.dot(&r.inv().unwrap()).dot(&h(b));
        max_abs(&(h(a).dot(x) + x.dot(a) - x.dot(&g).dot(x) + q)) / max_abs(x)
    };
    let dare_res = |a: &Array2<C>, b: &Array2<C>, q: &Array2<C>, r: &Array2<C>, x: &Array2<C>| {
        let k = (r + &h(b).dot(x).dot(b)).inv().unwrap().dot(&h(b)).dot(x).dot(a);
        max_abs(&(h(a).dot(x).dot(a) - x - h(a).dot(x).dot(b).dot(&k) + q)) / max_abs(x)
    };

    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row = |what: &str, diff: f64, res: Option<(f64, f64)>| {
        let (ours, theirs) = res.map_or((String::new(), String::new()), |(o, t)| (fmt_diff(o), fmt_diff(t)));
        rows.push(vec![what.to_string(), fmt_diff(diff), ours, theirs]);
    };
    row("`expm`, 20 × 20, ‖A‖₁ ≈ 4.6 (`linalg.expm`)", rel_diff(&expm(&a).unwrap(), &result("expm A", (n, n))), None);
    row("`expm`, 20 × 20, ‖A‖₁ ≈ 54, entries up to 4 × 10⁵", rel_diff(&expm(&big).unwrap(), &result("expm Big", (n, n))), None);
    row("`expm`, complex 20 × 20", rel_diff(&expm(&ac).unwrap(), &cresult("expm Ac", (n, n))), None);
    let x = solve_sylvester(&a, &sm, &f).unwrap();
    let want = result("sylvester", (n, 15));
    let syl = |x: &Array2<f64>| max_abs(&(a.dot(x) + x.dot(&sm) - &f)) / max_abs(x);
    row("`solve_sylvester`, 20 × 20 and 15 × 15", rel_diff(&x, &want), Some((syl(&x), syl(&want))));
    let x = solve_continuous_lyapunov(&stable, &q).unwrap();
    let want = result("continuous lyapunov", (n, n));
    row("`solve_continuous_lyapunov`, 20 × 20", rel_diff(&x, &want), Some((lyap_c(&tc(&stable), &tc(&q), &tc(&x)), lyap_c(&tc(&stable), &tc(&q), &tc(&want)))));
    let shifted = &ac - &(eye.clone() * C::new(1.5, 0.0));
    let x = solve_continuous_lyapunov(&shifted, &qh).unwrap();
    let want = cresult("continuous lyapunov complex", (n, n));
    row("`solve_continuous_lyapunov`, complex", rel_diff(&x, &want), Some((lyap_c(&shifted, &qh, &x), lyap_c(&shifted, &qh, &want))));
    let x = solve_discrete_lyapunov(&contr, &q).unwrap();
    let want = result("discrete lyapunov", (n, n));
    row("`solve_discrete_lyapunov`, 20 × 20, spectral radius 0.8", rel_diff(&x, &want), Some((lyap_d(&tc(&contr), &tc(&q), &tc(&x)), lyap_d(&tc(&contr), &tc(&q), &tc(&want)))));
    let x = solve_continuous_are(&a, &b, &q, &r).unwrap();
    let want = result("care", (n, n));
    row("`solve_continuous_are`, 20 states, 4 inputs", rel_diff(&x, &want), Some((care_res(&tc(&a), &tc(&b), &tc(&q), &tc(&r), &tc(&x)), care_res(&tc(&a), &tc(&b), &tc(&q), &tc(&r), &tc(&want)))));
    let x = solve_continuous_are(&ac, &tc(&b), &qh, &tc(&r)).unwrap();
    let want = cresult("care complex", (n, n));
    row("`solve_continuous_are`, complex", rel_diff(&x, &want), Some((care_res(&ac, &tc(&b), &qh, &tc(&r), &x), care_res(&ac, &tc(&b), &qh, &tc(&r), &want))));
    let x = solve_discrete_are(&a, &b, &q, &r).unwrap();
    let want = result("dare", (n, n));
    row("`solve_discrete_are`, 20 states, 4 inputs", rel_diff(&x, &want), Some((dare_res(&tc(&a), &tc(&b), &tc(&q), &tc(&r), &tc(&x)), dare_res(&tc(&a), &tc(&b), &tc(&q), &tc(&r), &tc(&want)))));
    write(CHAPTER, "matrix_equations_scipy.md", &md_table(&["computation (SciPy reference)", "difference", "residual, library", "residual, SciPy"], &rows));
}

/// The Taylor series summed until its terms stop changing the sum, against `expm`, on Moler and
/// Van Loan's example `A = [[−49, 24], [−64, 31]]` with eigenvalues −1 and −17.
fn taylor_table() {
    let a = array![[-49.0, 24.0], [-64.0, 31.0]];
    // A = V diag(−1, −17) V⁻¹ with V = [[1, 3], [2, 4]].
    let v = array![[1.0, 3.0], [2.0, 4.0]];
    let vinv = v.inv().unwrap();
    let mut rows = Vec::new();
    for t in [0.1f64, 0.5, 1.0, 2.0] {
        let at = &a * t;
        let exact = v.dot(&Array2::from_diag(&array![(-t).exp(), (-17.0 * t).exp()])).dot(&vinv);
        let mut sum = Array2::<f64>::eye(2);
        let mut term = Array2::<f64>::eye(2);
        let mut largest: f64 = 1.0;
        for k in 1..400 {
            term = term.dot(&at) / k as f64;
            largest = largest.max(term.iter().fold(0.0, |m, x: &f64| m.max(x.abs())));
            let next = &sum + &term;
            if next == sum {
                break;
            }
            sum = next;
        }
        let err = |m: &Array2<f64>| m.iter().zip(exact.iter()).fold(0.0f64, |e, (x, y)| e.max((x - y).abs())) / exact.iter().fold(0.0f64, |e, y: &f64| e.max(y.abs()));
        rows.push(vec![format!("{t}"), format!("{largest:.1e}"), fmt_diff(err(&sum)), fmt_diff(err(&expm(&at).unwrap()))]);
    }
    write(CHAPTER, "expm_taylor.md", &md_table(&["t", "largest Taylor term", "error, Taylor series", "error, `expm`"], &rows));
}

/// Discretises a continuous system with a zero-order hold: `[[A, B], [0, 0]] T` exponentiated
/// holds `e^{AT}` and `∫₀ᵀ e^{As} ds B` in its top blocks.
fn discretise(a: &Array2<f64>, b: &Array2<f64>, dt: f64) -> (Array2<f64>, Array2<f64>) {
    let (n, m) = (a.nrows(), b.ncols());
    let mut big = Array2::<f64>::zeros((n + m, n + m));
    big.slice_mut(s![..n, ..n]).assign(&(a * dt));
    big.slice_mut(s![..n, n..]).assign(&(b * dt));
    let e = expm(&big).unwrap();
    (e.slice(s![..n, ..n]).to_owned(), e.slice(s![..n, n..]).to_owned())
}

fn lqr_figure() {
    // A cart on a frictionless track: position and velocity, force as input.
    let a = array![[0.0, 1.0], [0.0, 0.0]];
    let b = array![[0.0], [1.0]];
    let dt = 0.1;
    let (ad, bd) = discretise(&a, &b, dt);
    let q = array![[1.0, 0.0], [0.0, 0.0]];
    let mut plot = Plot::new("LQR: a cart returned to the origin", "time (s)", "position");
    plot.legend = Legend::TopRight;
    let mut facts = Vec::new();
    for (i, rho) in [0.01, 0.1, 1.0].into_iter().enumerate() {
        let r = array![[rho]];
        let x = solve_discrete_are(&ad, &bd, &q, &r).unwrap();
        let k = (&r + &bd.t().dot(&x).dot(&bd)).inv().unwrap().dot(&bd.t()).dot(&x).dot(&ad);
        let closed = &ad - &bd.dot(&k);
        let radius = closed.eigvals().unwrap().iter().fold(0.0f64, |m, l| m.max(l.norm()));
        let mut state = array![1.0, 0.0];
        let mut pts = vec![(0.0, 1.0)];
        let mut peak_force: f64 = 0.0;
        for step in 1..=80 {
            let u = -k.dot(&state);
            peak_force = peak_force.max(u[0].abs());
            state = ad.dot(&state) + bd.dot(&u);
            pts.push((step as f64 * dt, state[0]));
        }
        plot.series.push(Series::new(&format!("R = {rho}"), pts, Style::Line, i));
        facts.push(vec![format!("{rho}"), format!("[{:.2}, {:.2}]", k[(0, 0)], k[(0, 1)]), format!("{radius:.3}"), format!("{peak_force:.2}")]);
    }
    write(CHAPTER, "lqr_cart.svg", &plot.render());
    write(CHAPTER, "lqr_cart.md", &md_table(&["R", "gain K", "spectral radius of A − BK", "largest force"], &facts));
}

fn kalman_steady_state() {
    use lib_jsl_dsp::discrete::controls::kalman::KalmanFilter;
    use lib_jsl_dsp::discrete::stream_operator::StreamOperator;
    let dt: f64 = 0.1;
    let f = array![[1.0, dt], [0.0, 1.0]];
    let hm = array![[1.0, 0.0]];
    let qn = array![[dt.powi(3) / 3.0, dt * dt / 2.0], [dt * dt / 2.0, dt]];
    let rn = array![[0.25]];
    // The predicted covariance solves the DARE of the dual system (Fᵀ, Hᵀ).
    let p = solve_discrete_are(&f.t().to_owned(), &hm.t().to_owned(), &qn, &rn).unwrap();
    let s = hm.dot(&p).dot(&hm.t()) + &rn;
    let posterior = &p - &p.dot(&hm.t()).dot(&s.inv().unwrap()).dot(&hm).dot(&p);
    let gain = p.dot(&hm.t()).dot(&s.inv().unwrap());
    let c = |m: &Array2<f64>| m.mapv(|v| C::new(v, 0.0));
    let mut kf = KalmanFilter::new(c(&f), c(&hm), c(&qn), c(&rn), c(&(Array2::<f64>::eye(2) * 100.0)), ndarray::Array1::zeros(2), None, None).unwrap();
    let mut steps_needed = 0;
    for k in 1..=500 {
        kf.process(&[ndarray::Array1::from(vec![C::new(0.0, 0.0)])]).unwrap();
        let diff = kf.estimate_covariance().iter().zip(posterior.iter()).fold(0.0, |m: f64, (a, b)| m.max((a.re - b).abs())) / posterior.iter().fold(0.0, |m: f64, b| m.max(b.abs()));
        if diff < 1e-12 {
            steps_needed = k;
            break;
        }
    }
    write(
        CHAPTER,
        "kalman_dare.md",
        &format!(
            "For the constant-velocity tracker ($T = 0.1$, $R = 0.25$), the steady-state gain from `solve_discrete_are` is $K = ({:.4}, {:.4})$, and the covariance of the running `KalmanFilter`, started from $P_0 = 100I$, agrees with it to $10^{{-12}}$ after {steps_needed} measurements.\n",
            gain[(0, 0)],
            gain[(1, 0)]
        ),
    );
}
