//! Figures and tables for "Radau, LSODA and event location" (the integration chapter): accuracy
//! and cost against SciPy on stiff and non-stiff problems, event times, and the step sizes and
//! method switches on the Van der Pol oscillator.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_integration::ivp::{Event, IvpMethod, IvpOptions, IvpSolution, solve_ivp};

const CHAPTER: &str = "integration";

type Rhs = Box<dyn Fn(f64, &[f64]) -> Vec<f64>>;
type Jac = Box<dyn Fn(f64, &[f64]) -> Vec<Vec<f64>>>;

pub fn generate() {
    scipy_reference();
    vdp_figures();
}

fn vdp(mu: f64) -> (Rhs, Jac) {
    (
        Box::new(move |_, y: &[f64]| vec![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]]),
        Box::new(move |_, y: &[f64]| vec![vec![0.0, 1.0], vec![-2.0 * mu * y[0] * y[1] - 1.0, mu * (1.0 - y[0] * y[0])]]),
    )
}

fn robertson() -> (Rhs, Jac) {
    (
        Box::new(|_, y: &[f64]| vec![-0.04 * y[0] + 1e4 * y[1] * y[2], 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1], 3e7 * y[1] * y[1]]),
        Box::new(|_, y: &[f64]| vec![vec![-0.04, 1e4 * y[2], 1e4 * y[1]], vec![0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]], vec![0.0, 6e7 * y[1], 0.0]]),
    )
}

fn arenstorf() -> Rhs {
    let m = 0.012277471;
    let mp = 1.0 - m;
    Box::new(move |_, y: &[f64]| {
        let d1 = ((y[0] + m).powi(2) + y[1] * y[1]).powf(1.5);
        let d2 = ((y[0] - mp).powi(2) + y[1] * y[1]).powf(1.5);
        vec![y[2], y[3], y[0] + 2.0 * y[3] - mp * (y[0] + m) / d1 - m * (y[0] - mp) / d2, y[1] - 2.0 * y[2] - mp * y[1] / d1 - m * y[1] / d2]
    })
}

const ARENSTORF_T: f64 = 17.065_216_560_157_96;
const ARENSTORF_Y0: [f64; 4] = [0.994, 0.0, 0.0, -2.001_585_106_379_082_5];

fn solve(name: &str, method: IvpMethod, rtol: f64, events: &[Event]) -> IvpSolution {
    let o = IvpOptions { rtol, atol: rtol * 1e-3, ..Default::default() };
    match name {
        "vdp" => {
            let (f, j) = vdp(100.0);
            solve_ivp(f, Some(j), (0.0, 200.0), &[2.0, 0.0], method, o, events).unwrap()
        }
        "robertson" => {
            let (f, j) = robertson();
            solve_ivp(f, Some(j), (0.0, 1e5), &[1.0, 0.0, 0.0], method, o, events).unwrap()
        }
        _ => solve_ivp(arenstorf(), None::<Jac>, (0.0, ARENSTORF_T), &ARENSTORF_Y0, method, o, events).unwrap(),
    }
}

fn rel_err(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0.0f64, |m, (a, b)| m.max((a - b).abs())) / scale
}

fn scipy_reference() {
    let refs = Refs::parse(include_str!("../data/ivp_reference.txt"));
    let mut rows = Vec::new();
    for (name, label) in [("vdp", "Van der Pol, μ = 100 (stiff)"), ("robertson", "Robertson kinetics (stiff)"), ("arenstorf", "Arenstorf orbit (non-stiff)")] {
        let exact = refs.values(&format!("exact {name}")).to_vec();
        for (method, key) in [(IvpMethod::Radau, "Radau"), (IvpMethod::Lsoda, "LSODA")] {
            for rtol in [1e-4, 1e-7] {
                let s = solve(name, method, rtol, &[]);
                let rkey = if rtol == 1e-4 { "0.0001" } else { "1e-07" };
                let r = refs.get(&format!("{key} {name} {rkey}"));
                let (steps, nfev) = (r[0][0], r[0][1]);
                rows.push(vec![
                    label.to_string(),
                    key.to_string(),
                    format!("{rtol:.0e}"),
                    format!("{} / {}", s.t.len() - 1, steps),
                    format!("{} / {}", s.nfev, nfev),
                    format!("{} / {}", fmt_diff(rel_err(s.y.last().unwrap(), &exact)), fmt_diff(rel_err(&r[1], &exact))),
                    if method == IvpMethod::Lsoda { format!("{}", s.switches.len()) } else { String::new() },
                ]);
            }
        }
    }
    write(CHAPTER, "ivp_scipy.md", &md_table(&["problem", "method", "rtol", "steps (library / SciPy)", "f evaluations (library / SciPy)", "final error (library / SciPy)", "method switches"], &rows));

    // Events.
    let mut rows = Vec::new();
    let exact = refs.values("events arenstorf exact").to_vec();
    for (method, key) in [(IvpMethod::Radau, "Radau"), (IvpMethod::Lsoda, "LSODA")] {
        let ev = Event::new(|_, y: &[f64]| y[1]);
        let s = solve("arenstorf", method, 1e-7, std::slice::from_ref(&ev));
        let got = &s.t_events[0];
        let scipy = refs.values(&format!("events arenstorf {key}"));
        // Compare the crossings both lists share; a drifting orbit can add a spurious last one.
        let err = |v: &[f64]| fmt_diff(v.iter().zip(&exact).fold(0.0f64, |m, (a, b)| m.max((a - b).abs())));
        rows.push(vec![format!("Arenstorf orbit: every crossing of y = 0 ({} in the reference)", exact.len()), key.to_string(), format!("{} (SciPy {})", got.len(), scipy.len()), err(got), err(scipy)]);
    }
    let exact = refs.values("events vdp exact")[0];
    for (method, key) in [(IvpMethod::Radau, "Radau"), (IvpMethod::Lsoda, "LSODA")] {
        let ev = Event { function: Box::new(|_, y: &[f64]| y[0] + 1.5), direction: -1.0, terminal: Some(1) };
        let s = solve("vdp", method, 1e-7, std::slice::from_ref(&ev));
        let scipy = refs.values(&format!("events vdp {key}"))[0];
        rows.push(vec!["Van der Pol: first time x falls through −1.5 (terminal)".into(), key.to_string(), "1".into(), fmt_diff((s.t_events[0][0] - exact).abs()), fmt_diff((scipy - exact).abs())]);
    }
    write(CHAPTER, "ivp_events.md", &md_table(&["event", "method", "events found", "largest time error (library)", "largest time error (SciPy)"], &rows));
}

fn vdp_figures() {
    let mut sol_plot = Plot::new("Van der Pol oscillator, μ = 100", "t", "x(t)");
    sol_plot.legend = Legend::BottomRight;
    let mut h_plot = Plot::new("Step sizes on the Van der Pol oscillator (rtol = 10⁻⁶)", "t", "step size");
    h_plot.log_y = true;
    h_plot.legend = Legend::BottomRight;
    let o = IvpOptions { rtol: 1e-6, atol: 1e-9, ..Default::default() };
    let (f, j) = vdp(100.0);
    let radau = solve_ivp(&f, Some(&j), (0.0, 200.0), &[2.0, 0.0], IvpMethod::Radau, o, &[]).unwrap();
    let lsoda = solve_ivp(&f, Some(&j), (0.0, 200.0), &[2.0, 0.0], IvpMethod::Lsoda, o, &[]).unwrap();
    let fine: Vec<f64> = (0..=2000).map(|i| i as f64 * 0.1).collect();
    sol_plot.series.push(Series::new("x(t), from the dense output", fine.iter().map(|&t| (t, radau.dense.evaluate(t)[0])).collect(), Style::Line, 0));
    sol_plot.series.push(Series::new("accepted Radau steps", radau.t.iter().zip(&radau.y).map(|(&t, y)| (t, y[0])).collect(), Style::Markers, 1));
    write(CHAPTER, "ivp_vdp.svg", &sol_plot.render());
    let steps = |s: &IvpSolution| s.t.windows(2).map(|w| (w[1], w[1] - w[0])).collect::<Vec<_>>();
    h_plot.series.push(Series::new("Radau", steps(&radau), Style::Line, 0));
    // LSODA: Adams steps and BDF steps in different colours.
    let mut stiff = false;
    let mut switches = lsoda.switches.iter().peekable();
    let mut adams: Vec<(f64, f64)> = Vec::new();
    let mut bdf: Vec<(f64, f64)> = Vec::new();
    for (t, h) in steps(&lsoda) {
        while let Some(&&(ts, to_bdf)) = switches.peek() {
            if ts < t {
                stiff = to_bdf;
                switches.next();
            } else {
                break;
            }
        }
        if stiff { bdf.push((t, h)) } else { adams.push((t, h)) }
    }
    h_plot.series.push(Series::new("LSODA, Adams steps", adams, Style::Markers, 1));
    h_plot.series.push(Series::new("LSODA, BDF steps", bdf, Style::Markers, 2));
    write(CHAPTER, "ivp_vdp_steps.svg", &h_plot.render());
    write(
        CHAPTER,
        "ivp_vdp_facts.md",
        &format!(
            "Over $0 \\le t \\le 200$ at $\\mathrm{{rtol}} = 10^{{-6}}$, Radau takes {} steps with {} function evaluations and {} LU factorisations; LSODA takes {} steps with {} evaluations, switching method {} times.",
            radau.t.len() - 1,
            radau.nfev,
            radau.nlu,
            lsoda.t.len() - 1,
            lsoda.nfev,
            lsoda.switches.len()
        ),
    );
}
