//! Tables and figures for release 2.15.0: `whiten`, `inconsistent`, optimal leaf ordering and
//! leaf order (unsupervised chapter), the `ZerosPolesGain` system type (continuous-time systems)
//! and the CODATA table (foundations), compared with SciPy.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_core::codata;
use lib_jsl_core::constants::{em, physics};
use lib_jsl_dsp::continuous::lti::Discretization;
use lib_jsl_dsp::continuous::zpk::ZerosPolesGain;
use lib_jsl_machine_learning::unsupervised::clustering::{Dendrogram, Linkage, agglomerate, whiten};
use ndarray::Array2;
use num_complex::Complex;

type C = Complex<f64>;

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/sys215_reference.txt"));
    cluster_table(&refs);
    zpk_table(&refs);
    codata_table(&refs);
    figures(&refs);
}

fn max_diff(a: impl IntoIterator<Item = f64>, b: &[f64]) -> f64 {
    a.into_iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn rel_diff(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
    max_diff(a.iter().copied(), b) / scale
}

fn distances(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    Array2::from_shape_fn((n, n), |(i, j)| x.row(i).iter().zip(x.row(j)).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt())
}

fn cluster_table(refs: &Refs) {
    let pts = refs.values("points");
    let x = Array2::from_shape_vec((pts.len() / 2, 2), pts.to_vec()).unwrap();
    let d = distances(&x);
    let mut rows = Vec::new();
    rows.push(vec!["`whiten`, 36 × 2 (features on scales 1 and 10)".into(), fmt_diff(max_diff(whiten(x.view()).iter().copied(), refs.values("whiten"))), "—".into(), "—".into(), "—".into()]);
    for (name, lk) in [("single", Linkage::Single), ("complete", Linkage::Complete), ("average", Linkage::Average), ("ward", Linkage::Ward)] {
        let zv = refs.values(&format!("{name} z"));
        let z = Array2::from_shape_vec((zv.len() / 4, 4), zv.to_vec()).unwrap();
        let sci = Dendrogram::from_linkage_matrix(z.view()).unwrap();
        // The library's own tree: the same heights and the same cophenetic matrix.
        let own = agglomerate(d.view(), lk).unwrap();
        let tree = max_diff(own.cophenetic_distances().iter().copied(), sci.cophenetic_distances().as_slice().unwrap()).max(max_diff(own.merges.iter().map(|m| m.distance), &z.column(2).to_vec()));
        let mut inc = 0.0f64;
        for depth in 1..=4 {
            let r = sci.inconsistent(depth).unwrap();
            let want = refs.values(&format!("{name} inconsistent {depth}"));
            inc = inc.max(max_diff(r.iter().copied(), want));
            // From the library's own tree too (the coefficient does not depend on child order).
            let r2 = own.inconsistent(depth).unwrap();
            inc = inc.max(max_diff(r2.iter().copied(), want));
        }
        let leaves: Vec<f64> = sci.leaves().iter().map(|&v| v as f64).collect();
        let same_leaves = leaves == refs.values(&format!("{name} leaves"));
        let olo_want = refs.get(&format!("{name} olo"));
        let olo = sci.optimal_leaf_ordering(d.view()).unwrap();
        let ol: Vec<f64> = olo.leaves().iter().map(|&v| v as f64).collect();
        // The library's own tree gives the same optimum (the tree is the same up to child order).
        assert!((own.optimal_leaf_ordering(d.view()).unwrap().leaf_path_length(d.view()) - olo.leaf_path_length(d.view())).abs() < 1e-9);
        rows.push(vec![
            format!("{name} linkage, 36 points"),
            fmt_diff(tree),
            fmt_diff(inc),
            if same_leaves { "identical".into() } else { "differ".into() },
            format!("{:.4} / {:.4}{}", olo.leaf_path_length(d.view()), olo_want[1][0], if ol == olo_want[0] { " (same order)" } else { "" }),
        ]);
    }
    write(
        "unsupervised",
        "sys215_cluster.md",
        &md_table(&["Case", "Tree (heights, cophenetic distances)", "`inconsistent`, depths 1–4", "`leaves_list`", "`optimal_leaf_ordering`: leaf path length, library / SciPy"], &rows),
    );
}

fn complexes(re: &[f64], im: &[f64]) -> Vec<C> {
    re.iter().zip(im).map(|(a, b)| C::new(*a, *b)).collect()
}

/// Largest distance from each of `got` to its nearest in `want` (orders differ).
fn matched(got: &[C], want: &[C]) -> f64 {
    if got.len() != want.len() {
        return f64::INFINITY;
    }
    got.iter().map(|g| want.iter().map(|w| (g - w).norm()).fold(f64::INFINITY, f64::min)).fold(0.0, f64::max)
}

fn zpk_table(refs: &Refs) {
    let w = refs.values("w");
    let t = refs.values("t");
    let mut rows = Vec::new();
    for (name, label) in [("butter", "Butterworth, order 6, low-pass"), ("cheby", "Chebyshev I, order 5, high-pass")] {
        let s = refs.get(&format!("{name} zpk"));
        let sys = ZerosPolesGain::new(complexes(&s[0], &s[1]), complexes(&s[2], &s[3]), s[4][0]);
        let f = refs.get(&format!("{name} freqresp"));
        let h = sys.freqresp(w);
        let hr: Vec<f64> = h.iter().map(|v| v.re).collect();
        let hi: Vec<f64> = h.iter().map(|v| v.im).collect();
        let scale = f[0].iter().zip(&f[1]).map(|(a, b)| a.hypot(*b)).fold(0.0f64, f64::max);
        let dh = max_diff(hr, &f[0]).max(max_diff(hi, &f[1])) / scale;
        let (mag, ph) = sys.bode(w);
        let st = refs.get(&format!("{name} step"));
        let ys = sys.step(t).unwrap();
        let yi = sys.impulse(t).unwrap();
        rows.push(vec![
            format!("{label}: `freqresp`; `bode` (dB, degrees); `step`, `impulse`"),
            format!("{}; {} / {}; {} / {}", fmt_diff(dh), fmt_diff(max_diff(mag, &f[2])), fmt_diff(max_diff(ph, &f[3])), fmt_diff(rel_diff(&ys, &st[0])), fmt_diff(rel_diff(&yi, &st[1]))),
        ]);
        for (m, method) in [("zoh", Discretization::ZeroOrderHold), ("bilinear", Discretization::Bilinear), ("foh", Discretization::FirstOrderHold)] {
            let r = refs.get(&format!("{name} {m}"));
            let dsys = sys.to_discrete(0.1, method).unwrap();
            let dp = matched(&dsys.poles, &complexes(&r[0], &r[1]));
            let dz = matched(&dsys.zeros, &complexes(&r[2], &r[3]));
            let dk = (dsys.gain - r[4][0]).abs() / r[4][0].abs();
            let ystep = dsys.dstep(60).unwrap();
            let yimp = dsys.dimpulse(60).unwrap();
            let wd: Vec<f64> = (0..30).map(|i| std::f64::consts::PI * i as f64 / 29.0).collect();
            let hd = dsys.freqresp(&wd);
            let sc = r[7].iter().zip(&r[8]).map(|(a, b)| a.hypot(*b)).fold(0.0f64, f64::max);
            let dhd = max_diff(hd.iter().map(|v| v.re), &r[7]).max(max_diff(hd.iter().map(|v| v.im), &r[8])) / sc;
            let mut cell = format!("{}; {}; {}; {} / {}; {}", fmt_diff(dp), fmt_diff(dz), fmt_diff(dk), fmt_diff(rel_diff(&ystep, &r[5])), fmt_diff(rel_diff(&yimp, &r[6])), fmt_diff(dhd));
            if m == "bilinear" {
                // The bilinear transform maps the zeros at s = ∞ (low-pass) or s = 0 (high-pass)
                // exactly to z = −1 or z = +1.
                let exact = if name == "butter" { -1.0 } else { 1.0 };
                let dist = |z: &[C]| z.iter().map(|z| (z - exact).norm()).fold(0.0f64, f64::max);
                cell += &format!(" (zeros exactly at {exact}: library off by {}, SciPy by {})", fmt_diff(dist(&dsys.zeros)), fmt_diff(dist(&complexes(&r[2], &r[3]))));
            }
            rows.push(vec![format!("… `to_discrete(0.1, {m})`: poles; zeros; gain; `dstep`, `dimpulse`; `dfreqresp`"), cell]);
        }
    }
    write("comms", "sys215_zpk.md", &md_table(&["System and operation", "Difference from SciPy (relative where scaled)"], &rows));
}

fn codata_table(refs: &Refs) {
    let c = refs.get("codata");
    let all: Vec<codata::PhysicalConstant> = codata::all().collect();
    let same = all.len() == c[0].len() && all.iter().zip(&c[0]).zip(&c[1]).all(|((k, v), u)| k.value == *v && k.uncertainty == *u);
    let mut rows = vec![vec![
        format!("`physical_constant`: values and uncertainties of all {} entries", all.len()),
        if same { "identical".into() } else { "differ".into() },
    ]];
    let counts: Vec<f64> = ["", "radius", "magn", "electron", "Planck"].iter().map(|s| codata::find(s).len() as f64).collect();
    rows.push(vec![
        "`find`: \"\", \"radius\", \"magn\", \"electron\", \"Planck\" (number of names)".into(),
        format!("{} ({})", if counts == refs.values("find") { "identical" } else { "differ" }, counts.iter().map(|v| format!("{v}")).collect::<Vec<_>>().join(", ")),
    ]);
    // The hand-written constants of `constants::physics` and `constants::em` against the table.
    let pairs: &[(&str, f64)] = &[
        ("speed of light in vacuum", physics::C),
        ("Planck constant", physics::H),
        ("reduced Planck constant", physics::HBAR),
        ("elementary charge", physics::E_CHARGE),
        ("Boltzmann constant", physics::K_B),
        ("Avogadro constant", physics::N_A),
        ("molar gas constant", physics::R_GAS),
        ("Faraday constant", physics::FARADAY),
        ("Stefan-Boltzmann constant", physics::SIGMA_SB),
        ("first radiation constant", physics::C1_RADIATION),
        ("second radiation constant", physics::C2_RADIATION),
        ("electron volt", physics::EV),
        ("Newtonian constant of gravitation", physics::G),
        ("fine-structure constant", physics::ALPHA),
        ("inverse fine-structure constant", physics::INV_ALPHA),
        ("electron mass", physics::M_E),
        ("proton mass", physics::M_P),
        ("neutron mass", physics::M_N),
        ("atomic mass constant", physics::AMU),
        ("proton-electron mass ratio", physics::MP_OVER_ME),
        ("Wien wavelength displacement law constant", physics::WIEN_B),
        ("Wien frequency displacement law constant", physics::WIEN_B_FREQ),
        ("Bohr radius", physics::BOHR_RADIUS),
        ("Compton wavelength", physics::COMPTON_WAVELENGTH),
        ("reduced Compton wavelength", physics::COMPTON_WAVELENGTH_REDUCED),
        ("classical electron radius", physics::CLASSICAL_ELECTRON_RADIUS),
        ("Thomson cross section", physics::THOMSON_CROSS_SECTION),
        ("Rydberg constant", physics::RYDBERG),
        ("Hartree energy", physics::HARTREE),
        ("electron mass energy equivalent", physics::ELECTRON_REST_ENERGY),
        ("proton mass energy equivalent", physics::PROTON_REST_ENERGY),
        ("Planck length", physics::PLANCK_LENGTH),
        ("Planck mass", physics::PLANCK_MASS),
        ("Planck time", physics::PLANCK_TIME),
        ("Planck temperature", physics::PLANCK_TEMPERATURE),
        ("standard acceleration of gravity", physics::STANDARD_GRAVITY),
        ("standard atmosphere", physics::STANDARD_ATMOSPHERE),
        ("vacuum mag. permeability", em::MU_0),
        ("vacuum electric permittivity", em::EPS_0),
        ("characteristic impedance of vacuum", em::Z_0),
        ("mag. flux quantum", em::FLUX_QUANTUM),
        ("Josephson constant", em::JOSEPHSON),
        ("von Klitzing constant", em::VON_KLITZING),
        ("conductance quantum", em::CONDUCTANCE_QUANTUM),
        ("Bohr magneton", em::BOHR_MAGNETON),
        ("nuclear magneton", em::NUCLEAR_MAGNETON),
    ];
    let (mut worst, mut worst_name, mut within) = (0.0f64, "", 0usize);
    for (name, v) in pairs {
        let k = codata::physical_constant(name).unwrap_or_else(|| panic!("{name}"));
        let rel = (v - k.value).abs() / k.value.abs();
        if rel > worst {
            (worst, worst_name) = (rel, name);
        }
        if (v - k.value).abs() <= k.uncertainty.max(4.0 * f64::EPSILON * k.value.abs()) {
            within += 1;
        }
    }
    rows.push(vec![
        format!("The library's {} hand-written constants (`physics`, `em`) against the table", pairs.len()),
        format!("{within} of {} within the CODATA uncertainty; largest relative difference {} ({worst_name})", pairs.len(), fmt_diff(worst)),
    ]);
    write("foundations", "sys215_codata.md", &md_table(&["Check", "Result against SciPy 1.17.1 (CODATA 2022)"], &rows));
}

fn horner(c: &[f64], x: C) -> C {
    c.iter().fold(C::new(0.0, 0.0), |acc, &a| acc * x + a)
}

fn figures(refs: &Refs) {
    // Butterworth magnitude |H(iω)|² = 1/(1 + ω^{2N}) evaluated from zeros/poles/gain and from
    // the expanded coefficients.
    let w: Vec<f64> = (0..400).map(|i| 10f64.powf(-1.0 + 2.0 * i as f64 / 399.0)).collect();
    let (mut zpk_err, mut tf_err) = (Vec::new(), Vec::new());
    for order in (2..=40).step_by(2) {
        let poles: Vec<C> = (0..order).map(|k| C::from_polar(1.0, std::f64::consts::PI * (2 * k + order + 1) as f64 / (2 * order) as f64)).collect();
        let sys = ZerosPolesGain::new(vec![], poles, 1.0);
        let (b, a) = sys.to_tf();
        let (mut ez, mut et) = (0.0f64, 0.0f64);
        for (&wi, hz) in w.iter().zip(sys.freqresp(&w)) {
            let exact = 1.0 / (1.0 + wi.powi(2 * order)).sqrt();
            let ht = horner(&b, C::new(0.0, wi)) / horner(&a, C::new(0.0, wi));
            ez = ez.max((hz.norm() - exact).abs() / exact);
            et = et.max((ht.norm() - exact).abs() / exact);
        }
        zpk_err.push((order as f64, ez.max(1e-17)));
        tf_err.push((order as f64, et.max(1e-17)));
    }
    let mut plot = Plot::new("Butterworth magnitude, 0.1 ≤ ω ≤ 10: factored form against expanded coefficients", "order N", "largest relative error in |H(iω)|");
    plot.log_y = true;
    plot.legend = Legend::TopLeft;
    plot.series.push(Series::new("ZerosPolesGain (factor by factor)", zpk_err, Style::LineMarkers, 0));
    plot.series.push(Series::new("transfer-function coefficients (Horner)", tf_err, Style::LineMarkers, 1));
    write("comms", "zpk_vs_tf.svg", &plot.render());

    // Dendrogram before and after optimal leaf ordering (average linkage).
    let pts = refs.values("points");
    let x = Array2::from_shape_vec((pts.len() / 2, 2), pts.to_vec()).unwrap();
    let d = distances(&x);
    let tree = agglomerate(d.view(), Linkage::Average).unwrap();
    let olo = tree.optimal_leaf_ordering(d.view()).unwrap();
    let mut plot = Plot::new("Average linkage: leaf order as built (top) and optimal (bottom, drawn downward)", "leaf position", "merge height");
    plot.legend = Legend::TopRight;
    plot.y_range = Some((-30.0, 40.0));
    for (k, (t, sign)) in [(&tree, 1.0), (&olo, -1.0)].into_iter().enumerate() {
        let n = t.points;
        let order = t.leaves();
        let label = format!("{}: leaf path length {:.1}", if k == 0 { "as built" } else { "optimal order" }, t.leaf_path_length(d.view()));
        let mut pos = vec![0.0; n + t.merges.len()];
        for (i, &p) in order.iter().enumerate() {
            pos[p] = i as f64;
        }
        let mut h = vec![0.0; n + t.merges.len()];
        for (i, m) in t.merges.iter().enumerate() {
            pos[n + i] = (pos[m.a] + pos[m.b]) / 2.0;
            h[n + i] = m.distance;
            let pts = vec![(pos[m.a], sign * h[m.a]), (pos[m.a], sign * m.distance), (pos[m.b], sign * m.distance), (pos[m.b], sign * h[m.b])];
            plot.series.push(Series::new(if i == 0 { &label } else { "" }, pts, Style::Line, k));
        }
    }
    write("unsupervised", "olo_dendrogram.svg", &plot.render());
}
