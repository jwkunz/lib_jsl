//! Figures and tables for the special functions chapter.
//!
//! The accuracy tables compare the library with reference values computed by mpmath at 50 digits
//! (`data/special_functions_reference.csv`, made by `data/make_refs.py`).

use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_special_functions::*;
use num_complex::Complex;

const CHAPTER: &str = "special_functions";
const REFERENCE: &str = include_str!("../data/special_functions_reference.csv");

/// Evaluates the library function named in the reference file.
fn eval(name: &str, a: &[f64]) -> f64 {
    let i = |k: usize| a[k].round() as i32;
    let u = |k: usize| a[k].round() as u32;
    let u64_ = |k: usize| a[k].round() as u64;
    match name {
        "gamma" => gamma(a[0]),
        "lngamma" => ln_gamma(a[0]),
        "digamma" => digamma(a[0]),
        "beta" => beta(a[0], a[1]),
        "lnbeta" => ln_beta(a[0], a[1]),
        "factf" => factorial_f64(u64_(0)),
        "lnfact" => ln_factorial(u64_(0)),
        "lnbinom" => ln_binomial_coefficient(u64_(0), u64_(1)),
        "gammainc" => incomplete_gamma(a[0], a[1]),
        "gammaincc" => incomplete_gamma_upper(a[0], a[1]),
        "betainc" => incomplete_beta(a[0], a[1], a[2]),
        "erf" => error_function(a[0]),
        "erfc" => erfc(a[0]),
        "erfinv" => erf_inv(a[0]),
        "dawson" => dawson_integral(a[0]),
        "e1" => exponential_integral(a[0]),
        "en" => exponential_integral_en(u(0), a[1]),
        "ei" => exponential_integral_ei(a[0]),
        "si" => sine_integral(a[0]),
        "ci" => cosine_integral(a[0]),
        "fresnels" => fresnel_integrals(a[0]).0,
        "fresnelc" => fresnel_integrals(a[0]).1,
        "besselj" => bessel_jy(a[0], a[1]).0,
        "bessely" => bessel_jy(a[0], a[1]).1,
        "besseli" => bessel_ik(a[0], a[1]).0,
        "besselk" => bessel_ik(a[0], a[1]).1,
        "besseljn" => bessel_j_integer(i(0), a[1]),
        "besselyn" => bessel_y_integer(i(0), a[1]),
        "besselin" => bessel_i_integer(i(0), a[1]),
        "besselkn" => bessel_k_integer(i(0), a[1]),
        "sphj" => bessel_spherical(i(0), a[1]),
        "sphy" => bessel_spherical_y(i(0), a[1]),
        "airyai" => airy_ai(a[0]),
        "airybi" => airy_bi(a[0]),
        "ellipk" => elliptic_integral(a[0]),
        "ellipe" => elliptic_integral_second(a[0]),
        "ellippi" => elliptic_integral_third(a[0], a[1]),
        "ellipf" => elliptic_f(a[0], a[1]),
        "ellipeinc" => elliptic_e(a[0], a[1]),
        "ellippiinc" => elliptic_pi(a[0], a[1], a[2]),
        "sn" => jacobian_elliptic(a[0], a[1]).0,
        "cn" => jacobian_elliptic(a[0], a[1]).1,
        "dn" => jacobian_elliptic(a[0], a[1]).2,
        "hyp2f1" => hypergeometric(a[0], a[1], a[2], a[3]),
        "hyp1f1" => hypergeometric_1f1(a[0], a[1], a[2]),
        "hyp0f1" => hypergeometric_0f1(a[0], a[1]),
        "legendre" => legendre(u(0), a[1]),
        "chebyt" => chebyshev_t(u(0), a[1]),
        "chebyu" => chebyshev_u(u(0), a[1]),
        "hermite" => hermite(u(0), a[1]),
        "hermitee" => hermite_he(u(0), a[1]),
        "laguerre" => laguerre(u(0), a[1], a[2]),
        "jacobi" => jacobi(u(0), a[1], a[2], a[3]),
        "legenp" => associated_legendre(u(0), i(1), a[2]),
        "ylmre" => spherical_harmonic(u(0), i(1), a[2], a[3]).re,
        "ylmim" => spherical_harmonic(u(0), i(1), a[2], a[3]).im,
        "ylmreal" => real_spherical_harmonic(u(0), i(1), a[2], a[3]),
        "zeta" => zeta(a[0]),
        "hurwitz" => hurwitz_zeta(a[0], a[1]),
        "polylog" => polylog(a[0], a[1]),
        "fermidirac" => fermi_dirac_integral(a[0], a[1]),
        "lambertw0" => lambert_w0(a[0]),
        "lambertwm1" => lambert_w_m1(a[0]),
        "lwre" => lambert_w(i(0), Complex::new(a[1], a[2])).re,
        "lwim" => lambert_w(i(0), Complex::new(a[1], a[2])).im,
        "ixlogx" => inverse_x_log_x(a[0]),
        "ixlogxlo" => inverse_x_log_x_lower(a[0]),
        "normcdf" => normal_cdf(a[0], 0.0, 1.0),
        "normq" => normal_quantile(a[0], 0.0, 1.0),
        "chi2q" => chi_square_quantile(a[0], a[1]),
        "tq" => student_t_quantile(a[0], a[1]),
        "tcdf" => student_t_cdf(a[0], a[1]),
        "fcdf" => f_cdf(a[0], a[1], a[2]),
        "chi2sf" => chi_square_survival(a[0], a[1]),
        "poispmf" => poisson_pmf(u64_(0), a[1]),
        "poiscdf" => poisson_cdf(u64_(0), a[1]),
        "binpmf" => binomial_pmf(u64_(0), u64_(1), a[2]),
        "bincdf" => binomial_cdf(u64_(0), u64_(1), a[2]),
        _ => panic!("unknown function {name}"),
    }
}

/// Writes the accuracy table for the given (reference name, displayed name) pairs.
fn accuracy_table(file: &str, functions: &[(&str, &str)]) {
    let mut table = String::from("| function | test points | median relative error | largest relative error | at arguments |\n|---|---:|---:|---:|---|\n");
    for &(key, label) in functions {
        let mut errors: Vec<(f64, String)> = vec![];
        for line in REFERENCE.lines().skip(1) {
            let mut parts = line.split(',');
            if parts.next() != Some(key) {
                continue;
            }
            let args_text = parts.next().unwrap();
            let args: Vec<f64> = args_text.split_whitespace().map(|v| v.parse().unwrap()).collect();
            let reference: f64 = parts.next().unwrap().parse().unwrap();
            let value = eval(key, &args);
            let err = ((value - reference) / reference).abs();
            let shown = args.iter().map(|v| format!("{v}")).collect::<Vec<_>>().join(", ");
            errors.push((if err.is_nan() { f64::INFINITY } else { err }, shown));
        }
        assert!(!errors.is_empty(), "no reference values for {key}");
        errors.sort_by(|a, b| a.0.total_cmp(&b.0));
        let median = errors[errors.len() / 2].0;
        let (worst, at) = errors.last().unwrap().clone();
        let fmt = |e: f64| if e == 0.0 { "0".to_string() } else { format!("{e:.0e}") };
        table.push_str(&format!("| {label} | {} | {} | {} | ({at}) |\n", errors.len(), fmt(median), fmt(worst)));
    }
    write(CHAPTER, file, &table);
}

fn curve(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> Vec<(f64, f64)> {
    (0..=n).map(|k| a + (b - a) * k as f64 / n as f64).map(|x| (x, f(x))).collect()
}

/// A curve split into pieces where |f| exceeds `cap` (to draw functions with poles).
fn pieces(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize, cap: f64) -> Vec<Vec<(f64, f64)>> {
    let mut out = vec![vec![]];
    for (x, y) in curve(f, a, b, n) {
        if y.is_finite() && y.abs() <= cap {
            out.last_mut().unwrap().push((x, y));
        } else if !out.last().unwrap().is_empty() {
            out.push(vec![]);
        }
    }
    out.into_iter().filter(|p| p.len() > 1).collect()
}

fn push_pieces(plot: &mut Plot, label: &str, ps: Vec<Vec<(f64, f64)>>, style: Style, color: usize) {
    for (k, p) in ps.into_iter().enumerate() {
        plot.series.push(Series::new(if k == 0 { label } else { "" }, p, style, color));
    }
}

fn gamma_figures() {
    let mut plot = Plot::new("The gamma function and its reciprocal", "x", "");
    plot.legend = Legend::BottomRight;
    plot.y_range = Some((-6.0, 6.0));
    push_pieces(&mut plot, "Γ(x)", pieces(gamma, -4.5, 5.0, 4000, 8.0), Style::Line, 0);
    plot.series.push(Series::new("1/Γ(x)", curve(|x| 1.0 / gamma(x), -4.5, 5.0, 1000), Style::Line, 1));
    plot.series.push(Series::new("(n − 1)! at x = n", (1..=4).map(|n| (n as f64, factorial_f64(n - 1))).collect(), Style::Markers, 6));
    write(CHAPTER, "gamma.svg", &plot.render());
    accuracy_table("gamma_accuracy.md", &[("gamma", "`gamma`"), ("lngamma", "`ln_gamma`"), ("digamma", "`digamma`"), ("beta", "`beta`"), ("lnbeta", "`ln_beta`"), ("factf", "`factorial_f64`"), ("lnfact", "`ln_factorial`"), ("lnbinom", "`ln_binomial_coefficient`")]);
}

fn incomplete_figures() {
    let mut plot = Plot::new("The regularised incomplete gamma function P(a, x)", "x", "P(a, x)");
    plot.legend = Legend::BottomRight;
    for (k, a) in [0.5, 1.0, 2.0, 5.0, 10.0].into_iter().enumerate() {
        plot.series.push(Series::new(&format!("a = {a}"), curve(|x| incomplete_gamma(a, x), 0.0, 20.0, 400), Style::Line, k));
    }
    write(CHAPTER, "incomplete_gamma.svg", &plot.render());
    let mut plot = Plot::new("The complementary error function: computed directly, or as 1 − erf(x)", "x", "");
    plot.log_y = true;
    plot.legend = Legend::BottomLeft;
    plot.y_range = Some((1e-45, 10.0));
    plot.series.push(Series::new("erfc(x), computed directly", curve(erfc, 0.0, 10.0, 400), Style::Line, 0));
    plot.series.push(Series::new("1 − erf(x): cancellation, then 0 beyond x ≈ 5.9", curve(|x| 1.0 - error_function(x), 0.0, 10.0, 400), Style::Dashed, 1));
    plot.series.push(Series::new("e^(−x²)/(x√π), the asymptote", curve(|x| (-x * x).exp() / (x * std::f64::consts::PI.sqrt()), 0.8, 10.0, 400), Style::Dashed, 7));
    write(CHAPTER, "erfc.svg", &plot.render());
    accuracy_table("incomplete_accuracy.md", &[("gammainc", "`incomplete_gamma` $P(a,x)$"), ("gammaincc", "`incomplete_gamma_upper` $Q(a,x)$"), ("betainc", "`incomplete_beta` $I_x(a,b)$"), ("erf", "`error_function`"), ("erfc", "`erfc`"), ("erfinv", "`erf_inv`"), ("dawson", "`dawson_integral`")]);
}

fn exponential_integral_figures() {
    let mut plot = Plot::new("The sine, cosine and exponential integrals", "x", "");
    plot.legend = Legend::BottomRight;
    plot.y_range = Some((-3.0, 3.0));
    plot.series.push(Series::new("Si(x)", curve(sine_integral, -15.0, 15.0, 600), Style::Line, 0));
    push_pieces(&mut plot, "Ci(x)", pieces(cosine_integral, 0.01, 15.0, 600, 3.0), Style::Line, 1);
    push_pieces(&mut plot, "Ei(x)", pieces(exponential_integral_ei, -15.0, 15.0, 1200, 3.0), Style::Line, 2);
    push_pieces(&mut plot, "E₁(x)", pieces(exponential_integral, 0.005, 15.0, 600, 3.0), Style::Line, 3);
    write(CHAPTER, "sine_cosine_integrals.svg", &plot.render());
    let mut plot = Plot::new("The Cornu spiral: (C(t), S(t)) for −8 ≤ t ≤ 8", "C(t)", "S(t)");
    plot.legend = Legend::TopLeft;
    plot.x_range = Some((-0.9, 0.9));
    plot.y_range = Some((-0.9, 0.9));
    let pts = (0..=3200).map(|k| -8.0 + 16.0 * k as f64 / 3200.0).map(|t| { let (s, c) = fresnel_integrals(t); (c, s) }).collect();
    plot.series.push(Series::new("", pts, Style::Line, 0));
    plot.series.push(Series::new("limits (±½, ±½)", vec![(0.5, 0.5), (-0.5, -0.5)], Style::Markers, 1));
    write(CHAPTER, "cornu_spiral.svg", &plot.render());
    accuracy_table("exponential_accuracy.md", &[("e1", "`exponential_integral` $E_1$"), ("en", "`exponential_integral_en` $E_n$"), ("ei", "`exponential_integral_ei`"), ("si", "`sine_integral`"), ("ci", "`cosine_integral`"), ("fresnels", "`fresnel_integrals` $S$"), ("fresnelc", "`fresnel_integrals` $C$")]);
}

fn bessel_figures() {
    let mut plot = Plot::new("Bessel functions of the first and second kinds", "x", "");
    plot.legend = Legend::BottomRight;
    plot.y_range = Some((-1.0, 1.05));
    for (k, n) in [0.0, 1.0, 2.0, 3.5].into_iter().enumerate() {
        plot.series.push(Series::new(&format!("J_{n}(x)"), curve(|x| bessel_fractional(n, x), 0.0, 20.0, 800), Style::Line, k));
    }
    for (k, n) in [0.0, 1.0].into_iter().enumerate() {
        push_pieces(&mut plot, &format!("Y_{n}(x)"), pieces(|x| bessel_y_fractional(n, x), 0.05, 20.0, 800, 1.0), Style::Dashed, k);
    }
    write(CHAPTER, "bessel_jy.svg", &plot.render());
    let mut plot = Plot::new("Modified Bessel functions and Airy functions", "x", "");
    plot.legend = Legend::TopLeft;
    plot.y_range = Some((-0.6, 2.5));
    for (k, n) in [0.0, 1.0].into_iter().enumerate() {
        plot.series.push(Series::new(&format!("I_{n}(x)"), curve(|x| bessel_i_fractional(n, x), 0.0, 3.0, 300), Style::Line, k));
        push_pieces(&mut plot, &format!("K_{n}(x)"), pieces(|x| bessel_k_fractional(n, x), 0.02, 3.0, 300, 2.5), Style::Dashed, k);
    }
    plot.series.push(Series::new("Ai(x), on −10 ≤ x ≤ 3", curve(airy_ai, -10.0, 3.0, 800), Style::Line, 2));
    push_pieces(&mut plot, "Bi(x)", pieces(airy_bi, -10.0, 3.0, 800, 2.5), Style::Dashed, 2);
    write(CHAPTER, "bessel_ik_airy.svg", &plot.render());
    accuracy_table("bessel_accuracy.md", &[
        ("besselj", "`bessel_jy` $J_\\nu$"), ("bessely", "`bessel_jy` $Y_\\nu$"), ("besseli", "`bessel_ik` $I_\\nu$"), ("besselk", "`bessel_ik` $K_\\nu$"),
        ("besseljn", "`bessel_j_integer`"), ("besselyn", "`bessel_y_integer`"), ("besselin", "`bessel_i_integer`"), ("besselkn", "`bessel_k_integer`"),
        ("sphj", "`bessel_spherical` $j_n$"), ("sphy", "`bessel_spherical_y` $y_n$"), ("airyai", "`airy_ai`"), ("airybi", "`airy_bi`"),
    ]);
}

fn elliptic_figures() {
    let mut plot = Plot::new("Complete elliptic integrals", "modulus k", "");
    plot.legend = Legend::TopLeft;
    plot.y_range = Some((0.8, 4.5));
    plot.series.push(Series::new("K(k)", curve(elliptic_integral, 0.0, 0.9995, 500), Style::Line, 0));
    plot.series.push(Series::new("E(k)", curve(elliptic_integral_second, 0.0, 1.0, 500), Style::Line, 1));
    plot.series.push(Series::new("π/2", vec![(0.0, std::f64::consts::FRAC_PI_2), (1.0, std::f64::consts::FRAC_PI_2)], Style::Dashed, 7));
    write(CHAPTER, "elliptic_ke.svg", &plot.render());
    let m: f64 = 0.95;
    let k4 = 4.0 * elliptic_integral(m.sqrt());
    let mut plot = Plot::new("Jacobian elliptic functions, m = 0.95 (period 4K = 11.39)", "u", "");
    plot.legend = Legend::BottomLeft;
    plot.y_range = Some((-1.1, 1.4));
    plot.series.push(Series::new("sn(u | m)", curve(|u| jacobian_elliptic(u, m).0, 0.0, 2.0 * k4, 800), Style::Line, 0));
    plot.series.push(Series::new("cn(u | m)", curve(|u| jacobian_elliptic(u, m).1, 0.0, 2.0 * k4, 800), Style::Line, 1));
    plot.series.push(Series::new("dn(u | m)", curve(|u| jacobian_elliptic(u, m).2, 0.0, 2.0 * k4, 800), Style::Line, 2));
    plot.series.push(Series::new("sin u, for comparison", curve(f64::sin, 0.0, 2.0 * k4, 800), Style::Dashed, 7));
    write(CHAPTER, "jacobi_elliptic.svg", &plot.render());
    accuracy_table("elliptic_accuracy.md", &[
        ("ellipk", "`elliptic_integral` $K(k)$"), ("ellipe", "`elliptic_integral_second` $E(k)$"), ("ellippi", "`elliptic_integral_third` $\\Pi(n,k)$"),
        ("ellipf", "`elliptic_f` $F(\\varphi,k)$"), ("ellipeinc", "`elliptic_e` $E(\\varphi,k)$"), ("ellippiinc", "`elliptic_pi` $\\Pi(\\varphi,n,k)$"),
        ("sn", "`jacobian_elliptic` sn"), ("cn", "`jacobian_elliptic` cn"), ("dn", "`jacobian_elliptic` dn"),
    ]);
}

fn hypergeometric_figures() {
    // Elementary functions as hypergeometric functions, evaluated through the library.
    let z: f64 = 0.37;
    let rows: Vec<(&str, f64, f64)> = vec![
        ("$\\ln(1 + z) = z\\,{}_2F_1(1, 1; 2; -z)$", (1.0 + z).ln(), z * hypergeometric(1.0, 1.0, 2.0, -z)),
        ("$\\arcsin z = z\\,{}_2F_1(\\tfrac12, \\tfrac12; \\tfrac32; z^2)$", z.asin(), z * hypergeometric(0.5, 0.5, 1.5, z * z)),
        ("$\\arctan z = z\\,{}_2F_1(\\tfrac12, 1; \\tfrac32; -z^2)$", z.atan(), z * hypergeometric(0.5, 1.0, 1.5, -z * z)),
        ("$(1 - z)^{-a} = {}_2F_1(a, b; b; z)$, $a = 2.5$", (1.0 - z).powf(-2.5), hypergeometric(2.5, 1.3, 1.3, z)),
        ("$P_5(1 - 2z) = {}_2F_1(-5, 6; 1; z)$", legendre(5, 1.0 - 2.0 * z), hypergeometric(-5.0, 6.0, 1.0, z)),
        ("$K(k) = \\tfrac\\pi2\\,{}_2F_1(\\tfrac12, \\tfrac12; 1; k^2)$", elliptic_integral(z), std::f64::consts::FRAC_PI_2 * hypergeometric(0.5, 0.5, 1.0, z * z)),
        ("$e^z = {}_1F_1(a; a; z)$", z.exp(), hypergeometric_1f1(1.7, 1.7, z)),
        ("$\\operatorname{erf} z = \\tfrac{2z}{\\sqrt\\pi}\\,{}_1F_1(\\tfrac12; \\tfrac32; -z^2)$", error_function(z), 2.0 * z / std::f64::consts::PI.sqrt() * hypergeometric_1f1(0.5, 1.5, -z * z)),
        ("$P(a, z) = \\tfrac{z^a e^{-z}}{\\Gamma(a+1)}\\,{}_1F_1(1; a+1; z)$, $a = 2$", incomplete_gamma(2.0, z), z * z * (-z).exp() / gamma(3.0) * hypergeometric_1f1(1.0, 3.0, z)),
        ("$\\cos 2\\sqrt z = {}_0F_1(; \\tfrac12; -z)$", (2.0 * z.sqrt()).cos(), hypergeometric_0f1(0.5, -z)),
        ("$J_1(2) = {}_0F_1(; 2; -1)$", bessel_j_integer(1, 2.0), hypergeometric_0f1(2.0, -1.0)),
    ];
    let mut table = String::from("| identity (at $z = 0.37$) | elementary side | hypergeometric side | relative difference |\n|---|---:|---:|---:|\n");
    for (name, a, b) in rows {
        table.push_str(&format!("| {name} | {a:.15} | {b:.15} | {:.0e} |\n", ((a - b) / a).abs()));
    }
    write(CHAPTER, "hypergeometric_identities.md", &table);
    let mut plot = Plot::new("Kummer's function ₁F₁(a; 2; x)", "x", "");
    plot.legend = Legend::TopLeft;
    plot.y_range = Some((-3.0, 6.0));
    for (k, a) in [-3.0, -1.0, 0.5, 1.0, 2.0].into_iter().enumerate() {
        plot.series.push(Series::new(&format!("a = {a}"), curve(|x| hypergeometric_1f1(a, 2.0, x), -6.0, 4.0, 400), Style::Line, k));
    }
    write(CHAPTER, "kummer.svg", &plot.render());
    accuracy_table("hypergeometric_accuracy.md", &[("hyp2f1", "`hypergeometric` ${}_2F_1$"), ("hyp1f1", "`hypergeometric_1f1` ${}_1F_1$"), ("hyp0f1", "`hypergeometric_0f1` ${}_0F_1$")]);
}

fn orthogonal_figures() {
    let mut plot = Plot::new("Legendre polynomials", "x", "Pₙ(x)");
    plot.legend = Legend::BottomRight;
    for n in 0..=5u32 {
        plot.series.push(Series::new(&format!("P_{n}"), curve(|x| legendre(n, x), -1.0, 1.0, 400), Style::Line, n as usize));
    }
    write(CHAPTER, "legendre.svg", &plot.render());
    let mut plot = Plot::new("Hermite functions Hₙ(x) e^(−x²/2) / √(2ⁿ n! √π)", "x", "");
    plot.legend = Legend::TopLeft;
    for n in [0u32, 1, 2, 5, 10] {
        let norm = (2f64.powi(n as i32) * factorial_f64(n as u64) * std::f64::consts::PI.sqrt()).sqrt();
        plot.series.push(Series::new(&format!("n = {n}"), curve(|x| hermite(n, x) * (-x * x / 2.0).exp() / norm, -6.0, 6.0, 600), Style::Line, n as usize % 7));
    }
    write(CHAPTER, "hermite_functions.svg", &plot.render());
    accuracy_table("orthogonal_accuracy.md", &[
        ("legendre", "`legendre` $P_n$"), ("chebyt", "`chebyshev_t` $T_n$"), ("chebyu", "`chebyshev_u` $U_n$"), ("hermite", "`hermite` $H_n$"), ("hermitee", "`hermite_he` $He_n$"),
        ("laguerre", "`laguerre` $L_n^{(1/2)}$"), ("jacobi", "`jacobi` $P_n^{(1/2,-0.3)}$"), ("legenp", "`associated_legendre` $P_l^m$"),
        ("ylmre", "`spherical_harmonic`, real part"), ("ylmim", "`spherical_harmonic`, imaginary part"), ("ylmreal", "`real_spherical_harmonic`"),
    ]);
}

fn zeta_figures() {
    let mut plot = Plot::new("The Riemann zeta function on the real line", "s", "ζ(s)");
    plot.legend = Legend::TopRight;
    plot.y_range = Some((-4.0, 6.0));
    push_pieces(&mut plot, "ζ(s)", pieces(zeta, -14.0, 8.0, 4400, 8.0), Style::Line, 0);
    plot.series.push(Series::new("trivial zeros s = −2, −4, …", (1..=7).map(|k| (-2.0 * k as f64, 0.0)).collect(), Style::Markers, 1));
    write(CHAPTER, "zeta.svg", &plot.render());
    let mut plot = Plot::new("Complete Fermi-Dirac integrals Fⱼ(x)", "x", "Fⱼ(x)");
    plot.log_y = true;
    plot.legend = Legend::TopLeft;
    plot.y_range = Some((1e-5, 1e4));
    for (k, j) in [-0.5, 0.5, 1.5, 3.0].into_iter().enumerate() {
        plot.series.push(Series::new(&format!("j = {j}"), curve(|x| fermi_dirac_integral(j, x), -10.0, 30.0, 400), Style::Line, k));
        plot.series.push(Series::new("", curve(|x| x.exp(), -10.0, 0.0, 50), Style::Dashed, 7));
        plot.series.push(Series::new("", curve(|x| x.powf(j + 1.0) / gamma(j + 2.0), 3.0, 30.0, 100), Style::Dashed, 7));
    }
    plot.series.push(Series::new("eˣ and x^(j+1)/Γ(j+2), the limits", vec![], Style::Dashed, 7));
    write(CHAPTER, "fermi_dirac.svg", &plot.render());
    accuracy_table("zeta_accuracy.md", &[("zeta", "`zeta`"), ("hurwitz", "`hurwitz_zeta`"), ("polylog", "`polylog`"), ("fermidirac", "`fermi_dirac_integral`")]);
}

fn lambert_figures() {
    let e = std::f64::consts::E;
    let mut plot = Plot::new("The real branches of the Lambert W function", "y", "W(y)");
    plot.legend = Legend::BottomRight;
    plot.y_range = Some((-5.0, 2.5));
    plot.series.push(Series::new("W₀(y)", curve(lambert_w0, -1.0 / e, 8.0, 800), Style::Line, 0));
    plot.series.push(Series::new("W₋₁(y)", curve(lambert_w_m1, -1.0 / e, -0.0005, 800), Style::Line, 1));
    plot.series.push(Series::new("branch point (−1/e, −1)", vec![(-1.0 / e, -1.0)], Style::Markers, 6));
    write(CHAPTER, "lambert_w.svg", &plot.render());
    accuracy_table("lambert_accuracy.md", &[("lambertw0", "`lambert_w0`"), ("lambertwm1", "`lambert_w_m1`"), ("lwre", "`lambert_w` (complex), real part"), ("lwim", "`lambert_w` (complex), imaginary part"), ("ixlogx", "`inverse_x_log_x`"), ("ixlogxlo", "`inverse_x_log_x_lower`")]);
}

fn statistical_figures() {
    let mut table = String::from("| distribution | 90% | 95% | 97.5% | 99% | 99.9% |\n|---|---:|---:|---:|---:|---:|\n");
    let ps = [0.9, 0.95, 0.975, 0.99, 0.999];
    let row = |name: &str, f: &dyn Fn(f64) -> f64| format!("| {name} | {} |\n", ps.iter().map(|&p| format!("{:.4}", f(p))).collect::<Vec<_>>().join(" | "));
    table.push_str(&row("standard normal", &|p| normal_quantile(p, 0.0, 1.0)));
    for nu in [1.0, 5.0, 10.0, 30.0] {
        table.push_str(&row(&format!("Student's $t$, $\\nu = {nu}$"), &|p| student_t_quantile(p, nu)));
    }
    for k in [1.0, 2.0, 5.0, 10.0] {
        table.push_str(&row(&format!("$\\chi^2$, $k = {k}$"), &|p| chi_square_quantile(p, k)));
    }
    write(CHAPTER, "quantiles.md", &table);
    let mut plot = Plot::new("Student's t distribution approaches the normal", "t", "CDF");
    plot.legend = Legend::BottomRight;
    for (k, nu) in [1.0, 2.0, 5.0, 30.0].into_iter().enumerate() {
        plot.series.push(Series::new(&format!("t, ν = {nu}"), curve(|t| student_t_cdf(t, nu), -5.0, 5.0, 400), Style::Line, k));
    }
    plot.series.push(Series::new("standard normal", curve(|t| normal_cdf(t, 0.0, 1.0), -5.0, 5.0, 400), Style::Dashed, 6));
    write(CHAPTER, "student_t.svg", &plot.render());
    accuracy_table("statistical_accuracy.md", &[
        ("normcdf", "`normal_cdf`"), ("normq", "`normal_quantile`"), ("tcdf", "`student_t_cdf`"), ("tq", "`student_t_quantile`"), ("chi2q", "`chi_square_quantile`"),
        ("chi2sf", "`chi_square_survival`"), ("fcdf", "`f_cdf`"), ("poispmf", "`poisson_pmf`"), ("poiscdf", "`poisson_cdf`"), ("binpmf", "`binomial_pmf`"), ("bincdf", "`binomial_cdf`"),
    ]);
}

pub fn generate() {
    gamma_figures();
    incomplete_figures();
    exponential_integral_figures();
    bessel_figures();
    elliptic_figures();
    hypergeometric_figures();
    orthogonal_figures();
    zeta_figures();
    lambert_figures();
    statistical_figures();
}
