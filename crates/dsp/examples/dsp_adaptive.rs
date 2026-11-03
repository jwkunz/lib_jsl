//! System identification with NLMS and RLS adaptive filters.

use lib_jsl_dsp::discrete::filters::adaptive::lms::LmsFilter;
use lib_jsl_dsp::discrete::filters::adaptive::rls::RlsFilter;
use num_complex::Complex;

fn main() {
    // An unknown 4-tap system, driven by a deterministic pseudo-random input.
    let plant = [0.6, -0.3, 0.2, 0.05];
    let mut state = 12345u64;
    let x: Vec<Complex<f64>> = (0..2000)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            Complex::new((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0, 0.0)
        })
        .collect();
    let d: Vec<Complex<f64>> = (0..x.len()).map(|n| (0..4).filter(|&k| k <= n).map(|k| x[n - k] * plant[k]).sum()).collect();
    let mut nlms = LmsFilter::new(4, 0.5, true).unwrap();
    let mut rls = RlsFilter::new(4, 0.999, 100.0).unwrap();
    for n in 0..x.len() {
        let (_, e1) = nlms.adapt(x[n], d[n]);
        let (_, e2) = rls.adapt(x[n], d[n]);
        if [10, 50, 200, 1999].contains(&n) {
            println!("n = {n:4}: |e| NLMS {:.1e}, RLS {:.1e}", e1.norm(), e2.norm());
        }
    }
    let show = |w: &[Complex<f64>]| w.iter().map(|c| format!("{:.4}", c.re)).collect::<Vec<_>>().join(", ");
    println!("NLMS weights [{}]", show(nlms.weights()));
    println!("RLS weights  [{}]", show(rls.weights()));
}
