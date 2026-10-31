use ndarray::Array1;
use num::Complex;

use crate::prelude::ErrorsJSL;

use super::design::Zpk;

/// Butterworth analog lowpass prototype (`buttap`) with cutoff at 1 rad/s.
pub fn buttap(order: usize) -> Result<Zpk, ErrorsJSL> {
    if order == 0 {
        return Err(ErrorsJSL::InvalidInputRange("order must be > 0"));
    }

    // Butterworth poles lie uniformly on the left-half unit semicircle. The
    // resulting prototype has maximally flat magnitude at DC and a -3 dB
    // cutoff at 1 rad/s.
    let poles = (0..order)
        .map(|k| {
            let theta = std::f64::consts::PI * (2.0 * k as f64 + order as f64 + 1.0)
                / (2.0 * order as f64);
            Complex::new(theta.cos(), theta.sin())
        })
        .collect::<Vec<_>>();

    Ok(Zpk {
        z: Array1::from_vec(vec![]),
        p: Array1::from_vec(poles),
        k: Complex::new(1.0, 0.0),
    })
}
