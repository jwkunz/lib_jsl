//! Window functions: shapes, figures of merit, and the periodic and symmetric forms.

use lib_jsl_dsp::discrete::windows::{WindowType, generate_window};

fn main() {
    for (name, kind) in [("Hann", WindowType::Hann), ("Blackman-Harris", WindowType::BlackmanHarris), ("Kaiser β = 8", WindowType::Kaiser { beta: 8.0 }), ("flat top", WindowType::Flattop)] {
        let w = generate_window(kind, 1024, false);
        let (s1, s2): (f64, f64) = (w.iter().sum(), w.iter().map(|v| v * v).sum());
        println!("{name:<16} coherent gain {:.4}, equivalent noise bandwidth {:.3} bins", s1 / 1024.0, 1024.0 * s2 / (s1 * s1));
    }
    // Symmetric windows suit filter design; periodic ones (sym = false) suit spectral analysis.
    println!("symmetric Hann(5): {:?}", generate_window(WindowType::Hann, 5, true));
    println!("periodic Hann(5):  {:?}", generate_window(WindowType::Hann, 5, false));
}
