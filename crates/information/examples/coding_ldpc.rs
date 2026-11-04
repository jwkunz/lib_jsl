//! A regular Gallager LDPC code decoded by belief propagation.

use lib_jsl_information::forward_error_correction::LdpcCode;

fn main() {
    let code = LdpcCode::gallager(96, 3, 6, 1).unwrap();
    println!("Gallager (3, 6) code: n = {}, k = {}, {} checks", code.n(), code.k(), code.num_checks());
    let info: Vec<u8> = (0..code.k()).map(|i| ((i * 5 + 3) % 7 % 2) as u8).collect();
    let cw = code.encode(&info).unwrap();
    println!("codeword satisfies every check: {}", code.is_codeword(&cw));
    // Channel LLRs of magnitude 2, with 8 bits received wrongly.
    let mut llr: Vec<f64> = cw.iter().map(|&b| if b == 0 { 2.0 } else { -2.0 }).collect();
    for i in (0..96).step_by(12) {
        llr[i] = -llr[i];
    }
    let d = code.decode(&llr, 50).unwrap();
    println!("8 errors: converged {}, after {} iterations, info recovered {}", d.converged, d.iterations, code.extract_info(&d.codeword).unwrap() == info);
}
