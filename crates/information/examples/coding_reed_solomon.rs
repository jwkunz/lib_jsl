//! Reed-Solomon RS(255, 223): correcting up to 16 byte errors, including long bursts.

use lib_jsl_information::forward_error_correction::ReedSolomon;

fn main() {
    let rs = ReedSolomon::new(255, 223).unwrap();
    let message: Vec<u8> = (0..223).map(|i| (i * 7 % 256) as u8).collect();
    let cw = rs.encode(&message).unwrap();
    println!("RS({}, {}): {} parity bytes, corrects {} byte errors", rs.n(), rs.k(), cw.len() - message.len(), rs.error_capacity());
    // A burst wiping out 16 consecutive bytes (128 bits).
    let mut rx = cw.clone();
    for b in &mut rx[100..116] {
        *b = !*b;
    }
    let (decoded, corrected) = rs.decode(&rx).unwrap();
    println!("16-byte burst: {corrected} bytes corrected, message recovered: {}", decoded == message);
    // One more error is beyond the code's capacity.
    rx[200] ^= 0x55;
    println!("17 errors: {:?}", rs.decode(&rx).map(|(_, c)| c));
    // A shortened code for short messages.
    let short = ReedSolomon::new(20, 12).unwrap();
    let cw = short.encode(b"Hello, RS!!!").unwrap();
    let mut rx = cw.clone();
    rx[0] = b'J';
    rx[15] ^= 0xFF;
    let (m, c) = short.decode(&rx).unwrap();
    println!("RS(20, 12): {} errors corrected -> {:?}", c, String::from_utf8(m).unwrap());
}
