//! Barker codes: binary phase codes whose aperiodic autocorrelation sidelobes never exceed 1 in
//! magnitude. They are used for frame synchronisation and radar pulse compression because the
//! correlation peak (equal to the code length) stands out from the sidelobes by a factor of
//! `length`. Barker codes exist only for lengths 2, 3, 4, 5, 7, 11 and 13.

use lib_jsl_core::ErrorsJSL;

/// The lengths for which a Barker code exists.
pub const BARKER_LENGTHS: [usize; 7] = [2, 3, 4, 5, 7, 11, 13];

/// The Barker code of the given length as `±1` chips.
pub fn barker_code(length: usize) -> Result<Vec<i8>, ErrorsJSL> {
    let code: &[i8] = match length {
        2 => &[1, -1],
        3 => &[1, 1, -1],
        4 => &[1, 1, -1, 1],
        5 => &[1, 1, 1, -1, 1],
        7 => &[1, 1, 1, -1, -1, 1, -1],
        11 => &[1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
        13 => &[1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
        _ => return Err(ErrorsJSL::InvalidInputRange("Barker codes exist only for lengths 2, 3, 4, 5, 7, 11 and 13")),
    };
    Ok(code.to_vec())
}

/// The aperiodic autocorrelation of a code at lags `-(n-1)..=n-1` (the peak is at index `n - 1`).
pub fn barker_autocorrelation(code: &[i8]) -> Vec<i32> {
    let n = code.len() as isize;
    (-(n - 1)..n)
        .map(|lag| {
            (0..n)
                .filter(|&i| i + lag >= 0 && i + lag < n)
                .map(|i| code[i as usize] as i32 * code[(i + lag) as usize] as i32)
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_code_has_unit_sidelobes_and_a_peak_equal_to_its_length() {
        for &n in &BARKER_LENGTHS {
            let code = barker_code(n).unwrap();
            assert_eq!(code.len(), n);
            assert!(code.iter().all(|&c| c == 1 || c == -1));
            let r = barker_autocorrelation(&code);
            assert_eq!(r.len(), 2 * n - 1);
            assert_eq!(r[n - 1], n as i32, "peak of the length-{n} code");
            for (i, &v) in r.iter().enumerate() {
                if i != n - 1 {
                    assert!(v.abs() <= 1, "length {n}, lag index {i}: sidelobe {v}");
                }
            }
        }
    }

    #[test]
    fn known_sequences() {
        assert_eq!(barker_code(13).unwrap(), vec![1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]);
        assert_eq!(barker_code(7).unwrap(), vec![1, 1, 1, -1, -1, 1, -1]);
        // The length-13 autocorrelation is 13 at zero lag and alternates 0, 1 away from it.
        let r = barker_autocorrelation(&barker_code(13).unwrap());
        assert_eq!(&r[12..16], &[13, 0, 1, 0]);
    }

    #[test]
    fn other_lengths_are_rejected() {
        for n in [0usize, 1, 6, 8, 9, 10, 12, 14, 100] {
            assert!(barker_code(n).is_err(), "length {n}");
        }
    }
}
