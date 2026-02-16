/// This function generates the taps for a derivative filter by convolving a derivative kernel (e.g. [-0.5,0,0.5]) with a low-pass filter designed using `firwin`. 
/// The `ftype` parameter controls the type of derivative filter (0 for first derivative, 1 for second derivative, etc.). The resulting taps are returned as a vector of f64 values.
/// The `cutoff` parameter allows you to control the frequency response of the generated derivative filter, which can be useful for designing filters that are tailored to specific applications or signal characteristics. 
/// The `numtaps` parameter specifies the number of taps in the resulting filter, which can affect the filter's performance and computational complexity. 
/// A larger number of taps can provide better frequency response but may also increase the computational cost of applying the filter to a signal. 

use crate::{
    dsp::{convolve::{ConvolveMethod, ConvolveMode, convolve}, sinc::generate_centered_sinc},
    prelude::{ErrorsJSL, IsAnalytic},
};

pub fn generate_derivative_taps(numtaps: usize, derivative_order: usize, cutoff: Option<f64>) -> Result<Vec<f64>, ErrorsJSL> {
    if numtaps < 3  {
        return Err(ErrorsJSL::InvalidInputRange("numtaps must be greater than or equal to 3"));
    }

    let cutoff = cutoff.unwrap_or(0.9); // Use a cutoff slightly below Nyquist to avoid aliasing
    let sinc_taps = generate_centered_sinc(numtaps-2, cutoff);
    let scale = 1.0/(sinc_taps.iter().map(|x| x.f_abs2()).sum::<f64>()*sinc_taps.len() as f64).sqrt(); // Normalize the sinc taps to have unit energy
    let sinc_taps: Vec<f64> = sinc_taps.into_iter().map(|x| x * scale).collect(); // Apply the normalization to the sinc taps
    let derivative_kernel = vec![0.5,0.0,-0.5]; // First derivative kernel (central difference)
    let mut result = Vec::new();
    for _ in 0..derivative_order {
        result = convolve(&sinc_taps, &derivative_kernel,ConvolveMode::Full, ConvolveMethod::Direct)?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_derivative_taps() {
        let n_taps = 13;
        let taps = generate_derivative_taps(13, 1, None).unwrap();
        let test_vector = (0..n_taps*2).map(|x| x as f64).collect::<Vec<f64>>();
        let convolved = convolve(&test_vector, &taps, ConvolveMode::Full, ConvolveMethod::Direct).unwrap();
        let middle_value = convolved[convolved.len()/2];
        assert!(middle_value.abs()-1.0 < 0.1, "The middle value of the convolved result should be near 1.0 for the first derivative kernel, got {middle_value}");
    }
}