use num::Complex;

use crate::dsp::stream_operator::{StreamOperator, StreamOperatorManagement};
/// This transformation applies a non-linear function to the input signal, which can be used for various purposes such as distortion effects in audio processing, or for implementing activation functions in neural networks. 
/// The non-linearity is applied element-wise to the input signal, and the output is of the same length as the input. 
/// The specific non-linear function can be defined by the user, allowing for flexibility in how the transformation is applied to the signal.

use crate::prelude::{ErrorsJSL, IsAnalytic, IsLinearOperatable};

/// The `sigmoid` function is a common non-linear activation function that maps input values to a range between 0 and 1. 
/// It is defined as `sigmoid(x) = 1 / (1 + exp(-x))`, where `exp` is the exponential function. 
/// The sigmoid function is often used in machine learning and neural networks to introduce non-linearity into the model, allowing it to learn complex patterns in the data. 
/// In the context of signal processing, applying a sigmoid non-linearity can create distortion effects or be used for dynamic range compression.
pub fn sigmoid<T: IsAnalytic>(x: T) -> T {
    T::one() / (T::one() + (-x).f_exp())
}

/// Hard clipping is a simple non-linear transformation that limits the amplitude of the input signal to a specified threshold.
pub fn hard_clip_real(x: f64, threshold: f64) -> f64 {  
    if x > threshold {
        threshold
    } else if x < -threshold {
        -threshold
    } else {
        x   
    }
}
pub fn hard_clip_complex(x: Complex<f64>, threshold: f64) -> Complex<f64> {  
    let mag = x.norm();
    let phase = x.arg();
    let clipped_mag = if mag > threshold { threshold } else { mag };
    Complex::from_polar(clipped_mag, phase)
}

/// Quantization is a non-linear transformation that maps a continuous range of values to a finite set of discrete levels.
/// This assumes the input is normalized to the range [-1, 1]. The `levels` parameter specifies the number of quantization levels, which determines the resolution of the quantization process.
/// For real-valued inputs, the quantization process involves dividing the range of possible input values into a specified number of levels and mapping each input value to the nearest quantization level. 
/// For complex-valued inputs, the quantization process is applied separately to the real and imaginary parts of the complex number, resulting in a quantized complex output where both the real and imaginary components have been transformed by the quantization process.
pub fn quantize_real(x: f64, levels: usize) -> f64 {
    let step = 2.0 / (levels as f64 - 1.0);
    let quantized = (x + 1.0) / step;
    let quantized_rounded = quantized.round();
    quantized_rounded * step - 1.0
}
pub fn quantize_complex(x: Complex<f64>, levels: usize) -> Complex<f64> {
    let real_quantized = quantize_real(x.re, levels);
    let imag_quantized = quantize_real(x.im, levels);
    Complex::new(real_quantized, imag_quantized)
}

/// Here is the struct which is simply a wrapper around a non-linearity function that implements the `StreamOperator` trait, allowing it to be used in a streaming context where it can process blocks of input samples and produce corresponding output samples. 
/// The `NonLinearityTransformer` struct takes a non-linearity function as an argument during initialization, and the `process` method applies this function to each sample in the input signal, producing  a transformed output signal.    
pub struct NonLinearityTransformer<T: IsLinearOperatable> {
    non_linearity_fn: fn(T) -> T,
}

impl<T: IsAnalytic> NonLinearityTransformer<T> {
    pub fn new(non_linearity_fn: Option<fn(T) -> T>) -> Self {
        Self {
            non_linearity_fn: non_linearity_fn.unwrap_or(sigmoid),
        }
    }
    pub fn evaluate(&self, x: T) -> T {
        (self.non_linearity_fn)(x)
    }
}


/// For real-valued inputs, the non-linearity is applied directly to each sample in the input signal, producing a real-valued output signal.
impl StreamOperatorManagement for NonLinearityTransformer<f64> {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl StreamOperator<f64, f64> for NonLinearityTransformer<f64> {
    fn process(&mut self, input: &[f64]) -> Result<Option<Vec<f64>>, ErrorsJSL> {
        if input.is_empty() {
            return Ok(None);
        }
        Ok(Some(input.iter().map(|&x| self.evaluate(x)).collect()))
    }
    fn flush(&mut self) -> Result<Option<Vec<f64>>, ErrorsJSL> {
        Ok(None)
    }
}

/// For complex-valued inputs, the non-linearity is applied separately to the real and imaginary parts of each sample, producing a complex-valued output signal where both the real and imaginary components have been transformed by the non-linear function.
impl StreamOperatorManagement for NonLinearityTransformer<Complex<f64>> {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl StreamOperator<Complex<f64>, Complex<f64>> for NonLinearityTransformer<Complex<f64>> {
    fn process(&mut self, input: &[Complex<f64>]) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        if input.is_empty() {
            return Ok(None);
        }
        Ok(Some(input.iter().map(|&x| self.evaluate(x)).collect()))
    }
    fn flush(&mut self) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn load_csv_real(path: &str) -> Vec<f64> {
        let path = format!("{}/src/dsp/test_data/{path}", env!("CARGO_MANIFEST_DIR"));
        fs::read_to_string(path)
            .expect("read golden csv")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<f64>().expect("valid f64 csv value"))
            .collect()
    }

    fn load_csv_complex(path: &str) -> Vec<Complex<f64>> {
        let path = format!("{}/src/dsp/test_data/{path}", env!("CARGO_MANIFEST_DIR"));
        fs::read_to_string(path)
            .expect("read golden csv")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let mut parts = l.split(',').map(|s| s.trim().parse::<f64>().expect("valid f64"));
                let re = parts.next().expect("real part");
                let im = parts.next().expect("imag part");
                Complex::new(re, im)
            })
            .collect()
    }

    fn assert_vec_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() <= tol, "actual={x}, golden={y}");
        }
    }

    fn assert_vec_complex_close(a: &[Complex<f64>], b: &[Complex<f64>], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x.re - y.re).abs() <= tol && (x.im - y.im).abs() <= tol,
                "actual={x}, golden={y}"
            );
        }
    }

    #[test]
    fn test_sigmoid_real_golden_len7() {
        let input = load_csv_real("nonlinearity_input_real_len7.csv");
        let golden = load_csv_real("nonlinearity_sigmoid_real_len7.csv");
        let output: Vec<f64> = input.into_iter().map(sigmoid).collect();
        assert_vec_close(&output, &golden, 1e-12);
    }

    #[test]
    fn test_hard_clip_real_golden_len7() {
        let input = load_csv_real("nonlinearity_input_real_len7.csv");
        let golden = load_csv_real("nonlinearity_hard_clip_real_thr0p8_len7.csv");
        let output: Vec<f64> = input.into_iter().map(|x| hard_clip_real(x, 0.8)).collect();
        assert_vec_close(&output, &golden, 1e-12);
    }

    #[test]
    fn test_hard_clip_complex_golden_len7() {
        let input = load_csv_complex("nonlinearity_input_complex_len7.csv");
        let golden = load_csv_complex("nonlinearity_hard_clip_complex_thr1p0_len7.csv");
        let output: Vec<Complex<f64>> = input
            .into_iter()
            .map(|x| hard_clip_complex(x, 1.0))
            .collect();
        assert_vec_complex_close(&output, &golden, 1e-12);
    }

    #[test]
    fn test_quantize_real_golden_len7() {
        let input = load_csv_real("nonlinearity_input_real_len7.csv");
        let golden = load_csv_real("nonlinearity_quantize_real_levels8_len7.csv");
        let output: Vec<f64> = input.into_iter().map(|x| quantize_real(x, 8)).collect();
        assert_vec_close(&output, &golden, 1e-12);
    }

    #[test]
    fn test_quantize_complex_golden_len7() {
        let input = load_csv_complex("nonlinearity_input_complex_len7.csv");
        let golden = load_csv_complex("nonlinearity_quantize_complex_levels8_len7.csv");
        let output: Vec<Complex<f64>> = input
            .into_iter()
            .map(|x| quantize_complex(x, 8))
            .collect();
        assert_vec_complex_close(&output, &golden, 1e-12);
    }

    #[test]
    fn test_non_linearity_transformer_sigmoid_real_golden_len7() {
        let input = load_csv_real("nonlinearity_input_real_len7.csv");
        let golden = load_csv_real("nonlinearity_sigmoid_real_len7.csv");
        let mut dut = NonLinearityTransformer::<f64>::new(None);
        let output = dut.process(&input).unwrap().unwrap();
        assert_vec_close(&output, &golden, 1e-12);
    }

    #[test]
    fn test_non_linearity_transformer_custom_complex_golden_len7() {
        let input = load_csv_complex("nonlinearity_input_complex_len7.csv");
        let golden = load_csv_complex("nonlinearity_hard_clip_complex_thr1p0_len7.csv");
        fn clip1(x: Complex<f64>) -> Complex<f64> {
            hard_clip_complex(x, 1.0)
        }
        let mut dut = NonLinearityTransformer::<Complex<f64>>::new(Some(clip1));
        let output = dut.process(&input).unwrap().unwrap();
        assert_vec_complex_close(&output, &golden, 1e-12);
    }
}
