use lib_jsl_core::ErrorsJSL;
/// The histogram will take incoming data, place it into discrete bins, and count the hits in each bin
/// The bin_labels are labeled by the minimum boundary of the bin
/// e.g. bin_value = 1, with a bin width of 1, corresponds to hits in  the interval [1,2).
pub struct Histogram{
    bin_labels : Vec<f64>,
    bin_counts : Vec<usize>,
    total : usize,
    bin_width : f64
}

impl Histogram{
    /// Creates a new histogram that spans `[bin_min, bin_max]` with `n_bins`
    /// Note the bin labels represent the lower edge, so bin_max is not represented in the bin_labels
    pub fn new(bin_min : f64, bin_max : f64, n_bins : usize)->Self{
        let span = bin_max-bin_min;
        let bin_width = span / (n_bins) as f64;
        let bin_labels = (0..n_bins).map(|x| x as f64*bin_width).collect();
        let bin_counts = (0..n_bins).map(|_| 0).collect();
        let total = 0;
        Histogram { bin_labels, bin_counts, total, bin_width}
    }
    /// Reset the counts in each bin
    pub fn reset(&mut self){
        self.bin_counts.iter_mut().for_each(|x| *x = 0);
    }
    /// Find the bin of the value and increment the count
    /// Will fail if the sample exceeds the bin_min and bin_max
    pub fn count(&mut self, value : f64) -> Result<(),ErrorsJSL>{
        let bin_index_raw = (value-self.bin_labels.first().expect("")) / self.bin_width;
        if bin_index_raw < 0.0{
            return Err(ErrorsJSL::InvalidInputRange("Input value is less than minimum bin value"));
        }
        else if bin_index_raw >= (self.bin_labels.len()+1) as f64{
            return Err(ErrorsJSL::InvalidInputRange("Input value is greater than maximum bin value"));
        }
        self.bin_counts[bin_index_raw.floor() as usize] += 1;
        self.total += 1;
        Ok(())
    }
    /// Getters
    pub fn get_bin_labels(&self) -> &Vec<f64>{
        &self.bin_labels
    }
    /// Get counts without dividing by total
    pub fn get_counts_raw(&self) -> &Vec<usize>{
        &self.bin_counts
    }
    /// Get counts after dividing by total.  Will fail if total = 0;
    pub fn get_counts_normalized(&self) -> Result<Vec<f64>,ErrorsJSL>{
        if self.total == 0{
            return Err(ErrorsJSL::RuntimeError("Not bin counts have yet been recorded"));
        }
        let scale = 1.0/self.total as f64;
        Ok(self.bin_counts.iter().map(|&x| (x as f64)*scale).collect())
    }
    /// Get the total
    pub fn get_total(&self) -> usize{
        self.total
    }
    /// Get the bin width
    pub fn get_bin_width(&self) -> f64{
        self.bin_width
    }
    /// These functions return vectors of tuples (bin_label,count)
    /// In either raw or normalized format
    pub fn get_counts_raw_paired(&self) -> Vec<(f64,usize)>{
        self.bin_labels.iter().zip(self.bin_counts.iter()).map(|(&x,&y)| (x,y)).collect::<Vec<(f64,usize)>>()
    }
    pub fn get_counts_normalized_paired(&self) -> Result<Vec<(f64,f64)>,ErrorsJSL>{
        let counts = self.get_counts_normalized()?;
        Ok(self.bin_labels.iter().zip(counts.iter()).map(|(&x,&y)| (x,y)).collect::<Vec<(f64,f64)>>())
    }
}

#[cfg(test)]
mod test{
    use super::*;
    use crate::{distributions::gaussian_distribution_box_muller_vec, uniform_generator::UniformRNG};
    #[test]
    fn test_histogram(){
        let mut rng = UniformRNG::from_seed(0);
        let mut dut = Histogram::new(0.0, 10.0, 10);
        let n_samples = 1_000_000;
        gaussian_distribution_box_muller_vec(n_samples,&mut rng, 5.0, 1.0).iter().for_each(|&x| {let _ = dut.count(x);});
        // The probabilities of N(5, 1) falling in each unit bin (Φ(k − 4) − Φ(k − 5)); the
        // sampling error with 10⁶ samples is below 5e-4 per bin.
        let expected = [3.17e-5, 1.3499e-3, 2.1400e-2, 1.35905e-1, 3.41345e-1, 3.41345e-1, 1.35905e-1, 2.1400e-2, 1.3499e-3, 3.17e-5];
        let counts_norm = dut.get_counts_normalized().unwrap();
        assert!(counts_norm.iter().zip(expected).all(|(c, e)| (c - e).abs() < 1e-3), "{counts_norm:?}");
        assert!((counts_norm.iter().sum::<f64>() - 1.0).abs() < 1e-4);
        dbg!(_ = dut.get_counts_normalized_paired());
    }   
}
