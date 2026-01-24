use crate::interpolation::interpolation_trait::InterpolationTrait;
use crate::prelude::*;

/// This stuct implements simple linear interpolation for y = f(x)

#[derive(Debug)]
pub struct LinearInterpolator<'a>{
    location : Option<usize>,
    x_values : VR1D<'a>,
    y_values : VR1D<'a>
}

impl<'a> InterpolationTrait<'a> for LinearInterpolator<'a>{
    fn get_x_values(&self) -> VR1D<'a>{
        self.x_values
    }
    fn set_x_values(&mut self,x : VR1D<'a>){
        self.x_values = x;
    }
    fn get_y_values(&self) -> VR1D<'a>{
        self.y_values
    }
    fn set_y_values(&mut self,y : VR1D<'a>){
        self.y_values = y;
    }
    fn get_cached_location(&self) -> Option<usize>{
        self.location
    }
    fn set_cached_location(&mut self,x : usize){
        self.location = Some(x)
    }

    /// This is the logic of the linear interpolator
    fn raw_interpolate(&mut self, jlo : usize, x : f64) -> Result<f64,ErrorsJSL> {
        
        let x_values = self.get_x_values();
        let y_values = self.get_y_values();
        if x_values[jlo] == x_values[jlo+1]{ // Table is defective, but can recover
            Ok(y_values[jlo])
        }else{
            let dx1 = x-x_values[jlo];
            let dx2 = x_values[jlo+1]-x_values[jlo];
            let dy = y_values[jlo+1]-y_values[jlo];
            let y = y_values[jlo];
            Ok(y + (dx1/dx2*dy))   
        }
    }
}

impl<'a> LinearInterpolator<'a>{
    /// Initialize with y_values = f(x_values)
    pub fn new(x_values : VR1D<'a>,   y_values : VR1D<'a>) -> Result<Self,ErrorsJSL>{
        let mut result = LinearInterpolator{
            x_values,
            y_values,
            location : None
        };
        result.verify_sizes(2)?;
        Ok(result)
    }
}

#[cfg(test)]
mod test{
    use super::*;

    // Helper for comparing floats
    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
    }

    #[test]
    fn tb_linear_interpolator() -> Result<(),ErrorsJSL>{

        let x = R1D::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let y = R1D::from_vec(vec![1.0, 2.0, 0.0, -1.0]);

        let mut dut = LinearInterpolator::new(x.view(), y.view())?;

        // Exact hits
        let y_interp = dut.interpolate_at(0.0)?;
        assert!(approx_eq(y_interp,1.0,1E-9));
        let y_interp = dut.interpolate_at(1.0)?;
        assert!(approx_eq(y_interp,2.0,1E-9));
        let y_interp = dut.interpolate_at(2.0)?;
        assert!(approx_eq(y_interp,0.0,1E-9));
        let y_interp = dut.interpolate_at(3.0)?;
        assert!(approx_eq(y_interp,-1.0,1E-9));

        // Interpolations
        let y_interp = dut.interpolate_at(0.5)?;
        assert!(approx_eq(y_interp,1.5,1E-9));
        let y_interp = dut.interpolate_at(1.5)?;
        assert!(approx_eq(y_interp,1.0,1E-9));
        let y_interp = dut.interpolate_at(2.5)?;
        assert!(approx_eq(y_interp,-0.5,1E-9));
        let y_interp = dut.interpolate_at(3.5)?;
        assert!(approx_eq(y_interp,-1.5,1E-9));
        let y_interp = dut.interpolate_at(-1.0)?;
        assert!(approx_eq(y_interp,0.0,1E-9));
            
        Ok(())
    }
}