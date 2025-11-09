use crate::interpolation::interpolation_trait::InterpolationTrait;
use crate::prelude::*;

#[derive(Debug)]
pub struct LinearInterpolator<'a>{
    jsav : usize,
    cor : bool,
    dj : usize,
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
    fn get_cached_correlation(&self) -> bool{
        self.cor
    }
    fn set_cached_correlation(&mut self,x : bool){
        self.cor = x
    }
    fn get_jsav(&self) -> usize{
        self.jsav
    }
    fn set_jsav(&mut self,x : usize){
        self.jsav = x
    }
    fn get_dj(&self) -> usize{
        self.dj
    }
    fn set_dj(&mut self, x : usize){
        self.dj = x
    }
    fn raw_interpolate(&mut self, jlo : usize, x : RealNumber) -> Result<RealNumber,ErrorsJSL> {
        
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
    pub fn new(x_values : VR1D<'a>,   y_values : VR1D<'a>) -> Result<Self,ErrorsJSL>{
        let mut result = LinearInterpolator{
            jsav:0,
            cor:false,
            dj: 0,
            x_values,
            y_values
        };
        result.configure()?;
        Ok(result)
    }
}

#[cfg(test)]
mod test{
    use super::*;

    #[test]
    fn tb_linear_interpolator() -> Result<(),ErrorsJSL>{

        let x = R1D::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let y = R1D::from_vec(vec![1.0, 2.0, 0.0, -1.0]);

        let mut dut = LinearInterpolator::new(x.view(), y.view())?;
        let y_interp = dut.interpolat_at(0.5)?;
        assert_eq!(y_interp,1.5);
        let y_interp = dut.interpolat_at(1.5)?;
        assert_eq!(y_interp,1.0);
        Ok(())
    }
}