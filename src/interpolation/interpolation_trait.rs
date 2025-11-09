use crate::prelude::*;

pub trait InterpolationTrait<'a> : Sized{

    /// Setters and getters for required fields
    fn get_x_values(&self) -> VR1D<'a>;
    fn set_x_values(&mut self,x : VR1D<'a>);
    fn get_y_values(&self) -> VR1D<'a>;
    fn set_y_values(&mut self,y : VR1D<'a>);
    fn get_cached_correlation(&self) -> bool;
    fn set_cached_correlation(&mut self,x : bool);
    fn get_jsav(&self) -> usize;
    fn set_jsav(&mut self,x : usize);
    fn get_dj(&self) -> usize;
    fn set_dj(&mut self, x : usize);

    /// Check if x range is ascending
    fn x_range_is_ascending(&self) -> bool{
        let x_values = self.get_x_values();
        x_values.last()>=x_values.first()
    }

    /// Configures the interpolator over input vector x and output vector y.
    /// Thie function checks to make sure x and y are the same length and greater than 2
    fn configure(&mut self) -> Result<(),ErrorsJSL>{
        let n = self.get_x_values().len();
        let mm = self.get_y_values().len();
        if (n != mm) || n < 2{
            return Err(ErrorsJSL::IncompatibleArraySizes((n,mm)));
        }
        self.set_jsav(0);
        self.set_cached_correlation(false);
        self.set_dj(std::cmp::min(1, (n as f64).powf(0.25).floor() as usize));
        return Ok(())
    }

    /// Given a value x, return a value j such that x is (insofar as possible) centered in the subrange x[j..j+mm-1], where x_values is the stored pointer.  
    /// The values in x_values must be monotonic, either increasing or decreasing.  The return value is not less thanb 0, nor greater than n-1.
    fn locate(&mut self, x : RealNumber) -> Result<usize,ErrorsJSL>{
        let x_values = self.get_x_values();
        let mut ju  = x_values.len()-1;
        let mut jm : usize;
        let mut jl = 0;
        // Is the table increasing?
        let ascnd = self.x_range_is_ascending();
        // Converge in on midpoint by bisection
        while ju-jl > 1{ 
            // Get the midpoint
            jm = ju+jl >> 1; 
            // Compare point with bounds
            if (x >= x_values[jm]) == ascnd{
                jl = jm; // Replace the lower limit
            }else{
                ju = jm; // Replace upper limit
            }
        }
        // Cache the hunt or locate decision for next time (speed up optimization)
        self.set_cached_correlation((jl as i64 - self.get_jsav() as i64).abs() as usize <= self.get_dj());
        self.set_jsav(jl);
        return Ok(jl)
    }

    /// Given a value at x, return a value j such tat x is (insofar as possible) centered in the subrrange x_values[j..j+mm-1].  
    /// The values in x_values must be monotonic, either increasing or decreasing. 
    /// The returned value is not less than 0, nor greater than n-1.
    fn hunt(&mut self, x : RealNumber) -> Result<usize,ErrorsJSL>{
        let x_values = self.get_x_values();
        let n = x_values.len();
        let mut jl = self.get_jsav();
        let mut ju : usize;
        let mut inc = 1;
        let mut jm : usize;
        // Is the table increasing?
        let ascnd = self.x_range_is_ascending();
        // Is the input guess useful? If not, go immidiately to bisection
        if jl > n-1{
            jl = 0;
            ju = n-1;
        }else{
            if (x >= x_values[jl]) == ascnd{ // Hunt up
                loop{
                    ju = jl + inc;
                    if ju >= n-1{
                        ju = n-1; // Off end of table
                        break;
                    }else if (x < x_values[ju]) == ascnd {
                        break; // Found bracket
                    }else{
                        // Not done yet
                        jl = ju;
                        inc += inc; // Increaing quadratically
                    }
                }
            }else{ // Hunt down
                ju = jl;
                loop{
                    if inc >= jl{
                        jl = 0;
                        break;
                    }
                    jl = jl - inc;
                    if (x >= x_values[jl]) == ascnd {
                        break; // Found bracket
                    }else{
                        // Not done yet
                        ju = jl;
                        inc += inc; // Increaing quadratically
                    }
                }
            }
        }
        // Hunt is done so do final bisection phase
        while ju-jl > 1{
            jm = (ju+jl) >> 1;
            if (x >= x_values[jm]) == ascnd{
                jl = jm;
            }else{
                ju = jm;
            }
        }

        // Cache the hunt or locate decision for next time (speed up optimization)
        self.set_cached_correlation((jl as i64 - self.get_jsav() as i64).abs() as usize <= self.get_dj());
        self.set_jsav(jl);
        return Ok(jl)
    }

    /// This is where the interpolation gets computed based on the methoed
    fn raw_interpolate(&mut self, jlo : usize, x : RealNumber) -> Result<RealNumber,ErrorsJSL>;

    /// Top level wrapper for evaluation
    fn interpolate_at(&mut self, x : RealNumber) -> Result<RealNumber,ErrorsJSL>{
        let jlo = if self.get_cached_correlation(){
            self.hunt(x)?
        }else{
            self.locate(x)?
        };
        self.raw_interpolate(jlo, x)
    }
}