use crate::prelude::*;

pub trait InterpolationTrait<'a> : Sized{

    /// Setters and getters for required fields
    fn get_x_values(&self) -> VR1D<'a>;
    fn set_x_values(&mut self,x : VR1D<'a>);
    fn get_y_values(&self) -> VR1D<'a>;
    fn set_y_values(&mut self,y : VR1D<'a>);
    fn get_cached_location(&self) -> Option<usize>;
    fn set_cached_location(&mut self,x : usize);

    /// Check if x range is ascending
    fn x_range_is_ascending(&self) -> bool{
        let x_values = self.get_x_values();
        x_values.last()>=x_values.first()
    }

    /// Thie function checks to make sure x and y are the same length and greater than 2
    fn verify_sizes(&mut self, limit : usize) -> Result<(),ErrorsJSL>{
        let n = self.get_x_values().len();
        let mm = self.get_y_values().len();
        if (n != mm) || n < limit{
            return Err(ErrorsJSL::IncompatibleArraySizes((n,mm)));
        }
        return Ok(())
    }

    /// Finds the index immidiately before x, optionally relying on the cache of the previous value
    /// The location will not exceed N-2 so it always indicates to two valid sequential indexes, even if at the end
    fn locate(&mut self, x : f64) -> Result<usize,ErrorsJSL>{
        let x_values = self.get_x_values();
        let mut jm : usize;

        // Check if cached value has the location 
        let mut jl = self.get_cached_location().unwrap_or(0);
        if x < x_values[jl]{
            jl = 0; // Unknown start point
        }
        // Either the next or the end
        let mut ju  = jl+1;
        if ju >= x_values.len(){
            return Ok(x_values.len()-2); // Start just before end
        }else if x < x_values[ju]{
            return Ok(jl); // Cache hit
        }else{
            ju = x_values.len()-1; // Unknown upper bound
        }

        // No cache hit, proceed with bisection
        while ju-jl > 1{ 
            // Get the midpoint
            jm = ju+jl >> 1; 
            // Compare point with bounds
            if x >= x_values[jm]{
                jl = jm; // Replace the lower limit
            }else{
                ju = jm; // Replace upper limit
            }
        }
        // Cache the hunt or locate decision for next time (speed up optimization)
        self.set_cached_location(jl);
        return Ok(std::cmp::min(jl,x_values.len()-1))
    }


    /// This is where the interpolation gets computed based on the methoed
    fn raw_interpolate(&mut self, jlo : usize, x : f64) -> Result<f64,ErrorsJSL>;

    /// Top level wrapper for evaluation
    fn interpolate_at(&mut self, x : f64) -> Result<f64,ErrorsJSL>{
        let jlo = self.locate(x)?;
        self.raw_interpolate(jlo, x)
    }
}