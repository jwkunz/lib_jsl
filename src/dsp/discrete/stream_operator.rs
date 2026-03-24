use crate::prelude::ErrorsJSL;

/// A stream operator trait indicates a standard interface for processing streams of data, such as audio or video streams. It defines a set of methods that can be implemented by different types of stream operators, allowing them to be used interchangeably in a streaming pipeline. 
/// The trait typically includes methods for initializing the operator, processing input data, and producing output data. 
/// By defining a common interface, the stream operator trait enables flexibility and modularity in the design of streaming applications, allowing developers to easily swap out different operators or add new ones without affecting the overall structure of the pipeline. 
/// This promotes code reuse and makes it easier to maintain and extend streaming applications over time.


pub trait StreamOperatorManagement{
    /// Resets the internal state of the operator, preparing it for a new stream of data. This method is typically called at the beginning of a new stream or when the operator needs to be reinitialized.
    fn reset(&mut self) -> Result<(), ErrorsJSL>;
    /// Finalizes the processing of the stream, performing any necessary cleanup or final computations. This method is typically called at the end of a stream or when the operator needs to be shut down.
    fn finalize(&mut self) -> Result<(), ErrorsJSL>;
}

pub trait StreamOperator<Input, Output> : StreamOperatorManagement {
    /// Processes a chunk of input data and produces a corresponding chunk of output data. The input and output types can be defined based on the specific requirements of the application, such as audio samples, video frames, or other types of data. 
    /// This method is called repeatedly as new data becomes available in the stream.
    fn process(&mut self, data_in: &[Input]) -> Result<Option<Vec<Output>>, ErrorsJSL>;
    /// Flushes any remaining data in the operator's internal buffers and produces any final output data. This method is typically called at the end of a stream or when the operator needs to be shut down, allowing it to perform any necessary cleanup or final computations before the stream is terminated.
    fn flush(&mut self) -> Result<Option<Vec<Output>>, ErrorsJSL>;
}