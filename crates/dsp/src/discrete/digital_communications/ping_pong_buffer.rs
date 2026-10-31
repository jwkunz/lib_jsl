//! Ping pong buffer.

pub struct PingPongBuffer<T> {
    _buf_a: Vec<T>,
    _buf_b: Vec<T>,
}
