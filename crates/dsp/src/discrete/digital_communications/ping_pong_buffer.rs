//! Ping-pong (double) buffer: one buffer fills with incoming samples while the other, already
//! full, is processed, so a producer and a consumer never touch the same memory.
//!
//! The producer pushes samples; when the active buffer holds `block_len` samples the roles swap and
//! that block becomes available to the consumer. If the producer fills a second block before the
//! consumer has released the first, the new block cannot be stored: the push reports an overrun and
//! the samples are dropped (and counted) rather than corrupting the block being read.
//!
//! See the book entry [DSP: synchronisation, communications, radar and controls](https://jwkunz.github.io/lib_jsl/book/comms/index.html#ping-pong-buffer).

use crate::discrete::stream_operator::{StreamOperator, StreamOperatorManagement};
use lib_jsl_core::ErrorsJSL;

/// The outcome of pushing samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PushOutcome {
    /// Number of blocks completed (and made ready) by this push.
    pub blocks_completed: usize,
    /// Number of samples dropped because the consumer had not released the ready block.
    pub dropped: usize,
}

/// A double buffer of two `block_len`-sample blocks.
#[derive(Clone, Debug)]
pub struct PingPongBuffer<T: Clone> {
    block_len: usize,
    /// The block being filled by the producer.
    filling: Vec<T>,
    /// The completed block waiting for the consumer, if any.
    ready: Option<Vec<T>>,
    overruns: usize,
}

impl<T: Clone> PingPongBuffer<T> {
    /// A buffer with blocks of `block_len` samples (at least 1).
    pub fn new(block_len: usize) -> Result<Self, ErrorsJSL> {
        if block_len == 0 {
            return Err(ErrorsJSL::InvalidInputRange("Block length must be at least 1"));
        }
        Ok(PingPongBuffer { block_len, filling: Vec::with_capacity(block_len), ready: None, overruns: 0 })
    }

    pub fn block_len(&self) -> usize {
        self.block_len
    }

    /// Pushes samples; completed blocks swap into the ready slot. While a block is waiting, the
    /// filling buffer can complete one more block but then further samples are dropped.
    pub fn push(&mut self, samples: &[T]) -> PushOutcome {
        let mut outcome = PushOutcome::default();
        for s in samples {
            if self.filling.len() == self.block_len {
                // A full block is parked in the filling slot because the ready slot was occupied.
                if self.ready.is_none() {
                    self.ready = Some(std::mem::replace(&mut self.filling, Vec::with_capacity(self.block_len)));
                } else {
                    outcome.dropped += 1;
                    continue;
                }
            }
            self.filling.push(s.clone());
            if self.filling.len() == self.block_len && self.ready.is_none() {
                self.ready = Some(std::mem::replace(&mut self.filling, Vec::with_capacity(self.block_len)));
                outcome.blocks_completed += 1;
            } else if self.filling.len() == self.block_len {
                outcome.blocks_completed += 1; // parked until the consumer frees the ready slot
            }
        }
        self.overruns += outcome.dropped;
        outcome
    }

    /// True if a completed block is waiting for the consumer.
    pub fn is_ready(&self) -> bool {
        self.ready.is_some()
    }

    /// The completed block, if any, without releasing it.
    pub fn peek(&self) -> Option<&[T]> {
        self.ready.as_deref()
    }

    /// Takes the completed block, freeing the slot (a parked full block, if any, becomes ready).
    pub fn take(&mut self) -> Option<Vec<T>> {
        let block = self.ready.take()?;
        if self.filling.len() == self.block_len {
            self.ready = Some(std::mem::replace(&mut self.filling, Vec::with_capacity(self.block_len)));
        }
        Some(block)
    }

    /// Number of samples currently in the filling buffer.
    pub fn filled(&self) -> usize {
        self.filling.len()
    }

    /// Total samples dropped over the buffer's lifetime.
    pub fn overruns(&self) -> usize {
        self.overruns
    }

    /// Discards all data and the overrun count.
    pub fn clear(&mut self) {
        self.filling.clear();
        self.ready = None;
        self.overruns = 0;
    }
}

impl<T: Clone> StreamOperatorManagement for PingPongBuffer<T> {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.clear();
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

/// As a stream operator the buffer re-blocks a stream: it emits whole blocks as they complete (so
/// the output is the input delayed by up to one block) and never overruns because it drains
/// immediately. `flush` emits the final partial block as-is.
impl<T: Clone> StreamOperator<T, T> for PingPongBuffer<T> {
    fn process(&mut self, data_in: &[T]) -> Result<Option<Vec<T>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        let mut out = Vec::new();
        for chunk in data_in.chunks(self.block_len) {
            self.push(chunk);
            while let Some(b) = self.take() {
                out.extend(b);
            }
        }
        Ok(if out.is_empty() { None } else { Some(out) })
    }

    fn flush(&mut self) -> Result<Option<Vec<T>>, ErrorsJSL> {
        let mut out = Vec::new();
        while let Some(b) = self.take() {
            out.extend(b);
        }
        out.append(&mut self.filling);
        Ok(if out.is_empty() { None } else { Some(out) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_become_ready_when_full() {
        let mut p = PingPongBuffer::new(4).unwrap();
        assert_eq!(p.push(&[1, 2, 3]), PushOutcome { blocks_completed: 0, dropped: 0 });
        assert!(!p.is_ready() && p.filled() == 3);
        assert_eq!(p.push(&[4, 5]).blocks_completed, 1);
        assert_eq!(p.peek(), Some(&[1, 2, 3, 4][..]));
        assert_eq!(p.filled(), 1, "the overflow went into the other buffer");
        assert_eq!(p.take(), Some(vec![1, 2, 3, 4]));
        assert!(!p.is_ready() && p.take().is_none());
    }

    #[test]
    fn producer_and_consumer_alternate_without_loss() {
        let mut p = PingPongBuffer::new(3).unwrap();
        let mut received = Vec::new();
        for chunk in (1..=30).collect::<Vec<i32>>().chunks(2) {
            p.push(chunk);
            while let Some(b) = p.take() {
                received.extend(b);
            }
        }
        assert_eq!(received, (1..=30).collect::<Vec<_>>());
        assert_eq!(p.overruns(), 0);
    }

    #[test]
    fn overruns_are_counted_and_the_waiting_block_is_untouched() {
        let mut p = PingPongBuffer::new(2).unwrap();
        let o = p.push(&[1, 2, 3, 4, 5, 6]); // block A ready, block B parked, [5, 6] dropped
        assert_eq!(o.dropped, 2);
        assert_eq!(p.overruns(), 2);
        assert_eq!(p.peek(), Some(&[1, 2][..]));
        assert_eq!(p.take(), Some(vec![1, 2]));
        assert_eq!(p.take(), Some(vec![3, 4]), "the parked block becomes ready once the slot frees");
        assert!(p.take().is_none());
        p.push(&[7, 8]);
        assert_eq!(p.take(), Some(vec![7, 8]));
    }

    #[test]
    fn stream_operator_reblocks_and_flush_emits_the_partial_block() {
        let mut p = PingPongBuffer::new(4).unwrap();
        assert!(p.process(&[1, 2, 3]).unwrap().is_none(), "no output until a block completes");
        assert_eq!(p.process(&[4, 5, 6, 7, 8, 9]).unwrap().unwrap(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(p.flush().unwrap().unwrap(), vec![9]);
        assert!(p.flush().unwrap().is_none());
        StreamOperatorManagement::reset(&mut p).unwrap();
        assert_eq!(p.filled(), 0);
    }

    #[test]
    fn invalid_block_length() {
        assert!(PingPongBuffer::<u8>::new(0).is_err());
    }
}
