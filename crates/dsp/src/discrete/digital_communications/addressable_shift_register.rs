//! Addressable shift register: a fixed-length delay line whose every stage can be read or written
//! by address, the building block of tapped delay lines, FIR structures and LFSR-like blocks.
//!
//! Address 0 is the newest stage and address `capacity - 1` the oldest. Pushing a value shifts every
//! stage by one (the old contents move to higher addresses) and returns the value that falls off
//! the end. The implementation is a circular buffer, so a push is `O(1)` regardless of length.

use crate::discrete::stream_operator::{StreamOperator, StreamOperatorManagement};
use lib_jsl_core::ErrorsJSL;

/// A shift register of `capacity` stages with random access to every stage.
#[derive(Clone, Debug, PartialEq)]
pub struct AddressableShiftRegister<T: Clone> {
    data: Vec<T>,
    /// Index in `data` of the newest stage (address 0).
    head: usize,
    initial: T,
}

impl<T: Clone> AddressableShiftRegister<T> {
    /// A register of `capacity` stages (at least 1), every stage holding `initial`.
    pub fn new(capacity: usize, initial: T) -> Result<Self, ErrorsJSL> {
        if capacity == 0 {
            return Err(ErrorsJSL::InvalidInputRange("A shift register needs at least one stage"));
        }
        Ok(AddressableShiftRegister { data: vec![initial.clone(); capacity], head: 0, initial })
    }

    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    fn index(&self, address: usize) -> usize {
        (self.head + address) % self.data.len()
    }

    /// Shifts in `value` at address 0 and returns the value shifted out of the last stage.
    pub fn push(&mut self, value: T) -> T {
        // The oldest stage is the one just before the head; overwriting it makes it the new head.
        self.head = (self.head + self.data.len() - 1) % self.data.len();
        std::mem::replace(&mut self.data[self.head], value)
    }

    /// The value at `address` (0 = newest), or `None` if out of range.
    pub fn get(&self, address: usize) -> Option<&T> {
        if address < self.data.len() { Some(&self.data[self.index(address)]) } else { None }
    }

    /// Overwrites the value at `address`; fails if the address is out of range.
    pub fn set(&mut self, address: usize, value: T) -> Result<(), ErrorsJSL> {
        if address >= self.data.len() {
            return Err(ErrorsJSL::InvalidInputRange("Shift register address out of range"));
        }
        let i = self.index(address);
        self.data[i] = value;
        Ok(())
    }

    /// The values at several addresses (a tapped delay line read). Fails on any bad address.
    pub fn taps(&self, addresses: &[usize]) -> Result<Vec<T>, ErrorsJSL> {
        addresses.iter().map(|&a| self.get(a).cloned().ok_or(ErrorsJSL::InvalidInputRange("Shift register address out of range"))).collect()
    }

    /// Iterates from the newest stage to the oldest.
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        (0..self.data.len()).map(move |a| &self.data[self.index(a)])
    }

    /// The contents ordered newest first.
    pub fn to_vec(&self) -> Vec<T> {
        self.iter().cloned().collect()
    }

    /// Resets every stage to the initial value.
    pub fn clear(&mut self) {
        for v in self.data.iter_mut() {
            *v = self.initial.clone();
        }
        self.head = 0;
    }
}

impl<T: Clone> StreamOperatorManagement for AddressableShiftRegister<T> {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.clear();
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

/// As a stream operator the register is a pure delay: each output is the input from `capacity`
/// samples earlier (the initial value for the first `capacity` outputs).
impl<T: Clone> StreamOperator<T, T> for AddressableShiftRegister<T> {
    fn process(&mut self, data_in: &[T]) -> Result<Option<Vec<T>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        Ok(Some(data_in.iter().map(|x| self.push(x.clone())).collect()))
    }

    fn flush(&mut self) -> Result<Option<Vec<T>>, ErrorsJSL> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_shifts_and_returns_the_dropped_value() {
        let mut r = AddressableShiftRegister::new(3, 0).unwrap();
        assert_eq!(r.push(1), 0);
        assert_eq!(r.push(2), 0);
        assert_eq!(r.push(3), 0);
        assert_eq!(r.to_vec(), vec![3, 2, 1]);
        assert_eq!(r.push(4), 1, "the oldest value falls off the end");
        assert_eq!(r.to_vec(), vec![4, 3, 2]);
        assert_eq!((r.get(0), r.get(2), r.get(3)), (Some(&4), Some(&2), None));
    }

    #[test]
    fn addressing_set_taps_and_wraparound() {
        let mut r = AddressableShiftRegister::new(5, 0.0).unwrap();
        for i in 1..=12 {
            r.push(i as f64);
        }
        assert_eq!(r.to_vec(), vec![12.0, 11.0, 10.0, 9.0, 8.0], "correct after many wraps of the circular buffer");
        r.set(2, -1.0).unwrap();
        assert_eq!(r.taps(&[0, 2, 4]).unwrap(), vec![12.0, -1.0, 8.0]);
        assert!(r.set(5, 0.0).is_err());
        assert!(r.taps(&[1, 9]).is_err());
        r.push(13.0);
        assert_eq!(r.to_vec(), vec![13.0, 12.0, 11.0, -1.0, 9.0], "writes shift along with the data");
    }

    #[test]
    fn behaves_like_a_delay_line_through_the_stream_interface() {
        let mut r = AddressableShiftRegister::new(4, 0).unwrap();
        let y = r.process(&[1, 2, 3, 4, 5, 6]).unwrap().unwrap();
        assert_eq!(y, vec![0, 0, 0, 0, 1, 2]);
        StreamOperatorManagement::reset(&mut r).unwrap();
        assert_eq!(r.to_vec(), vec![0; 4]);
        assert!(r.process(&[]).unwrap().is_none() && r.flush().unwrap().is_none());
    }

    #[test]
    fn single_stage_and_invalid_capacity() {
        let mut r = AddressableShiftRegister::new(1, 'a').unwrap();
        assert_eq!(r.push('b'), 'a');
        assert_eq!(r.push('c'), 'b');
        assert_eq!(r.to_vec(), vec!['c']);
        assert!(AddressableShiftRegister::new(0, 0).is_err());
    }
}
