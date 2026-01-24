use num::{
    Integer,
    traits::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, Num, One, PrimInt, Signed, Zero},
};
use std::ops::{Add, Div, Mul, Sub};
use std::{
    fmt,
    ops::{Neg, Rem},
};

/// A pedagogical integer type stored as decimal digits in a string.
///
/// All arithmetic is performed manually on the string representation, digit by digit.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct TextCodedInteger {
    /// Internal string representation, decimal digits only, optional leading '-' for negatives.
    text: String,
}

impl TextCodedInteger {
    /// Create a new TextCodedInteger from a string (decimal digits only).
    pub fn new<T: ToString>(value: T) -> Self {
        let s = value.to_string();
        if s.chars().all(|c| c.is_ascii_digit() || c == '-') {
            Self { text: s }
        } else {
            panic!("TextCodedInteger can only store decimal digits or a leading '-'");
        }
    }

    /// Helper to check if the number is negative
    fn is_negative(&self) -> bool {
        self.text.starts_with('-')
    }

    /// Helper to return unsigned digits string
    fn digits(&self) -> &str {
        if self.is_negative() {
            &self.text[1..]
        } else {
            &self.text
        }
    }

    /// Manual string addition of two non-negative numbers
    fn add_strings(a: &str, b: &str) -> String {
        let mut result = String::new();
        let mut carry = 0;

        let a_bytes = a.as_bytes();
        let b_bytes = b.as_bytes();

        let mut i = a_bytes.len() as isize - 1;
        let mut j = b_bytes.len() as isize - 1;

        while i >= 0 || j >= 0 || carry > 0 {
            let digit_a = if i >= 0 {
                (a_bytes[i as usize] - b'0') as u8
            } else {
                0
            };
            let digit_b = if j >= 0 {
                (b_bytes[j as usize] - b'0') as u8
            } else {
                0
            };

            let sum = digit_a + digit_b + carry;
            carry = sum / 10;
            result.push(((sum % 10) + b'0') as char);

            i -= 1;
            j -= 1;
        }

        result.chars().rev().collect()
    }

    /// Manual string subtraction: a >= b, non-negative numbers
    fn sub_strings(a: &str, b: &str) -> String {
        let mut result = String::new();
        let mut borrow = 0;

        let a_bytes = a.as_bytes();
        let b_bytes = b.as_bytes();

        let mut i = a_bytes.len() as isize - 1;
        let mut j = b_bytes.len() as isize - 1;

        while i >= 0 {
            let mut digit_a = (a_bytes[i as usize] - b'0') as i8 - borrow;
            let digit_b = if j >= 0 {
                (b_bytes[j as usize] - b'0') as i8
            } else {
                0
            };

            if digit_a < digit_b {
                digit_a += 10;
                borrow = 1;
            } else {
                borrow = 0;
            }

            result.push(((digit_a - digit_b) as u8 + b'0') as char);

            i -= 1;
            j -= 1;
        }

        // Remove leading zeros
        while result.len() > 1 && result.ends_with('0') {
            result.pop();
        }

        result.chars().rev().collect()
    }

    /// Compare absolute values: returns true if self >= other
    fn abs_ge(&self, other: &Self) -> bool {
        let a = self.digits();
        let b = other.digits();
        if a.len() != b.len() {
            a.len() > b.len()
        } else {
            a >= b
        }
    }

    /// Compare two numeric strings a and b (without signs)
    /// Returns true if a >= b
    fn numeric_ge(a: &str, b: &str) -> bool {
        if a.len() != b.len() {
            return a.len() > b.len();
        }
        a >= b // safe now because lengths are equal
    }

    /// Multiply a numeric string by a single digit (0-9)
    fn mul_string_digit(s: &str, d: u8) -> String {
        if d == 0 { return "0".to_string(); }
        let mut carry = 0;
        let mut result = String::new();
        for c in s.bytes().rev() {
            let prod = (c - b'0') as u16 * d as u16 + carry;
            carry = prod / 10;
            result.push(((prod % 10) as u8 + b'0') as char);
        }
        if carry > 0 {
            result.push_str(&carry.to_string().chars().rev().collect::<String>());
        }
        result.chars().rev().collect()
    }


}

// Display trait
impl fmt::Display for TextCodedInteger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text)
    }
}

// Manual arithmetic operators
impl Add for TextCodedInteger {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        match (self.is_negative(), rhs.is_negative()) {
            (false, false) => Self {
                text: Self::add_strings(&self.text, &rhs.text),
            },
            (true, true) => Self {
                text: format!("-{}", Self::add_strings(self.digits(), rhs.digits())),
            },
            (false, true) => {
                if self.abs_ge(&rhs) {
                    Self {
                        text: Self::sub_strings(&self.text, rhs.digits()),
                    }
                } else {
                    Self {
                        text: format!("-{}", Self::sub_strings(rhs.digits(), &self.text)),
                    }
                }
            }
            (true, false) => {
                if rhs.abs_ge(&self) {
                    Self {
                        text: Self::sub_strings(&rhs.text, self.digits()),
                    }
                } else {
                    Self {
                        text: format!("-{}", Self::sub_strings(self.digits(), &rhs.text)),
                    }
                }
            }
        }
    }
}

impl Sub for TextCodedInteger {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self + Self {
            text: if rhs.is_negative() {
                rhs.digits().to_string()
            } else {
                format!("-{}", rhs.text)
            },
        }
    }
}

impl Mul for TextCodedInteger {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let neg = self.is_negative() ^ rhs.is_negative();
        let a = self.digits();
        let b = rhs.digits();

        // Initialize result digits
        let mut result = vec![0; a.len() + b.len()];

        let a_bytes: Vec<u8> = a.bytes().rev().map(|b| b - b'0').collect();
        let b_bytes: Vec<u8> = b.bytes().rev().map(|b| b - b'0').collect();

        // Multiply each digit
        for i in 0..a_bytes.len() {
            let mut carry = 0;
            for j in 0..b_bytes.len() {
                let sum = result[i + j] + a_bytes[i] * b_bytes[j] + carry;
                result[i + j] = sum % 10;
                carry = sum / 10;
            }
            if carry > 0 {
                result[i + b_bytes.len()] += carry;
            }
        }

        // Remove leading zeros
        while result.len() > 1 && *result.last().unwrap() == 0 {
            result.pop();
        }

        let s: String = result
            .into_iter()
            .rev()
            .map(|d| (d + b'0') as char)
            .collect();
        if neg && s != "0" {
            Self::new(format!("-{}", s))
        } else {
            Self::new(s)
        }
    }
}

impl Neg for TextCodedInteger {
    type Output = Self;

    fn neg(self) -> Self::Output {
        if self.is_zero() {
            self
        } else if self.is_negative() {
            Self::new(self.digits())
        } else {
            Self::new(format!("-{}", self.text))
        }
    }
}

impl Div for TextCodedInteger {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let (q, _) = Self::div_mod_manual(&self, &rhs);
        q
    }
}

impl Rem for TextCodedInteger {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let (_, r) = Self::div_mod_manual(&self, &rhs);
        r
    }
}

impl TextCodedInteger {
    /// Manual long division: returns (quotient, remainder)
    /// Full manual long division: returns (quotient, remainder)
    pub fn div_mod_manual(dividend: &Self, divisor: &Self) -> (Self, Self) {
        if divisor.is_zero() {
            panic!("Division by zero");
        }

        let neg_quotient = dividend.is_negative() ^ divisor.is_negative();
        let neg_remainder = dividend.is_negative();

        let mut dividend_abs = dividend.digits().to_string();
        let divisor_abs = divisor.digits();

        if !Self::numeric_ge(&dividend_abs, divisor_abs) {
            return (Self::zero(), dividend.clone());
        }

        let mut quotient = String::new();
        let mut remainder = String::new();

        for c in dividend_abs.chars() {
            remainder.push(c);
            while remainder.starts_with('0') && remainder.len() > 1 {
                remainder.remove(0);
            }

            let mut q_digit = 0u8;
            while Self::numeric_ge(&remainder, divisor_abs) {
                remainder = Self::sub_strings(&remainder, divisor_abs);
                q_digit += 1;
            }

            quotient.push((q_digit + b'0') as char);
        }

        // Remove leading zeros in quotient
        while quotient.starts_with('0') && quotient.len() > 1 {
            quotient.remove(0);
        }

        let quotient = if neg_quotient && quotient != "0" {
            Self::new(format!("-{}", quotient))
        } else {
            Self::new(quotient)
        };

        let remainder = if neg_remainder && remainder != "0" {
            Self::new(format!("-{}", remainder))
        } else {
            Self::new(remainder)
        };

        (quotient, remainder)
    }
}

// Zero and One traits
impl Zero for TextCodedInteger {
    fn zero() -> Self {
        Self::new("0")
    }
    fn is_zero(&self) -> bool {
        self.digits() == "0"
    }
}

impl One for TextCodedInteger {
    fn one() -> Self {
        Self::new("1")
    }
}

// Num trait
impl Num for TextCodedInteger {
    type FromStrRadixErr = std::num::ParseIntError;

    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        if radix != 10 {
            panic!("TextCodedInteger only supports decimal strings for now");
        }
        Ok(Self::new(s))
    }
}

// Signed trait
impl Signed for TextCodedInteger {
    fn abs(&self) -> Self {
        if self.is_negative() {
            Self::new(self.digits())
        } else {
            self.clone()
        }
    }
    fn abs_sub(&self, rhs: &Self) -> Self {
        if self.abs_ge(rhs) {
            Self::new(Self::sub_strings(self.digits(), rhs.digits()))
        } else {
            Self::new(Self::sub_strings(rhs.digits(), self.digits()))
        }
    }
    fn signum(&self) -> Self {
        if self.is_zero() {
            Self::zero()
        } else if self.is_negative() {
            Self::new("-1")
        } else {
            Self::one()
        }
    }
    fn is_positive(&self) -> bool {
        !self.is_negative() && !self.is_zero()
    }
    fn is_negative(&self) -> bool {
        self.is_negative()
    }
}

// Checked arithmetic (optional for pedagogical version)
impl CheckedAdd for TextCodedInteger {
    fn checked_add(&self, rhs: &Self) -> Option<Self> {
        Some(self.clone() + rhs.clone())
    }
}
impl CheckedSub for TextCodedInteger {
    fn checked_sub(&self, rhs: &Self) -> Option<Self> {
        Some(self.clone() - rhs.clone())
    }
}
impl CheckedMul for TextCodedInteger {
    fn checked_mul(&self, _rhs: &Self) -> Option<Self> {
        unimplemented!()
    }
}
impl CheckedDiv for TextCodedInteger {
    fn checked_div(&self, _rhs: &Self) -> Option<Self> {
        unimplemented!()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_text_coded_integer() {
        let a_true = 12345;
        let b_true = -6789;
        let a = TextCodedInteger::new(a_true);
        let b = TextCodedInteger::new(b_true);

        let c = a.clone() + b.clone();
        assert_eq!(c, TextCodedInteger::new(a_true + b_true));
        let c = a.clone() - b.clone();
        assert_eq!(c, TextCodedInteger::new(a_true - b_true));
        let c = a.clone() * b.clone();
        assert_eq!(c, TextCodedInteger::new(a_true * b_true));
        let c = a.clone() / b.clone();
        assert_eq!(c, TextCodedInteger::new(a_true / b_true));
    }
}
