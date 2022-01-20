macro_rules! impl_rad {
    ($ty: ty, $f: ty, $noisy: ty, $pi: expr) => {
        impl $ty {
            /// 0
            pub const ZERO: Self = Self(<$noisy>::unchecked_new(0.0));
            /// A half turn around a circle, π in radians, 180°
            pub const HALF_TURN: Self = Self(<$noisy>::unchecked_new($pi));
            /// A full turn around a circle, 2×π in radians, 360°
            pub const FULL_TURN: Self = Self(<$noisy>::unchecked_new($pi * 2.0));

            /// Creates a new angle in radians.
            #[inline]
            pub fn new(val: $f) -> Self {
                Self(<$noisy>::new(val))
            }

            /// Creates a new angle, converting from a value in degrees.
            #[inline]
            pub fn from_deg(deg: $f) -> Self {
                Self(<$noisy>::new(deg) * $pi / 180.0)
            }

            /// Gets the value of this angle in radians.
            /// Should not be +-NaN or +-Infinity.
            #[inline]
            pub fn val(self) -> $f {
                self.0.raw()
            }
            /// Gets the value of this angle in degrees.
            /// Should not be +- NaN or +-Infinity.
            #[inline]
            pub fn deg(self) -> $f {
                self.0.raw() * 180.0 / $pi
            }
            /// Returns the magnitude (absolute value) of this angle in radians.
            #[inline]
            pub fn mag(self) -> $f {
                self.0.raw().abs()
            }
        }

        impl ::core::convert::From<$ty> for $f {
            #[inline]
            fn from(val: $ty) -> Self {
                val.0.raw()
            }
        }

        impl ::core::ops::Add for $ty {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                Self(self.0 + rhs.0)
            }
        }
        impl ::core::ops::Sub for $ty {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                Self(self.0 - rhs.0)
            }
        }
        impl ::core::ops::Neg for $ty {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self(-self.0)
            }
        }
        impl ::core::ops::Mul for $ty {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                Self(self.0 * rhs.0)
            }
        }
        impl ::core::ops::Div for $ty {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self {
                Self(self.0 / rhs.0)
            }
        }

        impl ::core::ops::AddAssign for $ty {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0
            }
        }
        impl ::core::ops::SubAssign for $ty {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0
            }
        }
        impl ::core::ops::MulAssign for $ty {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                self.0 *= rhs.0
            }
        }
        impl ::core::ops::DivAssign for $ty {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                self.0 /= rhs.0
            }
        }

        impl ::core::fmt::Debug for $ty {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Debug::fmt(&self.0, f)
            }
        }
        impl ::core::fmt::Display for $ty {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                write!(f, "{}π", self.val() / $pi)
            }
        }
    };
}

use noisy_float::types::{R32, R64};

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rad32(R32);
impl_rad!(Rad32, f32, R32, std::f32::consts::PI);

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rad64(noisy_float::types::R64);
impl_rad!(Rad64, f64, R64, std::f64::consts::PI);

#[cfg(test)]
mod tests {}
