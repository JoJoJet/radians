//! A crate for storing angles.
//!
//! Exports the types [`Rad32`] and [`Rad64`] for dealing with angles in radians,
//! as well as [`Wrap32`] and [`Wrap64`] for angles that automatically wrap around -π and +π.
//!
//! Supports custom formatting in terms of degrees, minutes, and seconds, via the `Rad{32, 64}`.deg() method.

use std::{
    fmt,
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// A floating-point number that serves as the backing value of an [`Angle`].
pub trait Float:
    Copy
    + PartialEq
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Neg<Output = Self>
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
{
    /// Additive identity, 0.
    const ZERO: Self;
    /// Archimedes’ constant (π)
    const PI: Self;
    /// π divided by two.
    const PI_OVER_TWO: Self;
    /// The full circle constant (τ)
    const TAU: Self;
    /// The number 60.
    const SIXTY: Self;
    /// The number 90.
    const NINETY: Self;
    /// The number 180.
    const ONE_EIGHTY: Self;
    /// The number 360.
    const THREE_SIXTY: Self;

    /// Minimum finite value.
    const MIN: Self;
    /// Maximum finite value.
    const MAX: Self;

    /// Returns whether this value is finite (not infinity, not NaN).
    fn is_finite(self) -> bool;
    fn is_sign_positive(self) -> bool;
    #[inline]
    fn is_sign_negative(self) -> bool {
        !self.is_sign_positive()
    }
    /// Modulus operation.
    fn rem_euclid(self, _: Self) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
    /// Truncates the fractional part from this value.
    fn trunc(self) -> Self;

    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;

    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, _: Self) -> Self;

    fn total_eq(self, _: Self) -> bool;

    /// Type that implements [`std::cmp::Ord`], which this floating point type can be trivially converted to.
    type Ord: Ord;
    /// Converts this floating point number to a type that implements total ordering. Conversion should be trivial.  
    ///
    /// For insight on how to implement this, check out the README of https://github.com/notriddle/rust-float-ord.
    /// Note that the implementation in that crate is slightly incorrect, as it will make 0.0 != -0.0
    fn to_ord(self) -> Self::Ord;
}

macro_rules! impl_float {
    ($f: ident, $i: ty) => {
        impl Float for $f {
            const ZERO: $f = 0.0;
            const PI: $f = std::$f::consts::PI;
            const PI_OVER_TWO: $f = std::$f::consts::PI / 2.0;
            const TAU: $f = std::$f::consts::TAU;
            const SIXTY: $f = 60.0;
            const NINETY: $f = 90.0;
            const ONE_EIGHTY: $f = 180.0;
            const THREE_SIXTY: $f = 360.0;
            const MIN: $f = $f::MIN;
            const MAX: $f = $f::MAX;

            #[inline]
            fn is_finite(self) -> bool {
                <$f>::is_finite(self)
            }
            #[inline]
            fn is_sign_positive(self) -> bool {
                <$f>::is_sign_positive(self)
            }
            #[inline]
            fn is_sign_negative(self) -> bool {
                <$f>::is_sign_negative(self)
            }
            #[inline]
            fn rem_euclid(self, rhs: Self) -> Self {
                <$f>::rem_euclid(self, rhs)
            }
            #[inline]
            fn abs(self) -> Self {
                <$f>::abs(self)
            }
            #[inline]
            fn trunc(self) -> Self {
                <$f>::trunc(self)
            }

            fn sin(self) -> Self {
                <$f>::sin(self)
            }
            #[inline]
            fn cos(self) -> Self {
                <$f>::cos(self)
            }
            #[inline]
            fn tan(self) -> Self {
                <$f>::tan(self)
            }

            #[inline]
            fn asin(self) -> Self {
                <$f>::asin(self)
            }
            #[inline]
            fn acos(self) -> Self {
                <$f>::acos(self)
            }
            #[inline]
            fn atan(self) -> Self {
                <$f>::atan(self)
            }
            #[inline]
            fn atan2(self, b: Self) -> Self {
                <$f>::atan2(self, b)
            }

            #[inline]
            fn total_eq(self, rhs: Self) -> bool {
                let a = self.to_bits();
                let b = rhs.to_bits();

                // disregard the sign bit when comparing zeros.
                if a << 1 == 0 {
                    return zero(b);

                    #[cold]
                    fn zero(b: $i) -> bool {
                        b << 1 == 0
                    }
                }
                // compare any other numbers directly.
                else {
                    a == b
                }
            }

            type Ord = $i;
            #[inline]
            fn to_ord(self) -> Self::Ord {
                // assert that the types have equal sizes.
                const _ASSERT: [(); std::mem::size_of::<$f>()] = [(); std::mem::size_of::<$i>()];

                const MSB: $i = 1 << (std::mem::size_of::<$f>() * 8 - 1);
                let bits = self.to_bits();
                if bits & MSB == 0 {
                    // if it's positive, flip the most significant bit.
                    bits | MSB
                } else {
                    // if its negative zero, pretend that its positive zero.
                    if bits << 1 == 0 {
                        return zero();

                        // Benchmarking shows that marking as cold provides a slight performance boost.
                        // This is because -0 is a special case.
                        #[cold]
                        fn zero() -> $i {
                            MSB // this is the result of flipping the most significant bit of +0
                        }
                    }
                    // if it's any other negative number, flip every bit
                    else {
                        !bits
                    }
                }
            }
        }
    };
}

impl_float!(f32, u32);
impl_float!(f64, u64);

/// A unit of measurement for an [`Angle`].
pub trait Unit<F: Float>: Sized {
    const DBG_NAME: &'static str;

    /// Number for a quarter turn around a circle in this unit space.
    const QUARTER_TURN: F;
    /// Number for a half turn around a circle in this unit space.
    const HALF_TURN: F;
    /// Number for a full turn around a circle in this unit space.
    const FULL_TURN: F;

    fn display(val: F, f: &mut fmt::Formatter) -> fmt::Result
    where
        F: fmt::Display;
}

/// An angle backed by a floating-point number.
#[repr(transparent)]
pub struct Angle<F: Float, U: Unit<F>>(F, PhantomData<U>);

impl<F: Float, U: Unit<F>> PartialOrd for Angle<F, U> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<F: Float, U: Unit<F>> Ord for Angle<F, U> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.to_ord().cmp(&other.0.to_ord())
    }
}

impl<F: Float + fmt::Debug, U: Unit<F>> fmt::Debug for Angle<F, U> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple(U::DBG_NAME).field(&self.0).finish()
    }
}

impl<F: Float, U: Unit<F>> Angle<F, U> {
    /// Zero angle, additive identity.
    pub const ZERO: Self = Self(F::ZERO, PhantomData);
    /// Quarter turn around a circle. Equal to π/2 radians or 90°.
    pub const QUARTER_TURN: Self = Self(U::QUARTER_TURN, PhantomData);
    /// Half turn around a circle. Equal to π radians or 180°.
    pub const HALF_TURN: Self = Self(U::HALF_TURN, PhantomData);
    /// Full turn around a circle. Equal to 2π radians or 360°.
    pub const FULL_TURN: Self = Self(U::FULL_TURN, PhantomData);
    /// Minimum finite angle.
    pub const MIN: Self = Self(F::MIN, PhantomData);
    /// Maximum finite angle.
    pub const MAX: Self = Self(F::MAX, PhantomData);

    /// Create a new angle from a raw value.
    /// # Panics
    /// If the value is non-finite (debug mode).
    #[inline]
    pub fn new(val: F) -> Self {
        debug_assert!(val.is_finite());
        Self(val, PhantomData)
    }

    /// Gets the value of this angle.
    #[inline]
    pub fn val_raw(self) -> F {
        self.0
    }
    #[inline]
    pub fn wrap(self) -> Wrap<F, U> {
        Wrap::wrap(self.0)
    }
    /// Returns the magnitude (absolute value) of this angle.
    #[inline]
    pub fn mag(mut self) -> Self {
        self.0 = self.0.abs();
        self
    }
}

pub struct Radians;
impl<F: Float> Unit<F> for Radians {
    const DBG_NAME: &'static str = "Rad";

    const QUARTER_TURN: F = F::PI_OVER_TWO;
    const HALF_TURN: F = F::PI;
    const FULL_TURN: F = F::TAU;

    fn display(val: F, f: &mut fmt::Formatter) -> fmt::Result
    where
        F: fmt::Display,
    {
        if val.is_sign_negative() {
            write!(f, "-")?;
        } else if f.sign_plus() {
            write!(f, "+")?;
        }
        let val = val.abs();
        if val == F::ZERO {
            if let Some(prec) = f.precision() {
                write!(f, "0.{:0<prec$}", 0)
            } else {
                write!(f, "0")
            }
        } else {
            if let Some(prec) = f.precision() {
                write!(f, "{:.prec$}π", val / F::PI)
            } else {
                write!(f, "{}π", val / F::PI)
            }
        }
    }
}

/// An angle measured in radians.
///
/// When formatting with [`Display`](fmt::Display), it will be shown as a multiple of π.
pub type Rad<F> = Angle<F, Radians>;

impl<F: Float> Rad<F> {
    /// Gets the value of this angle in radians.
    #[inline]
    pub fn val(self) -> F {
        self.0
    }
    /// Converts this angle to degrees.
    pub fn deg(self) -> Deg<F> {
        Deg::new(self.0 * (F::ONE_EIGHTY / F::PI))
    }
}

pub struct Degrees;
impl<F: Float> Unit<F> for Degrees {
    const DBG_NAME: &'static str = "Deg";

    const QUARTER_TURN: F = F::NINETY;
    const HALF_TURN: F = F::ONE_EIGHTY;
    const FULL_TURN: F = F::THREE_SIXTY;

    fn display(val: F, f: &mut fmt::Formatter) -> fmt::Result
    where
        F: fmt::Display,
    {
        let deg_frac = val.abs();
        let deg = deg_frac.trunc();

        let min_frac = (deg_frac - deg) * F::SIXTY;
        let min = min_frac.trunc();

        let sec_frac = (min_frac - min) * F::SIXTY;

        if val.is_sign_negative() {
            write!(f, "-")?;
        } else if f.sign_plus() {
            write!(f, "+")?;
        }
        write!(f, "{deg}°{min}'")?;
        if let Some(prec) = f.precision() {
            write!(f, "{sec_frac:.prec$}''")?;
        } else {
            write!(f, "{sec_frac}''")?;
        }

        Ok(())
    }
}

/// An angle measured in degrees.
pub type Deg<F> = Angle<F, Degrees>;

impl<F: Float> Deg<F> {
    /// Gets the value of this angle in degrees.
    #[inline]
    pub fn val(self) -> F {
        self.0
    }
    pub fn rad(self) -> Rad<F> {
        Rad::new(self.0 * (F::PI / F::ONE_EIGHTY))
    }
}

/// An angle that wraps between a negative half turn and a positive half turn.
#[repr(transparent)]
pub struct Wrap<F: Float, U: Unit<F>>(Angle<F, U>);

impl<F: Float + fmt::Debug, U: Unit<F>> fmt::Debug for Wrap<F, U> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Wrap").field(&self.0).finish()
    }
}

impl<F: Float, U: Unit<F>> Wrap<F, U> {
    /// Zero angle, additive identity.
    pub const ZERO: Self = Self(Angle::ZERO);
    /// Half turn around a circle. Equal to π/2 radians or 90°.
    pub const QUARTER_TURN: Self = Self(Angle::QUARTER_TURN);
    /// Half turn around a circle. Equal to π radians or 180°.
    pub const HALF_TURN: Self = Self(Angle::HALF_TURN);
    /// Full turn around a circle. Equal to 2π radians or 360°;
    pub const FULL_TURN: Angle<F, U> = Angle::FULL_TURN;

    /// Creates a new angle, wrapping between a negative half turn and a positive half turn.
    pub fn wrap(val: F) -> Self {
        let val = (-val + U::HALF_TURN).rem_euclid(U::FULL_TURN) - U::HALF_TURN;
        Self(Angle::new(-val))
    }
    /// Creates a new angle, without checking if it's in range.
    #[inline]
    fn new_unchecked(val: F) -> Self {
        Self(Angle::new(val))
    }

    /// Gets the value of this angle.
    #[inline]
    pub fn val_raw(self) -> F {
        self.0 .0
    }
    /// Gets the inner representation of this angle.
    #[inline]
    pub fn inner(self) -> Angle<F, U> {
        self.0
    }
    /// Returns the magnitude (absolute value) of this angle.
    #[inline]
    pub fn mag(self) -> Angle<F, U> {
        self.0.mag()
    }
}

impl<F: Float> Wrap<F, Radians> {
    /// Gets the value of this angle in radians.
    #[inline]
    pub fn val(self) -> F {
        self.0 .0
    }
}
impl<F: Float> Wrap<F, Degrees> {
    /// Gets the value of this angle in degrees.
    #[inline]
    pub fn val(self) -> F {
        self.0 .0
    }
}

impl<F: Float, U: Unit<F>> From<Wrap<F, U>> for Angle<F, U> {
    #[inline]
    fn from(val: Wrap<F, U>) -> Self {
        val.0
    }
}

macro_rules! impl_traits {
    ($ang: ident : $new: ident) => {
        impl<F: Float, U: Unit<F>> Default for $ang<F, U> {
            #[inline]
            fn default() -> Self {
                Self::$new(F::ZERO)
            }
        }

        impl<F: Float, U: Unit<F>> Clone for $ang<F, U> {
            #[inline]
            fn clone(&self) -> Self {
                Self::$new(self.val_raw())
            }
        }
        impl<F: Float, U: Unit<F>> Copy for $ang<F, U> {}

        impl<F: Float, U: Unit<F>> PartialEq for $ang<F, U> {
            #[inline]
            fn eq(&self, rhs: &Self) -> bool {
                self.val_raw().total_eq(rhs.val_raw())
            }
        }
        impl<F: Float, U: Unit<F>> Eq for $ang<F, U> {}

        impl<F: Float + fmt::Display, U: Unit<F>> fmt::Display for $ang<F, U> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                U::display(self.val_raw(), f)
            }
        }
    };
    ($($ang: ident : $new: ident),*) => {
        $(impl_traits!($ang : $new);)*
    }
}

impl_traits!(Angle: new, Wrap: new_unchecked);

macro_rules! impl_ops {
    ($ang: ident : $new: ident) => {
        impl<F: Float, U: Unit<F>, Rhs: Into<Angle<F, U>>> Add<Rhs> for $ang<F, U> {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Rhs) -> Self {
                Self::$new(self.val_raw() + rhs.into().val_raw())
            }
        }
        impl<F: Float, U: Unit<F>, Rhs: Into<Angle<F, U>>> AddAssign<Rhs> for $ang<F, U> {
            #[inline]
            fn add_assign(&mut self, rhs: Rhs) {
                *self = *self + rhs;
            }
        }

        impl<F: Float, U: Unit<F>, Rhs: Into<Angle<F, U>>> Sub<Rhs> for $ang<F, U> {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Rhs) -> Self {
                Self::$new(self.val_raw() - rhs.into().val_raw())
            }
        }
        impl<F: Float, U: Unit<F>, Rhs: Into<Angle<F, U>>> SubAssign<Rhs> for $ang<F, U> {
            #[inline]
            fn sub_assign(&mut self, rhs: Rhs) {
                *self = *self - rhs;
            }
        }

        impl<F: Float, U: Unit<F>> Neg for $ang<F, U> {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self::$new(-self.val_raw())
            }
        }

        impl<F: Float, U: Unit<F>> Mul<F> for $ang<F, U> {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: F) -> Self {
                Self::$new(self.val_raw() * rhs)
            }
        }
        impl<F: Float, U: Unit<F>> MulAssign<F> for $ang<F, U> {
            #[inline]
            fn mul_assign(&mut self, rhs: F) {
                *self = *self * rhs;
            }
        }

        impl<F: Float, U: Unit<F>> Div<F> for $ang<F, U> {
            type Output = Self;
            #[inline]
            fn div(self, rhs: F) -> Self {
                Self::$new(self.val_raw() / rhs)
            }
        }
        impl<F: Float, U: Unit<F>> DivAssign<F> for $ang<F, U> {
            #[inline]
            fn div_assign(&mut self, rhs: F) {
                *self = *self / rhs;
            }
        }
    };
    ($($ang: ident : $new: ident),*) => {
        $(impl_ops!($ang : $new);)*
    }
}

impl_ops!(Angle: new, Wrap: wrap);

macro_rules! impl_trig {
    ($ang: ident : $new: ident) => {
        impl<F: Float, U: Unit<F>> $ang<F, U> {
            /// Computes the sine of this angle.
            #[inline]
            pub fn sin(self) -> F {
                let rad = self.val_raw() * (F::PI / U::HALF_TURN);
                rad.sin()
            }
            /// Computes the cosine of this angle.
            #[inline]
            pub fn cos(self) -> F {
                let rad = self.val_raw() * (F::PI / U::HALF_TURN);
                rad.cos()
            }
            /// Computes both the sine and cosine of this angle.
            #[inline]
            pub fn sin_cos(self) -> (F, F) {
                let rad = self.val_raw() * (F::PI / U::HALF_TURN);
                (rad.sin(), rad.cos())
            }
            /// Computes the tangent of this angle.
            #[inline]
            pub fn tan(self) -> F {
                let rad = self.val_raw() * (F::PI / U::HALF_TURN);
                rad.tan()
            }

            /// Computes the arc-sine of the specified value.
            #[inline]
            pub fn asin(y: F) -> Self {
                let rad = y.asin();
                Self::$new(rad * (U::HALF_TURN / F::PI))
            }
            /// Computes the arc-cosine of the specified value.
            #[inline]
            pub fn acos(x: F) -> Self {
                let rad = x.acos();
                Self::$new(rad * (U::HALF_TURN / F::PI))
            }
            /// Computes the arc-tangent of the specified value.
            #[inline]
            pub fn atan(tan: F) -> Self {
                let rad = tan.atan();
                Self::$new(rad * (U::HALF_TURN / F::PI))
            }
            /// Computes the four-quadrant arc-tangent of the specified fraction.
            #[inline]
            pub fn atan2(y: F, x: F) -> Self {
                let rad = y.atan2(x);
                Self::$new(rad * (U::HALF_TURN / F::PI))
            }
        }
    };
    ($($ang: ident : $new: ident),*) => {
        $(impl_trig!($ang : $new);)*
    };
}

impl_trig!(Angle: new, Wrap: new_unchecked);

/// A 32-bit angle measured in radians.
///
/// This type is guaranteed to be finite (in debug mode). As such, it implements total equality and ordering.
///
/// When formatting with [`Display`](fmt::Display), it will be shown as a multiple of π.
pub type Rad32 = Rad<f32>;
/// A 64-bit angle measured in radians.
///
/// This type is guaranteed to be finite (in debug mode). As such, it implements total equality and ordering.
///
/// When formatting with [`Display`](fmt::Display), it will be shown as a multiple of π.
pub type Rad64 = Rad<f64>;

/// A 32-bit angle measured in degrees.
///
/// This type is guaranteed to be finite (in debug mode). As such, it implements total equality and ordering.
pub type Deg32 = Deg<f32>;
/// A 64-bit angle measured in degrees.
///
/// This type is guaranteed to be finite (in debug mode). As such, it implements total equality and ordering.
pub type Deg64 = Deg<f64>;

/// A 32-bit angle measured in radians that wraps between -π and +π.
///
/// This type is guaranteed to be finite (in debug mode). As such, it implements total equality.  
/// It intentionally does not implement any ordering, as the wrapping behavior of this type may
/// cause unexpected ordering. If you need ordering, use the method `inner` or `mag`.
pub type Wrap32 = Wrap<f32, Radians>;
/// A 64-bit angle measured in radians that wraps between -π and +π.
///
/// This type is guaranteed to be finite (in debug mode). As such, it implements total equality.  
/// It intentionally does not implement any ordering, as the wrapping behavior of this type may
/// cause unexpected ordering. If you need ordering, use the method `inner` or `mag`.
pub type Wrap64 = Wrap<f64, Radians>;

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_abs {
        ($lhs: expr, $rhs: expr, $ep: expr) => {
            assert!(($lhs - $rhs).abs() < $ep, "{} - {} >= {}", $lhs, $rhs, $ep)
        };
    }
    macro_rules! assert_epsilon {
        ($lhs: expr, $rhs: expr) => {
            assert_abs!($lhs, $rhs, std::f64::EPSILON)
        };
    }

    use std::f64::consts::{PI, SQRT_2};

    #[test]
    fn consts() {
        assert_epsilon!(Rad64::ZERO.val(), 0.0);
        assert_epsilon!(Rad64::QUARTER_TURN.val(), PI / 2.0);
        assert_epsilon!(Rad64::HALF_TURN.val(), PI);
        assert_epsilon!(Rad64::FULL_TURN.val(), 2.0 * PI);

        assert_epsilon!(Wrap64::ZERO.val(), 0.0);
        assert_epsilon!(Wrap64::QUARTER_TURN.val(), PI / 2.0);
        assert_epsilon!(Wrap64::HALF_TURN.val(), PI);
        assert_epsilon!(Wrap64::FULL_TURN.val(), 2.0 * PI);
    }

    #[test]
    fn ord() {
        assert!(Rad64::MIN < -Rad64::FULL_TURN);
        assert!(-Rad64::FULL_TURN < -Rad64::QUARTER_TURN);
        assert!(-Rad64::QUARTER_TURN < -Rad64::ZERO);
        assert!(-Rad64::ZERO == Rad64::ZERO);
        assert!(Rad64::ZERO < Rad64::QUARTER_TURN);
        assert!(Rad64::QUARTER_TURN < Rad64::FULL_TURN);
        assert!(Rad64::FULL_TURN < Rad64::MAX);

        assert!(Rad64::MAX > Rad64::FULL_TURN);
        assert!(Rad64::FULL_TURN > Rad64::QUARTER_TURN);
        assert!(Rad64::QUARTER_TURN > Rad64::ZERO);
        assert!(Rad64::ZERO == -Rad64::ZERO);
        assert!(-Rad64::ZERO > -Rad64::QUARTER_TURN);
        assert!(-Rad64::QUARTER_TURN > -Rad64::FULL_TURN);
        assert!(-Rad64::FULL_TURN > Rad64::MIN);
    }

    #[test]
    fn rad_ops() {
        let sum = Rad64::HALF_TURN + Rad64::HALF_TURN;
        assert_epsilon!(sum.val(), 2.0 * PI);

        let diff = Rad64::FULL_TURN - Rad64::HALF_TURN;
        assert_epsilon!(diff.val(), PI);

        let neg = -Rad64::HALF_TURN;
        assert_epsilon!(neg.val(), -PI);

        let prod = Rad64::HALF_TURN * 3.0;
        assert_epsilon!(prod.val(), PI * 3.0);

        let quot = Rad64::FULL_TURN / 3.0;
        assert_epsilon!(quot.val(), PI * 2.0 / 3.0);

        let mut val = Rad64::HALF_TURN;
        val += Rad64::HALF_TURN;
        assert_epsilon!(val.val(), 2.0 * PI);

        let mut val = Rad64::FULL_TURN;
        val -= Rad64::HALF_TURN;
        assert_epsilon!(val.val(), PI);

        let mut val = Rad64::HALF_TURN;
        val *= 3.0;
        assert_epsilon!(val.val(), 3.0 * PI);

        let mut val = Rad64::FULL_TURN;
        val /= 3.0;
        assert_epsilon!(val.val(), PI * 2.0 / 3.0);
    }

    #[test]
    fn convert() {
        assert_eq!(Rad64::HALF_TURN.deg(), Deg64::HALF_TURN);
        assert_eq!(Deg64::QUARTER_TURN.rad(), Rad64::QUARTER_TURN);
    }

    #[test]
    fn wrap() {
        let wrap = Rad64::HALF_TURN.wrap();
        assert_epsilon!(wrap.val(), PI);

        let wrap = (-Rad64::HALF_TURN).wrap();
        assert_epsilon!(wrap.val(), PI);

        let wrap = (Rad64::HALF_TURN * 1.5).wrap();
        assert_epsilon!(wrap.val(), -PI / 2.0);

        let wrap = (-Rad64::HALF_TURN * 1.5).wrap();
        assert_epsilon!(wrap.val(), PI / 2.0);
    }

    #[test]
    fn wrap_ops() {
        let sum = Wrap64::HALF_TURN + Wrap64::HALF_TURN;
        assert_epsilon!(sum.val(), 0.0);
        let sum = Wrap64::HALF_TURN + Wrap64::FULL_TURN;
        assert_epsilon!(sum.val(), PI);

        let diff = Wrap64::HALF_TURN - Wrap64::HALF_TURN;
        assert_epsilon!(diff.val(), 0.0);
        let diff = Wrap64::HALF_TURN - Wrap64::FULL_TURN;
        assert_epsilon!(diff.val(), PI);
        let diff = Wrap64::QUARTER_TURN - Wrap64::HALF_TURN;
        assert_epsilon!(diff.val(), -PI / 2.0);

        let neg = -Wrap64::HALF_TURN;
        assert_epsilon!(neg.val(), PI);

        let prod = Wrap64::QUARTER_TURN * 2.0;
        assert_epsilon!(prod.val(), PI);
        let prod = Wrap64::HALF_TURN * 3.0;
        assert_epsilon!(prod.val(), PI);
        let prod = Wrap64::QUARTER_TURN * -5.0;
        assert_epsilon!(prod.val(), -PI / 2.0);

        let quot = Wrap64::HALF_TURN / 2.0;
        assert_epsilon!(quot.val(), PI / 2.0);

        let mut val = Wrap64::QUARTER_TURN;
        val += Wrap64::HALF_TURN;
        assert_epsilon!(val.val(), -PI / 2.0);

        let mut val = Wrap64::HALF_TURN;
        val -= Wrap64::FULL_TURN;
        assert_epsilon!(val.val(), PI);
        let mut val = Wrap64::ZERO;
        val -= Wrap64::HALF_TURN;
        assert_epsilon!(val.val(), PI);

        let mut val = Wrap64::QUARTER_TURN;
        val *= 5.0;
        assert_epsilon!(val.val(), PI / 2.0);

        let mut val = Wrap64::QUARTER_TURN;
        val /= 2.0;
        assert_epsilon!(val.val(), PI / 4.0);
    }

    #[test]
    fn trig() {
        assert_epsilon!(Rad64::ZERO.sin(), 0.0);
        assert_epsilon!(Rad64::ZERO.cos(), 1.0);
        assert_epsilon!(Rad64::ZERO.tan(), 0.0);
        assert_epsilon!(Wrap64::ZERO.sin(), 0.0);
        assert_epsilon!(Wrap64::ZERO.cos(), 1.0);
        assert_epsilon!(Wrap64::ZERO.tan(), 0.0);

        assert_epsilon!((Rad64::HALF_TURN / 4.0).sin(), SQRT_2.recip());
        assert_epsilon!((Rad64::HALF_TURN / 4.0).cos(), SQRT_2.recip());
        assert_epsilon!((Rad64::HALF_TURN / 4.0).tan(), 1.0);
        assert_epsilon!((Wrap64::HALF_TURN / 4.0).sin(), SQRT_2.recip());
        assert_epsilon!((Wrap64::HALF_TURN / 4.0).cos(), SQRT_2.recip());
        assert_epsilon!((Wrap64::HALF_TURN / 4.0).tan(), 1.0);

        assert_epsilon!(Rad64::QUARTER_TURN.sin(), 1.0);
        assert_epsilon!(Rad64::QUARTER_TURN.cos(), 0.0);
        assert_epsilon!(Wrap64::QUARTER_TURN.sin(), 1.0);
        assert_epsilon!(Wrap64::QUARTER_TURN.cos(), 0.0);

        assert_epsilon!(Rad64::HALF_TURN.sin(), 0.0);
        assert_epsilon!(Rad64::HALF_TURN.cos(), -1.0);
        assert_epsilon!(Rad64::HALF_TURN.tan(), 0.0);
        assert_epsilon!(Wrap64::HALF_TURN.sin(), 0.0);
        assert_epsilon!(Wrap64::HALF_TURN.cos(), -1.0);
        assert_epsilon!(Wrap64::HALF_TURN.tan(), 0.0);

        assert_epsilon!((-Rad64::HALF_TURN / 4.0).sin(), -SQRT_2.recip());
        assert_epsilon!((-Rad64::HALF_TURN / 4.0).cos(), SQRT_2.recip());
        assert_epsilon!((-Rad64::HALF_TURN / 4.0).tan(), -1.0);
        assert_epsilon!((Wrap64::HALF_TURN / -4.0).sin(), -SQRT_2.recip());
        assert_epsilon!((Wrap64::HALF_TURN / -4.0).cos(), SQRT_2.recip());
        assert_epsilon!((Wrap64::HALF_TURN / -4.0).tan(), -1.0);

        assert_epsilon!((-Rad64::QUARTER_TURN).sin(), -1.0);
        assert_epsilon!((-Rad64::QUARTER_TURN).cos(), 0.0);
        assert_epsilon!((-Wrap64::QUARTER_TURN).sin(), -1.0);
        assert_epsilon!((-Wrap64::QUARTER_TURN).cos(), 0.0);
    }

    #[test]
    fn inverse_trig() {
        assert_epsilon!(Rad64::asin(0.0).val(), 0.0);
        assert_epsilon!(Rad64::acos(0.0).val(), PI / 2.0);
        assert_epsilon!(Wrap64::asin(0.0).val(), 0.0);
        assert_epsilon!(Wrap64::acos(0.0).val(), PI / 2.0);

        assert_epsilon!(Rad64::asin(SQRT_2 / 2.0).val(), PI / 4.0);
        assert_epsilon!(Rad64::acos(SQRT_2 / 2.0).val(), PI / 4.0);
        assert_epsilon!(Wrap64::asin(SQRT_2 / 2.0).val(), PI / 4.0);
        assert_epsilon!(Wrap64::acos(SQRT_2 / 2.0).val(), PI / 4.0);

        assert_epsilon!(Rad64::asin(1.0).val(), PI / 2.0);
        assert_epsilon!(Rad64::acos(1.0).val(), 0.0);
        assert_epsilon!(Wrap64::asin(1.0).val(), PI / 2.0);
        assert_epsilon!(Wrap64::acos(1.0).val(), 0.0);

        assert_epsilon!(Rad64::asin(-SQRT_2 / 2.0).val(), -PI / 4.0);
        assert_epsilon!(Rad64::acos(-SQRT_2 / 2.0).val(), 3.0 * PI / 4.0);
        assert_epsilon!(Wrap64::asin(-SQRT_2 / 2.0).val(), -PI / 4.0);
        assert_epsilon!(Wrap64::acos(-SQRT_2 / 2.0).val(), 3.0 * PI / 4.0);

        assert_epsilon!(Rad64::asin(-1.0).val(), -PI / 2.0);
        assert_epsilon!(Rad64::acos(-1.0).val(), PI);
        assert_epsilon!(Wrap64::asin(-1.0).val(), -PI / 2.0);
        assert_epsilon!(Wrap64::acos(-1.0).val(), PI);

        assert_epsilon!(Rad64::atan2(0.0, 1.0).val(), 0.0);
        assert_epsilon!(Rad64::atan2(SQRT_2 / 2.0, SQRT_2 / 2.0).val(), PI / 4.0);
        assert_epsilon!(Rad64::atan2(1.0, 0.0).val(), PI / 2.0);
        assert_epsilon!(
            Rad64::atan2(SQRT_2 / 2.0, -SQRT_2 / 2.0).val(),
            3.0 * PI / 4.0
        );
        assert_epsilon!(Rad64::atan2(0.0, -1.0).val(), PI);
        assert_epsilon!(Rad64::atan2(-SQRT_2 / 2.0, SQRT_2 / 2.0).val(), -PI / 4.0);
        assert_epsilon!(Rad64::atan2(-1.0, 0.0).val(), -PI / 2.0);
        assert_epsilon!(
            Rad64::atan2(-SQRT_2 / 2.0, -SQRT_2 / 2.0).val(),
            -3.0 * PI / 4.0
        );
    }

    #[test]
    fn display() {
        assert_eq!(format!("{}", Rad64::ZERO), "0");
        assert_eq!(format!("{:.4}", Rad64::ZERO), "0.0000");
        assert_eq!(format!("{}", Rad64::QUARTER_TURN), "0.5π");
        assert_eq!(format!("{:+.3}", Rad64::QUARTER_TURN), "+0.500π");
        assert_eq!(format!("{:.2}", -Rad64::HALF_TURN), "-1.00π");

        assert_eq!(format!("{}", Deg64::ZERO), "0°0'0''");
        assert_eq!(
            format!(
                "{:+.3}",
                Deg64::new(180.0 + 44.0 / 60.0 + 12.4567 / 60.0 / 60.0)
            ),
            "+180°44'12.457''"
        );
        assert_eq!(
            format!(
                "{:.2}",
                Deg64::new(-90.0 - 15.0 / 60.0 - 0.123 / 60.0 / 60.0)
            ),
            "-90°15'0.12''"
        );
    }
}
