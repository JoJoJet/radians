use std::{fmt, ops};

/// A floating-point number that serves as the backing value of an [`Angle`].
pub trait Float: num_traits::Float {
    /// Additive identity, 0.
    const ZERO: Self;
    /// Archimedes’ constant (π)
    const PI: Self;
    /// π divided by two.
    const PI_OVER_TWO: Self;
    /// The full circle constant (τ)
    const TAU: Self;

    /// Modulus operation.
    fn rem_euclid(self, _: Self) -> Self;
}

macro_rules! impl_float {
    ($f: ident) => {
        impl Float for $f {
            const ZERO: $f = 0.0;
            const PI: $f = std::$f::consts::PI;
            const PI_OVER_TWO: $f = std::$f::consts::PI / 2.0;
            const TAU: $f = std::$f::consts::TAU;

            #[inline]
            fn rem_euclid(self, rhs: Self) -> Self {
                <$f>::rem_euclid(self, rhs)
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);

/// An angle measured in radians.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct Rad<F: Float>(F);

impl<F: Float> Rad<F> {
    /// Zero angle, additive identity.
    pub const ZERO: Self = Self(F::ZERO);
    /// Quarter turn around a circle. Equal to π/2.
    pub const QUARTER_TURN: Self = Self(F::PI_OVER_TWO);
    /// Half turn around a circle. Equal to π.
    pub const HALF_TURN: Self = Self(F::PI);
    /// Full turn around a circle. Equal to 2π.
    pub const FULL_TURN: Self = Self(F::TAU);

    /// Creates a new angle in radians.
    /// # Panics
    /// If the value is non-finite (debug mode).
    #[inline]
    pub fn new(val: F) -> Self {
        debug_assert!(val.is_finite());
        Self(val)
    }

    /// Gets the value of this angle in radians.
    #[inline]
    pub fn val(self) -> F {
        self.0
    }
    /// Wraps this angle between -π and +π.
    #[inline]
    pub fn wrap(self) -> Wrap<F> {
        Wrap::wrap(self.0)
    }
    /// Returns the magnitude (absolute value) of this angle in radians.
    #[inline]
    pub fn mag(self) -> F {
        self.0.abs()
    }
}

impl<F: Float> ops::Add for Rad<F> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.0 + rhs.0)
    }
}
impl<F: Float + ops::AddAssign> ops::AddAssign for Rad<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        debug_assert!(self.0.is_finite());
    }
}
impl<F: Float> ops::Sub for Rad<F> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.0 - rhs.0)
    }
}
impl<F: Float + ops::SubAssign> ops::SubAssign for Rad<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        debug_assert!(self.0.is_finite());
    }
}
impl<F: Float> ops::Neg for Rad<F> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl<F: Float> ops::Mul<F> for Rad<F> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: F) -> Self {
        Self::new(self.0 * rhs)
    }
}
impl<F: Float + ops::MulAssign> ops::MulAssign<F> for Rad<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: F) {
        self.0 *= rhs;
        debug_assert!(self.0.is_finite());
    }
}
impl<F: Float> ops::Div<F> for Rad<F> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: F) -> Self {
        Self::new(self.0 / rhs)
    }
}
impl<F: Float + ops::DivAssign> ops::DivAssign<F> for Rad<F> {
    #[inline]
    fn div_assign(&mut self, rhs: F) {
        self.0 /= rhs;
        debug_assert!(self.0.is_finite());
    }
}

impl<F: Float + fmt::Debug> fmt::Debug for Rad<F> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}
impl<F: Float + fmt::Display> fmt::Display for Rad<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == F::ZERO {
            write!(f, "0")
        } else {
            write!(f, "{}π", self.val() / F::PI)
        }
    }
}

/// An angle that wraps between a negative half turn and a positive half turn.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Wrap<F: Float>(Rad<F>);

impl<F: Float> Wrap<F> {
    /// Zero angle, additive identity.
    pub const ZERO: Self = Self(Rad::ZERO);
    /// Half turn around a circle. Equal to π/2.
    pub const QUARTER_TURN: Self = Self(Rad::QUARTER_TURN);
    /// Half turn around a circle. Equal to π.
    pub const HALF_TURN: Self = Self(Rad::HALF_TURN);
    /// Full turn around a circle. Equal to 2π;
    pub const FULL_TURN: Rad<F> = Rad::FULL_TURN;

    /// Creates a new angle in radians, wrapping between -π and +π.
    pub fn wrap(val: F) -> Self {
        let val = (-val + F::PI).rem_euclid(F::TAU) - F::PI;
        Self(Rad::new(-val))
    }

    /// Gets the value of this angle in radians.
    #[inline]
    pub fn val(self) -> F {
        self.0.val()
    }
}

impl<F: Float> From<Wrap<F>> for Rad<F> {
    #[inline]
    fn from(val: Wrap<F>) -> Self {
        val.0
    }
}

impl<F: Float, Rhs: Into<Rad<F>>> ops::Add<Rhs> for Wrap<F> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Rhs) -> Self {
        Self::wrap(self.val() + rhs.into().val())
    }
}
impl<F: Float, Rhs: Into<Rad<F>>> ops::AddAssign<Rhs> for Wrap<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Rhs) {
        *self = *self + rhs
    }
}
impl<F: Float, Rhs: Into<Rad<F>>> ops::Sub<Rhs> for Wrap<F> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Rhs) -> Self {
        Self::wrap(self.val() + rhs.into().val())
    }
}
impl<F: Float, Rhs: Into<Rad<F>>> ops::SubAssign<Rhs> for Wrap<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Rhs) {
        *self = *self - rhs;
    }
}
impl<F: Float> ops::Neg for Wrap<F> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::wrap(-self.val())
    }
}

impl<F: Float> ops::Mul<F> for Wrap<F> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: F) -> Self {
        Self::wrap(self.val() * rhs)
    }
}
impl<F: Float> ops::MulAssign<F> for Wrap<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}
impl<F: Float> ops::Div<F> for Wrap<F> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: F) -> Self {
        Self::wrap(self.val() / rhs)
    }
}
impl<F: Float> ops::DivAssign<F> for Wrap<F> {
    #[inline]
    fn div_assign(&mut self, rhs: F) {
        *self = *self / rhs
    }
}

impl<F: Float + fmt::Display> fmt::Display for Wrap<F> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

pub type Rad32 = Rad<f32>;
pub type Rad64 = Rad<f64>;

pub type Wrap32 = Wrap<f32>;
pub type Wrap64 = Wrap<f64>;

#[cfg(test)]
mod tests {}
