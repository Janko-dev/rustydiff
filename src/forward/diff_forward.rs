use std::ops::{Add, Mul};
use num_traits::{Float, FloatConst, NumCast, One, ToPrimitive, Zero};

#[derive(Debug, Clone, Copy)]
pub struct F<X, D> {
    pub x: X,
    pub dx: D
}

impl <X, D> F<X, D> {
    pub fn new(x: X, dx: D) -> Self {
        F { x, dx }
    }
}

impl <X, D: Zero> F<X, D> {
    pub fn cst(x: X) -> Self {
        F { 
            x, 
            dx: D::zero()
        }
    }
}

impl <X, D: One> F<X, D> {
    pub fn var(x: X) -> Self {
        F { 
            x, 
            dx: D::one()
        }
    }
}

impl <X, D: Clone> F<X, D> {
    pub fn deriv(&mut self) -> D {
        self.dx.clone()
    }
}

impl<X: Add<X, Output = X>, D: Add<D, Output = D>> Add for F<X, D> {
    type Output = F<X, D>;

    fn add(self, rhs: Self) -> Self::Output {
        F {
            x: self.x + rhs.x,
            dx: self.dx + rhs.dx
        }
    }
}

impl<X, D> Mul for F<X, D> 
where 
    X: Mul + Clone, 
    D: Mul<X> + Clone,
    D::Output: Add
{
    type Output = F<X::Output, <<D as Mul<X>>::Output as Add>::Output>;

    fn mul(self, rhs: F<X, D>) -> Self::Output {
        F {
            x: self.x.clone() * rhs.x.clone(),
            dx: self.dx * rhs.x + rhs.dx * self.x
        }
    }
}