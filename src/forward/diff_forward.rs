use std::ops::{Add, Mul, Sub, Div, Neg};
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

impl<X, D> Add for F<X, D> 
where
    X: Add<X, Output = X>,
    D: Add<D, Output = D>
{
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
    X: Clone + Mul<Output = X>, 
    D: Clone + Mul<X, Output = D> + Add<Output = D>,
{
    type Output = F<X, D>;

    fn mul(self, rhs: F<X, D>) -> Self::Output {
        F {
            x: self.x.clone() * rhs.x.clone(),
            dx: self.dx * rhs.x + rhs.dx * self.x
        }
    }
}

impl<X, D> Div for F<X, D> 
where 
    X: Clone + Mul<Output = X> + Div<Output = X>, 
    D: Clone + Mul<X, Output = D> + Div<X, Output = D> + Sub<Output = D>,
{
    type Output = F<X, D>;

    fn div(self, rhs: F<X, D>) -> Self::Output {
        F {
            x: self.x.clone() / rhs.x.clone(),
            dx: (self.dx * rhs.x.clone() - rhs.dx * self.x) / (rhs.x.clone() * rhs.x) 
        }
    }
}

impl <X, D> Sub for F<X, D> 
where
    X: Sub<X, Output = X>,
    D: Sub<D, Output = D>
{
    type Output = F<X, D>;

    fn sub(self, rhs: Self) -> Self::Output {
        F {
            x: self.x - rhs.x,
            dx: self.dx - rhs.dx
        }
    }
}

impl <X, D> Neg for F<X, D> 
where
    X: Neg<Output = X>,
    D: Neg<Output = D>
{
    type Output = F<X, D>;

    fn neg(self) -> Self::Output {
        F {
            x: -self.x,
            dx: -self.dx
        }
    }
}

// impl<X, D> F<X, D> 
// where
//     X: Clone + Float,
//     D: Clone + Float
// {

//     fn powf(self, n: Self) -> Self {
//         F {
//             x: self.x.clone().powf(n.x.clone()),
//             dx: self.x
//         }
//     }
// } 


