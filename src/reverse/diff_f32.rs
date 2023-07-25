use super::tape::{Idx, Tape, Diff, EmptyOp, Var};
use std::ops::{*};
use std::fmt;

// Common scalar operators which can be extended.
#[derive(Debug)]
pub enum ScalarOps {
    Empty,
    Add(Idx, Idx),
    Sub(Idx, Idx),
    Mul(Idx, Idx),
    Pow(Idx, Idx),
    TanH(Idx),
    ReLU(Idx)
}

impl EmptyOp for ScalarOps {
    fn empty_operator() -> Self {
        ScalarOps::Empty
    }
}

// for readability
type Varf32<'a> = Var<'a, f32, ScalarOps>;

impl<'a> Add for &Varf32<'a> {
    type Output = Varf32<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        let res = self.data + rhs.data;
        Var {
            tape: self.tape,
            data: res,
            idx: self.tape.var_operator(res, ScalarOps::Add(self.idx, rhs.idx))
        }
    }
}

impl<'a> AddAssign for Varf32<'a> {

    fn add_assign(&mut self, rhs: Self) {
        let res = self.data + rhs.data;
        *self = Self {
            tape: self.tape,
            data: res,
            idx: self.tape.var_operator(res, ScalarOps::Add(self.idx, rhs.idx))
        }
    }
}

impl<'a> Sub for &Varf32<'a> {
    type Output = Varf32<'a>;

    fn sub(self, rhs: Self) -> Self::Output {
        let res = self.data - rhs.data;
        Var {
            tape: self.tape,
            data: res,
            idx: self.tape.var_operator(res, ScalarOps::Sub(self.idx, rhs.idx))
        }
    }
}

impl<'a> Mul for &Varf32<'a> {
    type Output = Varf32<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        let res = self.data * rhs.data;
        Var {
            tape: self.tape,
            data: res,
            idx: self.tape.var_operator(res, ScalarOps::Mul(self.idx, rhs.idx))
        }
    }
}

impl<'a> Varf32<'a> {
    pub fn powf(&self, n: &Varf32<'a>) -> Varf32<'a> {
        let res = self.data.powf(n.data);
        Var {
            tape: self.tape,
            data: res,
            idx: self.tape.var_operator(res, ScalarOps::Pow(self.idx, n.idx))
        }
    }

    pub fn tanh(&self) -> Varf32<'a> {
        let res = self.data.tanh();
        Var {
            tape: self.tape,
            data: res,
            idx: self.tape.var_operator(res, ScalarOps::TanH(self.idx))
        }
    }

    pub fn relu(&self) -> Varf32<'a> {
        let res = if self.data >= 0.0 {
            self.data
        } else {
            0.0
        };
        Var {
            tape: self.tape,
            data: res,
            idx: self.tape.var_operator(res, ScalarOps::ReLU(self.idx))
        }
    }
}

impl<'a> Diff<'a, f32, ScalarOps> for Varf32<'a> {

    // In the reverse mode autodiff for f32's, we traverse the tape in reversed order
    // and match against the enum operator variants to update the gradients of the parents
    // of each node in the graph. Afterwards, every node on the tape maintains the correct
    // partial derivative of the computation.     
    fn reverse(&self) {
        let mut nodes_ref = self.tape.nodes.borrow_mut();
        nodes_ref[self.idx].grad = 1.0;
        let len = nodes_ref.len();

        for i in (0..len).rev() {
            match nodes_ref[i].parents {
                ScalarOps::Add(a, b) => {
                    nodes_ref[a].grad += nodes_ref[i].grad * 1.0;
                    nodes_ref[b].grad += nodes_ref[i].grad * 1.0;
                },
                ScalarOps::Sub(a, b) => {
                    nodes_ref[a].grad += nodes_ref[i].grad * 1.0;
                    nodes_ref[b].grad += nodes_ref[i].grad * -1.0;
                },
                ScalarOps::Mul(a, b) => {
                    nodes_ref[a].grad += nodes_ref[i].grad * nodes_ref[b].data;
                    nodes_ref[b].grad += nodes_ref[i].grad * nodes_ref[a].data;
                },
                ScalarOps::Pow(a, b) => {
                    let data_a = nodes_ref[a].data;
                    let data_b = nodes_ref[b].data;
                    nodes_ref[a].grad += nodes_ref[i].grad * data_b * data_a.powf(data_b - 1.0);
                    nodes_ref[b].grad += nodes_ref[i].grad * data_a.ln() * data_a.powf(data_b);
                },
                ScalarOps::TanH(a) => {
                    let data_i = nodes_ref[i].data;
                    nodes_ref[a].grad += nodes_ref[i].grad * (1.0 - data_i*data_i);
                },
                ScalarOps::ReLU(a) => {
                    if nodes_ref[i].data > 0.0 {
                        nodes_ref[a].grad += nodes_ref[i].grad * 1.0;
                    }
                }
                ScalarOps::Empty => {}
            }
        }
    }
}

impl fmt::Display for Tape<f32, ScalarOps> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, node) in self.nodes.borrow().iter().enumerate() {
            write!(f, "(idx: {}, data: {}, grad: {})", i, node.data, node.grad)?;
            match node.parents {
                ScalarOps::Add(a, b) => {writeln!(f, " <-- add -- parents ({}, {})", a, b)?; },
                ScalarOps::Sub(a, b) => {writeln!(f, " <-- sub -- parents ({}, {})", a, b)?; },
                ScalarOps::Mul(a, b) => {writeln!(f, " <-- mul -- parents ({}, {})", a, b)?; },
                ScalarOps::Pow(a, b) => {writeln!(f, " <-- pow -- parents ({}, {})", a, b)?; },
                ScalarOps::Empty => {writeln!(f, "")?; },
                ScalarOps::ReLU(a) => {writeln!(f, " <-- relu -- parents ({})", a)?; },
                ScalarOps::TanH(a) => {writeln!(f, " <-- tanh -- parents ({})", a)?; },
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_equal(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-15
    }

    #[test]
    fn add_test() {
        let tp = Tape::<f32, ScalarOps>::new();
        let x = tp.var(5.0);
        let y = tp.var(2.0);
        let z = &x + &y;
        z.reverse();
        assert!(f32_equal(x.grad(), 1.0));
        assert!(f32_equal(y.grad(), 1.0));
        assert!(f32_equal(z.grad(), 1.0));
    }

    #[test]
    fn sub_test() {
        let tp = Tape::<f32, ScalarOps>::new();
        let x = tp.var(5.0);
        let y = tp.var(2.0);
        let z = &x - &y;
        z.reverse();
        assert!(f32_equal(x.grad(),  1.0));
        assert!(f32_equal(y.grad(), -1.0));
    }

    #[test]
    fn mul_test() {
        let tp = Tape::<f32, ScalarOps>::new();
        let x = tp.var(5.0);
        let y = tp.var(2.0);
        let z = &x * &y;
        z.reverse();
        assert!(f32_equal(x.grad(), 2.0));
        assert!(f32_equal(y.grad(), 5.0));
    }

    #[test]
    fn pow_test() {
        let tp = Tape::<f32, ScalarOps>::new();
        let x = tp.var(5.0);
        let y = tp.var(2.0);
        let z = x.powf(&y);
        z.reverse();
        // derivative of f(x) = x^2 is f'(x) = 2x
        assert!(f32_equal(x.grad(), 10.0));
        // derivative of f(y) = 5^y is f'(y) = ln(5) * 5^y 
        assert!(f32_equal(y.grad(), 5.0f32.ln() * 5.0f32.powf(2.0)));
    }

    #[test]
    fn tanh_test() {
        let tp = Tape::<f32, ScalarOps>::new();
        let x = tp.var(1.2);
        let y = tp.var(0.4);
        let z = (&x * &y).tanh();
        z.reverse();
        // derivative of f(x) = tanh(0.4x) is f'(x) = 0.4 * (1 - tanh^2(0.4x))
        assert!(f32_equal(x.grad(), 0.4f32 * (1.0 - (1.2 * 0.4f32).tanh().powf(2.0))));
        // derivative of f(y) = tanh(1.2y) is f'(y) = 1.2 * (1 - tanh^2(1.2y))
        assert!(f32_equal(y.grad(), 1.2f32 * (1.0 - (1.2 * 0.4f32).tanh().powf(2.0))));
    }

    #[test]
    fn relu_test() {
        let tp = Tape::<f32, ScalarOps>::new();
        let x = tp.var(-5.0);
        let y = tp.var(2.0);

        let z = &x.relu() + &y;
        z.reverse();
        assert!(f32_equal(x.grad(), 0.0));
        assert!(f32_equal(y.grad(), 1.0))
    }

    #[test]
    fn non_linear_activation() {
        let tp = Tape::<f32, ScalarOps>::new();
        let ws = vec![tp.var(0.4), tp.var(0.8), tp.var(0.1)];
        let inp = vec![tp.var(2.0), tp.var(4.0), tp.var(6.0)];

        let mut res = tp.var(0.0);
        for (w, i) in ws.iter().zip(inp.iter()) {
            res += w * i;
        }
        let res = res.tanh();
        res.reverse();

        // derivative w.r.t ws[1] = 0.8
        assert!(f32_equal(ws[1].grad(), 4.0 * (1.0 - (0.4f32*2.0 + 0.8*4.0 + 0.1*6.0).tanh().powf(2.0))));
    }
}