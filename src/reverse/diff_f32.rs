use super::tape::{Idx, Tape, Diff, EmptyOp, Var};
use std::ops::{*};
use std::fmt;

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

impl<'a> Add for Var<'a, f32, ScalarOps> {
    type Output = Var<'a, f32, ScalarOps>;

    fn add(self, rhs: Self) -> Self::Output {
        let res = self.data + rhs.data;
        Var {
            tape: self.tape,
            data: res,
            idx: self.tape.var_operator(res, ScalarOps::Add(self.idx, rhs.idx))
        }
    }
}

impl<'a> Sub for Var<'a, f32, ScalarOps> {
    type Output = Var<'a, f32, ScalarOps>;

    fn sub(self, rhs: Self) -> Self::Output {
        let res = self.data - rhs.data;
        Var {
            tape: self.tape,
            data: res,
            idx: self.tape.var_operator(res, ScalarOps::Sub(self.idx, rhs.idx))
        }
    }
}

impl<'a> Mul for Var<'a, f32, ScalarOps> {
    type Output = Var<'a, f32, ScalarOps>;

    fn mul(self, rhs: Self) -> Self::Output {
        let res = self.data * rhs.data;
        Var {
            tape: self.tape,
            data: res,
            idx: self.tape.var_operator(res, ScalarOps::Mul(self.idx, rhs.idx))
        }
    }
}

pub fn powf<'a>(x: Var<'a, f32, ScalarOps>, n: Var<'a, f32, ScalarOps>) -> Var<'a, f32, ScalarOps> {
    let res = x.data.powf(n.data);
    Var {
        tape: x.tape,
        data: res,
        idx: x.tape.var_operator(res, ScalarOps::Pow(x.idx, n.idx))
    }
}

pub fn tanh<'a>(x: Var<'a, f32, ScalarOps>) -> Var<'a, f32, ScalarOps> {
    let res = x.data.tanh();
    Var {
        tape: x.tape,
        data: res,
        idx: x.tape.var_operator(res, ScalarOps::TanH(x.idx))
    }
}

pub fn relu<'a>(x: Var<'a, f32, ScalarOps>) -> Var<'a, f32, ScalarOps> {
    let res = if x.data >= 0.0 {
        x.data
    } else {
        0.0
    };
    Var {
        tape: x.tape,
        data: res,
        idx: x.tape.var_operator(res, ScalarOps::ReLU(x.idx))
    }
}

impl<'a> Diff<'a, f32, ScalarOps> for Var<'a, f32, ScalarOps> {

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
                    nodes_ref[b].grad += nodes_ref[i].grad * data_a.log2() * data_a.powf(data_b);
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
            write!(f, "idx: {}, data: {}, grad: {}", i, node.data, node.grad)?;
            match node.parents {
                ScalarOps::Add(a, b) => {writeln!(f, " <-- add -- ({}, {})", a, b)?; },
                ScalarOps::Sub(a, b) => {writeln!(f, " <-- sub -- ({}, {})", a, b)?; },
                ScalarOps::Mul(a, b) => {writeln!(f, " <-- mul -- ({}, {})", a, b)?; },
                ScalarOps::Pow(a, b) => {writeln!(f, " <-- pow -- ({}, {})", a, b)?; },
                ScalarOps::Empty => {writeln!(f, "")?; },
                ScalarOps::ReLU(a) => {writeln!(f, " <-- relu -- ({})", a)?; },
                ScalarOps::TanH(a) => {writeln!(f, " <-- tanh -- ({})", a)?; },
            }
        }
        Ok(())
    }
}