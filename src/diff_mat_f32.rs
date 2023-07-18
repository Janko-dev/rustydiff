use std::ops::{Index, IndexMut};
use super::tape::{Idx, Tape, CompNode, Diff};
use std::fmt;

use ndarray::Array2;

#[derive(Debug)]
pub enum VectorOps {
    Empty,
    ElemAdd(Idx, Idx),
    ElemSub(Idx, Idx),
    ElemMul(Idx, Idx),
    ElemTanH(Idx),
    ElemReLU(Idx),
    Dot(Idx, Idx),
    Cross(Idx, Idx)
}

impl Index<Idx> for Tape<Array2<f32>, VectorOps> {
    type Output = CompNode<Array2<f32>, VectorOps>;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.vals[index]
    }
}

impl IndexMut<Idx> for Tape<Array2<f32>, VectorOps> {
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.vals[index]
    }
}

impl CompNode<Array2<f32>, VectorOps> {
    pub fn new(data: Array2<f32>) -> Self {
        let shape0 = data.shape()[0];
        let shape1 = data.shape()[1];
        Self { data, grad: Array2::zeros((shape0, shape1)), idx: 0, children: VectorOps::Empty }
    }
}

impl Tape<Array2<f32>, VectorOps> {
    pub fn new() -> Self {
        Self { vals: vec![]}
    }

    pub fn create_node(&mut self, data: Array2<f32>) -> Idx {
        let mut node = CompNode::<Array2<f32>, VectorOps>::new(data);
        node.idx = self.vals.len();
        self.vals.push(node);
        self.vals.len()-1
    }

    pub fn _reverse(&mut self, y: Idx) {
        let y_node = &self[y];

        match y_node.children {
            VectorOps::ElemAdd(a, b) => {

            },
            _ => {}
            // ScalarOps::Add(a, b) => {
            //     self[a].grad += self[y].grad * 1.0;
            //     self[b].grad += self[y].grad * 1.0;

            //     self._reverse(a);
            //     self._reverse(b);
            // },
            // ScalarOps::Sub(a, b) => {
            //     self[a].grad += self[y].grad * 1.0;
            //     self[b].grad += self[y].grad * -1.0;

            //     self._reverse(a);
            //     self._reverse(b);
            // },
            // ScalarOps::Mul(a, b) => {
            //     self[a].grad += self[y].grad * self[b].data;
            //     self[b].grad += self[y].grad * self[a].data;

            //     self._reverse(a);
            //     self._reverse(b);
            // },
            // ScalarOps::Pow(a, b) => {
            //     let l_data = self[a].data;
            //     let r_data = self[b].data;
            //     self[a].grad += self[y].grad * r_data * l_data.powf(r_data - 1.0);
            //     self[b].grad += self[y].grad * l_data.log2() * l_data.powf(r_data);

            //     self._reverse(a);
            //     self._reverse(b);
            // },
            // ScalarOps::TanH(a) => {
            //     self[a].grad += self[y].grad * (1.0 - self[y].grad*self[y].grad);

            //     self._reverse(a);
            // }
            // ScalarOps::ReLU(a) => {
            //     if self[y].data > 0.0 {
            //         self[a].grad += self[y].grad * 1.0;
            //     }
            //     self._reverse(a);
            // }
            // ScalarOps::Empty => {},
        }
        

    }

    pub fn add(&mut self, a: Idx, b: Idx) -> Idx {
        let res_array = self[a].data.clone() + self[b].data.clone();
        let res = self.create_node(res_array);
        self[res].children = VectorOps::ElemAdd(a, b);
        res
    }

    // pub fn mul(&mut self, a: Idx, b: Idx) -> Idx {
    //     let res = self.create_node(self[a].data * self[b].data);
    //     self[res].children = ScalarOps::Mul(a, b);
    //     res
    // }

    // pub fn sub(&mut self, a: Idx, b: Idx) -> Idx {
    //     let res = self.create_node(self[a].data - self[b].data);
    //     self[res].children = ScalarOps::Sub(a, b);
    //     res
    // }

    // pub fn pow(&mut self, a: Idx, b: Idx) -> Idx {
    //     let res = self.create_node(self[a].data.powf(self[b].data));
    //     self[res].children = ScalarOps::Pow(a, b);
    //     res
    // }
    
}

impl Diff for Tape<Array2<f32>, VectorOps> {

    fn reverse(&mut self, y: Idx) {
        self[y].grad = Array2::eye(self[y].data.shape()[0]);
        self._reverse(y);
    }
}

// impl fmt::Display for Tape<f32, ScalarOps> {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         for node in self.vals.iter() {
//             write!(f, "idx: {}, data: {}, grad: {}", node.idx, node.data, node.grad)?;
//             match node.children {
//                 ScalarOps::Add(a, b) => {writeln!(f, " <-- add -- ({}, {})", a, b)?; },
//                 ScalarOps::Sub(a, b) => {writeln!(f, " <-- sub -- ({}, {})", a, b)?; },
//                 ScalarOps::Mul(a, b) => {writeln!(f, " <-- mul -- ({}, {})", a, b)?; },
//                 ScalarOps::Pow(a, b) => {writeln!(f, " <-- pow -- ({}, {})", a, b)?; },
//                 ScalarOps::Empty => {writeln!(f, "")?; },
//                 ScalarOps::ReLU(a) => {writeln!(f, " <-- relu -- ({})", a)?; },
//                 ScalarOps::TanH(a) => {writeln!(f, " <-- tanh -- ({})", a)?; },
//             }
//         }
//         Ok(())
//     }
// }