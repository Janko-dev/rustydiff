use std::ops::{Index, IndexMut};
use super::tape::{Idx, Tape, CompNode, Diff};
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

impl Index<Idx> for Tape<f32, ScalarOps> {
    type Output = CompNode<f32, ScalarOps>;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.vals[index]
    }
}

impl IndexMut<Idx> for Tape<f32, ScalarOps> {
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.vals[index]
    }
}

impl CompNode<f32, ScalarOps> {
    pub fn new(data: f32) -> Self {
        Self { data, grad: 0.0, idx: 0, children: ScalarOps::Empty }
    }
}

impl Tape<f32, ScalarOps> {
    pub fn new() -> Self {
        Self { vals: vec![]}
    }

    pub fn create_node(&mut self, value: f32) -> Idx {
        let mut node = CompNode::<f32, ScalarOps>::new(value);
        node.idx = self.vals.len();
        self.vals.push(node);
        self.vals.len()-1
    }

    pub fn _reverse(&mut self, y: Idx) {
        let y_node = &self[y];

        match y_node.children {

            ScalarOps::Add(a, b) => {
                self[a].grad += self[y].grad * 1.0;
                self[b].grad += self[y].grad * 1.0;

                self._reverse(a);
                self._reverse(b);
            },
            ScalarOps::Sub(a, b) => {
                self[a].grad += self[y].grad * 1.0;
                self[b].grad += self[y].grad * -1.0;

                self._reverse(a);
                self._reverse(b);
            },
            ScalarOps::Mul(a, b) => {
                self[a].grad += self[y].grad * self[b].data;
                self[b].grad += self[y].grad * self[a].data;

                self._reverse(a);
                self._reverse(b);
            },
            ScalarOps::Pow(a, b) => {
                let l_data = self[a].data;
                let r_data = self[b].data;
                self[a].grad += self[y].grad * r_data * l_data.powf(r_data - 1.0);
                self[b].grad += self[y].grad * l_data.log2() * l_data.powf(r_data);

                self._reverse(a);
                self._reverse(b);
            },
            ScalarOps::TanH(a) => {
                self[a].grad += self[y].grad * (1.0 - self[y].grad*self[y].grad);

                self._reverse(a);
            }
            ScalarOps::ReLU(a) => {
                if self[y].data > 0.0 {
                    self[a].grad += self[y].grad * 1.0;
                }
                self._reverse(a);
            }
            ScalarOps::Empty => {},
        }
        

    }

    pub fn add(&mut self, a: Idx, b: Idx) -> Idx {
        let res = self.create_node(self[a].data + self[b].data);
        self[res].children = ScalarOps::Add(a, b);
        res
    }

    pub fn mul(&mut self, a: Idx, b: Idx) -> Idx {
        let res = self.create_node(self[a].data * self[b].data);
        self[res].children = ScalarOps::Mul(a, b);
        res
    }

    pub fn sub(&mut self, a: Idx, b: Idx) -> Idx {
        let res = self.create_node(self[a].data - self[b].data);
        self[res].children = ScalarOps::Sub(a, b);
        res
    }

    pub fn pow(&mut self, a: Idx, b: Idx) -> Idx {
        let res = self.create_node(self[a].data.powf(self[b].data));
        self[res].children = ScalarOps::Pow(a, b);
        res
    }

    pub fn relu(&mut self, a: Idx) -> Idx {
        let res = if self[a].data >= 0.0 {
            self.create_node(self[a].data)
        } else {
            self.create_node(0.0)
        };
        self[res].children = ScalarOps::ReLU(a);
        res
    }

    pub fn tanh(&mut self, a: Idx) -> Idx {
        let res = self.create_node(self[a].data.tanh());
        self[res].children = ScalarOps::ReLU(a);
        res
    }
    
}

impl Diff for Tape<f32, ScalarOps> {

    fn reverse(&mut self, y: Idx) {
        self[y].grad = 1.0;
        self._reverse(y);
    }
}

impl fmt::Display for Tape<f32, ScalarOps> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for node in self.vals.iter() {
            write!(f, "idx: {}, data: {}, grad: {}", node.idx, node.data, node.grad)?;
            match node.children {
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