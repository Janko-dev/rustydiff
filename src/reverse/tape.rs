use std::cell::RefCell;
use num_traits::Zero;

pub trait Diff<'a, T, OP> {
    fn reverse(&self);
}

pub trait EmptyOp {
    fn empty_operator() -> Self;
}

pub type Idx = usize;

#[derive(Debug)]
pub struct Var<'a, T, OP> {
    pub tape: &'a Tape<T, OP>,
    pub data: T,
    pub idx: Idx
}

#[derive(Debug)]
pub struct CompNode<T, OP> {
    pub data: T, 
    pub grad: T,
    pub parents: OP
}

#[derive(Debug)]
pub struct Tape<T, OP> {
    pub nodes: RefCell<Vec<CompNode<T, OP>>>
}

impl <'a, T: Zero + Clone + Copy, OP: EmptyOp> Tape<T, OP> {
    pub fn new() -> Self {
        Self { nodes: RefCell::new(vec![]) }
    }

    pub fn var(&'a self, data: T) -> Var<'a, T, OP> {
        Var {
            tape: self,
            data: data,
            idx: self.create_var(data)
        }
    }

    fn create_var(&self, data: T) -> Idx {
        let len = self.nodes.borrow().len();
        self.nodes.borrow_mut().push(
            CompNode { 
                data: data, 
                grad: Zero::zero(), 
                parents: OP::empty_operator() 
            }
        );
        len
    }

    pub fn var_operator(&self, data: T, op: OP) -> Idx {
        let mut nodes_ref_mut = self.nodes.borrow_mut();
        let len = nodes_ref_mut.len();
        nodes_ref_mut.push(
            CompNode {
                data,
                grad: Zero::zero(), 
                parents: op
            }
        );
        len
    }
}

// impl<'a, T, OP> Index<Idx> for Tape<'a, T, OP> {
//     type Output = CompNode<'a, T, OP>;

//     fn index(&self, index: Idx) -> &Self::Output {
//         // Ref::map(self.vals, |x| x);
//         let x = self.vals.()[index];
//         &x
//         // if let Some(n) = self.vals.get_mut().get(index) {
//         //     n
//         // } else {
//         //     panic!()
//         // }
//         // &self.vals.borrow()[index]
//     }
// }

// impl<'a, T, OP> IndexMut<Idx> for Tape<'a, T, OP> {
//     fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
//         &mut self.vals[index]
//     }
// }