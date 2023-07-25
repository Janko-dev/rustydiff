use std::cell::RefCell;
use num_traits::Zero;

pub trait Diff<'a, T, OP> {
    fn reverse(&self);
}

// refers to the operator for which its arity is 0, 
// i.e., the leaves of the computation graph
pub trait EmptyOp {
    fn empty_operator() -> Self;
}

pub type Idx = usize;

// Variable is independent of the actual computation graph.
// It has a reference to the tape, i.e., the comp graph, 
// however, it also owns the data which is useful for operator overloading.
#[derive(Debug)]
pub struct Var<'a, T, OP> {
    pub tape: &'a Tape<T, OP>,
    pub data: T,
    pub idx: Idx
}

// The computation node is a dual value of data and associated gradient 
// both of type T. The parents of a node refer to the specific operation
// that was performed to obtain the current value of data.
#[derive(Debug)]
pub struct CompNode<T, OP> {
    pub data: T, 
    pub grad: T,
    pub parents: OP
}

// The tape is a list of nodes within a RefCell, so that 
// borrowing rules are only enforced during runtime. 
// With this approach, it is possible to scatter references of the tape
// across many variables without the borrowing rules complicating things.  
#[derive(Debug)]
pub struct Tape<T, OP> {
    pub nodes: RefCell<Vec<CompNode<T, OP>>>
}

impl <T, OP> Tape<T, OP> 
where 
    T: Zero + Clone + Copy,
    OP: EmptyOp
{
    pub fn new() -> Self {
        Self { nodes: RefCell::new(vec![]) }
    }

    pub fn var<'a>(&'a self, data: T) -> Var<'a, T, OP> {
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

impl <'a, T, OP> Var<'a, T, OP> 
where 
    T: Clone
{
    pub fn grad(&self) -> T {
        let nodes_ref = self.tape.nodes.borrow();
        nodes_ref[self.idx].grad.clone()
    }
}