use std::ops::{Index, IndexMut};
use num_traits::Zero;

pub trait Diff {
    fn reverse(&mut self, y: Idx);
}

pub trait EmptyOp {
    fn empty_operator() -> Self;
}

pub type Idx = usize;

#[derive(Debug)]
pub struct CompNode<T, Operators> {
    pub data: T, 
    pub grad: T,
    pub idx: Idx,
    pub children: Operators
}

impl<T: Zero, Operators: EmptyOp> CompNode<T, Operators> {
    pub fn new(data: T) -> Self {
        Self { data, grad: T::zero(), idx: 0, children: EmptyOp::empty_operator() }
    }
}

#[derive(Debug)]
pub struct Tape<T, Operators> {
    pub vals: Vec<CompNode<T, Operators>>
}

impl<T, Operators> Index<Idx> for Tape<T, Operators> {
    type Output = CompNode<T, Operators>;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.vals[index]
    }
}

impl<T, Operators> IndexMut<Idx> for Tape<T, Operators> {
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.vals[index]
    }
}