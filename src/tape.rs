
pub trait Diff {
    fn reverse(&mut self, y: Idx);
}

pub type Idx = usize;



#[derive(Debug)]
pub struct CompNode<T, Operators> {
    pub data: T, 
    pub grad: T,
    pub idx: Idx,
    pub children: Operators
}

#[derive(Debug)]
pub struct Tape<T, Operators> {
    pub vals: Vec<CompNode<T, Operators>>
}