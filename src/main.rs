// use ndarray::Array3;
// use std::ops::{Index, IndexMut};


use rustydiff::autodiff::{self, Diff, ScalarOps};


// impl std::ops::Add for Idx {
//     type Output = Idx;

//     fn add(self, rhs: Self) -> Self::Output {
//         let res = 
        
//     }
// }

// struct Graph<T> {
//     nodes: Vec<NodeData<T>>,
//     edges: Vec<EdgeData>
// }

// struct NodeData<T> {
//     data: T,
//     index: Idx
// }

// struct EdgeData {
//     from: Idx,
//     to: Idx
// }

fn main() {
    let mut tp = autodiff::Tape::<f32, ScalarOps>::new();
    let mut a = tp.create_node(5.0);
    let b = tp.create_node(2.0);
    let c = tp.create_node(2.0);
    a = tp.add(a, b);
    let y = tp.mul(c, a);
    // (2 + 5) * 2 = 14
    // 
    tp.reverse(y);
    println!("{}", tp);
}   
