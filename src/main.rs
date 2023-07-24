// use ndarray::Array3;
// use std::ops::{Index, IndexMut};

use rustydiff::reverse::{Diff, ScalarOps, Tape, powf, relu, tanh};
// use rustydiff::forward::{F, FX};


// fn reverse_autodiff_demo() {
//     let mut tp = Tape::<f32, ScalarOps>::new();
//     let x = tp.create_node(2.0);
//     let cst = tp.create_node(5.0);
//     let xx = tp.mul(x, x);
//     let y  = tp.mul(cst, xx);
    
//     tp.reverse(y);
//     println!("Tape:\n{}", tp);
//     println!("f'(2) = {}", tp[x].grad);
// }

// fn forward_autodiff_demo() {
//     // f(x) = 5x^2
//     // f'(x) = 10x
//     let f1 = |x: F<f32, f32>| F::cst(5.0) * x * x;
//     // f'(2) = 10 * 2
//     println!("f(x) = 5x^2");
//     println!("f'(x) = 10x");
    
//     println!("f'(2) = {:?}", f1(F::var(2.0)).deriv());
// }

fn main() {
    // forward_autodiff_demo();
    // reverse_autodiff_demo();

    let tp = Tape::<f32, ScalarOps>::new();
    let x = tp.var(1.4);
    let y = tp.var(0.7);
    let z = tanh(x * y);

    z.reverse();
    println!("{}", tp);
}   
