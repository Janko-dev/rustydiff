// use ndarray::Array3;
// use std::ops::{Index, IndexMut};


use rustydiff::reverse::{Diff, ScalarOps, Tape};
use rustydiff::forward::F;


fn reverse_autodiff_demo() {
    let mut tp = Tape::<f32, ScalarOps>::new();
    let x = tp.create_node(2.0);
    let cst = tp.create_node(5.0);
    let xx = tp.mul(x, x);
    let y  = tp.mul(cst, xx);
    
    tp.reverse(y);
    println!("{}", tp);
    println!("x: {}, dx: {}", tp[x].data, tp[x].grad);
}

fn forward_autodiff_demo() {
    // f(x) = 5x^2
    // f'(x) = 10x
    let f1 = |x: F<f32, f32>| F::cst(5.0) * x * x;
    // f'(2) = 10 * 2
    println!("{:?}", f1(F::var(2.0)));
}

fn main() {
    // forward_autodiff_demo();
    // reverse_autodiff_demo();

    let f = |x: F<f32, f32>| F::cst(3.0) * x * x / F::cst(10.0);
    println!("f'(5) = {:?}", f(F::var(5.0)).deriv());

}   
