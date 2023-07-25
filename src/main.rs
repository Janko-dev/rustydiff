use rustydiff::reverse::{Diff, ScalarOps, Tape};
use rustydiff::forward::{F, FX};


fn reverse_autodiff_demo() {
    let tp = Tape::<f32, ScalarOps>::new();
    let x = tp.var(2.0);
    let exponent = tp.var(2.0);
    let cst = tp.var(5.0);
    let y = &cst * &x.powf(&exponent);
    
    y.reverse();
    println!("----------------------");
    println!("Reverse mode autodiff:");
    println!("example function f(x) = 5x^2");
    println!("                 f'(x) = 10x");
    println!("Tape of the computation graph:\n{}", tp);
    println!("f'(2) = {}", x.grad());
    println!("----------------------\n");
}

fn forward_autodiff_demo() {
    // derivative of f(x) = 5x^2 is f'(x) = 10x
    let f1 = |x: FX<f32>| F::cst(5.0) * x * x;
    // f'(2) = 10 * 2
    println!("----------------------");
    println!("Forward mode autodiff:");
    println!("example function f(x) = 5x^2");
    println!("                 f'(x) = 10x");
    
    println!("f'(2) = {:?}", f1(F::var(2.0)).deriv());
    println!("----------------------\n");
}

fn main() {
    forward_autodiff_demo();
    reverse_autodiff_demo();
}   
