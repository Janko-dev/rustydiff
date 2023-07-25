# Rustydiff
`rustydiff` is an [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) (`autodiff`) implementation written in Rust for educational and recreational purposes. Recently, I did a reverse mode autodiff implementation in C, for which the repository can be found [here](https://github.com/Janko-dev/autodiff). However, I was wondering whether I would run into issues doing the same implementation in Rust, as the implementation of directed acyclic graphs is considered somewhat difficult by the Rust community. That said, the resulting API is much cleaner than the C version. 

Currently, both `forward` and `reverse` mode autodiff are implemented on scalar `f32` values. In forward mode, the derivatives are calculated as the computations are performed. In reverse mode, the derivatices are calculated after all computations are perfomed. There is a backward pass through the computation graph, in which the gradients 'flow' from the roots, i.e., the results of the computation, to the leaves, i.e., the input variables.  

## Reverse mode example
```Rust
    use rustydiff::reverse::{Diff, ScalarOps, Tape};

    let tp = Tape::<f32, ScalarOps>::new();
    // weights
    let ws = vec![tp.var(0.4), tp.var(0.8), tp.var(0.1)];
    // inputs
    let inp = vec![tp.var(2.0), tp.var(4.0), tp.var(6.0)];

    let mut res = tp.var(0.0);
    // linear combination
    for (w, i) in ws.iter().zip(inp.iter()) {
        res += w * i;
    }
    // non-linear activation function
    let res = res.tanh();
    // backward pass
    res.reverse();

    // derivative w.r.t ws[1] = 0.8 is
    // f'(inp) = inp[1] * (1 - tanh^2(inp[0] * ws[0] + inp[1] * ws[1] + inp[2] * ws[2]))
    assert!(f32_equal(ws[1].grad(), 4.0 * (1.0 - (0.4f32*2.0 + 0.8*4.0 + 0.1*6.0).tanh().powf(2.0))));
```

## Forward mode
Forward mode is still incomplete as there are not many operators implemented. 
```Rust
    use rustydiff::forward::{F, FX};

    // derivative of f(x) = 5x^2 is f'(x) = 10x
    let f = |x: FX<f32>| F::cst(5.0) * x * x;

    assert!(f(F::var(2.0)).deriv() == 20.0);
```