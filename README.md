# rustydiff
Autodiff implementation in Rust for both `forward` and `reverse` mode automatic differentiation. This is still a proof of concept, for which I intend to find a better way to construct the computation graph in `reverse` mode, such that operator overloading can be applied and the `Tape` structure is implicit. `main.rs` contains examples for `forward` and `reverse` mode AD.  
