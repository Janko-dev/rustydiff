
mod tape;
mod diff_f32;
mod diff_mat_f32;

pub mod autodiff {
    pub use crate::tape::*;
    pub use crate::diff_f32::*;

}