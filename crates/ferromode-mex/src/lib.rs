//! MATLAB/Octave MEX bindings for Ferromode.

use mex::prelude::*;

/// Entry point for the MEX function.
#[mex_function]
fn ferromode_mex(_func: &mut Function) -> Result<()> {
    Ok(())
}
