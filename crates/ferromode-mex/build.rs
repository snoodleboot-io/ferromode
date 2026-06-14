//! Build script for ferromode-mex.
//!
//! Detects MATLAB vs Octave installation and sets correct link flags
//! and output extension.

use std::env;
use std::path::PathBuf;


fn main() {
    println!("cargo:rerun-if-changed=src/");

    let is_octave = env::var("FERROMODE_OCTAVE").is_ok();

    if is_octave {
        configure_octave();
    } else {
        configure_matlab();
    }
}

fn configure_matlab() {
    let target = env::var("TARGET").unwrap_or_default();

    if let Ok(matlab_root) = env::var("MATLAB_ROOT") {
        let lib_path =
            PathBuf::from(&matlab_root).join("bin").join(if target.contains("windows") {
                "win64"
            } else if target.contains("darwin") {
                "maci64"
            } else {
                "glnxa64"
            });

        if lib_path.exists() {
            println!("cargo:rustc-link-search=native={}", lib_path.display());
        }
    }
}

fn configure_octave() {
}
