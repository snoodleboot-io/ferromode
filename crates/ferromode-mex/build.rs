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

    let mex_ext = if target.contains("windows") {
        "mexw64"
    } else if target.contains("darwin") {
        "mexmaci64"
    } else {
        "mexa64"
    };

    println!("cargo:rustc-cdylib-link-arg=-Wl,-undefined,dynamic_lookup");

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

    println!("cargo:rustc-cdylib-link-arg=-shared");

    let out_dir = env::var("OUT_DIR").unwrap_or_else(|_| "target".to_string());
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());

    let output_dir = PathBuf::from(&out_dir)
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .unwrap_or_else(|| PathBuf::from("target"))
        .join(&profile);

    println!("cargo:rustc-link-arg=-o");
    println!("cargo:rustc-link-arg={}/libferromode_mex.{}", output_dir.display(), mex_ext);
}

fn configure_octave() {
    println!("cargo:rustc-cdylib-link-arg=-shared");
    println!("cargo:rustc-cdylib-link-arg=-Wl,-undefined,dynamic_lookup");

    let out_dir = env::var("OUT_DIR").unwrap_or_else(|_| "target".to_string());
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());

    let output_dir = PathBuf::from(&out_dir)
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .unwrap_or_else(|| PathBuf::from("target"))
        .join(&profile);

    println!("cargo:rustc-link-arg=-o");
    println!("cargo:rustc-link-arg={}/libferromode_mex.oct", output_dir.display());
}
