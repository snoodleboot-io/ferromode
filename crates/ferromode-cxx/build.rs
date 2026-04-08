fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("src/ferromode.cpp")
        .std("c++20")
        .include("include")
        .compile("ferromode-cxx");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/ferromode.cpp");
    println!("cargo:rerun-if-changed=include/ferromode.hpp");
}
