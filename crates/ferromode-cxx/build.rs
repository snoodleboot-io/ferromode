fn main() {
    // The crate is a plain cdylib that re-exports the core C ABI; no C++ is
    // compiled here. C++ consumers link the produced shared library against the
    // header in include/ferromode.hpp.
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=include/ferromode.hpp");
}
