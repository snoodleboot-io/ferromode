//! R bindings for Ferromode via extendr.

use extendr_api::prelude::*;

/// Return the version of the Ferromode R bindings.
#[extendr]
fn ferromode_r_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

extendr_module! {
    mod ferromode_r;
    fn ferromode_r_version;
}
