// We need to forward routine registration from C to Rust to avoid the linker
// removing the static library (and to give R an object file to compile, so the
// `ferromodeR.so` shared object is actually produced on install).

void R_init_ferromodeR_extendr(void *dll);

// Standard R package initialization
void R_init_ferromodeR(void *dll) {
    R_init_ferromodeR_extendr(dll);
}
