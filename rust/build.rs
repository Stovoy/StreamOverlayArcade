fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-env=LIBTORCH_STATIC=1");
} 
