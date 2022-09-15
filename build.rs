extern crate bindgen;

use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::process::Command;
use std::str;

fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    // FIXME: use relative path
    println!("cargo:rustc-link-search=/home/artemy/workspace/repos/rsml/cpp/toy/build/lib");

    // Tell cargo to tell rustc to link the system bzip2
    // shared library.
    println!("cargo:rustc-link-lib=static=ToyCAPI");
    println!("cargo:rustc-link-lib=static=MLIRToy");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        .clang_arg(format!(
            "-I{}",
            "/home/artemy/workspace/repos/rsml/cpp/toy/include/",
        ))
        .clang_arg(format!("-I{}", llvm_config("--includedir").unwrap()))
        .allowlist_function(".*toy.*")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn llvm_config(argument: &str) -> Result<String, Box<dyn Error>> {
    let call = format!("llvm-config --link-static {}", argument);

    Ok(str::from_utf8(
        &if cfg!(target_os = "windows") {
            Command::new("cmd").args(["/C", &call]).output()?
        } else {
            Command::new("sh").arg("-c").arg(&call).output()?
        }
        .stdout,
    )?
    .trim()
    .to_string())
}
