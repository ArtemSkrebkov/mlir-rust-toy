# MLIR RUST TOY

## Motivation

The content of this repo is a humble attempt to implement [Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) using Rust.

Fortunately, MLIR framework provides a decent C API and the current project uses the MLIR C API to build
a Rust wrapper around it to be able to implement the Toy language in Rust.
However, some parts still require building C++ project :)

The current state of `main` branch corresponds to [chapter 6](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/).
So curious user can play with building AST, MLIR IR as well as executing Toy language sources with and without optimizations.
There are a few bugs and some optimizations are missed since they are not yet available in C API.
Hopefully, they will be resolved in near future.

## Building & playing

1. Make sure that you have LLVM 14.0.0 installed - at this point, this is the only compatible version
2. Build ODS description for Toy
```
mkdir -p ./cpp/toy/build
cd ./cpp/toy/build
cmake ..
make
```
3. Build and test
```
cargo build
cargo test
```
4. Run the Toy compiler
```
cargo run --example toy-compiler -- --filename ./testdata/reshape_opt.toy --emit mlir --opt
```

## Inspiration

* [Inkwell](https://github.com/TheDan64/inkwell)
* [Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)
