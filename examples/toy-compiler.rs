use clap::Parser;
use rsml::context::Context;
use rsml::toy;
use rsml::toy::mlir_gen::MLIRGen;
use std::collections::HashMap;
use std::env;
use std::rc::Rc;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// path to a file to process
    #[clap(short, long, value_parser)]
    filename: String,
    /// Name of the person to greet
    #[clap(short, long, value_parser)]
    emit: String,
}

fn main() {
    let args = Args::parse();

    let content = std::fs::read_to_string(args.filename).unwrap();
    let mut prec = HashMap::with_capacity(6);

    prec.insert('=', 2);
    prec.insert('<', 10);
    prec.insert('+', 20);
    prec.insert('-', 20);
    prec.insert('*', 40);
    prec.insert('/', 40);

    let module = toy::parser::Parser::new(content, &mut prec)
        .parse_module()
        .unwrap();
    if args.emit == String::from("ast") {
        for fun in module.functions {
            println!("-> Function parsed: \n{:#?}\n", fun);
        }
    } else if args.emit == String::from("mlir") {
        let context = Rc::new(Context::default());
        let module = MLIRGen::new(Rc::clone(&context)).mlir_gen(module);
        module.dump();
    }
}