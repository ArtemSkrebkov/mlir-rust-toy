use clap::Parser;
use rsml::context::Context;
use rsml::pass_manager::PassManager;
use rsml::toy;
use rsml::toy::mlir_gen::MLIRGen;
use rsml::toy::toy_dialect::ToyDialect;
use std::collections::HashMap;
use std::rc::Rc;

/// a compiler for a language called Toy
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// path to a file to process
    #[clap(short, long, value_parser)]
    filename: String,
    /// Output of compiler. Possible values: ast, mlir
    #[clap(short, long, value_parser)]
    emit: String,
    /// Enable optimizations
    #[clap(short, long, value_parser, default_value_t = false)]
    opt: bool,
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
        let dialect = ToyDialect::new(&context);
        context.load_dialect(Box::new(dialect));
        let module = MLIRGen::new(Rc::clone(&context)).mlir_gen(module);
        if args.opt {
            let pass_manager = PassManager::new(Rc::clone(&context));
            let pass = PassManager::create_canonicalizer_pass();
            pass_manager.add_nested_pass(pass, "builtin.func");
            pass_manager.run(&module);
        }
        module.dump();
    }
}
