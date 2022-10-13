use clap::Parser;
use rsml::context::Context;
use rsml::operation::ModuleOp;
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

    if !args.filename.contains(".toy") && !args.filename.contains(".mlir") {
        panic!("Only .toy and .mlir supported!");
    }

    let content = std::fs::read_to_string(args.filename.clone()).unwrap();
    let mut prec = HashMap::with_capacity(6);

    prec.insert('=', 2);
    prec.insert('<', 10);
    prec.insert('+', 20);
    prec.insert('-', 20);
    prec.insert('*', 40);
    prec.insert('/', 40);

    let ast_module = if args.filename.contains(".toy") {
        Some(
            toy::parser::Parser::new(content.clone(), &mut prec)
                .parse_module()
                .unwrap(),
        )
    } else {
        None
    };
    if args.emit == String::from("ast") {
        for fun in ast_module.unwrap().functions {
            println!("-> Function parsed: \n{:#?}\n", fun);
        }
    } else if args.emit == String::from("mlir") || args.emit == String::from("mlir-affine") {
        let context = Rc::new(Context::default());
        let dialect = ToyDialect::new(&context);
        context.load_dialect(Box::new(dialect));
        let module = if let Some(module) = ast_module {
            MLIRGen::new(Rc::clone(&context)).mlir_gen(module)
        } else {
            ModuleOp::new_parsed(&context, &content)
        };
        let pass_manager = PassManager::new(Rc::clone(&context));
        let pass = PassManager::create_inliner_pass();
        pass_manager.add_owned_pass(pass);
        if args.opt {
            let pass = PassManager::create_canonicalizer_pass();
            pass_manager.add_nested_pass(pass, "builtin.func");

            let pass = PassManager::create_cse_pass();
            pass_manager.add_nested_pass(pass, "builtin.func");

            let pass = PassManager::create_shape_inference_pass();
            pass_manager.add_nested_pass(pass, "builtin.func");
        }

        if args.emit == String::from("mlir-affine") {
            let pass = PassManager::create_lower_to_affine_pass();
            pass_manager.add_nested_pass(pass, "builtin.func");
            // TODO: in original mlir tutorial, they add LoopFusion and MemRefDataFlowOpt
            // but those are not available currently in mlir-c api
        }
        pass_manager.run(&module);

        module.dump();
    }
}
