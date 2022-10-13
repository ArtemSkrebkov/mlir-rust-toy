pub mod block;
pub mod context;
pub mod dialect;
pub mod location;
pub mod misc;
pub mod op_builder;
pub mod operation;
pub mod pass_manager;
pub mod toy;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::rc::Rc;

    use crate::context::Context;
    use crate::dialect::StandardDialect;
    use crate::operation::ModuleOp;
    use crate::pass_manager::{self, PassManager};
    use crate::toy::mlir_gen::MLIRGen;
    use crate::toy::parser;
    use crate::toy::toy_dialect::ToyDialect;

    use test_case::test_case;

    #[test]
    fn create_context() {
        let _context = Context::default();
    }

    #[test]
    fn create_dialect() {
        let context = Context::default();
        let _dialect = ToyDialect::new(&context);
    }

    #[test]
    fn load_dialect() {
        let context = Context::default();
        let dialect = ToyDialect::new(&context);
        context.load_dialect(Box::new(dialect));

        let std_dialect = StandardDialect::new(&context);
        context.load_dialect(Box::new(std_dialect));
    }

    #[test_case("ast_empty", false; "when generate MLIR for empty main")]
    #[test_case("ast_tensor", false; "when generate MLIR with tensors")]
    #[test_case("ast", false; "when generate MLIR for complex file")]
    #[test_case("reshape_opt", true; "when optimizing reshape")]
    #[test_case("transpose_transpose_opt", true; "when optimizing transpose")]
    #[test_case("ast_tensor", true; "when inlining")]
    fn generate_mlir(filename: &str, is_opt: bool) {
        let filename = format!("testdata/{}.toy", filename);
        if filename.is_empty() {
            panic!("Cannot find file to read");
        }
        let content = std::fs::read_to_string(filename).unwrap();
        let mut prec = HashMap::with_capacity(6);

        prec.insert('=', 2);
        prec.insert('<', 10);
        prec.insert('+', 20);
        prec.insert('-', 20);
        prec.insert('*', 40);
        prec.insert('/', 40);

        let context = Rc::new(Context::default());
        let dialect = ToyDialect::new(&context);
        context.load_dialect(Box::new(dialect));
        let module = parser::Parser::new(content, &mut prec)
            .parse_module()
            .unwrap();
        let module = MLIRGen::new(Rc::clone(&context)).mlir_gen(module);

        if is_opt {
            let pass_manager = PassManager::new(Rc::clone(&context));
            let pass = PassManager::create_inliner_pass();
            pass_manager.add_owned_pass(pass);

            let pass = PassManager::create_canonicalizer_pass();
            pass_manager.add_nested_pass(pass, "builtin.func");

            let pass = PassManager::create_shape_inference_pass();
            pass_manager.add_nested_pass(pass, "builtin.func");

            pass_manager.run(&module);
        }
        println!("");
        module.dump();
        assert!(!module.block.operations.is_empty());
    }

    #[test]
    fn optimize_mlir() {
        let filename = "test_inliner";
        let filename = format!("testdata/{}.mlir", filename);
        if filename.is_empty() {
            panic!("Cannot find file to read");
        }
        let content = std::fs::read_to_string(filename).unwrap();
        let mut prec = HashMap::with_capacity(6);

        prec.insert('=', 2);
        prec.insert('<', 10);
        prec.insert('+', 20);
        prec.insert('-', 20);
        prec.insert('*', 40);
        prec.insert('/', 40);

        let context = Rc::new(Context::default());
        let dialect = ToyDialect::new(&context);
        context.load_dialect(Box::new(dialect));

        let module = ModuleOp::new_parsed(&context, &content);
        println!("before");
        module.dump();
        println!("");

        let pass_manager = PassManager::new(Rc::clone(&context));
        let pass = PassManager::create_inliner_pass();
        pass_manager.add_owned_pass(pass);

        let pass = PassManager::create_canonicalizer_pass();
        pass_manager.add_nested_pass(pass, "builtin.func");

        let pass = PassManager::create_cse_pass();
        pass_manager.add_nested_pass(pass, "builtin.func");

        let pass = PassManager::create_shape_inference_pass();
        pass_manager.add_nested_pass(pass, "builtin.func");

        pass_manager.run(&module);
        println!("after");
        module.dump();
    }

    #[test]
    fn lower_mlir_to_affine() {
        let filename = "test_lower_affine";
        let filename = format!("testdata/{}.mlir", filename);
        if filename.is_empty() {
            panic!("Cannot find file to read");
        }
        let content = std::fs::read_to_string(filename).unwrap();
        let mut prec = HashMap::with_capacity(6);

        prec.insert('=', 2);
        prec.insert('<', 10);
        prec.insert('+', 20);
        prec.insert('-', 20);
        prec.insert('*', 40);
        prec.insert('/', 40);

        let context = Rc::new(Context::default());
        let dialect = ToyDialect::new(&context);
        context.load_dialect(Box::new(dialect));

        let module = ModuleOp::new_parsed(&context, &content);
        println!("before");
        module.dump();
        println!("");

        let pass_manager = PassManager::new(Rc::clone(&context));
        let pass = PassManager::create_inliner_pass();
        pass_manager.add_owned_pass(pass);

        let pass = PassManager::create_canonicalizer_pass();
        pass_manager.add_nested_pass(pass, "builtin.func");

        let pass = PassManager::create_cse_pass();
        pass_manager.add_nested_pass(pass, "builtin.func");

        let pass = PassManager::create_shape_inference_pass();
        pass_manager.add_nested_pass(pass, "builtin.func");

        let pass = PassManager::create_lower_to_affine_pass();
        pass_manager.add_nested_pass(pass, "builtin.func");
        // TODO: in original mlir tutorial, they add LoopFusion and MemRefDataFlowOpt
        // but those are not available currently in mlir-c api

        pass_manager.run(&module);
        println!("after");
        module.dump();
    }
}
