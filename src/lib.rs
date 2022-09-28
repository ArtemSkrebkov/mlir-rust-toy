pub mod block;
pub mod context;
pub mod dialect;
pub mod location;
pub mod misc;
pub mod op_builder;
pub mod operation;
pub mod toy;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::rc::Rc;

    use crate::context::Context;
    use crate::dialect::StandardDialect;
    use crate::dialect::ToyDialect;
    use crate::toy::mlir_gen::MLIRGen;
    use crate::toy::parser;

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
        let mut context = Context::default();
        let dialect = ToyDialect::new(&context);
        context.load_dialect(Box::new(dialect));

        let std_dialect = StandardDialect::new(&context);
        context.load_dialect(Box::new(std_dialect));
    }

    #[test]
    fn generate_mlir_for_empty_ast() {
        let filename = "ast_empty.toy".to_string();
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
        // Need to make it mutable
        // let dialect = ToyDialect::new(&context);
        // context.load_dialect(Box::new(dialect));

        let module = parser::Parser::new(content, &mut prec)
            .parse_module()
            .unwrap();
        let module = MLIRGen::new(Rc::clone(&context)).mlir_gen(module);
        assert!(!module.block.operations.is_empty());
    }

    #[test]
    fn generate_mlir_for_ast_tensor() {
        let filename = "ast_tensor.toy".to_string();
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
        // Need to make it mutable
        let dialect = ToyDialect::new(&context);
        // context.load_dialect(Box::new(dialect));
        let module = parser::Parser::new(content, &mut prec)
            .parse_module()
            .unwrap();
        let module = MLIRGen::new(Rc::clone(&context)).mlir_gen(module);
        println!("Dump:");
        module.dump();
        assert!(!module.block.operations.is_empty());
    }
}
