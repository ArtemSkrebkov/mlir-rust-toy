pub mod parser;

use std::collections::HashMap;
use std::ffi::{CStr, CString};

use mlir_sys::{
    mlirContextCreate, mlirContextGetOrLoadDialect, mlirDialectHandleGetNamespace,
    mlirGetDialectHandle__std__, mlirLocationUnknownGet, mlirNoneTypeGet, mlirOperationCreate,
    mlirOperationGetResult, mlirOperationStateGet, mlirStringRefCreateFromCString,
};
use mlir_sys::{
    MlirContext, MlirDialectHandle, MlirLocation, MlirOperation, MlirOperationState, MlirType,
    MlirValue,
};

use crate::parser::Expr::{Binary, Call, ExprList, Number, Return, Tensor, VarDecl};
use parser::{Expr, Function, Module, Prototype};
pub trait Dialect {
    fn get_name(&self) -> String;
}

pub struct Context {
    instance: MlirContext,
    dialects: Vec<Box<dyn Dialect>>,
}

impl Context {
    pub fn new() -> Self {
        unsafe {
            let instance = mlirContextCreate();
            Self {
                instance,
                dialects: Vec::new(),
            }
        }
    }

    pub fn load_dialect(&mut self, dialect: Box<dyn Dialect>) {
        self.dialects.push(dialect);
        println!(
            "Dialect {} loaded",
            self.dialects.last().unwrap().get_name()
        );
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
pub struct ToyDialect {
    // context: &'a Context,
    // name
    ops: Vec<Box<dyn Op>>,
}

impl ToyDialect {
    pub fn new(_context: &Context) -> Self {
        // let ops = Vec::new();
        let ops: Vec<Box<dyn Op>> = vec![
            Box::new(ConstantOp::default()),
            Box::new(PrintOp::default()),
        ];
        Self { ops }
    }
}

impl Dialect for ToyDialect {
    fn get_name(&self) -> String {
        String::from("toy")
    }
}

pub struct StandardDialect {
    std_handle: MlirDialectHandle,
}

impl StandardDialect {
    pub fn new(_context: &Context) -> Self {
        unsafe {
            let std_handle = mlirGetDialectHandle__std__();
            let std = mlirContextGetOrLoadDialect(
                _context.instance,
                mlirDialectHandleGetNamespace(std_handle),
            );
            Self { std_handle }
        }
    }
}

impl Dialect for StandardDialect {
    fn get_name(&self) -> String {
        unsafe {
            let namespace = mlirDialectHandleGetNamespace(self.std_handle);
            let c_str: &CStr = unsafe { CStr::from_ptr(namespace.data) };
            let str_slice: &str = c_str.to_str().unwrap();
            let str_buf: String = str_slice.to_owned();
            str_buf
        }
    }
}

#[derive(Clone)]
struct OperationState {
    instance: MlirOperationState,
}

impl OperationState {
    fn new(name: &str, location: Location) -> Self {
        let string = CString::new(name).unwrap();
        let reference = unsafe { mlirStringRefCreateFromCString(string.as_ptr()) };
        let instance = unsafe { mlirOperationStateGet(reference, location.instance) };

        Self { instance }
    }
}

#[derive(Clone)]
pub struct Operation {
    state: OperationState,
}

impl Operation {
    fn new(state: OperationState) -> Self {
        Self { state }
    }
}

pub trait OneRegion {
    fn push_back(&mut self, operation: Box<Operation>);
}

#[derive(Clone)]
struct Block {
    operations: Vec<Box<Operation>>,
}

impl Default for Block {
    fn default() -> Self {
        Self {
            operations: Vec::new(),
        }
    }
}

struct ConstantOp {
    name: String,
    instance: MlirOperation,
}

impl ConstantOp {
    pub fn new() -> Self {
        Self::new_with_location(Location::new(Context::default()))
    }

    pub fn new_with_location(location: Location) -> Self {
        unsafe {
            // FIXME: duplication toy.constant
            let name = String::from("toy.constant");
            let mut state = OperationState::new("toy.constant", location);
            let p_state: *mut MlirOperationState = &mut state.instance;
            let instance = mlirOperationCreate(p_state);

            Self { name, instance }
        }
    }
}

impl Default for ConstantOp {
    fn default() -> Self {
        Self::new()
    }
}

impl Op for ConstantOp {
    fn build(&self) -> Self {
        Self::new()
    }
}

struct PrintOp {
    name: String,
}

impl PrintOp {
    pub fn new() -> Self {
        let name = String::from("Print");
        Self { name }
    }
}

impl Default for PrintOp {
    fn default() -> Self {
        Self::new()
    }
}

impl Op for PrintOp {
    fn build(&self) -> Self {
        Self::new()
    }
}

impl From<ConstantOp> for Value {
    fn from(op: ConstantOp) -> Self {
        // use op to construct Value
        let instance = unsafe { mlirOperationGetResult(op.instance, 0) };
        Value::new(instance)
    }
}

struct Location {
    instance: MlirLocation,
}

impl Location {
    pub fn new(context: Context) -> Self {
        let instance = unsafe { mlirLocationUnknownGet(context.instance) };
        Self { instance }
    }
}

#[derive(Clone)]
struct ModuleOp {
    module: MlirOperation,
    block: Block,
}

impl ModuleOp {
    pub fn new() -> Self {
        // FIXME: should be shared between all ops
        let context = Context::default();
        let location = Location::new(context);
        Self::new_with_location(location)
    }

    pub fn new_with_location(location: Location) -> Self {
        unsafe {
            let mut state = OperationState::new("builtin.module", location);
            let p_state: *mut MlirOperationState = &mut state.instance;
            let module = mlirOperationCreate(p_state);

            Self {
                module,
                block: Block::default(),
            }
        }
    }
}

impl OneRegion for ModuleOp {
    fn push_back(&mut self, operation: Box<Operation>) {
        self.block.operations.push(operation);
    }
}

impl Default for ModuleOp {
    fn default() -> Self {
        Self::new()
    }
}

struct FuncOp {
    function: MlirOperation,
    block: Block,
}

impl FuncOp {
    pub fn new_with_location(location: Location) -> Self {
        unsafe {
            let mut state = OperationState::new("builtin.func", location);
            let p_state: *mut MlirOperationState = &mut state.instance;
            let function = mlirOperationCreate(p_state);

            Self {
                function,
                block: Block::default(),
            }
        }
    }
}

pub trait Op {
    fn build(&self) -> Self
    where
        Self: Sized;
}
struct OpBuilder {}

impl OpBuilder {
    pub fn new(context: Context) -> Self {
        Self {}
    }

    pub fn unknown_loc(&self) -> Location {
        let context = Context::default();
        Location::new(context)
    }

    pub fn create_operation<OpTy: Op>(&self, op: OpTy) -> OpTy {
        op.build()
    }
}

#[derive(Clone)]
struct Type {
    instance: MlirType,
}

impl Type {
    fn new(context: Context) -> Self {
        let instance = unsafe { mlirNoneTypeGet(context.instance) };
        Self { instance }
    }
}

struct Value {
    instance: MlirValue,
}

impl Value {
    // TODO: make it available only for crate but not for user
    pub fn new(instance: MlirValue) -> Value {
        Self { instance }
    }
}

struct MLIRGen {
    module: ModuleOp,
    builder: OpBuilder,
}

impl MLIRGen {
    pub fn new(context: Context) -> Self {
        Self {
            module: ModuleOp::default(),
            builder: OpBuilder::new(context),
        }
    }
    // TODO: pass parsed result
    pub fn mlir_gen(&mut self, module_ast: Module) -> ModuleOp {
        self.module = ModuleOp::new_with_location(self.builder.unknown_loc());

        // TODO: implement Iterator for Module?
        for f in module_ast.functions {
            let func = self.mlir_gen_function(f);
            // add func into self.module
        }

        // TODO: verify self.module

        self.module.clone()
    }

    fn mlir_gen_function(&mut self, function_ast: Function) -> FuncOp {
        // varScopeTable
        let var_scope: HashMap<&str, Value> = HashMap::new();
        let function: FuncOp = self.mlir_gen_prototype(function_ast.prototype.clone());

        let entry_block = function.block.clone();
        // TODO: getter for prototype
        let proto_args = function_ast.prototype.args.clone();
        // TODO: declare all the function arguments in the symbol table
        // TODO: implement builder method
        // self.builder.set_insertion_point_to_start(entry_block);

        self.mlir_gen_expression(function_ast.body.unwrap());
        // function.erase();

        // TODO: handle return op

        function
    }

    fn mlir_gen_prototype(&mut self, prototype_ast: Prototype) -> FuncOp {
        // FIXME: construct location from AST location
        let location = Location::new(Context::default());
        // FIXME: convert VarType to mlirType
        // let arg_types = vec![Type::new(Context::default()); prototype_ast.args.len()];
        // let func_type = self.builder.get_function_type(arg_types, llvm::None);

        // FIXME: create function op with prototype name and funcType
        FuncOp::new_with_location(location)
    }

    fn mlir_gen_expression(&mut self, expr: Expr) {
        // NB: this clone is used for collect_data method
        // there should be a way to avoid this
        let clone_expr = expr.clone();
        match expr {
            ExprList { expressions } => {
                for expr in expressions {
                    self.mlir_gen_expression(*expr.clone());
                    // println!("ExprList: {:#?}", expr);
                    println!("ExprList");
                }
            }
            VarDecl { name, value } => {
                println!("VarDecl: {:#?} {}", value, name);
                // expr
            }
            Tensor {
                location,
                values,
                dims,
            } => {
                println!("Tensor: ");
                let size = dims.iter().product();
                let mut data: Vec<f64> = Vec::new();
                data.reserve(size);
                self.collect_data(clone_expr, &mut data);
                let ty = Type::new(Context::default());
                let value = Value::from(self.builder.create_operation(ConstantOp::new()));
            }
            Number(num) => {
                println!("Num {}", num);
            }
            Call { fn_name, args } => {
                println!("Call");
            }
            Return {
                location,
                expression,
            } => {
                println!("Return");
            }
            Binary { op, left, right } => {
                println!("Binary");
            }

            _ => {
                panic!("Unknown expression");
            }
        }
    }

    fn collect_data(&self, expr: Expr, data: &mut Vec<f64>) {
        match expr {
            Tensor {
                location: _,
                values,
                dims: _,
            } => {
                for v in values {
                    self.collect_data(v.clone(), data);
                }
            }
            Number(num) => {
                data.push(num);
            }
            _ => {
                panic!("Unexpected expression");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn generate_mlir() {
        let filename = "ast.toy".to_string();
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

        let context = Context::default();
        let module = parser::Parser::new(content, &mut prec)
            .parse_module()
            .unwrap();
        let module = MLIRGen::new(context).mlir_gen(module);
        assert!(module.block.operations.is_empty());
    }
}
