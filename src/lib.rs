pub mod parser;

use std::collections::HashMap;
use std::ffi::{CStr, CString};

use std::rc::Rc;

use mlir_sys::{
    mlirAttributeParseGet, mlirBlockCreate, mlirBlockInsertOwnedOperation,
    mlirBlockInsertOwnedOperationAfter, mlirContextAppendDialectRegistry, mlirContextCreate,
    mlirContextGetNumLoadedDialects, mlirContextGetNumRegisteredDialects,
    mlirContextGetOrLoadDialect, mlirContextSetAllowUnregisteredDialects,
    mlirDialectHandleGetNamespace, mlirDialectRegistryCreate, mlirGetDialectHandle__std__,
    mlirIdentifierGet, mlirLocationGetContext, mlirLocationUnknownGet, mlirModuleCreateEmpty,
    mlirModuleGetBody, mlirModuleGetOperation, mlirNamedAttributeGet, mlirNoneTypeGet,
    mlirOperationCreate, mlirOperationDump, mlirOperationGetBlock, mlirOperationGetResult,
    mlirOperationStateAddAttributes, mlirOperationStateAddOwnedRegions, mlirOperationStateGet,
    mlirRegionAppendOwnedBlock, mlirRegionCreate, mlirRegisterAllDialects,
    mlirStringRefCreateFromCString,
};
use mlir_sys::{
    MlirBlock, MlirContext, MlirDialectHandle, MlirLocation, MlirModule, MlirNamedAttribute,
    MlirOperation, MlirOperationState, MlirRegion, MlirType, MlirValue,
};

use crate::parser::Expr::{Binary, Call, ExprList, Number, Return, Tensor, VarDecl, Variable};
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
            // FIXME: let's register all dialects
            // let registry = mlirDialectRegistryCreate();
            println!(
                "Registered dialects num {}",
                mlirContextGetNumLoadedDialects(instance)
            );
            println!(
                "Load dialects num {}",
                mlirContextGetNumLoadedDialects(instance)
            );
            // FIXME: make dialects to be registered separately
            mlirRegisterAllDialects(instance);
            mlirContextSetAllowUnregisteredDialects(instance, true);
            println!(
                "Registered dialects num {}",
                mlirContextGetNumLoadedDialects(instance)
            );
            println!(
                "Load dialects num {}",
                mlirContextGetNumLoadedDialects(instance)
            );
            // mlirContextAppendDialectRegistry(instance, registry);
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
        let ops: Vec<Box<dyn Op>> = vec![Box::new(PrintOp::default())];
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
    // NB: to make string live long enough
    string: CString,
}

impl OperationState {
    fn new(name: &str, location: Location) -> Self {
        let string = CString::new(name).unwrap();
        let reference = unsafe { mlirStringRefCreateFromCString(string.as_ptr()) };
        let instance = unsafe { mlirOperationStateGet(reference, location.instance) };

        Self { instance, string }
        // Self { instance }
    }
}

#[derive(Clone)]
pub struct Operation {
    state: OperationState,
    instance: MlirOperation,
}

impl Operation {
    fn new(mut state: OperationState) -> Self {
        let p_state: *mut MlirOperationState = &mut state.instance;
        let instance = unsafe { mlirOperationCreate(p_state) };
        Self { state, instance }
    }
}

pub trait OneRegion {
    fn push_back(&mut self, operation: Box<Operation>);
}

#[derive(Clone)]
struct Block {
    operations: Vec<Box<Operation>>,
    instance: MlirBlock,
}

impl Block {
    pub fn new(mlir_block: MlirBlock) -> Self {
        Self {
            operations: Vec::new(),
            instance: mlir_block,
        }
    }
}

#[derive(Clone)]
struct ConstantOp {
    name: String,
    instance: MlirOperation,
    value: f64,
    state: OperationState,
}

impl ConstantOp {
    pub fn new(location: Location, value: f64) -> Self {
        unsafe {
            // FIXME: duplication toy.constant
            let name = String::from("toy.constant");
            let mut state = OperationState::new("toy.constant", location);
            let p_state: *mut MlirOperationState = &mut state.instance;
            let instance = mlirOperationCreate(p_state);

            Self {
                name,
                instance,
                value,
                state,
            }
        }
    }
}

impl From<ConstantOp> for Value {
    fn from(op: ConstantOp) -> Self {
        // use op to construct Value
        let instance = unsafe { mlirOperationGetResult(op.instance, 0) };
        Value::new(instance)
    }
}

impl From<ConstantOp> for Operation {
    fn from(op: ConstantOp) -> Self {
        Self {
            state: op.state,
            instance: op.instance,
        }
    }
}

#[derive(Clone)]
struct TransposeOp {
    name: String,
    instance: MlirOperation,
    input: Value,
    state: OperationState,
}

// TODO: implementation looks the same to ConstantOp, besides the name
impl TransposeOp {
    pub fn new(location: Location, input: Value) -> Self {
        unsafe {
            // FIXME: duplication toy.constant
            let name = String::from("toy.transpose");
            let mut state = OperationState::new("toy.transpose", location);
            let p_state: *mut MlirOperationState = &mut state.instance;
            let instance = mlirOperationCreate(p_state);

            Self {
                name,
                instance,
                input,
                state,
            }
        }
    }
}

impl From<TransposeOp> for Value {
    fn from(op: TransposeOp) -> Self {
        // use op to construct Value
        let instance = unsafe { mlirOperationGetResult(op.instance, 0) };
        Value::new(instance)
    }
}

impl From<TransposeOp> for Operation {
    fn from(op: TransposeOp) -> Self {
        Self {
            state: op.state,
            instance: op.instance,
        }
    }
}

#[derive(Clone)]
struct GenericCallOp {
    name: String,
    instance: MlirOperation,
    callee: String,
    operands: Vec<Value>,
    state: OperationState,
}

// TODO: implementation looks the same to ConstantOp, besides the name
impl GenericCallOp {
    pub fn new(location: Location, callee: String, operands: Vec<Value>) -> Self {
        unsafe {
            // FIXME: duplication toy.constant
            let name = String::from("toy.generic_call");
            let mut state = OperationState::new("toy.generic_call", location);
            let p_state: *mut MlirOperationState = &mut state.instance;
            let instance = mlirOperationCreate(p_state);

            Self {
                name,
                instance,
                callee,
                operands,
                state,
            }
        }
    }
}

impl From<GenericCallOp> for Value {
    fn from(op: GenericCallOp) -> Self {
        // use op to construct Value
        let instance = unsafe { mlirOperationGetResult(op.instance, 0) };
        Value::new(instance)
    }
}

impl From<GenericCallOp> for Operation {
    fn from(op: GenericCallOp) -> Self {
        Self {
            state: op.state,
            instance: op.instance,
        }
    }
}

struct ReturnOp {
    name: String,
    instance: MlirOperation,
    input: Value,
}

impl ReturnOp {
    pub fn new(location: Location, input: Value) -> Self {
        unsafe {
            // FIXME: duplication toy.constant
            let name = String::from("toy.return");
            let mut state = OperationState::new("toy.return", location);
            let p_state: *mut MlirOperationState = &mut state.instance;
            let instance = mlirOperationCreate(p_state);

            Self {
                name,
                instance,
                input,
            }
        }
    }
}

impl From<ReturnOp> for Value {
    fn from(op: ReturnOp) -> Self {
        // use op to construct Value
        let instance = unsafe { mlirOperationGetResult(op.instance, 0) };
        Value::new(instance)
    }
}

struct AddOp {
    name: String,
    instance: MlirOperation,
    lhs: Value,
    rhs: Value,
}

impl AddOp {
    pub fn new(location: Location, lhs: Value, rhs: Value) -> Self {
        unsafe {
            // FIXME: duplication toy.constant
            let name = String::from("toy.add");
            let mut state = OperationState::new("toy.add", location);
            let p_state: *mut MlirOperationState = &mut state.instance;
            let instance = mlirOperationCreate(p_state);

            Self {
                name,
                instance,
                lhs,
                rhs,
            }
        }
    }
}

impl From<AddOp> for Value {
    fn from(op: AddOp) -> Self {
        // use op to construct Value
        let instance = unsafe { mlirOperationGetResult(op.instance, 0) };
        Value::new(instance)
    }
}

struct MulOp {
    name: String,
    instance: MlirOperation,
    lhs: Value,
    rhs: Value,
}

impl MulOp {
    pub fn new(location: Location, lhs: Value, rhs: Value) -> Self {
        unsafe {
            // FIXME: duplication toy.constant
            let name = String::from("toy.mul");
            let mut state = OperationState::new("toy.mul", location);
            let p_state: *mut MlirOperationState = &mut state.instance;
            let instance = mlirOperationCreate(p_state);

            Self {
                name,
                instance,
                lhs,
                rhs,
            }
        }
    }
}

impl From<MulOp> for Value {
    fn from(op: MulOp) -> Self {
        // use op to construct Value
        let instance = unsafe { mlirOperationGetResult(op.instance, 0) };
        Value::new(instance)
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

struct Location {
    instance: MlirLocation,
}

impl Location {
    pub fn new(context: &Context) -> Self {
        let instance = unsafe { mlirLocationUnknownGet(context.instance) };
        Self { instance }
    }
}

#[derive(Clone)]
struct ModuleOp {
    instance: MlirModule,
    // state: OperationState,
    block: Block,
    pos: isize,
}

impl ModuleOp {
    pub fn new(location: Location) -> Self {
        unsafe {
            // let mut state = OperationState::new("builtin.module", location);
            // let p_state: *mut MlirOperationState = &mut state.instance;
            // let instance = mlirOperationCreate(p_state);
            let instance = mlirModuleCreateEmpty(location.instance);
            let mlir_block = mlirModuleGetBody(instance);

            Self {
                instance,
                // state,
                block: Block::new(mlir_block),
                pos: 0,
            }
        }
    }

    pub fn dump(&self) {
        unsafe {
            let mlir_operation = mlirModuleGetOperation(self.instance);
            mlirOperationDump(mlir_operation);
        };
    }
}

impl OneRegion for ModuleOp {
    fn push_back(&mut self, operation: Box<Operation>) {
        unsafe {
            mlirBlockInsertOwnedOperation(self.block.instance, self.pos, operation.instance);
            self.pos += 1;
        }
        self.block.operations.push(operation);
    }
}

struct FuncOp {
    instance: MlirOperation,
    state: OperationState,
    block: Rc<Block>,
    name: CString,
}

impl FuncOp {
    pub fn new(mut location: Location, name: &str) -> Self {
        unsafe {
            let mlir_context = mlirLocationGetContext(location.instance);
            let mlir_region = mlirRegionCreate();
            let p_mlir_region: *const MlirRegion = &mlir_region;
            let num_args = 0;
            let mut args = mlirNoneTypeGet(mlir_context);
            let p_args: *mut MlirType = &mut args;

            let p_locs = &mut location.instance;

            let mlir_block = mlirBlockCreate(num_args, p_args, p_locs);
            mlirRegionAppendOwnedBlock(mlir_region, mlir_block);

            let string_func_attr = CString::new("() -> ()").unwrap();
            let func_type_attr = mlirAttributeParseGet(
                mlir_context,
                mlirStringRefCreateFromCString(string_func_attr.as_ptr()),
            );

            let mut string = String::from("\"");
            string += name;
            string += "\"";

            let string_func_name_attr = CString::new(string).unwrap();
            let func_name_attr = mlirAttributeParseGet(
                mlir_context,
                mlirStringRefCreateFromCString(string_func_name_attr.as_ptr()),
            );

            let string_type_id = CString::new("type").unwrap();
            let type_id = mlirIdentifierGet(
                mlir_context,
                mlirStringRefCreateFromCString(string_type_id.as_ptr()),
            );

            let string_sym_name_id = CString::new("sym_name").unwrap();
            let sym_name_id = mlirIdentifierGet(
                mlir_context,
                mlirStringRefCreateFromCString(string_sym_name_id.as_ptr()),
            );
            let func_attrs: [MlirNamedAttribute; 2] = [
                mlirNamedAttributeGet(type_id, func_type_attr),
                mlirNamedAttributeGet(sym_name_id, func_name_attr),
            ];
            let p_func_attrs = func_attrs.as_ptr();

            let mut state = OperationState::new("builtin.func", location);
            let p_state: *mut MlirOperationState = &mut state.instance;

            mlirOperationStateAddAttributes(p_state, 2, p_func_attrs);
            mlirOperationStateAddOwnedRegions(p_state, 1, p_mlir_region);

            let instance = mlirOperationCreate(p_state);

            if instance.ptr.is_null() {
                panic!("Cannot create FuncOp");
            }

            Self {
                instance,
                state,
                block: Rc::new(Block::new(mlir_block)),
                name: string_func_name_attr,
            }
        }
    }
}

impl From<FuncOp> for Operation {
    fn from(op: FuncOp) -> Self {
        Self {
            state: op.state,
            instance: op.instance,
        }
    }
}

pub trait Op {
    fn build(&self) -> Self
    where
        Self: Sized;
}

#[derive(Clone)]
struct Type {
    instance: MlirType,
}

impl Type {
    fn new(context: &Context) -> Self {
        let instance = unsafe { mlirNoneTypeGet(context.instance) };
        Self { instance }
    }
}

#[derive(Clone)]
struct Value {
    instance: MlirValue,
}

impl Value {
    // TODO: make it available only for crate but not for user
    pub fn new(instance: MlirValue) -> Value {
        Self { instance }
    }
}

struct OpBuilder {
    block: Option<Rc<Block>>,
    pos: isize,
}

impl<'ctx> OpBuilder {
    pub fn set_insertion_point(&mut self, block: Rc<Block>, pos: isize) {
        self.block = Some(block);
        self.pos = pos;
    }

    pub fn insert(&mut self, operation: Operation) {
        unsafe {
            let block: &Block = self.block.as_ref().unwrap();
            mlirBlockInsertOwnedOperation(block.instance, self.pos, operation.instance);
            self.pos += 1;
        }
    }
}

struct MLIRGen {
    module: ModuleOp,
    symbol_table: HashMap<String, Value>,
    context: Context,
    builder: OpBuilder,
}

impl<'ctx> MLIRGen {
    pub fn new(context: Context) -> Self {
        Self {
            module: ModuleOp::new(Location::new(&context)),
            symbol_table: HashMap::new(),
            context,
            builder: OpBuilder {
                block: Option::None,
                pos: 0,
            },
        }
    }
    // TODO: pass parsed result
    pub fn mlir_gen(&mut self, module_ast: Module) -> ModuleOp {
        self.module = ModuleOp::new(Location::new(&self.context));

        // TODO: implement Iterator for Module?
        for f in module_ast.functions {
            let func = self.mlir_gen_function(f);
            self.module.push_back(Box::new(Operation::from(func)));
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
        self.builder
            .set_insertion_point(Rc::clone(&function.block), 0);
        //
        // mlirBlockInsertOwnedOperationAfter(function.block.instance, reference, operation)
        // mlirOperationGetBlock(op)

        self.mlir_gen_expression(function_ast.body.unwrap());
        // function.erase();

        // TODO: handle return op

        function
    }

    fn mlir_gen_prototype(&mut self, prototype_ast: Prototype) -> FuncOp {
        // FIXME: construct location from AST location
        let location = Location::new(&self.context);
        // FIXME: convert VarType to mlirType
        // let arg_types = vec![Type::new(Context::default()); prototype_ast.args.len()];
        // let func_type = self.builder.get_function_type(arg_types, llvm::None);

        // FIXME: create function op with prototype name and funcType
        FuncOp::new(location, &prototype_ast.name)
    }

    fn declare(&mut self, name: String, value: Value) {
        self.symbol_table.insert(name, value);
    }

    fn mlir_gen_expression(&mut self, expr: Expr) -> Result<Value, &'static str> {
        // NB: this clone is used for collect_data method
        // there should be a way to avoid this
        let clone_expr = expr.clone();
        match expr {
            ExprList { expressions } => {
                for expr in expressions {
                    let _value = self.mlir_gen_expression(*expr.clone());
                }
                Err("ExprList not implemented")
            }
            VarDecl { name, value } => {
                let value = self.mlir_gen_expression(*value).unwrap();
                // TODO: reshape op
                // declare variable in the symbol table
                self.declare(name, value.clone());
                Ok(value)
            }
            Variable(name) => {
                if self.symbol_table.contains_key(&name) {
                    let value = (*self.symbol_table.get(&name).unwrap()).clone();
                    return Ok(value);
                }
                Err("Variable is not found")
                // extract variable from symbol table
            }
            Tensor {
                location,
                values,
                dims,
            } => {
                let size = dims.iter().product();
                let mut data: Vec<f64> = Vec::new();
                data.reserve(size);
                self.collect_data(clone_expr, &mut data);
                let ty = Type::new(&self.context);
                // TODO: contruct ConstanOp with array
                let op = ConstantOp::new(Location::new(&self.context), data[0]);
                self.builder.insert(Operation::from(op.clone()));
                Ok(Value::from(op))
            }
            Number(num) => {
                let location = Location::new(&self.context);
                let op = ConstantOp::new(location, num);
                self.builder.insert(Operation::from(op.clone()));
                Ok(Value::from(op))
            }
            Call { fn_name, args } => {
                let location = Location::new(&self.context);
                let mut operands: Vec<Value> = Vec::new();
                for arg in &args {
                    let arg = self.mlir_gen_expression(arg.clone()).unwrap();
                    operands.push(arg);
                }
                if fn_name == "transpose" {
                    if args.len() != 1 {
                        panic!("MLIR codegen encountered an error: toy.transpose does not accept multiple args");
                    }
                    let op = TransposeOp::new(location, operands[0].clone());
                    self.builder.insert(Operation::from(op.clone()));
                    return Ok(Value::from(op));
                }

                let op = GenericCallOp::new(location, fn_name, operands);
                self.builder.insert(Operation::from(op.clone()));
                Ok(Value::from(op))
            }
            Return {
                location: _,
                expression,
            } => {
                let location = Location::new(&self.context);
                let value = self.mlir_gen_expression(*expression).unwrap();
                Ok(Value::from(ReturnOp::new(location, value)))
            }

            Binary { op, left, right } => {
                let lhs = self.mlir_gen_expression(*left).unwrap();
                let rhs = self.mlir_gen_expression(*right).unwrap();
                let location = Location::new(&self.context);
                match op {
                    '+' => Ok(Value::from(AddOp::new(location, lhs, rhs))),
                    '*' => Ok(Value::from(MulOp::new(location, lhs, rhs))),
                    _ => Err("Invalid binary operation"),
                }
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

        let mut context = Context::default();
        let dialect = ToyDialect::new(&context);
        context.load_dialect(Box::new(dialect));

        let module = parser::Parser::new(content, &mut prec)
            .parse_module()
            .unwrap();
        let module = MLIRGen::new(context).mlir_gen(module);
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

        let context = Context::default();
        let module = parser::Parser::new(content, &mut prec)
            .parse_module()
            .unwrap();
        let module = MLIRGen::new(context).mlir_gen(module);
        println!("Dump:");
        module.dump();
        assert!(!module.block.operations.is_empty());
    }
}
