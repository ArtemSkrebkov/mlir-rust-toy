pub mod block;
pub mod context;
pub mod dialect;
pub mod location;
pub mod misc;
pub mod operation;
pub mod parser;
pub mod toy;

use std::collections::HashMap;
use std::ffi::CString;

use std::rc::Rc;

use crate::context::Context;
// use crate::dialect::Dialect;
use crate::block::Block;
use crate::location::Location;
use crate::misc::{Attribute, Type, Value};
use crate::operation::{FuncOp, ModuleOp, OneRegion, Operation, OperationState};

use mlir_sys::{
    mlirAttributeGetNull, mlirBlockGetArgument, mlirBlockInsertOwnedOperation,
    mlirDenseElementsAttrDoubleGet, mlirF64TypeGet, mlirFlatSymbolRefAttrGet, mlirFunctionTypeGet,
    mlirIdentifierGet, mlirLocationGetContext, mlirNamedAttributeGet, mlirOperationCreate,
    mlirOperationGetResult, mlirOperationStateAddOperands, mlirOperationStateAddResults,
    mlirRankedTensorTypeGet, mlirStringRefCreateFromCString, mlirUnrankedTensorTypeGet,
    mlirValueDump,
};
use mlir_sys::{MlirNamedAttribute, MlirOperation, MlirOperationState, MlirType, MlirValue};

use crate::parser::Expr::{
    Binary, Call, ExprList, Number, Print, Return, Tensor, VarDecl, Variable,
};
use parser::{Expr, Function, Module, Prototype};

#[derive(Clone)]
struct ConstantOp {
    name: String,
    instance: MlirOperation,
    value: f64,
    state: OperationState,
    location: Location,
}

impl ConstantOp {
    pub fn new(location: Location) -> Self {
        unsafe {
            // FIXME: duplication toy.constant
            let name = String::from("toy.constant");
            let mut state = OperationState::new("toy.constant", location.clone());
            let p_state: *mut MlirOperationState = &mut state.instance;
            let instance = mlirOperationCreate(p_state);

            Self {
                name,
                instance,
                value: 0.0,
                state,
                location,
            }
        }
    }

    pub fn with_value(&mut self, value: f64) -> &mut Self {
        self.value = value;
        self
    }

    pub fn with_result(&mut self, result_type: Type) -> &mut Self {
        let results: Vec<MlirType> = vec![result_type.instance; 1];
        let p_state: *mut MlirOperationState = &mut self.state.instance;
        unsafe { mlirOperationStateAddResults(p_state, 1, results.as_ptr()) };
        self
    }

    pub(crate) fn with_attribute(&mut self, data_attr: Attribute) -> &mut Self {
        unsafe {
            let mlir_context = mlirLocationGetContext(self.location.instance);
            let p_state: *mut MlirOperationState = &mut self.state.instance;
            let string_value_id = CString::new("value").unwrap();
            let value_id = mlirIdentifierGet(
                mlir_context,
                mlirStringRefCreateFromCString(string_value_id.as_ptr()),
            );
            let named_data_attr = mlirNamedAttributeGet(value_id, data_attr.instance);
            let p_named_data_attr: *const MlirNamedAttribute = &named_data_attr;
            mlirOperationStateAddAttributes(p_state, 1, p_named_data_attr)
        }
        self
    }

    pub(crate) fn build(&mut self) -> &mut Self {
        unsafe {
            // FIXME: here we create op one more time despite it was created in new method
            let p_state: *mut MlirOperationState = &mut self.state.instance;
            let instance = mlirOperationCreate(p_state);
            self.instance = instance;

            self
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

impl From<&mut ConstantOp> for Value {
    fn from(op: &mut ConstantOp) -> Self {
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
    pub fn new(location: Location, input: Value, result_type: Type) -> Self {
        unsafe {
            // FIXME: duplication toy.constant
            let name = String::from("toy.transpose");
            let mut state = OperationState::new("toy.transpose", location);
            let p_state: *mut MlirOperationState = &mut state.instance;

            // TODO: extract to builder
            let p_operands: *const MlirValue = &input.instance;
            mlirOperationStateAddOperands(p_state, 1, p_operands);

            // TODO: extract to builder
            let results: Vec<MlirType> = vec![result_type.instance; 1];
            unsafe { mlirOperationStateAddResults(p_state, 1, results.as_ptr()) };

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
    // callee: String,
    // operands: Vec<Value>,
    state: OperationState,
}

// TODO: implementation looks the same to ConstantOp, besides the name
impl GenericCallOp {
    pub fn new(
        location: Location,
        callee: String,
        operands: Vec<Value>,
        result_type: Type,
    ) -> Self {
        unsafe {
            // FIXME: duplication toy.constant
            let mlir_context = mlirLocationGetContext(location.instance);
            let name = String::from("toy.generic_call");
            let mut state = OperationState::new("toy.generic_call", location);
            let p_state: *mut MlirOperationState = &mut state.instance;

            let operands: Vec<MlirValue> = operands.into_iter().map(|v| v.instance).collect();
            let p_operands: *const MlirValue = operands.as_ptr();
            mlirOperationStateAddOperands(p_state, operands.len() as isize, p_operands);

            let p_results: *const MlirType = &result_type.instance;
            mlirOperationStateAddResults(p_state, 1, p_results);

            // let mlir_context = mlirLocationGetContext(location.instance);

            let string_callee_id = CString::new("callee").unwrap();
            let callee_id = mlirIdentifierGet(
                mlir_context,
                mlirStringRefCreateFromCString(string_callee_id.as_ptr()),
            );

            let s = CString::new(callee).unwrap();
            let data_attr =
                mlirFlatSymbolRefAttrGet(mlir_context, mlirStringRefCreateFromCString(s.as_ptr()));

            let named_data_attr = mlirNamedAttributeGet(callee_id, data_attr);
            let p_named_data_attr: *const MlirNamedAttribute = &named_data_attr;

            mlirOperationStateAddAttributes(p_state, 1, p_named_data_attr);
            let instance = mlirOperationCreate(p_state);

            Self {
                name,
                instance,
                // callee,
                // operands,
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

#[derive(Clone)]
struct ReturnOp {
    name: String,
    instance: MlirOperation,
    input: Value,
    state: OperationState,
}

impl ReturnOp {
    pub fn new(location: Location, input: Value) -> Self {
        unsafe {
            // FIXME: duplication toy.constant
            let name = String::from("toy.return");
            let mut state = OperationState::new("toy.return", location);
            let p_state: *mut MlirOperationState = &mut state.instance;

            // TODO: let extract to builder
            let p_operands: *const MlirValue = &input.instance;
            mlirOperationStateAddOperands(p_state, 1, p_operands);

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

impl From<ReturnOp> for Value {
    fn from(op: ReturnOp) -> Self {
        // use op to construct Value
        let instance = unsafe { mlirOperationGetResult(op.instance, 0) };
        Value::new(instance)
    }
}

impl From<ReturnOp> for Operation {
    fn from(op: ReturnOp) -> Self {
        Self {
            state: op.state,
            instance: op.instance,
        }
    }
}

#[derive(Clone)]
struct AddOp {
    name: String,
    instance: MlirOperation,
    lhs: Value,
    rhs: Value,
    state: OperationState,
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
                state,
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

impl From<AddOp> for Operation {
    fn from(op: AddOp) -> Self {
        Self {
            state: op.state,
            instance: op.instance,
        }
    }
}

#[derive(Clone)]
struct MulOp {
    name: String,
    instance: MlirOperation,
    lhs: Value,
    rhs: Value,
    state: OperationState,
}

impl MulOp {
    pub fn new(location: Location, lhs: Value, rhs: Value, result_type: Type) -> Self {
        unsafe {
            // FIXME: duplication toy.constant
            let name = String::from("toy.mul");
            let mut state = OperationState::new("toy.mul", location);
            let p_state: *mut MlirOperationState = &mut state.instance;

            let operands = vec![lhs.instance, rhs.instance];
            let p_operands = operands.as_ptr();
            mlirOperationStateAddOperands(p_state, operands.len() as isize, p_operands);

            let p_results: *const MlirType = &result_type.instance;
            mlirOperationStateAddResults(p_state, 1, p_results);

            let instance = mlirOperationCreate(p_state);

            Self {
                name,
                instance,
                lhs,
                rhs,
                state,
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

impl From<MulOp> for Operation {
    fn from(op: MulOp) -> Self {
        Self {
            state: op.state,
            instance: op.instance,
        }
    }
}

#[derive(Clone)]
struct PrintOp {
    name: String,
    instance: MlirOperation,
    input: Value,
    state: OperationState,
}

impl PrintOp {
    pub fn new(location: Location, input: Value) -> Self {
        unsafe {
            // FIXME: duplication toy.constant
            let name = String::from("toy.print");
            let mut state = OperationState::new("toy.print", location);
            let p_state: *mut MlirOperationState = &mut state.instance;

            // TODO: let extract to builder
            let p_operands: *const MlirValue = &input.instance;
            mlirOperationStateAddOperands(p_state, 1, p_operands);

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

impl From<PrintOp> for Value {
    fn from(op: PrintOp) -> Self {
        // use op to construct Value
        let instance = unsafe { mlirOperationGetResult(op.instance, 0) };
        Value::new(instance)
    }
}

impl From<PrintOp> for Operation {
    fn from(op: PrintOp) -> Self {
        Self {
            state: op.state,
            instance: op.instance,
        }
    }
}

struct OpBuilder {
    context: Rc<Context>,
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

    fn get_f64_type(&self) -> Type {
        unsafe { Type::from(mlirF64TypeGet(self.context.instance)) }
    }

    // TODO: redundant copies of dims
    fn get_ranked_tensor_type(&self, dims: Vec<usize>, elem_ty: Type) -> Type {
        let rank: isize = dims.len() as isize;
        let shape: Vec<i64> = dims.into_iter().map(|x| x as i64).collect();
        let p_shape = shape.as_ptr();
        // NB: not sure what else can be used as enconding, so passing mlirAttributeGetNull for now
        unsafe {
            Type::from(mlirRankedTensorTypeGet(
                rank,
                p_shape,
                elem_ty.instance,
                mlirAttributeGetNull(),
            ))
        }
    }

    fn get_dense_elements_attr(&self, data_ty: Type, data: Vec<f64>) -> Attribute {
        unsafe {
            Attribute::from(mlirDenseElementsAttrDoubleGet(
                data_ty.instance,
                data.len() as isize,
                data.as_ptr(),
            ))
        }
    }

    fn get_unranked_tensor_type(&self, elem_type: Type) -> Type {
        unsafe { Type::from(mlirUnrankedTensorTypeGet(elem_type.instance)) }
    }

    fn get_function_type(&self, arg_types: Vec<Type>, result_types: Vec<Type>) -> Type {
        let num_inputs = arg_types.len() as isize;
        let num_result = result_types.len() as isize;
        let args: Vec<MlirType> = arg_types.into_iter().map(|x| x.instance).collect();
        let p_args: *const MlirType = args.as_ptr();
        let results: Vec<MlirType> = result_types.into_iter().map(|x| x.instance).collect();
        let p_results: *const MlirType = results.as_ptr();

        unsafe {
            Type::from(mlirFunctionTypeGet(
                self.context.instance,
                num_inputs,
                p_args,
                num_result,
                p_results,
            ))
        }
    }
}

struct MLIRGen {
    module: ModuleOp,
    symbol_table: HashMap<String, Value>,
    context: Rc<Context>,
    builder: OpBuilder,
}

impl<'ctx> MLIRGen {
    pub fn new(context: Rc<Context>) -> Self {
        Self {
            module: ModuleOp::new(Location::new(&*context)),
            symbol_table: HashMap::new(),
            context: Rc::clone(&context),
            builder: OpBuilder {
                block: Option::None,
                pos: 0,
                context: Rc::clone(&context),
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

        self.module.clone()
    }

    fn mlir_gen_function(&mut self, function_ast: Function) -> FuncOp {
        // varScopeTable
        // let var_scope: HashMap<&str, Value> = HashMap::new();
        let function: FuncOp = self.mlir_gen_prototype(function_ast.prototype.clone());

        let entry_block = function.block.clone();
        // TODO: getter for prototype
        let proto_args = function_ast.prototype.args.clone();
        // FIXME: just for the sake of adding something into the table
        // need to review how it is done in original Toy tutorial
        let mut pos = 0;
        for arg in proto_args {
            unsafe {
                let mlir_arg_value = mlirBlockGetArgument(entry_block.instance, pos);
                self.declare(arg, Value::new(mlir_arg_value));
                pos += 1;
            }
        }
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
        let arg_types = vec![self.get_type(Vec::new()); prototype_ast.args.len()];
        let func_type = self.builder.get_function_type(arg_types, Vec::new());

        // TODO: make it builder
        FuncOp::new(location, &prototype_ast.name, func_type)
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
                // TODO: contruct ConstanOp with array
                let elem_ty = self.builder.get_f64_type();
                let data_ty = self.builder.get_ranked_tensor_type(dims, elem_ty);
                let data_attr: Attribute =
                    self.builder.get_dense_elements_attr(data_ty.clone(), data);
                // TODO: a separate builder for constructing a type?
                let mut op = ConstantOp::new(Location::new(&self.context));
                op.with_result(data_ty).with_attribute(data_attr).build();
                self.builder.insert(Operation::from(op.clone()));
                Ok(Value::from(op))
            }
            Number(num) => {
                let location = Location::new(&self.context);
                let mut op = ConstantOp::new(location);
                op.with_value(num);
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
                    let op = TransposeOp::new(
                        location,
                        operands[0].clone(),
                        self.builder
                            .get_unranked_tensor_type(self.builder.get_f64_type()),
                    );
                    self.builder.insert(Operation::from(op.clone()));
                    let value = Value::from(op);

                    return Ok(value);
                    // return Ok(Value::from(op));
                }

                let result_type = self
                    .builder
                    .get_unranked_tensor_type(self.builder.get_f64_type());
                let op = GenericCallOp::new(location, fn_name, operands, result_type);
                self.builder.insert(Operation::from(op.clone()));
                let value = Value::from(op);
                unsafe { mlirValueDump(value.instance) };
                Ok(value)
            }
            Return {
                location: _,
                expression,
            } => {
                let location = Location::new(&self.context);
                let value = self.mlir_gen_expression(*expression).unwrap();
                let op = ReturnOp::new(location, value);
                self.builder.insert(Operation::from(op.clone()));
                Ok(Value::from(op))
            }

            Binary { op, left, right } => {
                let lhs = self.mlir_gen_expression(*left).unwrap();
                let rhs = self.mlir_gen_expression(*right).unwrap();
                let result_type = self
                    .builder
                    .get_unranked_tensor_type(self.builder.get_f64_type());
                let location = Location::new(&self.context);
                match op {
                    '+' => {
                        let op = AddOp::new(location, lhs, rhs);
                        self.builder.insert(Operation::from(op.clone()));
                        Ok(Value::from(op))
                    }
                    '*' => {
                        let op = MulOp::new(location, lhs, rhs, result_type);
                        self.builder.insert(Operation::from(op.clone()));
                        Ok(Value::from(op))
                    }
                    _ => Err("Invalid binary operation"),
                }
            }

            Print {
                location: _,
                expression,
            } => {
                let location = Location::new(&self.context);
                let value = self.mlir_gen_expression(*expression).unwrap();
                let op = PrintOp::new(location, value);
                self.builder.insert(Operation::from(op.clone()));
                Ok(Value::from(op))
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

    fn get_type(&self, shape: Vec<usize>) -> Type {
        if shape.is_empty() {
            return self
                .builder
                .get_unranked_tensor_type(self.builder.get_f64_type());
        }

        self.builder
            .get_ranked_tensor_type(shape, self.builder.get_f64_type())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;
    use crate::context::Context;
    use crate::dialect::StandardDialect;
    use crate::dialect::ToyDialect;
    use crate::operation::ModuleOp;

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
