pub mod parser;
pub mod toy;

use std::collections::HashMap;
use std::ffi::{CStr, CString};

use libc::CS;
use std::rc::Rc;
use toy::mlirGetDialectHandle__toy__;

use mlir_sys::{
    mlirAttributeGetNull, mlirAttributeParseGet, mlirBlockAddArgument, mlirBlockCreate,
    mlirBlockGetArgument, mlirBlockInsertOwnedOperation, mlirBlockInsertOwnedOperationAfter,
    mlirContextAppendDialectRegistry, mlirContextCreate, mlirContextGetNumLoadedDialects,
    mlirContextGetNumRegisteredDialects, mlirContextGetOrLoadDialect,
    mlirContextSetAllowUnregisteredDialects, mlirDenseElementsAttrDoubleGet,
    mlirDenseElementsAttrGet, mlirDenseElementsAttrGetDoubleValue, mlirDialectHandleGetNamespace,
    mlirDialectHandleInsertDialect, mlirDialectHandleLoadDialect, mlirDialectHandleRegisterDialect,
    mlirDialectRegistryCreate, mlirF64TypeGet, mlirFlatSymbolRefAttrGet, mlirFunctionTypeGet,
    mlirFunctionTypeGetInput, mlirFunctionTypeGetNumInputs, mlirFunctionTypeGetNumResults,
    mlirFunctionTypeGetResult, mlirGetDialectHandle__std__, mlirIdentifierGet, mlirIdentifierStr,
    mlirLocationGetContext, mlirLocationUnknownGet, mlirModuleCreateEmpty, mlirModuleGetBody,
    mlirModuleGetOperation, mlirNamedAttributeGet, mlirNoneTypeGet, mlirOperationCreate,
    mlirOperationDump, mlirOperationGetAttributeByName, mlirOperationGetBlock,
    mlirOperationGetName, mlirOperationGetResult, mlirOperationStateAddAttributes,
    mlirOperationStateAddOperands, mlirOperationStateAddOwnedRegions, mlirOperationStateAddResults,
    mlirOperationStateGet, mlirRankedTensorTypeGet, mlirRankedTensorTypeGetEncoding,
    mlirRegionAppendOwnedBlock, mlirRegionCreate, mlirRegisterAllDialects,
    mlirRegisterLinalgLinalgLowerToLoops, mlirShapedTypeGetDimSize, mlirShapedTypeGetRank,
    mlirStringAttrGet, mlirStringRefCreateFromCString, mlirSymbolRefAttrGet, mlirTypeDump,
    mlirTypeIsAFunction, mlirTypeIsANone, mlirTypeIsARankedTensor, mlirTypeIsATensor,
    mlirTypeIsAUnrankedTensor, mlirTypeParseGet, mlirUnrankedTensorTypeGet, mlirValueDump,
    mlirValueGetType, mlirValueIsABlockArgument,
};
use mlir_sys::{
    MlirAttribute, MlirBlock, MlirContext, MlirDialectHandle, MlirLocation, MlirModule,
    MlirNamedAttribute, MlirOperation, MlirOperationState, MlirRegion, MlirType, MlirValue,
};

use crate::parser::Expr::{
    Binary, Call, ExprList, Number, Print, Return, Tensor, VarDecl, Variable,
};
use parser::{Expr, Function, Module, Prototype};
pub trait Dialect {
    fn get_name(&self) -> String;
}

pub struct Context {
    instance: MlirContext,
    dialects: Vec<Box<dyn Dialect>>,
}

impl From<toy::MlirDialectHandle> for mlir_sys::MlirDialectHandle {
    fn from(dialect: toy::MlirDialectHandle) -> Self {
        Self { ptr: dialect.ptr }
    }
}

impl Context {
    pub fn new() -> Self {
        unsafe {
            let instance = mlirContextCreate();
            // FIXME: make dialects to be registered separately
            mlirRegisterAllDialects(instance);
            let handle = mlir_sys::MlirDialectHandle::from(mlirGetDialectHandle__toy__());
            mlirDialectHandleRegisterDialect(handle, instance);
            mlirDialectHandleLoadDialect(handle, instance);
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
    name: CString,
}

impl ToyDialect {
    pub fn new(context: &Context) -> Self {
        // let ops = Vec::new();
        let name = CString::new("toy").unwrap();
        unsafe {
            let dialect = mlirContextGetOrLoadDialect(
                context.instance,
                mlirStringRefCreateFromCString(name.as_ptr()),
            );
            // mlirDialectHandleInsertDialect(arg1, arg2)
            if dialect.ptr.is_null() {
                panic!("Cannot load Toy dialect");
            }
            mlirRegisterLinalgLinalgLowerToLoops()
        }
        Self { name }
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

#[derive(Clone)]
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

#[derive(Clone)]
struct FuncOp {
    instance: MlirOperation,
    state: OperationState,
    block: Rc<Block>,
    name: CString,
}

impl FuncOp {
    pub fn new(mut location: Location, name: &str, func_type: Type) -> Self {
        unsafe {
            let mlir_context = mlirLocationGetContext(location.instance);
            let mlir_region = mlirRegionCreate();
            let p_mlir_region: *const MlirRegion = &mlir_region;

            let arg_string_types: Vec<String> = Self::extract_arg_types(&func_type).unwrap();
            let arg_cstring_types: Vec<CString> = arg_string_types
                .into_iter()
                .map(|x| CString::new(x).unwrap())
                .collect();
            let num_args = arg_cstring_types.len() as isize;

            let args: Vec<MlirType> = arg_cstring_types
                .into_iter()
                .map(|x| mlirTypeParseGet(mlir_context, mlirStringRefCreateFromCString(x.as_ptr())))
                .collect();

            let p_args: *const MlirType = args.as_ptr();
            let p_locs = &mut location.instance;

            let mlir_block = mlirBlockCreate(num_args, p_args, p_locs);
            mlirRegionAppendOwnedBlock(mlir_region, mlir_block);

            let func_type_string: String = Self::serialize_func_type(&func_type).unwrap();

            let string_func_attr = CString::new(func_type_string).unwrap();
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

    fn serialize_func_type(func_type: &Type) -> Result<String, &'static str> {
        unsafe {
            if !mlirTypeIsAFunction(func_type.instance) {
                return Err("Provided type is not a function type");
            }

            let mut result = String::new();
            result.push('(');
            let num_inputs = mlirFunctionTypeGetNumInputs(func_type.instance);
            for pos in 0..num_inputs {
                let arg = mlirFunctionTypeGetInput(func_type.instance, pos);
                if mlirTypeIsATensor(arg) {
                    result.push_str("tensor<");
                    if mlirTypeIsARankedTensor(arg) {
                        let rank = mlirShapedTypeGetRank(arg);
                        for r in 0..rank {
                            let dim = mlirShapedTypeGetDimSize(arg, r as isize);
                            result.push_str(&dim.to_string());
                            result.push(',');
                        }
                    } else if mlirTypeIsAUnrankedTensor(arg) {
                        // FIXME: parse elem type
                        result.push_str("?xf64");
                    }
                    result.push('>');
                }
                if pos < num_inputs - 1 {
                    result.push(',')
                }
            }
            result.push(')');
            result.push_str(" -> ");
            result.push('(');
            let num_results = mlirFunctionTypeGetNumResults(func_type.instance);
            for pos in 0..num_results {
                let arg = mlirFunctionTypeGetResult(func_type.instance, pos);
                if mlirTypeIsATensor(arg) {
                    if mlirTypeIsARankedTensor(arg) {
                        let rank = mlirShapedTypeGetRank(arg);
                        for r in 0..rank {
                            let dim = mlirShapedTypeGetDimSize(arg, r as isize);
                            result.push_str(&dim.to_string());
                            result.push(',');
                        }
                    }
                }
            }
            result.push(')');
            Ok(result)
        }
    }

    fn extract_arg_types(func_type: &Type) -> Result<Vec<String>, &'static str> {
        unsafe {
            if !mlirTypeIsAFunction(func_type.instance) {
                return Err("Provided type is not a function type");
            }

            let mut result = Vec::new();
            let num_inputs = mlirFunctionTypeGetNumInputs(func_type.instance);
            for pos in 0..num_inputs {
                let mut arg_string = String::new();
                let arg = mlirFunctionTypeGetInput(func_type.instance, pos);
                if mlirTypeIsATensor(arg) {
                    arg_string.push_str("tensor<");
                    if mlirTypeIsARankedTensor(arg) {
                        let rank = mlirShapedTypeGetRank(arg);
                        for r in 0..rank {
                            let dim = mlirShapedTypeGetDimSize(arg, r as isize);
                            arg_string.push_str(&dim.to_string());
                            arg_string.push(',');
                        }
                    } else if mlirTypeIsAUnrankedTensor(arg) {
                        // FIXME: parse elem type
                        arg_string.push_str("?xf64");
                    }
                    arg_string.push('>');
                }
                result.push(arg_string);
            }

            Ok(result)
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

impl From<FuncOp> for Value {
    fn from(op: FuncOp) -> Self {
        // use op to construct Value
        let instance = unsafe { mlirOperationGetResult(op.instance, 0) };
        Value::new(instance)
    }
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

impl From<MlirType> for Type {
    fn from(instance: MlirType) -> Self {
        Self { instance }
    }
}

#[derive(Clone)]
struct Attribute {
    instance: MlirAttribute,
}

impl From<MlirAttribute> for Attribute {
    fn from(instance: MlirAttribute) -> Self {
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
