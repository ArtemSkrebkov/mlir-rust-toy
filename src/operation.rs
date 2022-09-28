use crate::block::Block;
use crate::location::Location;
use crate::misc::{Type, Value};

use mlir_sys::{
    mlirAttributeParseGet, mlirBlockCreate, mlirBlockInsertOwnedOperation,
    mlirFunctionTypeGetInput, mlirFunctionTypeGetNumInputs, mlirFunctionTypeGetNumResults,
    mlirFunctionTypeGetResult, mlirIdentifierGet, mlirLocationGetContext, mlirModuleCreateEmpty,
    mlirModuleGetBody, mlirModuleGetOperation, mlirNamedAttributeGet, mlirOperationCreate,
    mlirOperationDump, mlirOperationGetResult, mlirOperationStateAddAttributes,
    mlirOperationStateAddOwnedRegions, mlirOperationStateGet, mlirRegionAppendOwnedBlock,
    mlirRegionCreate, mlirShapedTypeGetDimSize, mlirShapedTypeGetRank,
    mlirStringRefCreateFromCString, mlirTypeIsAFunction, mlirTypeIsARankedTensor,
    mlirTypeIsATensor, mlirTypeIsAUnrankedTensor, mlirTypeParseGet, MlirModule, MlirNamedAttribute,
    MlirOperation, MlirOperationState, MlirRegion, MlirType,
};
use std::ffi::CString;
use std::rc::Rc;

#[derive(Clone)]
pub struct OperationState {
    pub(crate) instance: MlirOperationState,
    // NB: to make string live long enough
    string: CString,
}

impl OperationState {
    pub fn new(name: &str, location: Location) -> Self {
        let string = CString::new(name).unwrap();
        let reference = unsafe { mlirStringRefCreateFromCString(string.as_ptr()) };
        let instance = unsafe { mlirOperationStateGet(reference, location.instance) };

        Self { instance, string }
        // Self { instance }
    }
}

#[derive(Clone)]
pub struct Operation {
    pub(crate) state: OperationState,
    pub(crate) instance: MlirOperation,
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
pub struct ModuleOp {
    instance: MlirModule,
    // state: OperationState,
    // TODO: make it private? better to have accessor methods
    pub(crate) block: Block,
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
pub struct FuncOp {
    instance: MlirOperation,
    state: OperationState,
    pub(crate) block: Rc<Block>,
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
