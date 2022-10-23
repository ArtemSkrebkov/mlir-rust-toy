use crate::block::{Block, Region};
use crate::context::Context;
use crate::location::Location;
use crate::misc::{Attribute, NamedAttribute, Type, Value};

use mlir_sys::{
    mlirAttributeParseGet, mlirBlockCreate, mlirBlockInsertOwnedOperation,
    mlirFunctionTypeGetInput, mlirFunctionTypeGetNumInputs, mlirFunctionTypeGetNumResults,
    mlirFunctionTypeGetResult, mlirIdentifierStr, mlirLocationGetContext, mlirModuleCreateEmpty,
    mlirModuleCreateParse, mlirModuleGetBody, mlirModuleGetOperation, mlirOperationCreate,
    mlirOperationDump, mlirOperationGetAttributeByName, mlirOperationGetContext,
    mlirOperationGetName, mlirOperationGetNumOperands, mlirOperationGetResult,
    mlirOperationSetAttributeByName, mlirOperationStateAddAttributes,
    mlirOperationStateAddOperands, mlirOperationStateAddOwnedRegions, mlirOperationStateAddResults,
    mlirOperationStateGet, mlirRegionAppendOwnedBlock, mlirRegionCreate, mlirShapedTypeGetDimSize,
    mlirShapedTypeGetRank, mlirStringAttrGet, mlirStringRefCreateFromCString,
    mlirSymbolTableCreate, mlirSymbolTableGetVisibilityAttributeName, mlirTypeIsAFunction,
    mlirTypeIsARankedTensor, mlirTypeIsATensor, mlirTypeIsAUnrankedTensor, mlirTypeParseGet,
    MlirModule, MlirNamedAttribute, MlirOperation, MlirOperationState, MlirRegion, MlirType,
    MlirValue,
};
use std::ffi::{CStr, CString};
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
    }

    pub fn add_results(&mut self, results: Vec<Type>) {
        let results: Vec<MlirType> = results.into_iter().map(|x| x.instance).collect();
        let p_state: *mut MlirOperationState = &mut self.instance;

        unsafe { mlirOperationStateAddResults(p_state, results.len() as isize, results.as_ptr()) };
    }

    pub fn add_attributes(&mut self, attrs: Vec<NamedAttribute>) {
        let attrs: Vec<MlirNamedAttribute> = attrs.into_iter().map(|x| x.instance).collect();
        let p_state: *mut MlirOperationState = &mut self.instance;
        let p_named_attr: *const MlirNamedAttribute = attrs.as_ptr();
        unsafe {
            mlirOperationStateAddAttributes(p_state, attrs.len() as isize, p_named_attr);
        }
    }

    pub fn add_operands(&mut self, operands: Vec<Value>) {
        let operands: Vec<MlirValue> = operands.into_iter().map(|x| x.instance).collect();

        let p_state: *mut MlirOperationState = &mut self.instance;
        let p_operands: *const MlirValue = operands.as_ptr();
        unsafe {
            mlirOperationStateAddOperands(p_state, operands.len() as isize, p_operands);
        }
    }

    fn add_owned_regions(&mut self, regions: Vec<Region>) {
        let regions: Vec<MlirRegion> = regions.into_iter().map(|x| x.instance).collect();

        let p_state: *mut MlirOperationState = &mut self.instance;
        let p_regions: *const MlirRegion = regions.as_ptr();
        unsafe {
            mlirOperationStateAddOwnedRegions(p_state, regions.len() as isize, p_regions);
        }
    }
}

#[derive(Clone)]
pub struct Operation {
    pub(crate) instance: MlirOperation,
}

impl Operation {
    pub fn new(state: &mut OperationState) -> Self {
        let p_state: *mut MlirOperationState = &mut state.instance;
        let instance = unsafe { mlirOperationCreate(p_state) };
        Self { instance }
    }

    pub fn name(&self) -> String {
        unsafe {
            let ident = mlirOperationGetName(self.instance);
            let mlir_name = mlirIdentifierStr(ident);
            let c_str: &CStr = CStr::from_ptr(mlir_name.data);
            let str_slice: &str = c_str.to_str().unwrap();
            let str_buf: String = str_slice.to_owned();
            str_buf
        }
    }

    pub(crate) fn num_operands(&self) -> usize {
        let num = unsafe { mlirOperationGetNumOperands(self.instance) };

        num as usize
    }
}

impl From<MlirOperation> for Operation {
    fn from(mlir_operation: MlirOperation) -> Self {
        Operation {
            instance: mlir_operation,
        }
    }
}

impl From<Operation> for Value {
    fn from(operation: Operation) -> Self {
        let instance = unsafe { mlirOperationGetResult(operation.instance, 0) };
        Value::new(instance)
    }
}

impl From<&mut Operation> for Value {
    fn from(operation: &mut Operation) -> Self {
        let instance = unsafe { mlirOperationGetResult(operation.instance, 0) };
        Value::new(instance)
    }
}

pub trait OneRegion {
    fn push_back(&mut self, operation: Box<Operation>);
}

#[derive(Clone)]
pub struct ModuleOp {
    pub(crate) instance: MlirModule,
    // state: OperationState,
    // TODO: make it private? better to have accessor methods
    pub(crate) block: Block,
    pos: isize,
}

impl ModuleOp {
    pub fn new(location: Location) -> Self {
        unsafe {
            let instance = mlirModuleCreateEmpty(location.instance);
            let mlir_block = mlirModuleGetBody(instance);

            Self {
                instance,
                block: Block::from(mlir_block),
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

    pub fn new_parsed(context: &Context, content: &str) -> Self {
        unsafe {
            let content = CString::new(content).unwrap();
            let instance = mlirModuleCreateParse(
                context.instance,
                mlirStringRefCreateFromCString(content.as_ptr()),
            );
            let mlir_block = mlirModuleGetBody(instance);

            // FIXME: creates a memory leak, since it is not removed
            let _mlir_symbol_table = mlirSymbolTableCreate(mlirModuleGetOperation(instance));

            Self {
                instance,
                block: Block::from(mlir_block),
                pos: 0,
            }
        }
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
    pub(crate) operation: Operation,
    pub(crate) block: Rc<Block>,
}

impl FuncOp {
    pub fn new(mut location: Location, name: &str, func_type: Type, exported: bool) -> Self {
        unsafe {
            let mlir_context = mlirLocationGetContext(location.instance);
            let mlir_region = mlirRegionCreate();

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

            let mut string = String::from("\"");
            string += name;
            string += "\"";

            let type_attr = NamedAttribute::new(
                "type",
                Attribute::new_parsed(location.context(), &func_type_string),
            );
            let func_name_attr = NamedAttribute::new(
                "sym_name",
                Attribute::new_parsed(location.context(), &string),
            );
            let mut attributes = vec![type_attr, func_name_attr];
            if exported {
                let c_emit_id_attr = NamedAttribute::new(
                    "llvm.emit_c_interface",
                    Attribute::new_unit(location.context()),
                );
                attributes.push(c_emit_id_attr);
            }

            let region = Region::from(mlir_region);
            let block = Block::from(mlir_block);

            Self::create(location, block, region, attributes)
        }
    }

    fn create(
        location: Location,
        block: Block,
        region: Region,
        attributes: Vec<NamedAttribute>,
    ) -> Self {
        let mut state = OperationState::new("builtin.func", location);
        state.add_attributes(attributes);
        state.add_owned_regions(vec![region; 1]);

        let operation = Operation::new(&mut state);

        Self {
            operation,
            block: Rc::new(block),
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
                        // TODO: parse elem type
                        result.push_str("*xf64");
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
                    result.push_str("tensor<");
                    if mlirTypeIsARankedTensor(arg) {
                        let rank = mlirShapedTypeGetRank(arg);
                        for r in 0..rank {
                            let dim = mlirShapedTypeGetDimSize(arg, r as isize);
                            result.push_str(&dim.to_string());
                            result.push(',');
                        }
                    } else if mlirTypeIsAUnrankedTensor(arg) {
                        // TODO: parse elem type
                        result.push_str("*xf64");
                    }
                    result.push('>');
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
                        // TODO: parse elem type
                        arg_string.push_str("*xf64");
                    }
                    arg_string.push('>');
                }
                result.push(arg_string);
            }

            Ok(result)
        }
    }

    // FIXME: should be handled by symbol table once introduced
    // it should be mutable ref since we actually change object
    pub fn set_private(&self) {
        unsafe {
            let mlir_attr_name = mlirSymbolTableGetVisibilityAttributeName();
            let _mlir_cur_vis_attr =
                mlirOperationGetAttributeByName(self.operation.instance, mlir_attr_name);
            // TODO: add a validity check for mlir_cur_vis_attr
            let str = CString::new("private").unwrap();
            let mlir_context = mlirOperationGetContext(self.operation.instance);
            let mlir_new_vis_attr =
                mlirStringAttrGet(mlir_context, mlirStringRefCreateFromCString(str.as_ptr()));
            mlirOperationSetAttributeByName(
                self.operation.instance,
                mlir_attr_name,
                mlir_new_vis_attr,
            );
        }
    }

    pub fn set_type(&self, func_type: &Type) {
        let type_attr_name = CString::new("type").unwrap();
        let func_type_string: String = Self::serialize_func_type(&func_type).unwrap();
        let func_type_string = CString::new(&*func_type_string).unwrap();

        unsafe {
            let mlir_context = mlirOperationGetContext(self.operation.instance);
            let mlir_new_type_attr = mlirAttributeParseGet(
                mlir_context,
                mlirStringRefCreateFromCString(func_type_string.as_ptr()),
            );
            let mlir_attr_name = mlirStringRefCreateFromCString(type_attr_name.as_ptr());
            mlirOperationSetAttributeByName(
                self.operation.instance,
                mlir_attr_name,
                mlir_new_type_attr,
            );
        }
    }
}
