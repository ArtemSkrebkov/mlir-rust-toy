use mlir_sys::{
    mlirFlatSymbolRefAttrGet, mlirIdentifierGet, mlirLocationGetContext, mlirNamedAttributeGet,
    mlirOperationCreate, mlirOperationGetResult, mlirOperationStateAddAttributes,
    mlirOperationStateAddOperands, mlirOperationStateAddResults, mlirStringRefCreateFromCString,
    MlirNamedAttribute, MlirOperation, MlirOperationState, MlirType, MlirValue,
};

use crate::location::Location;
use crate::misc::{Attribute, Type, Value};
use crate::operation::{Operation, OperationState};

use std::ffi::CString;
#[derive(Clone)]
pub struct ConstantOp {
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
pub struct TransposeOp {
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
pub struct GenericCallOp {
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
pub struct ReturnOp {
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
pub(crate) struct AddOp {
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
pub(crate) struct MulOp {
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
pub(crate) struct PrintOp {
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
