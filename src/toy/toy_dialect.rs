use mlir_sys::{
    mlirFlatSymbolRefAttrGet, mlirIdentifierGet, mlirLocationGetContext, mlirNamedAttributeGet,
    mlirOperationCreate, mlirOperationGetResult, mlirOperationStateAddAttributes,
    mlirOperationStateAddOperands, mlirOperationStateAddResults, mlirStringRefCreateFromCString,
    MlirDialectHandle, MlirNamedAttribute, MlirOperation, MlirOperationState, MlirType, MlirValue,
};

use crate::context::Context;
use crate::dialect::Dialect;
use crate::location::Location;
use crate::misc::{Attribute, NamedAttribute, Type, Value};
use crate::operation::{self, Operation, OperationState};
use crate::toy;

use std::ffi::CString;

use crate::toy::ffi::mlirGetDialectHandle__toy__;

impl From<toy::ffi::MlirDialectHandle> for mlir_sys::MlirDialectHandle {
    fn from(dialect: toy::ffi::MlirDialectHandle) -> Self {
        Self { ptr: dialect.ptr }
    }
}

pub struct ToyDialect {
    name: CString,
    instance: MlirDialectHandle,
}

impl ToyDialect {
    pub fn new(_context: &Context) -> Self {
        let name = CString::new("toy").unwrap();
        unsafe {
            let instance = mlir_sys::MlirDialectHandle::from(mlirGetDialectHandle__toy__());
            if instance.ptr.is_null() {
                panic!("Cannot load Toy dialect");
            }
            Self { name, instance }
        }
    }
}

impl Dialect for ToyDialect {
    fn get_name(&self) -> String {
        String::from("toy")
    }

    fn handle(&self) -> MlirDialectHandle {
        self.instance
    }
}
#[derive(Clone)]
pub struct ConstantOp {
    name: String,
    location: Location,
    state: OperationState,
    // TODO: add accessor?
    pub operation: Operation,
    value: f64,
}

impl ConstantOp {
    pub fn new(location: Location) -> Self {
        let name = String::from("toy.constant");
        let state = OperationState::new("toy.constant", location.clone());
        let operation = Operation::new(state.clone());

        Self {
            name,
            operation,
            value: 0.0,
            state,
            location,
        }
    }

    pub fn with_value(&mut self, value: f64) -> &mut Self {
        self.value = value;
        self
    }

    pub fn with_result(&mut self, result_type: Type) -> &mut Self {
        let results = vec![result_type; 1];
        self.state.add_results(results);
        self
    }

    pub fn with_attribute(&mut self, attr: Attribute) -> &mut Self {
        let named_attr = NamedAttribute::new("value", attr);
        self.state.add_attributes(vec![named_attr; 1]);
        self
    }

    pub(crate) fn build(&mut self) -> &mut Self {
        // FIXME: here we create op one more time despite it was created in new method
        self.operation = Operation::new(self.state.clone());

        self
    }
}

impl From<ConstantOp> for Operation {
    fn from(op: ConstantOp) -> Self {
        Self {
            state: op.state,
            instance: op.operation.instance,
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

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use crate::op_builder::OpBuilder;

    use super::*;

    #[test]
    fn create_constant() {
        let context = Rc::new(Context::default());
        // Need to make it mutable
        let dialect = ToyDialect::new(&context);
        context.load_dialect(Box::new(dialect));
        let location = Location::new(&context);
        let mut constant = ConstantOp::new(location);

        let op_builder = OpBuilder::new(None, 0, context);
        let result_type = op_builder.get_f64_type();
        let ty = op_builder.get_f64_type();
        let ty = op_builder.get_ranked_tensor_type(vec![3, 1], ty);
        let attr: Attribute = op_builder.get_dense_elements_attr(ty.clone(), vec![1.0, 1.0, 1.0]);
        constant
            .with_value(1.0)
            .with_result(result_type)
            .with_attribute(attr)
            .build();
    }
}
