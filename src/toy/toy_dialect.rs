use mlir_sys::MlirDialectHandle;

use crate::context::Context;
use crate::dialect::Dialect;
use crate::location::Location;
use crate::misc::{Attribute, NamedAttribute, Type, Value};
use crate::operation::{Operation, OperationState};
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
    state: OperationState,
    pub operation: Operation,
}

// TODO: implementation looks the same to ConstantOp, besides the name
impl TransposeOp {
    pub fn new(location: Location, input: Value, result_type: Type) -> Self {
        let name = String::from("toy.transpose");
        let mut state = OperationState::new("toy.transpose", location);

        state.add_results(vec![result_type; 1]);
        state.add_operands(vec![input; 1]);

        let operation = Operation::new(state.clone());

        Self {
            name,
            state,
            operation,
        }
    }
}

impl From<TransposeOp> for Operation {
    fn from(op: TransposeOp) -> Self {
        Self {
            state: op.state,
            instance: op.operation.instance,
        }
    }
}

#[derive(Clone)]
pub struct GenericCallOp {
    name: String,
    callee: String,
    state: OperationState,
    pub operation: Operation,
}

impl GenericCallOp {
    pub fn new(
        location: Location,
        callee: String,
        operands: Vec<Value>,
        result_type: Type,
    ) -> Self {
        let name = String::from("toy.generic_call");
        let mut state = OperationState::new("toy.generic_call", location.clone());
        state.add_operands(operands);
        state.add_results(vec![result_type; 1]);

        let attr = Attribute::new_flat_symbol_ref(location.context(), &callee);
        let named_attr = NamedAttribute::new("callee", attr);

        state.add_attributes(vec![named_attr; 1]);
        let operation = Operation::new(state.clone());

        Self {
            name,
            state,
            callee,
            operation,
        }
    }
}

impl From<GenericCallOp> for Operation {
    fn from(op: GenericCallOp) -> Self {
        Self {
            state: op.state,
            instance: op.operation.instance,
        }
    }
}

#[derive(Clone)]
pub struct ReturnOp {
    name: String,
    state: OperationState,
    pub operation: Operation,
}

impl ReturnOp {
    pub fn new(location: Location, input: Value) -> Self {
        let name = String::from("toy.return");
        let mut state = OperationState::new("toy.return", location);
        state.add_operands(vec![input; 1]);

        let operation = Operation::new(state.clone());

        Self {
            name,
            state,
            operation,
        }
    }
}

impl From<ReturnOp> for Operation {
    fn from(op: ReturnOp) -> Self {
        Self {
            state: op.state,
            instance: op.operation.instance,
        }
    }
}

#[derive(Clone)]
pub(crate) struct AddOp {
    name: String,
    state: OperationState,
    pub operation: Operation,
}

impl AddOp {
    pub fn new(location: Location, lhs: Value, rhs: Value, result_type: Type) -> Self {
        let name = String::from("toy.add");
        let mut state = OperationState::new("toy.add", location);

        let operands = vec![lhs, rhs];
        state.add_operands(operands);

        state.add_results(vec![result_type; 1]);

        let operation = Operation::new(state.clone());

        Self {
            name,
            state,
            operation,
        }
    }
}

impl From<AddOp> for Operation {
    fn from(op: AddOp) -> Self {
        Self {
            state: op.state,
            instance: op.operation.instance,
        }
    }
}

#[derive(Clone)]
pub(crate) struct MulOp {
    name: String,
    state: OperationState,
    pub operation: Operation,
}

impl MulOp {
    pub fn new(location: Location, lhs: Value, rhs: Value, result_type: Type) -> Self {
        let name = String::from("toy.mul");
        let mut state = OperationState::new("toy.mul", location);

        let operands = vec![lhs, rhs];
        state.add_operands(operands);
        state.add_results(vec![result_type; 1]);

        let operation = Operation::new(state.clone());

        Self {
            name,
            state,
            operation,
        }
    }
}

impl From<MulOp> for Operation {
    fn from(op: MulOp) -> Self {
        Self {
            state: op.state,
            instance: op.operation.instance,
        }
    }
}

#[derive(Clone)]
pub(crate) struct PrintOp {
    name: String,
    state: OperationState,
    pub operation: Operation,
}

impl PrintOp {
    pub fn new(location: Location, input: Value) -> Self {
        let name = String::from("toy.print");
        let mut state = OperationState::new("toy.print", location);

        state.add_operands(vec![input; 1]);
        let operation = Operation::new(state.clone());

        Self {
            name,
            state,
            operation,
        }
    }
}

impl From<PrintOp> for Operation {
    fn from(op: PrintOp) -> Self {
        Self {
            state: op.state,
            instance: op.operation.instance,
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
        let dialect = ToyDialect::new(&context);
        context.load_dialect(Box::new(dialect));
        let location = Location::new(Rc::clone(&context));
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
