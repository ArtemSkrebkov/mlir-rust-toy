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

pub struct ConstantOpBuilder {
    state: OperationState,
}

impl ConstantOpBuilder {
    pub fn new(location: Location) -> Self {
        let state = OperationState::new("toy.constant", location.clone());
        Self { state }
    }

    pub fn result(&mut self, result_type: Type) -> &mut Self {
        let results = vec![result_type; 1];
        self.state.add_results(results);
        self
    }

    pub fn attribute(&mut self, attr: Attribute) -> &mut Self {
        let named_attr = NamedAttribute::new("value", attr);
        self.state.add_attributes(vec![named_attr; 1]);
        self
    }

    pub fn build(&mut self) -> Operation {
        Operation::new(&mut self.state)
    }
}

pub struct TransposeOpBuilder {
    state: OperationState,
}

impl TransposeOpBuilder {
    pub fn new(location: Location) -> Self {
        let state = OperationState::new("toy.transpose", location);

        Self { state }
    }

    pub fn result(&mut self, result_type: Type) -> &mut Self {
        let results = vec![result_type; 1];
        self.state.add_results(results);
        self
    }

    pub fn input(&mut self, input: Value) -> &mut Self {
        self.state.add_operands(vec![input; 1]);
        self
    }

    pub fn build(&mut self) -> Operation {
        Operation::new(&mut self.state)
    }
}

pub struct GenericCallOpBuilder {
    state: OperationState,
    location: Location,
}

impl GenericCallOpBuilder {
    pub fn new(location: Location) -> Self {
        let state = OperationState::new("toy.generic_call", location.clone());

        Self { state, location }
    }

    pub fn operands(&mut self, operands: Vec<Value>) -> &mut Self {
        self.state.add_operands(operands);
        self
    }

    pub fn result(&mut self, result_type: Type) -> &mut Self {
        let results = vec![result_type; 1];
        self.state.add_results(results);
        self
    }

    pub fn callee(&mut self, callee: &str) -> &mut Self {
        let attr = Attribute::new_flat_symbol_ref(self.location.context(), callee);
        let named_attr = NamedAttribute::new("callee", attr);

        self.state.add_attributes(vec![named_attr; 1]);
        self
    }

    pub fn build(&mut self) -> Operation {
        Operation::new(&mut self.state)
    }
}

pub struct ReturnOpBuilder {
    state: OperationState,
}

impl ReturnOpBuilder {
    pub fn new(location: Location) -> Self {
        let state = OperationState::new("toy.return", location);

        Self { state }
    }

    pub fn input(&mut self, input: Value) -> &mut Self {
        self.state.add_operands(vec![input; 1]);
        self
    }

    pub fn build(&mut self) -> Operation {
        Operation::new(&mut self.state)
    }
}

pub struct AddOpBuilder {
    state: OperationState,
}

impl AddOpBuilder {
    pub fn new(location: Location) -> Self {
        let state = OperationState::new("toy.add", location);

        Self { state }
    }

    pub fn operands(&mut self, lhs: Value, rhs: Value) -> &mut Self {
        self.state.add_operands([lhs, rhs].to_vec());
        self
    }

    pub fn result(&mut self, result_type: Type) -> &mut Self {
        let results = vec![result_type; 1];
        self.state.add_results(results);
        self
    }

    pub fn build(&mut self) -> Operation {
        Operation::new(&mut self.state)
    }
}

pub struct MulOpBuilder {
    state: OperationState,
}

impl MulOpBuilder {
    pub fn new(location: Location) -> Self {
        let state = OperationState::new("toy.mul", location);

        Self { state }
    }

    pub fn operands(&mut self, lhs: Value, rhs: Value) -> &mut Self {
        self.state.add_operands([lhs, rhs].to_vec());
        self
    }

    pub fn result(&mut self, result_type: Type) -> &mut Self {
        let results = vec![result_type; 1];
        self.state.add_results(results);
        self
    }

    pub fn build(&mut self) -> Operation {
        Operation::new(&mut self.state)
    }
}

pub struct PrintOpBuilder {
    state: OperationState,
}

impl PrintOpBuilder {
    pub fn new(location: Location) -> Self {
        let state = OperationState::new("toy.print", location);

        Self { state }
    }

    pub fn input(&mut self, input: Value) -> &mut Self {
        self.state.add_operands(vec![input; 1]);
        self
    }

    pub fn build(&mut self) -> Operation {
        Operation::new(&mut self.state)
    }
}

pub struct ReshapeOpBuilder {
    state: OperationState,
}

impl ReshapeOpBuilder {
    pub fn new(location: Location) -> Self {
        let state = OperationState::new("toy.reshape", location);

        Self { state }
    }
    pub fn input(&mut self, input: Value) -> &mut Self {
        self.state.add_operands(vec![input; 1]);
        self
    }

    pub fn result(&mut self, result_type: Type) -> &mut Self {
        let results = vec![result_type; 1];
        self.state.add_results(results);
        self
    }

    pub fn build(&mut self) -> Operation {
        Operation::new(&mut self.state)
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

        let op_builder = OpBuilder::new(None, 0, context);
        let result_type = op_builder.get_f64_type();
        let ty = op_builder.get_f64_type();
        let ty = op_builder.get_ranked_tensor_type(vec![3, 1], ty);
        let attr: Attribute = op_builder.get_dense_elements_attr(ty.clone(), vec![1.0, 1.0, 1.0]);
        let _constant = ConstantOpBuilder::new(location)
            .result(result_type)
            .attribute(attr)
            .build();
    }
}
