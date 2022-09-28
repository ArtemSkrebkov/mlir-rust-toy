use crate::context::Context;
use mlir_sys::{mlirNoneTypeGet, MlirAttribute, MlirType, MlirValue};

#[derive(Clone)]
pub struct Type {
    pub(crate) instance: MlirType,
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
pub struct Attribute {
    pub(crate) instance: MlirAttribute,
}

impl From<MlirAttribute> for Attribute {
    fn from(instance: MlirAttribute) -> Self {
        Self { instance }
    }
}

#[derive(Clone)]
pub struct Value {
    pub(crate) instance: MlirValue,
}

impl Value {
    // TODO: make it available only for crate but not for user
    pub fn new(instance: MlirValue) -> Value {
        Self { instance }
    }
}
