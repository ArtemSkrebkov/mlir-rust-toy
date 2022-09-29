use std::ffi::CString;

use crate::context::Context;
use mlir_sys::{
    mlirAttributeGetContext, mlirIdentifierGet, mlirNamedAttributeGet, mlirNoneTypeGet,
    mlirStringRefCreateFromCString, MlirAttribute, MlirNamedAttribute, MlirType, MlirValue,
};

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

#[derive(Clone)]
pub struct NamedAttribute {
    pub(crate) name: String,
    pub(crate) attr: Attribute,
    pub(crate) instance: MlirNamedAttribute,
}

impl NamedAttribute {
    pub fn new(name: &str, attr: Attribute) -> Self {
        // TODO: probably better to store CString or do not store it at all
        let name = String::from(name);
        let c_name = CString::new(name.clone()).unwrap();
        unsafe {
            let mlir_context = mlirAttributeGetContext(attr.instance);
            let id = mlirIdentifierGet(
                mlir_context,
                mlirStringRefCreateFromCString(c_name.as_ptr()),
            );
            let instance = mlirNamedAttributeGet(id, attr.instance);

            Self {
                name,
                attr,
                instance,
            }
        }
    }
}
