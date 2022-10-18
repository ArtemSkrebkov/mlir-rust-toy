use std::ffi::CString;

use crate::context::Context;
use mlir_sys::{
    mlirAttributeGetContext, mlirAttributeParseGet, mlirFlatSymbolRefAttrGet, mlirIdentifierGet,
    mlirNamedAttributeGet, mlirNoneTypeGet, mlirStringRefCreateFromCString, mlirUnitAttrGet,
    MlirAttribute, MlirNamedAttribute, MlirType, MlirValue,
};

#[derive(Clone)]
pub struct Type {
    pub(crate) instance: MlirType,
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
    pub(crate) fn new(instance: MlirValue) -> Value {
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
impl Attribute {
    pub fn new_flat_symbol_ref(context: &Context, symbol: &str) -> Attribute {
        let symbol = CString::new(symbol).unwrap();
        let instance = unsafe {
            mlirFlatSymbolRefAttrGet(
                context.instance,
                mlirStringRefCreateFromCString(symbol.as_ptr()),
            )
        };

        Self { instance }
    }

    pub fn new_unit(context: &Context) -> Attribute {
        let instance = unsafe { mlirUnitAttrGet(context.instance) };
        Self { instance }
    }

    pub fn new_parsed(context: &Context, string: &str) -> Attribute {
        let string = CString::new(string).unwrap();
        let instance = unsafe {
            mlirAttributeParseGet(
                context.instance,
                mlirStringRefCreateFromCString(string.as_ptr()),
            )
        };

        Self { instance }
    }
}
