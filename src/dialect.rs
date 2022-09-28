use mlir_sys::{
    mlirContextGetOrLoadDialect, mlirDialectHandleGetNamespace, mlirGetDialectHandle__std__,
    mlirStringRefCreateFromCString, MlirDialectHandle,
};

use crate::context::Context;
use crate::toy;
use std::ffi::{CStr, CString};

pub trait Dialect {
    fn get_name(&self) -> String;
}

impl From<toy::ffi::MlirDialectHandle> for mlir_sys::MlirDialectHandle {
    fn from(dialect: toy::ffi::MlirDialectHandle) -> Self {
        Self { ptr: dialect.ptr }
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
