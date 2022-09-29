use mlir_sys::{
    mlirContextGetOrLoadDialect, mlirDialectHandleGetNamespace, mlirGetDialectHandle__std__,
    mlirStringRefCreateFromCString, MlirDialectHandle,
};

use crate::toy::ffi::mlirGetDialectHandle__toy__;

use crate::context::Context;
use crate::toy;
use std::ffi::{CStr, CString};

pub trait Dialect {
    fn get_name(&self) -> String;
    fn handle(&self) -> MlirDialectHandle;
}

pub struct StandardDialect {
    instance: MlirDialectHandle,
}

impl StandardDialect {
    pub fn new(_context: &Context) -> Self {
        unsafe {
            let instance = mlirGetDialectHandle__std__();
            Self { instance }
        }
    }
}

impl Dialect for StandardDialect {
    fn get_name(&self) -> String {
        unsafe {
            let namespace = mlirDialectHandleGetNamespace(self.instance);
            let c_str: &CStr = unsafe { CStr::from_ptr(namespace.data) };
            let str_slice: &str = c_str.to_str().unwrap();
            let str_buf: String = str_slice.to_owned();
            str_buf
        }
    }

    fn handle(&self) -> MlirDialectHandle {
        self.instance
    }
}
