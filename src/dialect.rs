use mlir_sys::{mlirDialectHandleGetNamespace, mlirGetDialectHandle__std__, MlirDialectHandle};

use crate::context::Context;
use std::ffi::CStr;

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
            let c_str: &CStr = CStr::from_ptr(namespace.data);
            let str_slice: &str = c_str.to_str().unwrap();
            let str_buf: String = str_slice.to_owned();
            str_buf
        }
    }

    fn handle(&self) -> MlirDialectHandle {
        self.instance
    }
}
