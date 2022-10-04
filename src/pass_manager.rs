use std::{ffi::CString, rc::Rc};

use mlir_sys::{
    mlirCreateTransformsCanonicalizer, mlirOpPassManagerAddOwnedPass, mlirPassManagerCreate,
    mlirPassManagerGetNestedUnder, mlirPassManagerRun, mlirStringRefCreateFromCString, MlirPass,
    MlirPassManager,
};

use crate::context::Context;

pub struct PassManager {
    instance: MlirPassManager,
}

impl PassManager {
    pub fn new(context: Rc<Context>) -> Self {
        let instance = unsafe { mlirPassManagerCreate(context.instance) };

        Self { instance }
    }

    pub fn create_canonicalizer_pass() -> Pass {
        let mlir_pass = unsafe { mlirCreateTransformsCanonicalizer() };
        Pass {
            instance: mlir_pass,
        }
    }

    pub fn add_nested_pass(&self, pass: Pass, op_name: &str) {
        unsafe {
            let op_name = CString::new(op_name).unwrap();
            let mlir_op_manager = mlirPassManagerGetNestedUnder(
                self.instance,
                mlirStringRefCreateFromCString(op_name.as_ptr()),
            );
            mlirOpPassManagerAddOwnedPass(mlir_op_manager, pass.instance);
        }
    }

    pub fn run(&self, module: &crate::operation::ModuleOp) {
        // TODO: check returned value
        unsafe { mlirPassManagerRun(self.instance, module.instance) };
    }
}

pub struct Pass {
    instance: MlirPass,
}
