use std::{ffi::CString, rc::Rc};

use mlir_sys::{
    mlirCreateTransformsCSE, mlirCreateTransformsCanonicalizer, mlirCreateTransformsInliner,
    mlirOpPassManagerAddOwnedPass, mlirPassManagerAddOwnedPass, mlirPassManagerCreate,
    mlirPassManagerGetNestedUnder, mlirPassManagerRun, mlirStringRefCreateFromCString, MlirPass,
    MlirPassManager,
};

use crate::context::Context;
use crate::toy;
use crate::toy::ffi::mlirToyCreateShapeInference;

impl From<toy::ffi::MlirPass> for mlir_sys::MlirPass {
    fn from(dialect: toy::ffi::MlirPass) -> Self {
        Self { ptr: dialect.ptr }
    }
}

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

    pub fn create_inliner_pass() -> Pass {
        let mlir_pass = unsafe { mlirCreateTransformsInliner() };
        Pass {
            instance: mlir_pass,
        }
    }

    pub fn create_cse_pass() -> Pass {
        let mlir_pass = unsafe { mlirCreateTransformsCSE() };
        Pass {
            instance: mlir_pass,
        }
    }

    // TODO: this is pass created by Toy dialect and need to be extracted from common passes
    pub fn create_shape_inference_pass() -> Pass {
        let mlir_pass = unsafe { mlirToyCreateShapeInference() };
        Pass {
            instance: mlir_sys::MlirPass::from(mlir_pass),
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

    pub fn add_owned_pass(&self, pass: Pass) {
        unsafe {
            mlirPassManagerAddOwnedPass(self.instance, pass.instance);
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
