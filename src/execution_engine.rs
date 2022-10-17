use std::{ffi::CString, os::raw::c_void};

use mlir_sys::{
    mlirExecutionEngineCreate, mlirExecutionEngineInvokePacked, mlirRegisterAllLLVMTranslations,
    mlirStringRefCreateFromCString, MlirExecutionEngine, MlirStringRef,
};

use crate::{context::Context, operation::ModuleOp};

pub struct ExecutionEngine {
    instance: MlirExecutionEngine,
}

impl ExecutionEngine {
    pub fn new(context: &Context, module: &ModuleOp) -> Self {
        unsafe {
            mlirRegisterAllLLVMTranslations(context.instance);
            let opt_level = 2;
            let num_paths = 0;
            let shared_libs_paths_cstr = CString::new("").unwrap();
            let shared_libs_paths = mlirStringRefCreateFromCString(shared_libs_paths_cstr.as_ptr());
            let shared_libs_paths_ptr: *const MlirStringRef = &shared_libs_paths;
            let instance = mlirExecutionEngineCreate(
                module.instance,
                opt_level,
                num_paths,
                shared_libs_paths_ptr,
            );

            Self { instance }
        }
    }

    pub fn run(&self, func_name: &str) {
        let name = CString::new(func_name).unwrap();
        unsafe {
            let mut args: *mut c_void = std::ptr::null_mut();
            let args_ptr: *mut *mut c_void = &mut args;
            let _ = mlirExecutionEngineInvokePacked(
                self.instance,
                mlirStringRefCreateFromCString(name.as_ptr()),
                args_ptr,
            );
        }
    }
}
