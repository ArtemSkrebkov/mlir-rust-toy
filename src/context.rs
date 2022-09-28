use crate::dialect::Dialect;
use mlir_sys::{
    mlirContextCreate, mlirDialectHandleLoadDialect, mlirDialectHandleRegisterDialect,
    mlirRegisterAllDialects, MlirContext,
};

use crate::toy::mlirGetDialectHandle__toy__;

pub struct Context {
    pub(crate) instance: MlirContext,
    dialects: Vec<Box<dyn Dialect>>,
}

impl Context {
    pub fn new() -> Self {
        unsafe {
            let instance = mlirContextCreate();
            // FIXME: make dialects to be registered separately
            mlirRegisterAllDialects(instance);
            let handle = mlir_sys::MlirDialectHandle::from(mlirGetDialectHandle__toy__());
            mlirDialectHandleRegisterDialect(handle, instance);
            mlirDialectHandleLoadDialect(handle, instance);
            Self {
                instance,
                dialects: Vec::new(),
            }
        }
    }

    pub fn load_dialect(&mut self, dialect: Box<dyn Dialect>) {
        self.dialects.push(dialect);
        println!(
            "Dialect {} loaded",
            self.dialects.last().unwrap().get_name()
        );
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
