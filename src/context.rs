use crate::dialect::Dialect;
use mlir_sys::{
    mlirContextCreate, mlirDialectHandleLoadDialect, mlirDialectHandleRegisterDialect, MlirContext,
};

pub struct Context {
    pub(crate) instance: MlirContext,
}

impl Context {
    pub fn new() -> Self {
        unsafe {
            let instance = mlirContextCreate();
            Self { instance }
        }
    }

    pub fn load_dialect(&self, dialect: Box<dyn Dialect>) {
        unsafe {
            mlirDialectHandleRegisterDialect(dialect.handle(), self.instance);
            mlirDialectHandleLoadDialect(dialect.handle(), self.instance);
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
