use std::rc::Rc;

use crate::context::Context;
use mlir_sys::{mlirLocationUnknownGet, MlirLocation};

#[derive(Clone)]
pub struct Location {
    pub(crate) instance: MlirLocation,
    pub(crate) context: Rc<Context>,
}

impl Location {
    pub fn new(context: Rc<Context>) -> Self {
        let instance = unsafe { mlirLocationUnknownGet(context.instance) };
        Self { instance, context }
    }

    pub fn context(&self) -> &Context {
        &self.context
    }
}
