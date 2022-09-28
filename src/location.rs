use crate::context::Context;
use mlir_sys::{mlirLocationUnknownGet, MlirLocation};

#[derive(Clone)]
pub struct Location {
    pub(crate) instance: MlirLocation,
}

impl Location {
    pub fn new(context: &Context) -> Self {
        let instance = unsafe { mlirLocationUnknownGet(context.instance) };
        Self { instance }
    }
}
