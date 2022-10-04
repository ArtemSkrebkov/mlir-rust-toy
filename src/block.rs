use crate::operation::Operation;
use mlir_sys::MlirBlock;

#[derive(Clone)]
pub struct Block {
    pub(crate) operations: Vec<Box<Operation>>,
    pub(crate) instance: MlirBlock,
}

impl Block {
    pub fn new(mlir_block: MlirBlock) -> Self {
        Self {
            operations: Vec::new(),
            instance: mlir_block,
        }
    }
}
