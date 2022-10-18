use crate::operation::Operation;
use mlir_sys::{MlirBlock, MlirRegion};

#[derive(Clone)]
pub struct Block {
    pub(crate) operations: Vec<Box<Operation>>,
    pub(crate) instance: MlirBlock,
}

impl From<MlirBlock> for Block {
    fn from(mlir_block: MlirBlock) -> Self {
        Self {
            instance: mlir_block,
            operations: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct Region {
    pub instance: MlirRegion,
}

impl From<MlirRegion> for Region {
    fn from(mlir_region: MlirRegion) -> Self {
        Self {
            instance: mlir_region,
        }
    }
}
