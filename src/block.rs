use crate::operation::Operation;
use mlir_sys::{
    mlirBlockGetFirstOperation, mlirBlockGetTerminator, mlirOperationGetNextInBlock, MlirBlock,
    MlirRegion,
};

#[derive(Clone)]
pub struct Block {
    pub(crate) operations: Vec<Box<Operation>>,
    pub(crate) instance: MlirBlock,
}

impl Block {
    // TODO: better way is to implement iterator
    pub fn back(&self) -> Operation {
        unsafe {
            let mut cur = mlirBlockGetFirstOperation(self.instance);
            let mut next = mlirOperationGetNextInBlock(cur);
            while !next.ptr.is_null() {
                cur = next;
                next = mlirOperationGetNextInBlock(next);
            }
            if cur.ptr.is_null() {
                println!("Broken terminator op");
            }
            Operation::from(cur)
        }
    }
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
