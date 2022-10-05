use mlir_sys::{
    mlirAttributeGetNull, mlirBlockInsertOwnedOperation, mlirDenseElementsAttrDoubleGet,
    mlirF64TypeGet, mlirFloatAttrDoubleGet, mlirFunctionTypeGet, mlirRankedTensorTypeGet,
    mlirUnrankedTensorTypeGet, MlirType,
};

use crate::block::Block;
use crate::context::Context;
use crate::misc::{Attribute, Type};
use crate::operation::Operation;
use std::rc::Rc;

pub struct OpBuilder {
    context: Rc<Context>,
    block: Option<Rc<Block>>,
    pos: isize,
}

impl<'ctx> OpBuilder {
    pub fn set_insertion_point(&mut self, block: Rc<Block>, pos: isize) {
        self.block = Some(block);
        self.pos = pos;
    }

    pub fn insert(&mut self, operation: Operation) {
        unsafe {
            let block: &Block = self.block.as_ref().unwrap();
            mlirBlockInsertOwnedOperation(block.instance, self.pos, operation.instance);
            self.pos += 1;
        }
    }

    pub fn get_f64_type(&self) -> Type {
        unsafe { Type::from(mlirF64TypeGet(self.context.instance)) }
    }

    // TODO: redundant copies of dims
    pub fn get_ranked_tensor_type(&self, dims: Vec<usize>, elem_ty: Type) -> Type {
        let rank: isize = dims.len() as isize;
        let shape: Vec<i64> = dims.into_iter().map(|x| x as i64).collect();
        let p_shape = shape.as_ptr();
        // NB: not sure what else can be used as enconding, so passing mlirAttributeGetNull for now
        unsafe {
            Type::from(mlirRankedTensorTypeGet(
                rank,
                p_shape,
                elem_ty.instance,
                mlirAttributeGetNull(),
            ))
        }
    }

    pub fn get_dense_elements_attr(&self, data_ty: Type, data: Vec<f64>) -> Attribute {
        unsafe {
            Attribute::from(mlirDenseElementsAttrDoubleGet(
                data_ty.instance,
                data.len() as isize,
                data.as_ptr(),
            ))
        }
    }

    pub fn get_float_attr_double(&self, data_ty: Type, data: f64) -> Attribute {
        unsafe {
            Attribute::from(mlirFloatAttrDoubleGet(
                self.context.instance,
                data_ty.instance,
                data,
            ))
        }
    }

    pub fn get_unranked_tensor_type(&self, elem_type: Type) -> Type {
        unsafe { Type::from(mlirUnrankedTensorTypeGet(elem_type.instance)) }
    }

    pub fn get_function_type(&self, arg_types: Vec<Type>, result_types: Vec<Type>) -> Type {
        let num_inputs = arg_types.len() as isize;
        let num_result = result_types.len() as isize;
        let args: Vec<MlirType> = arg_types.into_iter().map(|x| x.instance).collect();
        let p_args: *const MlirType = args.as_ptr();
        let results: Vec<MlirType> = result_types.into_iter().map(|x| x.instance).collect();
        let p_results: *const MlirType = results.as_ptr();

        unsafe {
            Type::from(mlirFunctionTypeGet(
                self.context.instance,
                num_inputs,
                p_args,
                num_result,
                p_results,
            ))
        }
    }

    pub fn new(block: Option<Rc<Block>>, pos: isize, context: Rc<Context>) -> Self {
        Self {
            context,
            block,
            pos,
        }
    }
}
