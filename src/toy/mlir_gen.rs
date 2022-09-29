use std::collections::HashMap;
use std::rc::Rc;

use mlir_sys::{mlirBlockGetArgument, mlirValueDump};

use crate::context::Context;
use crate::location::Location;
use crate::misc::{Attribute, Type, Value};
use crate::op_builder::OpBuilder;
use crate::operation::{FuncOp, ModuleOp, OneRegion, Operation};

use crate::toy::parser::Expr::{
    Binary, Call, ExprList, Number, Print, Return, Tensor, VarDecl, Variable,
};

use crate::toy::parser::{Expr, Function, Module, Prototype};

use crate::toy::toy_dialect::{
    AddOp, ConstantOp, GenericCallOp, MulOp, PrintOp, ReturnOp, TransposeOp,
};

pub struct MLIRGen {
    module: ModuleOp,
    symbol_table: HashMap<String, Value>,
    context: Rc<Context>,
    builder: OpBuilder,
}

impl<'ctx> MLIRGen {
    pub fn new(context: Rc<Context>) -> Self {
        Self {
            module: ModuleOp::new(Location::new(&*context)),
            symbol_table: HashMap::new(),
            context: Rc::clone(&context),
            builder: OpBuilder::new(Option::None, 0, Rc::clone(&context)),
        }
    }
    // TODO: pass parsed result
    pub fn mlir_gen(&mut self, module_ast: Module) -> ModuleOp {
        self.module = ModuleOp::new(Location::new(&self.context));

        // TODO: implement Iterator for Module?
        for f in module_ast.functions {
            let func = self.mlir_gen_function(f);
            self.module.push_back(Box::new(Operation::from(func)));
            // add func into self.module
        }

        self.module.clone()
    }

    fn mlir_gen_function(&mut self, function_ast: Function) -> FuncOp {
        // varScopeTable
        // let var_scope: HashMap<&str, Value> = HashMap::new();
        let function: FuncOp = self.mlir_gen_prototype(function_ast.prototype.clone());

        let entry_block = function.block.clone();
        // TODO: getter for prototype
        let proto_args = function_ast.prototype.args.clone();
        // FIXME: just for the sake of adding something into the table
        // need to review how it is done in original Toy tutorial
        let mut pos = 0;
        for arg in proto_args {
            unsafe {
                let mlir_arg_value = mlirBlockGetArgument(entry_block.instance, pos);
                self.declare(arg, Value::new(mlir_arg_value));
                pos += 1;
            }
        }
        // TODO: declare all the function arguments in the symbol table
        // TODO: implement builder method
        // self.builder.set_insertion_point_to_start(entry_block);
        self.builder
            .set_insertion_point(Rc::clone(&function.block), 0);
        //
        // mlirBlockInsertOwnedOperationAfter(function.block.instance, reference, operation)
        // mlirOperationGetBlock(op)

        self.mlir_gen_expression(function_ast.body.unwrap());
        // function.erase();

        // TODO: handle return op

        function
    }

    fn mlir_gen_prototype(&mut self, prototype_ast: Prototype) -> FuncOp {
        // FIXME: construct location from AST location
        let location = Location::new(&self.context);
        let arg_types = vec![self.get_type(Vec::new()); prototype_ast.args.len()];
        let func_type = self.builder.get_function_type(arg_types, Vec::new());

        // TODO: make it builder
        FuncOp::new(location, &prototype_ast.name, func_type)
    }

    fn declare(&mut self, name: String, value: Value) {
        self.symbol_table.insert(name, value);
    }

    fn mlir_gen_expression(&mut self, expr: Expr) -> Result<Value, &'static str> {
        // NB: this clone is used for collect_data method
        // there should be a way to avoid this
        let clone_expr = expr.clone();
        match expr {
            ExprList { expressions } => {
                for expr in expressions {
                    let _value = self.mlir_gen_expression(*expr.clone());
                }
                Err("ExprList not implemented")
            }
            VarDecl { name, value } => {
                let value = self.mlir_gen_expression(*value).unwrap();
                // TODO: reshape op
                // declare variable in the symbol table
                self.declare(name, value.clone());
                Ok(value)
            }
            Variable(name) => {
                if self.symbol_table.contains_key(&name) {
                    let value = (*self.symbol_table.get(&name).unwrap()).clone();
                    return Ok(value);
                }
                Err("Variable is not found")
                // extract variable from symbol table
            }
            Tensor {
                location,
                values,
                dims,
            } => {
                let size = dims.iter().product();
                let mut data: Vec<f64> = Vec::new();
                data.reserve(size);
                self.collect_data(clone_expr, &mut data);
                // TODO: contruct ConstanOp with array
                let elem_ty = self.builder.get_f64_type();
                let data_ty = self.builder.get_ranked_tensor_type(dims, elem_ty);
                let data_attr: Attribute =
                    self.builder.get_dense_elements_attr(data_ty.clone(), data);
                // TODO: a separate builder for constructing a type?
                let mut op = ConstantOp::new(Location::new(&self.context));
                op.with_result(data_ty).with_attribute(data_attr).build();
                self.builder.insert(Operation::from(op.clone()));
                Ok(Value::from(op.operation))
            }
            Number(num) => {
                let location = Location::new(&self.context);
                let mut op = ConstantOp::new(location);
                op.with_value(num);
                self.builder.insert(Operation::from(op.clone()));
                Ok(Value::from(op.operation))
            }
            Call { fn_name, args } => {
                let location = Location::new(&self.context);
                let mut operands: Vec<Value> = Vec::new();
                for arg in &args {
                    let arg = self.mlir_gen_expression(arg.clone()).unwrap();
                    operands.push(arg);
                }
                if fn_name == "transpose" {
                    if args.len() != 1 {
                        panic!("MLIR codegen encountered an error: toy.transpose does not accept multiple args");
                    }
                    let op = TransposeOp::new(
                        location,
                        operands[0].clone(),
                        self.builder
                            .get_unranked_tensor_type(self.builder.get_f64_type()),
                    );
                    self.builder.insert(Operation::from(op.clone()));
                    let value = Value::from(op);

                    return Ok(value);
                    // return Ok(Value::from(op));
                }

                let result_type = self
                    .builder
                    .get_unranked_tensor_type(self.builder.get_f64_type());
                let op = GenericCallOp::new(location, fn_name, operands, result_type);
                self.builder.insert(Operation::from(op.clone()));
                let value = Value::from(op);
                unsafe { mlirValueDump(value.instance) };
                Ok(value)
            }
            Return {
                location: _,
                expression,
            } => {
                let location = Location::new(&self.context);
                let value = self.mlir_gen_expression(*expression).unwrap();
                let op = ReturnOp::new(location, value);
                self.builder.insert(Operation::from(op.clone()));
                Ok(Value::from(op))
            }

            Binary { op, left, right } => {
                let lhs = self.mlir_gen_expression(*left).unwrap();
                let rhs = self.mlir_gen_expression(*right).unwrap();
                let result_type = self
                    .builder
                    .get_unranked_tensor_type(self.builder.get_f64_type());
                let location = Location::new(&self.context);
                match op {
                    '+' => {
                        let op = AddOp::new(location, lhs, rhs);
                        self.builder.insert(Operation::from(op.clone()));
                        Ok(Value::from(op))
                    }
                    '*' => {
                        let op = MulOp::new(location, lhs, rhs, result_type);
                        self.builder.insert(Operation::from(op.clone()));
                        Ok(Value::from(op))
                    }
                    _ => Err("Invalid binary operation"),
                }
            }

            Print {
                location: _,
                expression,
            } => {
                let location = Location::new(&self.context);
                let value = self.mlir_gen_expression(*expression).unwrap();
                let op = PrintOp::new(location, value);
                self.builder.insert(Operation::from(op.clone()));
                Ok(Value::from(op))
            }
            _ => {
                panic!("Unknown expression");
            }
        }
    }

    fn collect_data(&self, expr: Expr, data: &mut Vec<f64>) {
        match expr {
            Tensor {
                location: _,
                values,
                dims: _,
            } => {
                for v in values {
                    self.collect_data(v.clone(), data);
                }
            }
            Number(num) => {
                data.push(num);
            }
            _ => {
                panic!("Unexpected expression");
            }
        }
    }

    fn get_type(&self, shape: Vec<usize>) -> Type {
        if shape.is_empty() {
            return self
                .builder
                .get_unranked_tensor_type(self.builder.get_f64_type());
        }

        self.builder
            .get_ranked_tensor_type(shape, self.builder.get_f64_type())
    }
}
