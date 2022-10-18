use std::collections::HashMap;
use std::rc::Rc;

use mlir_sys::mlirBlockGetArgument;

use crate::context::Context;
use crate::location::Location;
use crate::misc::{Attribute, Type, Value};
use crate::op_builder::OpBuilder;
use crate::operation::{FuncOp, ModuleOp, OneRegion, Operation};

use crate::toy::parser::Expr::{
    Binary, Call, ExprList, Number, Print, Return, Tensor, VarDecl, Variable,
};

use crate::toy::parser::{Expr, Function, Module, Prototype};

use super::toy_dialect::{
    AddOpBuilder, ConstantOpBuilder, GenericCallOpBuilder, MulOpBuilder, PrintOpBuilder,
    ReshapeOpBuilder, ReturnOpBuilder, TransposeOpBuilder,
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
            module: ModuleOp::new(Location::new(Rc::clone(&context))),
            symbol_table: HashMap::new(),
            context: Rc::clone(&context),
            builder: OpBuilder::new(Option::None, 0, Rc::clone(&context)),
        }
    }

    pub fn mlir_gen(&mut self, module_ast: Module) -> ModuleOp {
        self.module = ModuleOp::new(Location::new(Rc::clone(&self.context)));

        // TODO: implement Iterator for Module?
        for f in module_ast.functions {
            let func = self.mlir_gen_function(f);
            self.module.push_back(Box::new(func.operation.clone()));
        }

        self.module.clone()
    }

    fn mlir_gen_function(&mut self, function_ast: Function) -> FuncOp {
        let function: FuncOp = self.mlir_gen_prototype(function_ast.prototype.clone());

        let entry_block = function.block.clone();
        let proto_args = function_ast.prototype.args.clone();
        let mut pos = 0;
        for arg in proto_args {
            unsafe {
                let mlir_arg_value = mlirBlockGetArgument(entry_block.instance, pos);
                self.declare(arg, Value::new(mlir_arg_value));
                pos += 1;
            }
        }
        self.builder
            .set_insertion_point(Rc::clone(&function.block), 0);

        let _ = self.mlir_gen_expression(function_ast.body.unwrap());

        if function_ast.prototype.name != String::from("main") {
            function.set_private();
        }
        // TODO: function type should be calculated based on the return value
        function
    }

    fn mlir_gen_prototype(&mut self, prototype_ast: Prototype) -> FuncOp {
        // TODO: construct location from AST location
        let location = Location::new(Rc::clone(&self.context));
        let arg_types = vec![self.get_type(Vec::new()); prototype_ast.args.len()];
        let func_type = self.builder.get_function_type(arg_types, Vec::new());
        // NB: only main function is exported to outside
        let exported = if prototype_ast.name == String::from("main") {
            true
        } else {
            false
        };

        FuncOp::new(location, &prototype_ast.name, func_type, exported)
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
            VarDecl {
                name,
                var_type,
                value,
            } => {
                let mut value = self.mlir_gen_expression(*value).unwrap();
                if !var_type.shape.is_empty() {
                    let location = Location::new(Rc::clone(&self.context));
                    let var_type = self.get_type(var_type.shape);
                    let op = ReshapeOpBuilder::new(location)
                        .result(var_type)
                        .input(value.clone())
                        .build();
                    self.builder.insert(op.clone());
                    value = Value::from(op);
                }
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
                location: _,
                values: _, // read by collect_data method
                dims,
            } => {
                let size = dims.iter().product();
                let mut data: Vec<f64> = Vec::new();
                data.reserve(size);
                self.collect_data(clone_expr, &mut data);

                let elem_ty = self.builder.get_f64_type();
                let data_ty = self.builder.get_ranked_tensor_type(dims, elem_ty);
                let data_attr: Attribute =
                    self.builder.get_dense_elements_attr(data_ty.clone(), data);
                let mut op = ConstantOpBuilder::new(Location::new(Rc::clone(&self.context)))
                    .result(data_ty)
                    .attribute(data_attr)
                    .build();
                self.builder.insert(op.clone());
                Ok(Value::from(op))
            }
            Number(num) => {
                let location = Location::new(Rc::clone(&self.context));
                // FIXME: consider constant as a tensor with shape 1
                // otherwise, getting a conversion error
                let elem_ty = self.builder.get_f64_type();
                let elem_ty = self.builder.get_ranked_tensor_type(vec![1], elem_ty);
                let elem_attr: Attribute = self
                    .builder
                    .get_dense_elements_attr(elem_ty.clone(), vec![num]);

                let op = ConstantOpBuilder::new(location)
                    .result(elem_ty)
                    .attribute(elem_attr)
                    .build();

                self.builder.insert(op.clone());
                Ok(Value::from(op))
            }
            Call { fn_name, args } => {
                let location = Location::new(Rc::clone(&self.context));
                let mut operands: Vec<Value> = Vec::new();
                for arg in &args {
                    let arg = self.mlir_gen_expression(arg.clone()).unwrap();
                    operands.push(arg);
                }
                if fn_name == "transpose" {
                    if args.len() != 1 {
                        panic!("MLIR codegen encountered an error: toy.transpose does not accept multiple args");
                    }
                    let op = TransposeOpBuilder::new(location)
                        .input(operands[0].clone())
                        .result(
                            self.builder
                                .get_unranked_tensor_type(self.builder.get_f64_type()),
                        )
                        .build();
                    self.builder.insert(op.clone());
                    let value = Value::from(op);

                    return Ok(value);
                }

                let result_type = self
                    .builder
                    .get_unranked_tensor_type(self.builder.get_f64_type());
                let op = GenericCallOpBuilder::new(location)
                    .callee(&fn_name)
                    .operands(operands)
                    .result(result_type)
                    .build();
                self.builder.insert(op.clone());
                let value = Value::from(op);
                Ok(value)
            }
            Return {
                location: _,
                expression,
            } => {
                let location = Location::new(Rc::clone(&self.context));
                if let Some(expr) = expression {
                    let value = self.mlir_gen_expression(*expr).unwrap();
                    let op = ReturnOpBuilder::new(location).input(value).build();
                    self.builder.insert(op.clone());
                    return Ok(Value::from(op));
                } else {
                    let op = ReturnOpBuilder::new(location).build();
                    self.builder.insert(op.clone());
                    return Ok(Value::from(op));
                }
            }

            Binary { op, left, right } => {
                let lhs = self.mlir_gen_expression(*left).unwrap();
                let rhs = self.mlir_gen_expression(*right).unwrap();
                let result_type = self
                    .builder
                    .get_unranked_tensor_type(self.builder.get_f64_type());
                let location = Location::new(Rc::clone(&self.context));
                match op {
                    '+' => {
                        let op = AddOpBuilder::new(location)
                            .operands(lhs, rhs)
                            .result(result_type)
                            .build();
                        self.builder.insert(op.clone());
                        Ok(Value::from(op))
                    }
                    '*' => {
                        let op = MulOpBuilder::new(location)
                            .operands(lhs, rhs)
                            .result(result_type)
                            .build();
                        self.builder.insert(op.clone());
                        Ok(Value::from(op))
                    }
                    _ => Err("Invalid binary operation"),
                }
            }

            Print {
                location: _,
                expression,
            } => {
                let location = Location::new(Rc::clone(&self.context));
                let value = self.mlir_gen_expression(*expression).unwrap();
                let op = PrintOpBuilder::new(location).input(value).build();
                self.builder.insert(op.clone());
                Ok(Value::from(op))
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
