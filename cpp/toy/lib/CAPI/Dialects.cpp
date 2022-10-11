//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Toy-c/Dialect.h"

#include "Toy/Dialect.h"
#include "mlir/CAPI/Registration.h"

#include "mlir/CAPI/Wrap.h"
#include "Toy/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/CAPI/Pass.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Toy, toy,
                                      mlir::toy::ToyDialect)

MlirPass mlirToyCreateShapeInference() {
  return wrap(mlir::toy::createShapeInferencePass().release());
}
