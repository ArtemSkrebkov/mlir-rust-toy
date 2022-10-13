//===- Dialects.h - CAPI for dialects -----------------------------*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_C_DIALECTS_H
#define STANDALONE_C_DIALECTS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir-c/Registration.h"
#include "mlir-c/Pass.h"


#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirPass mlirToyCreateLowerToAffine();

MLIR_CAPI_EXPORTED MlirPass mlirToyCreateShapeInference();

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Toy, toy);


#ifdef __cplusplus
}
#endif

#endif // STANDALONE_C_DIALECTS_H
