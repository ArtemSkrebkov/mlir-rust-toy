set(LLVM_TARGET_DEFINITIONS ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCombineIncGen)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_mlir_dialect_library(MLIRToy
        ShapeInferencePass.cpp
        Dialect.cpp
        ToyCombine.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Toy

        DEPENDS
            ToyOpsIncGen
            ToyShapeInferenceInterfaceIncGen
            ToyCombineIncGen

	LINK_LIBS PUBLIC
    MLIRAnalysis
    MLIRCastInterfaces
    MLIRCallInterfaces
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTransforms)
