set(LLVM_TARGET_DEFINITIONS ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCombineIncGen)

add_mlir_dialect_library(MLIRToy
        Dialect.cpp
        ToyCombine.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Toy

        DEPENDS
            ToyOpsIncGen
            ToyCombineIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
