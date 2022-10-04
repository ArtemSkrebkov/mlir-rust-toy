add_mlir_dialect_library(MLIRToy
        Dialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Toy

        DEPENDS
            ToyOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
