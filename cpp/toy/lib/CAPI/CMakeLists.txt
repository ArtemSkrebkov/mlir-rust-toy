add_mlir_public_c_api_library(ToyCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRToy
)
