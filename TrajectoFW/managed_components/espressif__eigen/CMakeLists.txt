idf_component_register(SRCS "dummy.c"
                       INCLUDE_DIRS "eigen"
                       REQUIRES)

target_compile_definitions(${COMPONENT_LIB} INTERFACE EIGEN_NO_DEBUG)