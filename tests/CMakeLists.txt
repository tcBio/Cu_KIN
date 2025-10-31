set(TEST_SOURCES
    test_main.cu
)

add_executable(gpu_kinship_tests ${TEST_SOURCES})

target_include_directories(gpu_kinship_tests
    PRIVATE
        ${KINSHIP_SRC_DIR}
        ${KINSHIP_UTILS_DIR}
)

target_link_libraries(gpu_kinship_tests PRIVATE
    gpu_kinship_core
)

set_target_properties(gpu_kinship_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_test(NAME gpu_kinship_smoke COMMAND gpu_kinship_tests)
