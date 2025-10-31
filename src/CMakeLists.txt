set(KINSHIP_CORE_SOURCES
    bit_encode.cpp
    normalize.cpp
    gpu_kinship.cu
)

add_library(gpu_kinship_core STATIC ${KINSHIP_CORE_SOURCES})

target_include_directories(gpu_kinship_core
    PRIVATE
        ${KINSHIP_SRC_DIR}
        ${KINSHIP_UTILS_DIR}
)

target_compile_features(gpu_kinship_core PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(gpu_kinship_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
    )
else()
    target_compile_options(gpu_kinship_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    )
endif()

if(KINSHIP_ENABLE_FAST_MATH)
    target_compile_options(gpu_kinship_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
endif()

set_target_properties(gpu_kinship_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(gpu_kinship main.cu)

target_link_libraries(gpu_kinship PRIVATE gpu_kinship_core)

target_include_directories(gpu_kinship
    PRIVATE
        ${KINSHIP_SRC_DIR}
        ${KINSHIP_UTILS_DIR}
)

if(MSVC)
    target_compile_options(gpu_kinship PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
    )
else()
    target_compile_options(gpu_kinship PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    )
endif()

set_target_properties(gpu_kinship PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
