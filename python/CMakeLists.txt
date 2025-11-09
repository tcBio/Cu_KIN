# Python bindings for cuKINSHIP-Lite using pybind11

# Find pybind11
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via find_package, trying FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Python module
pybind11_add_module(_cukinship bindings.cpp)

# Link against the core library
target_link_libraries(_cukinship PRIVATE gpu_kinship_core)

# Include directories
target_include_directories(_cukinship PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/utils
)

# Set output directory
set_target_properties(_cukinship PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cukinship"
)

# Install
install(TARGETS _cukinship
    LIBRARY DESTINATION cukinship
)
