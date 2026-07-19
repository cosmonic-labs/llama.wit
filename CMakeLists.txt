cmake_minimum_required(VERSION 3.21)
project(llama-cpp-wit C CXX)

include(FetchContent)

# Upstream llama.cpp, pinned to a release (https://github.com/ggml-org/llama.cpp/releases).
set(LLAMA_GIT_TAG b10068 CACHE STRING "llama.cpp release tag to build")

# Trim llama.cpp to just the static CPU libs we link.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE       OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP       OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL        OFF CACHE BOOL "" FORCE)

FetchContent_Declare(llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG ${LLAMA_GIT_TAG}
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(llama)

set(GEN ${CMAKE_CURRENT_SOURCE_DIR}/src/generated)

# Regenerate the committed C++ bindings from wit/. Run on demand after a WIT change
add_custom_target(regenerate-bindings
    COMMAND wit-bindgen cpp ${CMAKE_CURRENT_SOURCE_DIR}/wit --world provider --out-dir ${GEN}
    # Move the generated exports-*.h includes (emitted at the top) below the record
    # definitions they reference, else provider_cpp.h won't compile. Resource-agnostic:
    # relocates whatever exports-* includes exist to just before the closing #endif.
    # TODO: remove once wit-bindgen issue #1658 is fixed
    # https://github.com/bytecodealliance/wit-bindgen/issues/1658
    COMMAND awk -v out=${GEN}/provider_cpp.h.tmp
        [[/#include "exports-/{m=m $0 ORS;next} /^#endif/&&!d{printf "%s",m>out;d=1} {print>out}]]
        ${GEN}/provider_cpp.h
    COMMAND ${CMAKE_COMMAND} -E rename ${GEN}/provider_cpp.h.tmp ${GEN}/provider_cpp.h
    VERBATIM
)

add_executable(provider
    src/inference.cpp
    src/generated/provider.cpp
    src/generated/provider_component_type.o
)
set_target_properties(provider PROPERTIES OUTPUT_NAME llama.wasm SUFFIX "")
target_compile_features(provider PRIVATE cxx_std_23)
target_include_directories(provider PRIVATE src/generated)
# library component, no main()
target_link_options(provider PRIVATE -mexec-model=reactor)
target_link_libraries(provider PRIVATE llama ggml ggml-cpu ggml-base)
