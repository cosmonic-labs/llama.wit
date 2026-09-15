cmake_minimum_required(VERSION 3.21)
project(llama-cpp-wit C CXX)

include(FetchContent)

# https://releases.llvm.org/20.1.0/docs/ReleaseNotes.html#changes-to-the-webassembly-backend
# > this option is turned on by default for the moment
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-mllvm;-wasm-use-legacy-eh=false>")

# Upstream llama.cpp, pinned to a release (https://github.com/ggml-org/llama.cpp/releases).
set(LLAMA_GIT_TAG b10068 CACHE STRING "llama.cpp release tag to build")

# Trim llama.cpp to just the static CPU libs we link.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE       OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP       OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL        OFF CACHE BOOL "" FORCE)

# Not warnings on upstream llama.cpp
set(LLAMA_ALL_WARNINGS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG ${LLAMA_GIT_TAG}
    GIT_SHALLOW TRUE
    ${_llama_patch}
)
FetchContent_MakeAvailable(llama)

set(GEN ${CMAKE_CURRENT_SOURCE_DIR}/src/generated)

# Regenerate the committed C bindings from wit/. Run on demand after a WIT change.
add_custom_target(regenerate-bindings
    COMMAND wit-bindgen c ${CMAKE_CURRENT_SOURCE_DIR}/wit --world provider --out-dir ${GEN}
    VERBATIM
)

add_executable(provider
    src/inference.c
    src/generated/provider.c
    src/generated/provider_component_type.o
)
# Sources are C, but llama.cpp's static libs are C++ — link with the C++ driver so
# the C++ runtime (libc++/libunwind) is pulled in.
set_target_properties(provider PROPERTIES OUTPUT_NAME llama-cpp.wasm SUFFIX "" LINKER_LANGUAGE CXX)
target_include_directories(provider PRIVATE src/generated)
# library component, no main()
target_link_options(provider PRIVATE -mexec-model=reactor)
target_link_libraries(provider PRIVATE llama ggml ggml-cpu ggml-base)
if(LLAMA_WIT_WEBGPU)
    # ggml-webgpu carries the wasi-webgpu-headers bindings + imports_component_type.o,
    # which add the wasi:webgpu imports to the final component.
    target_link_libraries(provider PRIVATE ggml-webgpu)
endif()
