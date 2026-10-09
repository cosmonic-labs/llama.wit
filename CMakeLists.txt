cmake_minimum_required(VERSION 3.24)
project(llama-cpp-wit C CXX)

include(FetchContent)

# https://releases.llvm.org/20.1.0/docs/ReleaseNotes.html#changes-to-the-webassembly-backend
# > this option is turned on by default for the moment
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-mllvm;-wasm-use-legacy-eh=false>")

set(LLAMA_GIT_TAG b11514 CACHE STRING "llama.cpp release tag to build")

# Trim llama.cpp to just the static libs we link.
set(BUILD_SHARED_LIBS  OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE        OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP        OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL    OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL         OFF CACHE BOOL "" FORCE)
set(LLAMA_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_WEBGPU        ON  CACHE BOOL "" FORCE)

set(WASI_WEBGPU_HEADERS_REPOSITORY "https://github.com/wasi-gfx/wasi-webgpu-headers.git" CACHE STRING "wasi-webgpu-headers git remote")
set(WASI_WEBGPU_HEADERS_TAG "917765edd7cb8f8cf95bda1217514c78f71b582b" CACHE STRING "wasi-webgpu-headers git ref")
set(DAWN_HEADERS_URL "https://github.com/google/dawn/releases/download/v20260825.221656/dawn-headers-6de3790c93dab8baac0555db186420adacc395be.tar.gz"
    CACHE STRING "dawn-headers release tarball")
set(DAWN_HEADERS_SHA256 "929f0548ff6a0d728d4a39e036139694211671a38d838f9f2d7e22181a6f98f2"
    CACHE STRING "SHA256 of DAWN_HEADERS_URL")

# Download only, we compile its files via dawn::webgpu_dawn below
FetchContent_Declare(wasi_webgpu_headers
    GIT_REPOSITORY ${WASI_WEBGPU_HEADERS_REPOSITORY}
    GIT_TAG        ${WASI_WEBGPU_HEADERS_TAG}
    SOURCE_SUBDIR  _skip_configure_)

# OVERRIDE_FIND_PACKAGE to satisfy ggml-webgpu's find_package(Dawn REQUIRED)
FetchContent_Declare(dawn
    URL      ${DAWN_HEADERS_URL}
    URL_HASH SHA256=${DAWN_HEADERS_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    OVERRIDE_FIND_PACKAGE)

FetchContent_MakeAvailable(wasi_webgpu_headers dawn)

# Stands in for the Dawn library ggml-webgpu links: the wasi:webgpu glue, no Dawn runtime
add_library(dawn::webgpu_dawn INTERFACE IMPORTED)
set_target_properties(dawn::webgpu_dawn PROPERTIES
    INTERFACE_SOURCES
        "${wasi_webgpu_headers_SOURCE_DIR}/webgpu.c;${wasi_webgpu_headers_SOURCE_DIR}/async_futures.c;${wasi_webgpu_headers_SOURCE_DIR}/imports.c;${wasi_webgpu_headers_SOURCE_DIR}/imports_component_type.o"
    INTERFACE_INCLUDE_DIRECTORIES
        "${dawn_SOURCE_DIR}/include/webgpu_upstream;${wasi_webgpu_headers_SOURCE_DIR}"
)

FetchContent_Declare(llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG ${LLAMA_GIT_TAG}
    GIT_SHALLOW TRUE
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
set_target_properties(provider PROPERTIES OUTPUT_NAME llama-cpp.wasm)
target_include_directories(provider PRIVATE src/generated)
target_link_options(provider PRIVATE -mexec-model=reactor)
target_link_libraries(provider PRIVATE llama)
