cmake_minimum_required(VERSION 3.24)
project(llama-cpp-wit C CXX)

include(FetchContent)

# patches/ applies to exactly this tag: move them together.
set(LLAMA_GIT_TAG b12135 CACHE STRING "llama.cpp release tag to build")
option(LLAMA_WIT_JSON_SCHEMA "Build the provider-with-json-schema world, which also exports llama.cpp's JSON Schema to GBNF converter" OFF)

# Trim llama.cpp to just the static libs we link.
set(BUILD_SHARED_LIBS  OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE        OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP        OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL    OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL         OFF CACHE BOOL "" FORCE)
set(LLAMA_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_WEBGPU        ON  CACHE BOOL "" FORCE)

set(WASI_WEBGPU_HEADERS_REPOSITORY "https://github.com/wasi-gfx/wasi-webgpu-headers.git" CACHE STRING "wasi-webgpu-headers git remote")
set(WASI_WEBGPU_HEADERS_TAG "5163b36ce9d679a33a331c5d82703c4a2db1a600" CACHE STRING "wasi-webgpu-headers git ref")
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

file(GLOB LLAMA_PATCHES ${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch)
list(SORT LLAMA_PATCHES)
# Resets the checkout first, so a rerun applies the series to a clean tree.
FetchContent_Declare(llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG ${LLAMA_GIT_TAG}
    GIT_SHALLOW TRUE
    PATCH_COMMAND git checkout -q -- .
          COMMAND git apply ${LLAMA_PATCHES}
)
FetchContent_MakeAvailable(llama)

set(GEN ${CMAKE_CURRENT_SOURCE_DIR}/src/generated)

# Regenerate the committed C bindings from wit/. Run on demand after a WIT
# change. provider-with-json-schema's are also named `provider`, so inference.c
# builds against either.
add_custom_target(regenerate-bindings
    COMMAND wit-bindgen c ${CMAKE_CURRENT_SOURCE_DIR}/wit --world provider --out-dir ${GEN}
    COMMAND wit-bindgen c ${CMAKE_CURRENT_SOURCE_DIR}/wit --world provider-with-json-schema --rename-world provider --out-dir ${GEN}/provider-with-json-schema
    VERBATIM
)

if(LLAMA_WIT_JSON_SCHEMA)
    set(WORLD_GEN ${GEN}/provider-with-json-schema)
else()
    set(WORLD_GEN ${GEN})
endif()

add_executable(provider
    src/inference.c
    ${WORLD_GEN}/provider.c
    ${WORLD_GEN}/provider_component_type.o
)
set_target_properties(provider PROPERTIES OUTPUT_NAME llama-cpp.wasm)
target_include_directories(provider PRIVATE ${WORLD_GEN})
if(LLAMA_WIT_JSON_SCHEMA)
    target_sources(provider PRIVATE
        src/grammar.cpp
        # llama.cpp's JSON Schema to GBNF converter, without the rest of its common library
        ${llama_SOURCE_DIR}/common/json-schema-to-grammar.cpp
        ${llama_SOURCE_DIR}/common/json-schema.cpp
        ${llama_SOURCE_DIR}/common/json.cpp
        ${llama_SOURCE_DIR}/common/trie.cpp
        ${llama_SOURCE_DIR}/common/unicode.cpp
    )
    target_include_directories(provider PRIVATE ${llama_SOURCE_DIR}/common ${llama_SOURCE_DIR}/vendor)
endif()
target_link_options(provider PRIVATE -mexec-model=reactor)
target_link_libraries(provider PRIVATE llama)
