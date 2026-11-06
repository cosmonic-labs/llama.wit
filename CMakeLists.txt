cmake_minimum_required(VERSION 3.24)
project(llama-cpp-wit C CXX)

# Fetched dependencies live in deps/ (gitignored) instead of inside each build
# tree, so the build-* dirs share one checkout. Override with -DFETCHCONTENT_BASE_DIR.
set(FETCHCONTENT_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps CACHE PATH "Where fetched dependencies are unpacked")

include(FetchContent)

# https://releases.llvm.org/20.1.0/docs/ReleaseNotes.html#changes-to-the-webassembly-backend
# > this option is turned on by default for the moment
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-mllvm;-wasm-use-legacy-eh=false>")

set(LLAMA_GIT_TAG b11514 CACHE STRING "llama.cpp release tag to build")
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

# Patches applied to the fetched llama.cpp tree, in order, so models larger
# than linear memory, and over 4 GiB, load:
# - 0002: visit tensors in file-offset order so a forward-only stream-backed
#   FILE* never needs to seek backwards. A tied output, which shares token_embd's
#   bytes, is copied from it rather than read twice.
# - 0003: ggml-webgpu advertises async uploads + events + a host buffer type on
#   wasi, so the loader stages GPU-bound tensors through 4 x 1 MiB buffers
#   instead of a host copy of each whole tensor.
# - 0004: file offsets and sizes as uint64_t instead of size_t (llama_file,
#   llama_tensor_weight, gguf's data/tensor offset getters). size_t is 32-bit
#   on wasm32, so without this any GGUF file over 4 GiB fails to load.
# - 0005: ggml-webgpu caps the buffer-type max size on wasm32. The adapter's
#   maxStorageBufferBindingSize narrows to nearly SIZE_MAX in a 32-bit size_t,
#   which overflows the buffer-splitting arithmetic in ggml-alloc.c, so any
#   model whose GPU-resident weights exceed 4 GiB aborts while laying out
#   tensors. See the patch for the full accounting.
# - 0006: ggml-webgpu stages op params on the CPU and uploads them once per
#   submit batch instead of one queue.write-buffer per dispatch (~766 hostcalls
#   per token on a 40-layer model), and raises that batch from 64 to 256.
# - 0007: ggml-webgpu caches bind groups and bind group layouts, which were
#   rebuilt for every dispatch (two more hostcalls per node). Invalidated when
#   any buffer is freed.
# - 0008: ggml-webgpu flushes queued writes during loading. queue.WriteBuffer
#   stages its source until the next submit, and a model load submits nothing,
#   so the weights are held twice and peak memory measures ~2.15x the model --
#   enough to put a 13 GiB model over a 32 GiB machine.
set(_patches
    ${CMAKE_CURRENT_SOURCE_DIR}/patches/0002-stream-ordered-tensor-loading.patch
    ${CMAKE_CURRENT_SOURCE_DIR}/patches/0003-ggml-webgpu-async-uploads.patch
    ${CMAKE_CURRENT_SOURCE_DIR}/patches/0004-64-bit-file-offsets.patch
    ${CMAKE_CURRENT_SOURCE_DIR}/patches/0005-webgpu-wasm32-max-buffer-size.patch
    ${CMAKE_CURRENT_SOURCE_DIR}/patches/0006-webgpu-batch-param-uploads.patch
    ${CMAKE_CURRENT_SOURCE_DIR}/patches/0007-webgpu-cache-bind-groups.patch
    ${CMAKE_CURRENT_SOURCE_DIR}/patches/0008-webgpu-flush-pending-writes.patch)
# Idempotent: detection uses the same tool and fuzz as application. (A patch that
# landed with fuzz is not reverse-detectable by strict `git apply`, and
# `patch --forward` then exits 1 on "previously applied", failing populate.)
# Each patch's hash rides along in the command: FetchContent re-runs the patch
# step only when the command changes, so without it editing a patch file would
# never reach an already-populated tree.
# NOTE: no literal ';' in the command — CMake would split it into list items.
set(_apply "true")
foreach(_patch IN LISTS _patches)
    file(SHA256 ${_patch} _patch_hash)
    string(APPEND _apply " && (patch -p1 --reverse --dry-run --force --fuzz=3 < '${_patch}' >/dev/null 2>&1 || patch -p1 --forward --fuzz=3 < '${_patch}') && true ${_patch_hash}")
endforeach()
# ...and re-run configure itself when a patch file changes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_patches})

# llama is the one dependency we add_subdirectory, so its build tree must stay
# per-build-dir; only the checkout is shared via deps/.
FetchContent_Declare(llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG ${LLAMA_GIT_TAG}
    GIT_SHALLOW TRUE
    BINARY_DIR ${CMAKE_BINARY_DIR}/llama-build
    PATCH_COMMAND sh -c "${_apply}"
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
