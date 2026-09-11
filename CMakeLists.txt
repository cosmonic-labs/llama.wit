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

# Optional wasi:webgpu GPU backend. When ON, patch upstream llama.cpp with the
# WASI branch of ggml-webgpu (see patches/0001-ggml-webgpu-wasi.patch, a verbatim
# copy of the fork's "with WebGPU" commit) and build ggml-webgpu against the
# wasi-webgpu-headers + dawn_wasi_webgpu_cpp glue. Off by default -> plain CPU build.
option(LLAMA_WIT_WEBGPU "Build with the wasi:webgpu GPU backend" OFF)

# The webgpu glue is fetched from GitHub by default. Both repos commit their
# generated bindings (imports*.c, imports_component_type.o, webgpu_cpp.h), so a
# shallow clone is self-contained. To build against a local checkout instead,
# point these at it (e.g. -DWASI_WEBGPU_HEADERS_DIR=/path/to/checkout) and the
# matching fetch is skipped.
set(WASI_WEBGPU_HEADERS_DIR "" CACHE PATH "Local wasi-webgpu-headers checkout (empty = fetch)")
set(DAWN_WASI_WEBGPU_DIR    "" CACHE PATH "Local dawn_wasi_webgpu_cpp checkout (empty = fetch)")
set(WASI_WEBGPU_HEADERS_REPOSITORY "https://github.com/MendyBerger/wasi-webgpu-headers.git" CACHE STRING "wasi-webgpu-headers git remote")
set(WASI_WEBGPU_HEADERS_TAG        "wasm-cg-demo" CACHE STRING "wasi-webgpu-headers git ref")
set(DAWN_WASI_WEBGPU_REPOSITORY    "https://github.com/MendyBerger/dawn_wasi_webgpu_cpp.git" CACHE STRING "dawn_wasi_webgpu_cpp git remote")
set(DAWN_WASI_WEBGPU_TAG           "wasm-cg-demo" CACHE STRING "dawn_wasi_webgpu_cpp git ref")

set(_llama_patch "")
if(LLAMA_WIT_WEBGPU)
    set(GGML_WEBGPU ON CACHE BOOL "" FORCE)
    # ggml-webgpu + the wasi-webgpu-headers glue pull mmap/getpid/process-clocks,
    # which WASI lacks; use wasi-libc's emulation (matches the fork's wasi toolchain).
    # Applied before MakeAvailable so the ggml subdir targets inherit them.
    add_compile_definitions(_WASI_EMULATED_MMAN _WASI_EMULATED_GETPID _WASI_EMULATED_PROCESS_CLOCKS)
    add_link_options(-lwasi-emulated-mman -lwasi-emulated-getpid -lwasi-emulated-process-clocks)

    # Fetch the webgpu glue from GitHub unless a local checkout was supplied.
    # SOURCE_SUBDIR points at a nonexistent dir so MakeAvailable only populates the
    # sources (we consume files directly) and never add_subdirectory's their CMake.
    if(NOT WASI_WEBGPU_HEADERS_DIR)
        FetchContent_Declare(wasi_webgpu_headers
            GIT_REPOSITORY ${WASI_WEBGPU_HEADERS_REPOSITORY}
            GIT_TAG        ${WASI_WEBGPU_HEADERS_TAG}
            GIT_SHALLOW TRUE
            SOURCE_SUBDIR  _skip_configure_)
        FetchContent_MakeAvailable(wasi_webgpu_headers)
        set(WASI_WEBGPU_HEADERS_DIR ${wasi_webgpu_headers_SOURCE_DIR} CACHE PATH "" FORCE)
    endif()
    if(NOT DAWN_WASI_WEBGPU_DIR)
        FetchContent_Declare(dawn_wasi_webgpu_cpp
            GIT_REPOSITORY ${DAWN_WASI_WEBGPU_REPOSITORY}
            GIT_TAG        ${DAWN_WASI_WEBGPU_TAG}
            GIT_SHALLOW TRUE
            SOURCE_SUBDIR  _skip_configure_)
        FetchContent_MakeAvailable(dawn_wasi_webgpu_cpp)
        set(DAWN_WASI_WEBGPU_DIR ${dawn_wasi_webgpu_cpp_SOURCE_DIR} CACHE PATH "" FORCE)
    endif()

    # Apply the WASI ggml-webgpu overlay idempotently: skip if already applied,
    # else apply (fuzz-tolerant, since the pinned tag may drift from the base the
    # patch was cut against). Runs in the populated source dir.
    set(_patch ${CMAKE_CURRENT_SOURCE_DIR}/patches/0001-ggml-webgpu-wasi.patch)
    set(_llama_patch PATCH_COMMAND sh -c
        "git apply --reverse --check '${_patch}' 2>/dev/null || patch -p1 --forward --fuzz=3 < '${_patch}'")
endif()

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
