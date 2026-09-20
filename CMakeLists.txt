cmake_minimum_required(VERSION 3.21)
project(llama-cpp-wit C CXX)

# Fetched dependencies live in deps/ (gitignored) instead of inside each build
# tree, so the build-* dirs share one checkout. Override with -DFETCHCONTENT_BASE_DIR.
set(FETCHCONTENT_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps CACHE PATH "Where fetched dependencies are unpacked")

include(FetchContent)

# https://releases.llvm.org/20.1.0/docs/ReleaseNotes.html#changes-to-the-webassembly-backend
# > this option is turned on by default for the moment
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-mllvm;-wasm-use-legacy-eh=false>")

# Upstream llama.cpp, pinned to a release (https://github.com/ggml-org/llama.cpp/releases).
set(LLAMA_GIT_TAG b10951 CACHE STRING "llama.cpp release tag to build")

# Trim llama.cpp to just the static CPU libs we link.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE       OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP       OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL        OFF CACHE BOOL "" FORCE)

# Optional wasi:webgpu GPU backend. The build-system side is done from out here
# (see the Dawn_DIR shim below), so the only thing patched into ggml for this is
# patches/0001 — source-level #ifdefs, which
# https://github.com/ggml-org/llama.cpp/pull/27069 upstreams verbatim.
option(LLAMA_WIT_WEBGPU "Build with the wasi:webgpu GPU backend" OFF)

# The webgpu glue, in two pinned pieces:
#   wasi_webgpu_headers  webgpu.h implemented over wit-bindgen'd wasi:webgpu imports
#                        (commits its bindings, so a shallow clone is self-contained).
#                        Its webgpu/ submodule supplies upstream webgpu.h, the C API
#                        that both it and Dawn's wrapper compile against.
#   dawn_headers         Dawn's prebuilt release headers, for the generated
#                        upstream-flavored webgpu_cpp.h in include/webgpu_upstream
#                        (https://github.com/google/dawn/pull/80 — no Dawn runtime
#                        is linked, only that header tree is used)
# To build against a local checkout instead, point FetchContent at it, e.g.
# -DFETCHCONTENT_SOURCE_DIR_WASI_WEBGPU_HEADERS=/path/to/checkout.
# TODO: back to wasi-gfx/wasi-webgpu-headers once the p3-update-webgpu.h branch
# is opened as a PR and merged. It implements the wgpu* entry points ggml needs
# (AdapterGetInfo, QueueOnSubmittedWorkDone, InstanceHasWGSLLanguageFeature,
# SurfaceRelease) and carries webgpu.h at upstream main.
set(WASI_WEBGPU_HEADERS_REPOSITORY "https://github.com/MendyBerger/wasi-webgpu-headers.git" CACHE STRING "wasi-webgpu-headers git remote")
set(WASI_WEBGPU_HEADERS_TAG "ef983a33573d7e6170cf8ae3714cc2956de5f636" CACHE STRING "wasi-webgpu-headers git ref (branch p3-update-webgpu.h)")
set(DAWN_HEADERS_URL "https://github.com/google/dawn/releases/download/v20260825.221656/dawn-headers-6de3790c93dab8baac0555db186420adacc395be.tar.gz"
    CACHE STRING "dawn-headers release tarball")
set(DAWN_HEADERS_SHA256 "929f0548ff6a0d728d4a39e036139694211671a38d838f9f2d7e22181a6f98f2"
    CACHE STRING "SHA256 of DAWN_HEADERS_URL")

if(LLAMA_WIT_WEBGPU)
    set(GGML_WEBGPU ON CACHE BOOL "" FORCE)

    # SOURCE_SUBDIR points at a nonexistent dir so MakeAvailable only populates
    # the sources (we consume files directly) and never add_subdirectory's them.
    # The webgpu/ submodule is wanted: it puts webgpu.h where <webgpu/webgpu.h>
    # resolves against the checkout root, which is what webgpu.c and Dawn's
    # webgpu_cpp.h both include.
    FetchContent_Declare(wasi_webgpu_headers
        GIT_REPOSITORY ${WASI_WEBGPU_HEADERS_REPOSITORY}
        GIT_TAG        ${WASI_WEBGPU_HEADERS_TAG}
        SOURCE_SUBDIR  _skip_configure_)

    FetchContent_Declare(dawn_headers
        URL      ${DAWN_HEADERS_URL}
        URL_HASH SHA256=${DAWN_HEADERS_SHA256}
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE)

    FetchContent_MakeAvailable(wasi_webgpu_headers dawn_headers)

    # Read by cmake/wasi-dawn-shim/DawnConfig.cmake.
    set(WASI_WEBGPU_HEADERS_DIR ${wasi_webgpu_headers_SOURCE_DIR} CACHE PATH "" FORCE)
    set(DAWN_HEADERS_DIR        ${dawn_headers_SOURCE_DIR} CACHE PATH "" FORCE)

    # ggml-webgpu's CMakeLists has no WASI branch — off Emscripten it just does
    # find_package(Dawn REQUIRED) and links dawn::webgpu_dawn. Rather than patch
    # that, we answer the find_package with a shim config that defines
    # dawn::webgpu_dawn as the wasi:webgpu glue. See that file for the details.
    set(Dawn_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake/wasi-dawn-shim CACHE PATH "" FORCE)
endif()

# Patches applied to the fetched llama.cpp tree, idempotently. Detection uses the
# same tool and fuzz as application: a patch that once landed with fuzz is not
# reverse-detectable by `git apply` (strict), and `patch --forward` then exits 1
# on "previously applied" — which would fail the populate step forever.
# - 0002 (always): visit tensors in file-offset order so a forward-only
#   stream-backed FILE* never needs to seek backwards.
# - 0001 (webgpu builds): __wasi__ guards on Dawn-native-only features.
set(_patches ${CMAKE_CURRENT_SOURCE_DIR}/patches/0002-stream-ordered-tensor-loading.patch)
if(LLAMA_WIT_WEBGPU)
    list(PREPEND _patches ${CMAKE_CURRENT_SOURCE_DIR}/patches/0001-ggml-webgpu-wasi.patch)
endif()
# NOTE: no literal ';' in the command — CMake would split it into list items.
set(_apply "true")
foreach(_patch IN LISTS _patches)
    string(APPEND _apply " && (patch -p1 --reverse --dry-run --force --fuzz=3 < '${_patch}' >/dev/null 2>&1 || patch -p1 --forward --fuzz=3 < '${_patch}')")
endforeach()
set(_llama_patch PATCH_COMMAND sh -c "${_apply}")

# llama is the one dependency we add_subdirectory, so its build tree must stay
# per-build-dir; only the checkout is shared via deps/.
FetchContent_Declare(llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG ${LLAMA_GIT_TAG}
    GIT_SHALLOW TRUE
    BINARY_DIR ${CMAKE_BINARY_DIR}/llama-build
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
