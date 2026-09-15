# A stand-in for Dawn, so ggml-webgpu can be built for wasi without patching it.
#
# ggml/src/ggml-webgpu/CMakeLists.txt has no WASI branch: for any non-Emscripten
# target it does
#
#     find_package(Dawn REQUIRED)
#     set(DawnWebGPU_TARGET dawn::webgpu_dawn)
#     ...
#     target_link_libraries(ggml-webgpu PRIVATE ${DawnWebGPU_TARGET})
#
# Pointing Dawn_DIR at this directory turns that into the wiring we actually
# want. Everything rides in on the usage requirements of one INTERFACE IMPORTED
# target, so no part of llama.cpp's build files has to be edited:
#
#   INTERFACE_SOURCES          become sources of ggml-webgpu itself
#   INTERFACE_INCLUDE_DIRECTORIES  put Dawn's upstream-flavored webgpu_cpp.h and
#                              wasi-webgpu-headers' webgpu.h on its include path
#
# No Dawn runtime is linked. The name is Dawn only because that is the name
# ggml's find_package() asks for.

if(NOT CMAKE_SYSTEM_NAME STREQUAL "WASI")
    message(FATAL_ERROR
        "The wasi:webgpu Dawn shim was selected for a non-WASI build "
        "(CMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}). Unset Dawn_DIR to use a real Dawn.")
endif()

foreach(_var WASI_WEBGPU_HEADERS_DIR DAWN_HEADERS_DIR)
    if(NOT ${_var})
        message(FATAL_ERROR "wasi:webgpu build needs ${_var}")
    endif()
endforeach()

add_library(dawn::webgpu_dawn INTERFACE IMPORTED)

# imports_component_type.o carries the wasi:webgpu import types, so linking it
# is what makes the final binary componentizable.
set_target_properties(dawn::webgpu_dawn PROPERTIES
    INTERFACE_SOURCES
        "${WASI_WEBGPU_HEADERS_DIR}/webgpu.c;${WASI_WEBGPU_HEADERS_DIR}/async_futures.c;${WASI_WEBGPU_HEADERS_DIR}/imports.c;${WASI_WEBGPU_HEADERS_DIR}/imports_component_type.o"
    # Dawn's upstream-flavored webgpu_cpp.h first; webgpu.h comes from
    # wasi-webgpu-headers' webgpu/ submodule at the checkout root.
    INTERFACE_INCLUDE_DIRECTORIES
        "${DAWN_HEADERS_DIR}/include/webgpu_upstream;${WASI_WEBGPU_HEADERS_DIR}"
)
