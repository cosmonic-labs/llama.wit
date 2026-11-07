# wasm32-wasip3 toolchain for llama.cpp. Builds this repo's provider component,
# and llama.cpp's own tools (llama-bench, llama-simple) as wasip3 commands.
# Requires WASI_SDK_PATH pointing at a wasi-sdk 34+ install.
set(WASI_SDK_PATH $ENV{WASI_SDK_PATH})
if(NOT WASI_SDK_PATH OR NOT EXISTS "${WASI_SDK_PATH}/bin/clang")
    message(FATAL_ERROR "Set WASI_SDK_PATH to a valid wasi-sdk (need bin/clang). Got: ${WASI_SDK_PATH}")
endif()

set(CMAKE_SYSTEM_NAME WASI)
set(CMAKE_SYSTEM_VERSION 1)
# Makes ggml-cpu take its "wasm" arch branch (arch/wasm/quants.c).
set(CMAKE_SYSTEM_PROCESSOR wasm32)

set(CMAKE_C_COMPILER   "${WASI_SDK_PATH}/bin/clang")
set(CMAKE_CXX_COMPILER "${WASI_SDK_PATH}/bin/clang++")
set(CMAKE_AR           "${WASI_SDK_PATH}/bin/llvm-ar")
set(CMAKE_RANLIB       "${WASI_SDK_PATH}/bin/llvm-ranlib")
set(CMAKE_C_COMPILER_TARGET   wasm32-wasip3)
set(CMAKE_CXX_COMPILER_TARGET wasm32-wasip3)

set(CMAKE_SYSROOT "${WASI_SDK_PATH}/share/wasi-sysroot")
set(CMAKE_FIND_ROOT_PATH "${WASI_SDK_PATH}/share/wasi-sysroot")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Headers wasi-libc lacks, for llama.cpp's common library and tools.
# -idirafter: real sysroot headers always win over these shims.
set(_wasi_shims "${CMAKE_CURRENT_LIST_DIR}/wasi-shims")
# -I: these deliberately shadow a sysroot header. See the files for why.
set(_wasi_shims_force "${CMAKE_CURRENT_LIST_DIR}/wasi-shims-force")

# Single-threaded model. _WASI_EMULATED_PROCESS_CLOCKS lets common.cpp include <sys/resource.h>.
set(_wasi_flags "-mthread-model single -msimd128 -D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_PROCESS_CLOCKS -I ${_wasi_shims_force} -idirafter ${_wasi_shims}")
set(CMAKE_C_FLAGS_INIT   "${_wasi_flags}")
# C++ exceptions via the wasm EH proposal. -wasm-use-legacy-eh=false selects the
# standardized exnref encoding (https://releases.llvm.org/20.1.0/docs/ReleaseNotes.html#changes-to-the-webassembly-backend).
# -include errno.h/stdlib.h: some llama.cpp sources use errno/getenv without including them.
set(CMAKE_CXX_FLAGS_INIT "${_wasi_flags} -fwasm-exceptions -mllvm -wasm-use-legacy-eh=false -include errno.h -include stdlib.h")
# -ldl required for dlopen/dlsym/dlclose stubs
# stack-size: llama_decode overflows wasm-ld's 64KiB default (traps near address 0)
set(_wasi_link "-fwasm-exceptions -lunwind -ldl -Wl,-z,stack-size=16777216 -lwasi-emulated-signal -lwasi-emulated-mman -lwasi-emulated-process-clocks")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_wasi_link}")

# Skip FindThreads' probe (ggml calls find_package(Threads REQUIRED)): it references pthread_cancel/pthread_exit, which wasi-libc doesn't declare, so it won't compile.
# This makes FindThreads succeed with an empty-link Threads::Threads. The pthread stubs are in libc, and we run single-threaded anyway.
# TODO: drop once ggml stops requiring Threads on WASI.
set(CMAKE_HAVE_LIBC_PTHREAD 1)
