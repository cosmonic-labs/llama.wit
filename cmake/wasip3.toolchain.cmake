# ensure wasi-sdk
set(WASI_SDK_PATH $ENV{WASI_SDK_PATH})
if(NOT WASI_SDK_PATH OR NOT EXISTS "${WASI_SDK_PATH}/bin/clang")
    message(FATAL_ERROR "Set WASI_SDK_PATH to a valid wasi-sdk (need bin/clang). Got: ${WASI_SDK_PATH}")
endif()

# set wasi target
set(CMAKE_SYSTEM_NAME WASI)
set(CMAKE_SYSTEM_VERSION 1)
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

# compile + link flags
# llama.cpp uses mmap/clock/signal - each needs both the define below and the matching -l
set(_wasi_flags "-mthread-model single -msimd128 -D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_PROCESS_CLOCKS")
set(CMAKE_C_FLAGS_INIT   "${_wasi_flags}")
# -include new: ggml/src/ggml-cpu/ops.h reads std::hardware_destructive_interference_size
# behind a bare __cpp_lib_hardware_interference_size check, without including <new>. ggml's
# precompiled header pulls in <vector>, which defines the macro but not the constant.
set(CMAKE_CXX_FLAGS_INIT "${_wasi_flags} -fwasm-exceptions -include errno.h -include stdlib.h -include new")
# -ldl required for dlopen/dlsym/dlclose stubs
# stack-size: llama_decode overflows wasm-ld's 64KiB default (traps near address 0)
set(_wasi_link "-fwasm-exceptions -lunwind -ldl -Wl,-z,stack-size=8388608 -lwasi-emulated-signal -lwasi-emulated-mman -lwasi-emulated-process-clocks")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_wasi_link}")

# Skip FindThreads' probe (ggml calls find_package(Threads REQUIRED)): it references pthread_cancel/pthread_exit, which wasi-libc doesn't declare, so it won't compile.
# This makes FindThreads succeed with an empty-link Threads::Threads. The pthread stubs are in libc, and we run single-threaded anyway.
# TODO: drop once ggml stops requiring Threads on WASI.
set(CMAKE_HAVE_LIBC_PTHREAD 1)
