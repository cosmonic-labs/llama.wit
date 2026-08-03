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
set(_wasi_flags "-mthread-model single -msimd128 -D_WASI_EMULATED_SIGNAL")
set(CMAKE_C_FLAGS_INIT   "${_wasi_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_wasi_flags} -fwasm-exceptions -mllvm -wasm-use-legacy-eh=false -include errno.h -include stdlib.h")
# -ldl required for dlopen/dlsym/dlclose stubs
# stack-size: wasm-ld's default shadow stack is 64KiB, which llama.cpp (deep
# C++ call chains, e.g. the webgpu backend's completion callbacks) overflows —
# overflow traps as "index/memory access out of bounds" near address 0. Match
# native llama.cpp's 8MiB thread stacks.
set(_wasi_link "-fwasm-exceptions -lunwind -ldl -Wl,-z,stack-size=8388608")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_wasi_link}")

# Skip FindThreads' probe (ggml calls find_package(Threads REQUIRED)): it references pthread_cancel/pthread_exit, which wasi-libc doesn't declare, so it won't compile.
# This makes FindThreads succeed with an empty-link Threads::Threads. The pthread stubs are in libc, and we run single-threaded anyway.
# TODO: drop once ggml stops requiring Threads on WASI.
set(CMAKE_HAVE_LIBC_PTHREAD 1)
