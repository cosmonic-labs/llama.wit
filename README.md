# llama.wit: a wit wrapper around llama.cpp

## Compile

1. point to your local wasi-sdk >= v34
```shell
export WASI_SDK_PATH=/path/to/wasi-sdk
```

2. Compile
```shell
cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/wasip3.toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Output at `build/llama.wasm`

## Generate Bindings
```shell
cmake --build build --target regenerate-bindings
```
