# llama.wit: a wit wrapper around llama.cpp

## Compile

1. point to your local wasi-sdk >= v34
```shell
export WASI_SDK_PATH=/path/to/wasi-sdk
```

2. Compile
```shell
cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/wasip3.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DLLAMA_WIT_WEBGPU=ON
cmake --build build
```

Output at `build/llama-cpp.wasm`

Dependencies are fetched into `deps/` (gitignored), which is shared across build
directories. Override with `-DFETCHCONTENT_BASE_DIR`.

## WebGPU backend

`-DLLAMA_WIT_WEBGPU=ON` builds the GPU backend against `wasi:webgpu` instead of
Dawn: `wasi-webgpu-headers` implements `webgpu.h` over wit-bindgen'd imports, and
the C++ wrapper is the upstream-flavored `webgpu_cpp.h` from a `dawn-headers`
release ([dawn#80](https://github.com/google/dawn/pull/80)) — no Dawn runtime is
linked.

Output at `build-webgpu/llama-cpp.wasm`, with the `wasi:webgpu` imports in place.

ggml-webgpu's own build files are not modified. Off Emscripten they just call
`find_package(Dawn REQUIRED)` and link `dawn::webgpu_dawn`, so we answer that
with `cmake/wasi-dawn-shim/DawnConfig.cmake`, which defines that target as the
wasi:webgpu glue instead. The only thing patched in llama.cpp is the source-level
`__wasi__` guards in `patches/0001`, which llama.cpp#27069 upstreams verbatim.

This currently pins the `p3-update-webgpu.h` branch of a `wasi-webgpu-headers`
fork; switch back to `wasi-gfx` once that lands upstream.

## Generate Bindings
```shell
cmake --build build --target regenerate-bindings
```

CI regenerates the bindings with wit-bindgen 0.59.0 (`WIT_BINDGEN_VERSION` in
[the workflow](.github/workflows/ci.yml)) and fails if they differ from what is committed,
so regenerate with that version.

## Test
```shell
./tests/run.sh
```

## Releases

[CI](.github/workflows/ci.yml) builds the WebGPU component on every pull request and push to
`main`. It checks that the committed bindings match `wit/`, that the component exports the
interface declared there, and that it imports `wasi:webgpu`. The package version in
[`wit/llama.wit`](wit/llama.wit) is the release version: a merge to `main` that bumps it
publishes the component to `ghcr.io/cosmonic-labs/cosmonic/llama-cpp` with a signed
build-provenance attestation.

To release, bump `package cosmonic:llama-cpp@<version>;`,
[regenerate the bindings](#generate-bindings) and merge. Merges that leave the version alone
still build but publish nothing, and a published version is never overwritten.

```shell
wash oci pull ghcr.io/cosmonic-labs/cosmonic/llama-cpp:<version> llama-cpp.wasm
gh attestation verify oci://ghcr.io/cosmonic-labs/cosmonic/llama-cpp:<version> --owner cosmonic-labs
```
