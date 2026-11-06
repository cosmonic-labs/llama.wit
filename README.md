# llama.wit: a wit wrapper around llama.cpp

Run llama.cpp from any WebAssembly component. llama.wit aims to expose llama.cpp's public API - llama.h - as WIT.

There's also an optional JSON Schema to GBNF helper, since it's so useful for structured output (see below).

## Compile

1. point to your local wasi-sdk >= v34
```shell
export WASI_SDK_PATH=/path/to/wasi-sdk
```

2. Compile
```shell
wash build
```

This runs the CMake configure + build from [`.wash/config.yaml`](.wash/config.yaml):
```shell
cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/wasip3.toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Output at `build/llama-cpp.wasm`

## JSON Schema converter (optional)

By default the component implements the `provider` world, which exports only `api`: a
wrapper around `llama.h`. `-DLLAMA_WIT_JSON_SCHEMA=ON` (or `LLAMA_WIT_JSON_SCHEMA=ON wash build`)
builds the `provider-with-json-schema` world instead, which also exports `json-schema`:
llama.cpp's own JSON Schema to GBNF converter, from its common library, for
`sampler.init-grammar`.

## WebGPU backend

The component always builds llama.cpp's `ggml-webgpu` backend, so it imports
`wasi:webgpu/webgpu@0.3.0-rc.2` and the host must provide it. Set `n-gpu-layers > 0` in
`model-params` to offload layers to the GPU. The test harness provides it with Wasmtime.

Instead of Dawn, the backend links [wasi-webgpu-headers](https://github.com/wasi-gfx/wasi-webgpu-headers),
which implements `webgpu.h` over `wasi:webgpu`, plus the headers from a Dawn release. Both are
fetched and pinned in [CMakeLists.txt](CMakeLists.txt) (`WASI_WEBGPU_HEADERS_TAG`, `DAWN_HEADERS_URL`).
To build against a local checkout instead, use CMake's
`-DFETCHCONTENT_SOURCE_DIR_WASI_WEBGPU_HEADERS=/path/...`.

## llama.cpp patches

The build fetches llama.cpp at `LLAMA_GIT_TAG` and applies [`patches/`](patches) in order.
They are what llama.cpp needs to run as a wasm32 component, and are kept here so every
consumer builds the same source:

| patch | what it does |
|---|---|
| 0001 | `GGML_WEBGPU_PORTABLE` keeps a native ggml-webgpu build off Dawn-only features, so it runs the same shaders as the wasm build |
| 0002, 0003 | llama.cpp's common library builds on WASI: no process priority, no file locks |
| 0004 | 64-bit GGUF file offsets on WASI, so a model past 4 GiB loads in a wasm32 guest |
| 0005 | model reads in 64 MiB chunks on WASI: a component-model stream read traps at 256 MiB |
| 0006 | ggml-webgpu's buffer size limit is capped at `SIZE_MAX/2` on 32-bit targets, so weight buffers split below 4 GiB |
| 0007 | ggml-webgpu lowercases the adapter vendor, so its dp4a path also turns on for "NVIDIA" from wgpu-native |
| 0008 | the loader visits tensors in file order across every context, so a stream-backed `FILE *` only seeks forward; a tied output is copied from `token_embd` instead of read twice |

The patches apply to exactly that tag, so move them together. With
`-DFETCHCONTENT_SOURCE_DIR_LLAMA=/path/...` the local checkout is used as it is, unpatched.

[`cmake/wasip3.toolchain.cmake`](cmake/wasip3.toolchain.cmake) also builds llama.cpp's own
tools (`llama-bench`, `llama-simple`) for wasm32-wasip3, with the headers wasi-libc lacks in
`cmake/wasi-shims*`.

## Large models

`model.create` takes the GGUF as a `stream<u8>` plus its size, and reads it in order, so
the file is never held in linear memory whole.
Models over 4 GiB load, as long as the weights that stay on the CPU
fit in wasm32's 4 GiB of linear memory. The llama.cpp changes this needs are in
[`patches/`](patches), applied to the fetched checkout and listed above.

Fetched dependencies are unpacked into `deps/` and shared by every build directory.

## Generate Bindings
```shell
cmake --build build --target regenerate-bindings
```

This regenerates `provider` into `src/generated/` and `provider-with-json-schema` into
`src/generated/provider-with-json-schema/`. Both are named `provider`, so `src/inference.c`
builds against either.

CI regenerates the bindings with wit-bindgen 0.59.0 (`WIT_BINDGEN_VERSION` in
[the workflow](.github/workflows/ci.yml)) and fails if they differ from what is committed,
so regenerate with that version.

## Test

Install the `wasm32-wasip2` Rust target and `wac`, then build the component before
running the harness. The host needs a WebGPU adapter.

```shell
rustup target add wasm32-wasip2
./tests/run.sh
```

The default suite uses a 1.1 MiB model to check token bytes, chat templates, context limits,
logit bias, grammar constraints, embeddings, and prompt reuse against fresh inference.
When the component was built with `LLAMA_WIT_JSON_SCHEMA=ON`, it also tests the converter.
Models are cached in `tests/guest-dir/models`.

```shell
./tests/run.sh --live
./tests/run.sh --live --filter prompt-reuse
```

The live suite also downloads Qwen 0.5B, Llama 1B, and BGE small. It checks repeated
GPU model swaps, larger-model prompt reuse, and embedding dimensions and similarity.
CI compiles the guest and tests the runner's failure reporting; GPU tests run locally.

## Releases

[CI](.github/workflows/ci.yml) builds the WebGPU component for both worlds on every pull
request and push to `main`. It checks that the committed bindings match `wit/`, that each
component exports the interfaces its world declares, and that it imports `wasi:webgpu`. The
package version in [`wit/llama.wit`](wit/llama.wit) is the release version: a merge to `main`
that bumps it publishes, each with a signed build-provenance attestation:
- `provider` to `ghcr.io/cosmonic-labs/cosmonic/llama-cpp`
- `provider-with-json-schema` to `ghcr.io/cosmonic-labs/cosmonic/llama-cpp-with-json-schema`

To release, bump `package cosmonic:llama-cpp@<version>;`,
[regenerate the bindings](#generate-bindings) and merge. Merges that leave the version alone
still build but publish nothing, and a published version is never overwritten.

```shell
wash oci pull ghcr.io/cosmonic-labs/cosmonic/llama-cpp:<version> llama-cpp.wasm
gh attestation verify oci://ghcr.io/cosmonic-labs/cosmonic/llama-cpp:<version> --owner cosmonic-labs
```
