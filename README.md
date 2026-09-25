# llama.wit: a wit wrapper around llama.cpp

## Compile

1. point to your local wasi-sdk >= v34
```shell
export WASI_SDK_PATH=/path/to/wasi-sdk
```

2. Compile
```shell
wash build
```

This builds the [WebGPU](#webgpu-backend-optional-experimental) variant, which is what CI
publishes, by running the CMake configure + build from [`.wash/config.yaml`](.wash/config.yaml):
```shell
cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/wasip3.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DLLAMA_WIT_WEBGPU=ON
cmake --build build
```

For a CPU-only build, run those commands without `-DLLAMA_WIT_WEBGPU=ON` (in a fresh build
directory: the WebGPU patch is applied only when llama.cpp is first fetched).

Output at `build/llama-cpp.wasm`

## WebGPU backend (optional, experimental)

Builds the `ggml-webgpu` backend for WASI so the component offloads to the host's
`wasi:webgpu` implementation (set `n-gpu-layers > 0` in `model-params`). This
replicates the fork's "with WebGPU" commit as an overlay patch on the pinned
llama.cpp release — see [patches/0001-ggml-webgpu-wasi.patch](patches/0001-ggml-webgpu-wasi.patch).

The two webgpu deps are fetched from GitHub automatically, pinned to commits on their
`wasm-cg-demo` branches (`WASI_WEBGPU_HEADERS_TAG` / `DAWN_WASI_WEBGPU_TAG` in
[CMakeLists.txt](CMakeLists.txt)), so no extra flags are needed:
- [wasi-webgpu-headers](https://github.com/MendyBerger/wasi-webgpu-headers/tree/wasm-cg-demo) — `webgpu.h` over `wasi:webgpu@0.3.0-rc.2`
- [dawn_wasi_webgpu_cpp](https://github.com/MendyBerger/dawn_wasi_webgpu_cpp/tree/wasm-cg-demo) — Dawn's `webgpu_cpp.h` wrapper for WASI

```shell
cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/wasip3.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DLLAMA_WIT_WEBGPU=ON
cmake --build build
```

To build against local checkouts instead of fetching, point at them (this skips
the fetch): `-DWASI_WEBGPU_HEADERS_DIR=/path/... -DDAWN_WASI_WEBGPU_DIR=/path/...`.
Override the fetched refs with `-DWASI_WEBGPU_HEADERS_TAG=...` / `-DDAWN_WASI_WEBGPU_TAG=...`.

The resulting `llama-cpp.wasm` additionally imports `wasi:webgpu/webgpu@0.3.0-rc.2`,
which the host must provide. This path is unproven
end-to-end on wasip3; expect to shake out linker/componentization issues (e.g.
duplicate `cabi_realloc` across the two `component-type` objects). If the patch
ever fails to apply against a newer `LLAMA_GIT_TAG`, pin `-DLLAMA_GIT_TAG=b9886`
(the tag the WebGPU work was based on) or refresh the patch.

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
