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
