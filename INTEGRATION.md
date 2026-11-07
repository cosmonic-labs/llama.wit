# integration/control-demo

The branch Control builds its LLM demo from until the pieces below land on
`main` and upstream. It is `main` plus the branches listed here, merged and
rebuilt into one llama.cpp patch series for b12135. Don't merge it to `main`:
land each piece through its own PR, then drop it here.

| piece | source | head merged |
|---|---|---|
| shared patch series, llama.cpp b12135 | #7 `build/shared-llama-patches` | add56fb |
| large-model support (`model.create` over `stream<u8>`) | `large-models` | df4b8e5 |

`large-models` targets b11514. Here its stream-ordered loading patch is ported
to b12135 as `patches/0008`, its four ggml-webgpu patches are `0009`-`0012`
unchanged, and its 64-bit offset and wasm32 buffer-size patches are dropped
for the series' `0004` and `0006`.

## API

`wit/llama.wit` is `cosmonic:llama-cpp@0.8.0-rc.N`: main's 0.7.0 with
`large-models`' change to `model.create`, which takes
`data: stream<u8>, size: u64` and returns the load's progress as a
`stream<f32>` beside a future for the model, and with `memory-clear` and the
embedding getters async, since they wait on the GPU. The prerelease suffix
keeps it off every version main will release.

## Publishing

Publish the WIT package first, by hand, so consumers fetch both interfaces
with wkg instead of copying `wit/`:

```shell
export WKG_CONFIG_FILE=wkg-config.toml
wkg build -o llama-cpp.wasm
wkg publish llama-cpp.wasm
```

It lands at `ghcr.io/cosmonic-labs/cosmonic/llama-cpp:<version>`, where wkg
resolves `cosmonic:llama-cpp`. Then push here: CI builds both worlds and
publishes `ghcr.io/cosmonic-labs/cosmonic/llama-cpp-with-json-schema:<version>`
with a provenance attestation. It leaves the `llama-cpp` tag to the WIT
package rather than pushing the api-only component over it. CI refuses any
version without a `-rc.N` suffix. A published version never moves, so bump `N`
in `wit/llama.wit`, regenerate the bindings, and commit them together to
publish a change.

## Using it from Control

Control's `llama-inference` imports `api` and `json-schema` at this version,
fetches the WIT package with wkg, and pulls `llama-cpp-with-json-schema` at the
same version as its engine.
