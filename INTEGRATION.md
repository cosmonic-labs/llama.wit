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

`wit/llama.wit` is `cosmonic:llama-cpp@0.8.0-control-demo.N`: main's 0.7.0
with `large-models`' change to `model.create`, which takes
`data: stream<u8>, size: u64`. The prerelease suffix keeps it off every
version main will release.

## Publishing

A push here publishes both worlds to GHCR, like a merge to main, as
`ghcr.io/cosmonic-labs/cosmonic/llama-cpp:<version>` and
`.../llama-cpp-with-json-schema:<version>`. CI refuses any version without a
`-control-demo.N` suffix. A published version never moves, so bump `N` in
`wit/llama.wit`, regenerate the bindings, and commit them together to publish
a change.

## Using it from Control

Control's `llama-inference` imports `api` and `json-schema` at this version and
pins `LLM_LLAMA_ENGINE_VERSION` to it. Its Makefile pulls
`llama-cpp-with-json-schema`.
