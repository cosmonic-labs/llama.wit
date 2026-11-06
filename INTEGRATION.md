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

`wit/llama.wit` is `cosmonic:llama-cpp@0.7.0` with `large-models`' change to
`model.create`, which now takes `data: stream<u8>, size: u64`. That is a
breaking change still numbered 0.7.0, and it needs a version bump before it
reaches `main`. Control's `llama-inference` imports `cosmonic:llama-cpp/api@0.6.0`,
so it has to move to this API before it can use this build.

## Using it from Control

CI doesn't publish from this branch. Dispatch `ci.yml` on it, take the
`llama-cpp` artifact (or run `wash build` here), and put the component where
Control's build looks first:

```sh
cp build/llama-cpp.wasm "$CONTROL/components/rust/build/llama-cpp.wasm"
make -C "$CONTROL" build-llm-app
```
