#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GUEST=./target/wasm32-wasip2/release/guest.wasm
PROVIDER=../build/llama-cpp.wasm
HARNESS=./target/harness.wasm

# Test the converter too when the provider was built with LLAMA_WIT_JSON_SCHEMA.
# The ${FEATURES[@]+...} form below keeps an empty array working under set -u on bash < 4.4.
FEATURES=()
# Captured first: under pipefail, grep -q exiting early can fail wasm-tools with EPIPE.
WIT=$(wasm-tools component wit "$PROVIDER")
if grep -q 'export cosmonic:llama-cpp/json-schema@' <<<"$WIT"; then
  FEATURES=(--features json-schema)
fi

cargo build --manifest-path ./guest/Cargo.toml --target wasm32-wasip2 --release --locked ${FEATURES[@]+"${FEATURES[@]}"}

wac plug "$GUEST" --plug "$PROVIDER" -o "$HARNESS"

cargo run --manifest-path ./runtime/Cargo.toml --release --locked -- --path "$HARNESS" "$@"
