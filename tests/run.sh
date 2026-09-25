#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GUEST=./target/wasm32-wasip2/release/guest.wasm
PROVIDER=../build/llama-cpp.wasm
HARNESS=./target/harness.wasm

cargo build --manifest-path ./guest/Cargo.toml --target wasm32-wasip2 --release

wac plug "$GUEST" --plug "$PROVIDER" -o "$HARNESS"

cargo run --manifest-path ./runtime/Cargo.toml --release -- --path "$HARNESS"
