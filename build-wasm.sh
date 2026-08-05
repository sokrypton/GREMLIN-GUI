#!/bin/sh
# Rebuild gremlin.wasm from gremlin.c. The .wasm is committed, so users never
# need to run this -- it exists so the binary is reproducible from source.
#
# -ffast-math is deliberately NOT used: reassociation would change the summation
# order and break agreement with the JS reference.
set -e
cd "$(dirname "$0")"
clang --target=wasm32 -O3 -msimd128 -nostdlib \
      -Wl,--no-entry -Wl,--export-dynamic -Wl,--import-memory \
      -Wl,--initial-memory=1048576 \
      -o gremlin.wasm gremlin.c
ls -l gremlin.wasm
